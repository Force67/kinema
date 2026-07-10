// kinema unit tests: synthetic data only, no game assets. Exits non-zero on
// the first failure so it slots into ctest.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "kinema/kinema.h"

namespace {

using namespace kinema;

int failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

#define CHECK_NEAR(a, b, eps) CHECK(std::abs((a) - (b)) <= (eps))

Quat AxisAngle(f32 x, f32 y, f32 z, f32 angle) {
  f32 len = std::sqrt(x * x + y * y + z * z);
  f32 s = std::sin(angle * 0.5f) / (len > 0 ? len : 1.0f);
  return Quat{x * s, y * s, z * s, std::cos(angle * 0.5f)};
}

f32 QuatError(const Quat& a, const Quat& b) {
  f32 dot = std::abs(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
  return 1.0f - std::min(dot, 1.0f);  // 0 = identical orientation
}

// Round-trip: analytic tracks through the builder and sampler.
void TestClipRoundTrip() {
  constexpr u32 kTracks = 8, kFrames = 61;
  constexpr f32 kRate = 30.0f;
  ClipBuilder builder(kTracks, kFrames, kRate);
  auto truth = [](u32 track, f32 time, Vec3* t, Quat* r, f32* s) {
    if (track == 0) {  // constant track
      *t = Vec3{1, 2, 3};
      *r = AxisAngle(0, 0, 1, 0.5f);
      *s = 1.0f;
      return;
    }
    f32 phase = time * (1.0f + 0.3f * static_cast<f32>(track));
    *t = Vec3{50.0f * std::sin(phase), 10.0f * static_cast<f32>(track),
              25.0f * std::cos(phase * 0.7f)};
    *r = AxisAngle(0.2f, 1, 0.1f * static_cast<f32>(track), 2.0f * std::sin(phase * 0.5f));
    *s = 1.0f + 0.25f * std::sin(phase);
  };
  for (u32 f = 0; f < kFrames; ++f) {
    for (u32 track = 0; track < kTracks; ++track) {
      Vec3 t;
      Quat r;
      f32 s;
      truth(track, static_cast<f32>(f) / kRate, &t, &r, &s);
      builder.SetSample(f, track, t, r, s);
    }
  }
  builder.AddEvent("FootLeft", 0.4f);
  builder.AddEvent("FootRight", 1.1f);
  builder.AddRootKey(2.0f, Vec3{0, 100, 0});
  OwnedClip clip(builder.Build());
  CHECK(static_cast<bool>(clip));
  CHECK(clip->num_tracks() == kTracks);
  CHECK_NEAR(clip->duration(), (kFrames - 1) / kRate, 1e-5f);

  PoseArena arena(kTracks, 1);
  PoseView pose = arena.Acquire();
  f32 worst_t = 0, worst_q = 0, worst_s = 0;
  for (int i = 0; i <= 200; ++i) {
    f32 time = clip->duration() * static_cast<f32>(i) / 200.0f;
    clip->Sample(time, pose);
    // Ground truth is the piecewise-linear interpolation of the uniform
    // samples - exactly what the codec stores - so the deltas below measure
    // pure quantization error, not source-curve linearization.
    f32 x = time * kRate;
    u32 k = std::min(static_cast<u32>(x), kFrames - 2);
    f32 a = x - static_cast<f32>(k);
    for (u32 track = 0; track < kTracks; ++track) {
      Vec3 t0, t1;
      Quat r0, r1;
      f32 s0, s1;
      truth(track, static_cast<f32>(k) / kRate, &t0, &r0, &s0);
      truth(track, static_cast<f32>(k + 1) / kRate, &t1, &r1, &s1);
      Vec3 t{t0.x + (t1.x - t0.x) * a, t0.y + (t1.y - t0.y) * a, t0.z + (t1.z - t0.z) * a};
      f32 s = s0 + (s1 - s0) * a;
      worst_t = std::max({worst_t, std::abs(pose.translation[track].x - t.x),
                          std::abs(pose.translation[track].y - t.y),
                          std::abs(pose.translation[track].z - t.z)});
      // Quat ground truth at key times only (nlerp between differs from the
      // codec's component lerp by < quantization for adjacent frames).
      if (a < 1e-4f) worst_q = std::max(worst_q, QuatError(pose.rotation[track], r0));
      worst_s = std::max(worst_s, std::abs(pose.scale[track] - s));
    }
  }
  CHECK(worst_t < 0.005f);  // 16-bit over a 100-unit range
  CHECK(worst_q < 1e-4f);
  CHECK(worst_s < 0.001f);

  // Constant track must be exact (stored full precision).
  clip->Sample(1.234f, pose);
  CHECK(pose.translation[0].x == 1.0f && pose.translation[0].y == 2.0f);

  // Events: range, boundary and wrap semantics.
  int hits = 0;
  clip->EventsInRange(0.0f, 0.5f, [&](const ClipEvent& e) {
    ++hits;
    CHECK(e.name_hash == HashName("FootLeft"));
  });
  CHECK(hits == 1);
  hits = 0;
  clip->EventsInRange(1.5f, 0.5f, [&](const ClipEvent&) { ++hits; });  // wrap
  CHECK(hits == 1);

  // Root motion: linear ramp, delta with wrap.
  Vec3 half = clip->RootTranslation(1.0f);
  CHECK_NEAR(half.y, 50.0f, 1e-3f);
  Vec3 wrap = clip->RootDelta(1.5f, 0.5f);
  CHECK_NEAR(wrap.y, 50.0f, 1.0f);  // 25 to end + 25 past start (duration 2s)

  // Blob survives a copy (relocatable).
  std::vector<u8> copy(clip.bytes());
  auto view = Clip::FromBlob(copy.data(), copy.size());
  CHECK(view.has_value());
  view->Sample(0.5f, pose);
}

void TestBlendKernels() {
  constexpr u32 kBones = 4;
  PoseArena arena(kBones, 4);
  PoseView a = arena.Acquire(), b = arena.Acquire(), dst = arena.Acquire();
  for (u32 i = 0; i < kBones; ++i) {
    a.translation[i] = Vec3{0, 0, 0};
    a.rotation[i] = Quat{};
    a.scale[i] = 1;
    b.translation[i] = Vec3{2, 4, 6};
    b.rotation[i] = AxisAngle(0, 1, 0, 1.0f);
    b.scale[i] = 3;
  }
  BlendPoses(a, b, 0.5f, dst);
  CHECK_NEAR(dst.translation[0].y, 2.0f, 1e-6f);
  CHECK_NEAR(dst.scale[0], 2.0f, 1e-6f);
  CHECK_NEAR(QuatError(dst.rotation[0], AxisAngle(0, 1, 0, 0.5f)), 0.0f, 1e-4f);

  // Double cover: blending toward -q must not go the long way round.
  Quat neg{-b.rotation[0].x, -b.rotation[0].y, -b.rotation[0].z, -b.rotation[0].w};
  for (u32 i = 0; i < kBones; ++i) b.rotation[i] = neg;
  BlendPoses(a, b, 0.5f, dst);
  CHECK_NEAR(QuatError(dst.rotation[0], AxisAngle(0, 1, 0, 0.5f)), 0.0f, 1e-4f);

  // Additive identity at weight 0; full application at weight 1.
  for (u32 i = 0; i < kBones; ++i) {
    b.translation[i] = Vec3{1, 0, 0};
    b.rotation[i] = AxisAngle(1, 0, 0, 0.4f);
    b.scale[i] = 1.5f;
  }
  ApplyAdditive(a, b, 0.0f, dst);
  CHECK_NEAR(dst.translation[0].x, 0.0f, 1e-6f);
  CHECK_NEAR(QuatError(dst.rotation[0], Quat{}), 0.0f, 1e-6f);
  ApplyAdditive(a, b, 1.0f, dst);
  CHECK_NEAR(dst.translation[0].x, 1.0f, 1e-6f);
  CHECK_NEAR(QuatError(dst.rotation[0], AxisAngle(1, 0, 0, 0.4f)), 0.0f, 1e-4f);

  // Masked blend: bone 1 pinned to a.
  f32 mask[kBones] = {1, 0, 1, 1};
  for (u32 i = 0; i < kBones; ++i) b.translation[i] = Vec3{2, 4, 6};
  BlendPosesMasked(a, b, 1.0f, mask, dst);
  CHECK_NEAR(dst.translation[0].y, 4.0f, 1e-6f);
  CHECK_NEAR(dst.translation[1].y, 0.0f, 1e-6f);
}

void TestProgram() {
  constexpr u32 kTracks = 2;
  ClipBuilder ba(kTracks, 2, 30.0f), bb(kTracks, 2, 30.0f);
  for (u32 f = 0; f < 2; ++f) {
    for (u32 t = 0; t < kTracks; ++t) {
      ba.SetSample(f, t, Vec3{0, 0, 0}, Quat{}, 1);
      bb.SetSample(f, t, Vec3{10, 0, 0}, Quat{}, 1);
    }
  }
  OwnedClip ca(ba.Build()), cb(bb.Build());
  PoseArena arena(kTracks, 3);
  PoseOp ops[3];
  ops[0] = {.kind = PoseOp::Kind::kSample, .dst = 0, .clip = ca.get(), .time = 0};
  ops[1] = {.kind = PoseOp::Kind::kSample, .dst = 1, .clip = cb.get(), .time = 0};
  ops[2] = {.kind = PoseOp::Kind::kBlend, .dst = 2, .a = 0, .b = 1, .alpha = 0.25f};
  PoseView out = ExecuteProgram(ops, 3, arena);
  CHECK_NEAR(out.translation[0].x, 2.5f, 1e-5f);
}

void TestInertializer() {
  constexpr u32 kBones = 2;
  PoseArena arena(kBones, 3);
  PoseView from = arena.Acquire(), to = arena.Acquire(), pose = arena.Acquire();
  for (u32 i = 0; i < kBones; ++i) {
    from.translation[i] = Vec3{1, 0, 0};
    from.rotation[i] = AxisAngle(0, 0, 1, 0.6f);
    from.scale[i] = 2;
    to.translation[i] = Vec3{0, 0, 0};
    to.rotation[i] = Quat{};
    to.scale[i] = 1;
  }
  Inertializer inert;
  inert.Init(kBones);
  inert.Begin(from, to, 0.25f);
  // Immediately after the switch the offset restores the old pose...
  CopyPose(to, pose);
  inert.Apply(pose, 0.0f);
  CHECK_NEAR(pose.translation[0].x, 1.0f, 1e-4f);
  CHECK_NEAR(QuatError(pose.rotation[0], from.rotation[0]), 0.0f, 1e-4f);
  // ...decays monotonically...
  f32 prev = 1.0f;
  for (int i = 0; i < 10; ++i) {
    CopyPose(to, pose);
    inert.Apply(pose, 0.025f);
    CHECK(pose.translation[0].x <= prev + 1e-5f);
    prev = pose.translation[0].x;
  }
  // ...and lands exactly on the new pose.
  CopyPose(to, pose);
  bool active = inert.Apply(pose, 1.0f);
  CHECK(!active);
  CHECK_NEAR(pose.translation[0].x, 0.0f, 1e-5f);
}

void TestModelSpace() {
  Skeleton skel;
  skel.parents = {-1, 0, 1};
  skel.name_hashes = {HashName("root"), HashName("mid"), HashName("tip")};
  skel.bind_translation.resize(3);
  skel.bind_rotation.resize(3);
  skel.bind_scale.assign(3, 1.0f);
  PoseArena arena(3, 1);
  PoseView local = arena.Acquire();
  local.translation[0] = Vec3{0, 1, 0};
  local.rotation[0] = AxisAngle(0, 0, 1, 1.57079633f);  // +90 deg about z
  local.scale[0] = 1;
  local.translation[1] = Vec3{1, 0, 0};
  local.rotation[1] = Quat{};
  local.scale[1] = 2;
  local.translation[2] = Vec3{1, 0, 0};
  local.rotation[2] = Quat{};
  local.scale[2] = 1;
  Vec3 mt[3];
  Quat mr[3];
  f32 ms[3];
  ComputeModelSpace(skel, local, mt, mr, ms);
  // mid: root rotates +x into +y.
  CHECK_NEAR(mt[1].x, 0.0f, 1e-5f);
  CHECK_NEAR(mt[1].y, 2.0f, 1e-5f);
  // tip: mid's scale 2 stretches its child offset, still rotated into +y.
  CHECK_NEAR(mt[2].y, 4.0f, 1e-5f);
  CHECK_NEAR(ms[2], 2.0f, 1e-6f);
  CHECK(skel.Find(HashName("tip")) == 2);
}

// A single-track clip that holds a constant translation.x - handy for
// exercising blend selection/weights independent of sampling.
OwnedClip MakeConstClip(f32 val, u32 tracks = 1) {
  ClipBuilder b(tracks, 2, 30.0f);
  for (u32 f = 0; f < 2; ++f)
    for (u32 t = 0; t < tracks; ++t) b.SetSample(f, t, Vec3{val, 0, 0}, Quat{}, 1);
  return OwnedClip(b.Build());
}

void TestBlendSpace1D() {
  OwnedClip c0 = MakeConstClip(0), c1 = MakeConstClip(10), c2 = MakeConstClip(20);
  BlendSpace bs(BlendSpace::Dim::k1D);
  // Add out of order to exercise Finalize's sort.
  bs.Add(c2.get(), 2.0f).Add(c0.get(), 0.0f).Add(c1.get(), 1.0f);
  bs.Finalize();

  PoseArena arena(1, 2);
  PoseView dst = arena.At(0), scratch = arena.At(1);
  auto val = [&](f32 x) {
    EvalBlendSpace(bs, x, 0.0f, 0.0f, dst, scratch);
    return dst.translation[0].x;
  };
  CHECK_NEAR(val(0.0f), 0.0f, 1e-4f);   // at sample
  CHECK_NEAR(val(1.0f), 10.0f, 1e-4f);  // at sample
  CHECK_NEAR(val(0.5f), 5.0f, 1e-4f);   // between
  CHECK_NEAR(val(1.5f), 15.0f, 1e-4f);  // between
  CHECK_NEAR(val(-1.0f), 0.0f, 1e-4f);  // clamped below
  CHECK_NEAR(val(9.0f), 20.0f, 1e-4f);  // clamped above
}

void TestBlendSpace2D() {
  OwnedClip a = MakeConstClip(0), b = MakeConstClip(10), c = MakeConstClip(100);
  BlendSpace bs(BlendSpace::Dim::k2D);
  bs.Add(a.get(), 0.0f, 0.0f).Add(b.get(), 1.0f, 0.0f).Add(c.get(), 0.0f, 1.0f);
  bs.Finalize();

  PoseArena arena(1, 2);
  PoseView dst = arena.At(0), scratch = arena.At(1);
  auto val = [&](f32 x, f32 y) {
    EvalBlendSpace(bs, x, y, 0.0f, dst, scratch);
    return dst.translation[0].x;
  };
  CHECK_NEAR(val(0.0f, 0.0f), 0.0f, 1e-4f);      // pure A
  CHECK_NEAR(val(1.0f, 0.0f), 10.0f, 1e-4f);     // pure B
  CHECK_NEAR(val(0.0f, 1.0f), 100.0f, 1e-4f);    // pure C
  CHECK_NEAR(val(0.5f, 0.0f), 5.0f, 1e-4f);      // A/B midpoint on the A-B edge
  // Outside every band: nearest-sample fallback stays finite and sane.
  f32 outside = val(-5.0f, -5.0f);
  CHECK_NEAR(outside, 0.0f, 1e-4f);
}

void TestAdditiveRoundTrip() {
  constexpr u32 kTracks = 4, kFrames = 31;
  constexpr f32 kRate = 30.0f;
  ClipBuilder sb(kTracks, kFrames, kRate);
  auto truth = [](u32 track, f32 t, Vec3* tr, Quat* r, f32* s) {
    f32 p = t * (0.5f + 0.4f * static_cast<f32>(track));
    *tr = Vec3{3.0f * std::sin(p), 2.0f * static_cast<f32>(track), std::cos(p)};
    *r = AxisAngle(0.3f, 1.0f, 0.2f, 0.8f * std::sin(p));
    *s = 1.0f + 0.1f * std::sin(p);
  };
  for (u32 f = 0; f < kFrames; ++f)
    for (u32 t = 0; t < kTracks; ++t) {
      Vec3 tr;
      Quat r;
      f32 s;
      truth(t, static_cast<f32>(f) / kRate, &tr, &r, &s);
      sb.SetSample(f, t, tr, r, s);
    }
  OwnedClip source(sb.Build());
  OwnedClip additive = MakeAdditiveClipFromFirstFrame(*source.get());
  CHECK(static_cast<bool>(additive));
  CHECK(additive->additive());

  PoseArena arena(kTracks, 4);
  PoseView ref = arena.At(0), add = arena.At(1), truthp = arena.At(2), recon = arena.At(3);
  source->Sample(0.0f, ref);  // reference used for the bake

  f32 worst_t = 0, worst_q = 0;
  for (int i = 0; i <= 30; ++i) {
    f32 time = source->duration() * static_cast<f32>(i) / 30.0f;
    source->Sample(time, truthp);
    additive->Sample(time, add);
    // base + (clip - ref) at weight 1 must reproduce the source pose.
    ApplyAdditive(ref, add, 1.0f, recon);
    for (u32 t = 0; t < kTracks; ++t) {
      worst_t = std::max({worst_t, std::abs(recon.translation[t].x - truthp.translation[t].x),
                          std::abs(recon.translation[t].y - truthp.translation[t].y),
                          std::abs(recon.translation[t].z - truthp.translation[t].z)});
      worst_q = std::max(worst_q, QuatError(recon.rotation[t], truthp.rotation[t]));
    }
  }
  CHECK(worst_t < 0.01f);
  CHECK(worst_q < 1e-3f);

  // Weight 0 must be a no-op (pose == reference).
  additive->Sample(source->duration() * 0.5f, add);
  ApplyAdditive(ref, add, 0.0f, recon);
  for (u32 t = 0; t < kTracks; ++t)
    CHECK_NEAR(recon.translation[t].x, ref.translation[t].x, 1e-5f);
}

void TestBoneMask() {
  Skeleton skel;
  skel.parents = {-1, 0, 1, 0};  // 0=root, 1=child(0), 2=child(1), 3=child(0)
  BoneMask mask(4, 0.0f);
  mask.SetChain(skel, 1, 1.0f, /*descendants=*/true);
  CHECK_NEAR(mask.data()[0], 0.0f, 0.0f);  // root untouched
  CHECK_NEAR(mask.data()[1], 1.0f, 0.0f);  // selected
  CHECK_NEAR(mask.data()[2], 1.0f, 0.0f);  // descendant
  CHECK_NEAR(mask.data()[3], 0.0f, 0.0f);  // sibling branch untouched

  // Drive a masked blend: only masked bones move toward b.
  PoseArena arena(4, 3);
  PoseView a = arena.At(0), b = arena.At(1), dst = arena.At(2);
  for (u32 i = 0; i < 4; ++i) {
    a.translation[i] = Vec3{0, 0, 0};
    a.rotation[i] = Quat{};
    a.scale[i] = 1;
    b.translation[i] = Vec3{8, 0, 0};
    b.rotation[i] = Quat{};
    b.scale[i] = 1;
  }
  BlendPosesMasked(a, b, 1.0f, mask.data(), dst);
  CHECK_NEAR(dst.translation[0].x, 0.0f, 1e-5f);  // root pinned to a
  CHECK_NEAR(dst.translation[1].x, 8.0f, 1e-5f);  // masked -> b
  CHECK_NEAR(dst.translation[2].x, 8.0f, 1e-5f);
  CHECK_NEAR(dst.translation[3].x, 0.0f, 1e-5f);  // sibling pinned
}

void TestSyncGroup() {
  // Walk (1.0s) and run (0.5s), each with two footfall markers. Different
  // absolute times, same marker count -> the group keeps them foot-synced.
  f32 walk[2] = {0.25f, 0.75f};
  f32 run[2] = {0.10f, 0.35f};
  SyncGroup group;
  group.AddClipMarkers(walk, 2, 1.0f);
  group.AddClipMarkers(run, 2, 0.5f);
  group.Reset(0.0f);

  // At phase 0 both sit on their first marker.
  CHECK_NEAR(group.LocalTime(0), 0.25f, 1e-4f);
  CHECK_NEAR(group.LocalTime(1), 0.10f, 1e-4f);

  // Advance halfway to the next marker on the walk (leader) clock: 0.25s of a
  // 0.5s leader segment -> phase 0.5. Both clips are half-way between their
  // markers, so followers are time-scaled to stay in step.
  group.Advance(0.25f, /*leader=*/0);
  CHECK_NEAR(group.phase(), 0.5f, 1e-3f);
  CHECK_NEAR(group.LocalTime(0), 0.5f, 1e-3f);     // 0.25 + 0.5*(0.75-0.25)
  CHECK_NEAR(group.LocalTime(1), 0.225f, 1e-3f);   // 0.10 + 0.5*(0.35-0.10)

  // Advance the rest of the segment: both land exactly on their second marker.
  group.Advance(0.25f, /*leader=*/0);
  CHECK_NEAR(group.phase(), 1.0f, 1e-3f);
  CHECK_NEAR(group.LocalTime(0), 0.75f, 1e-3f);
  CHECK_NEAR(group.LocalTime(1), 0.35f, 1e-3f);
}

void TestProgramParams() {
  // One program, driven entirely by a per-frame parameter block: no ops are
  // rebuilt between evaluations.
  OwnedClip c0 = MakeConstClip(0), c1 = MakeConstClip(10);
  BlendSpace bs(BlendSpace::Dim::k1D);
  bs.Add(c0.get(), 0.0f).Add(c1.get(), 1.0f);
  bs.Finalize();

  PoseArena arena(1, 3);
  PoseOp ops[3];
  ops[0] = {.kind = PoseOp::Kind::kSample, .dst = 0, .clip = c0.get(), .time_param = 0};
  ops[1] = {.kind = PoseOp::Kind::kSample, .dst = 1, .clip = c1.get(), .time_param = 0};
  ops[2] = {.kind = PoseOp::Kind::kBlend, .dst = 2, .a = 0, .b = 1, .alpha_param = 1};

  f32 buf[2];
  PoseParams params{buf, 2};
  buf[0] = 0.0f;
  buf[1] = 0.25f;
  PoseView out = ExecuteProgram(ops, 3, arena, &params);
  CHECK_NEAR(out.translation[0].x, 2.5f, 1e-4f);

  // Same ops, new params -> new result.
  buf[1] = 0.75f;
  arena.Reset();
  out = ExecuteProgram(ops, 3, arena, &params);
  CHECK_NEAR(out.translation[0].x, 7.5f, 1e-4f);

  // Blend-space op driven by a coord param, using register 1 as scratch.
  PoseOp bsops[1];
  bsops[0] = {.kind = PoseOp::Kind::kBlendSpace,
              .dst = 0,
              .a = 1,
              .space = &bs,
              .time_param = 2,
              .coord_param = 3};
  f32 bbuf[4] = {0, 0, 0.0f /*phase*/, 0.5f /*coord*/};
  PoseParams bparams{bbuf, 4};
  arena.Reset();
  out = ExecuteProgram(bsops, 1, arena, &bparams);
  CHECK_NEAR(out.translation[0].x, 5.0f, 1e-4f);
  bbuf[3] = 0.9f;
  arena.Reset();
  out = ExecuteProgram(bsops, 1, arena, &bparams);
  CHECK_NEAR(out.translation[0].x, 9.0f, 1e-4f);
}

// A longer constant clip with a chosen duration (frames at 30 fps), used to
// drive state clocks and exit-time windows deterministically.
OwnedClip MakeClipDur(f32 val, f32 dur, u32 tracks = 1) {
  u32 frames = static_cast<u32>(std::round(dur * 30.0f)) + 1;
  ClipBuilder b(tracks, frames, 30.0f);
  for (u32 f = 0; f < frames; ++f)
    for (u32 t = 0; t < tracks; ++t) b.SetSample(f, t, Vec3{val, 0, 0}, Quat{}, 1);
  return OwnedClip(b.Build());
}

// Constant-pose clip with a linear +x root ramp (speed vx over duration dur).
OwnedClip MakeMoverClip(f32 vx, f32 dur) {
  u32 frames = static_cast<u32>(std::round(dur * 30.0f)) + 1;
  ClipBuilder b(1, frames, 30.0f);
  for (u32 f = 0; f < frames; ++f) b.SetSample(f, 0, Vec3{0, 0, 0}, Quat{}, 1);
  b.AddRootKey(dur * 0.5f, Vec3{vx * dur * 0.5f, 0, 0});
  b.AddRootKey(dur, Vec3{vx * dur, 0, 0});
  return OwnedClip(b.Build());
}

void TestRootMotionLoop() {
  OwnedClip clip = MakeMoverClip(10.0f, 1.0f);  // duration 1s, 10 u/s along +x
  CHECK_NEAR(clip->duration(), 1.0f, 1e-4f);
  CHECK_NEAR(clip->RootTranslation(0.5f).x, 5.0f, 1e-3f);

  // No wrap: straightforward forward advance.
  CHECK_NEAR(clip->RootDeltaLooped(0.2f, 0.3f).x, 3.0f, 1e-3f);
  // Wrap over the loop seam (t1 < t0 modulo duration): 0.8 -> end (2) + 0..0.2 (2).
  CHECK_NEAR(clip->RootDeltaLooped(0.8f, 0.4f).x, 4.0f, 1e-3f);
  CHECK_NEAR(clip->RootDelta(0.8f, 0.2f).x, 4.0f, 1e-3f);  // agree with single-wrap form
  // Multi-loop: dt greater than the duration accumulates whole-loop travel.
  CHECK_NEAR(clip->RootDeltaLooped(0.0f, 2.5f).x, 25.0f, 1e-2f);
  CHECK_NEAR(clip->RootDeltaLooped(0.5f, 3.0f).x, 30.0f, 1e-2f);
}

void TestRangedEvents() {
  // A v1 clip carries no ranged events and still loads.
  OwnedClip plain = MakeClipDur(0.0f, 1.0f);
  CHECK(plain->num_ranged_events() == 0);

  ClipBuilder b(1, 31, 30.0f);
  for (u32 f = 0; f < 31; ++f) b.SetSample(f, 0, Vec3{0, 0, 0}, Quat{}, 1);
  b.AddRangedEvent("Attack", 0.3f, 0.7f);
  b.AddRangedEvent("Early", 0.05f, 0.2f);
  b.AddRangedEvent("Late", 0.8f, 0.95f);
  OwnedClip clip(b.Build());
  CHECK(static_cast<bool>(clip));
  CHECK(clip->num_ranged_events() == 3);

  // Blob relocatability with the v2 ranged block.
  std::vector<u8> copy(clip.bytes());
  auto view = Clip::FromBlob(copy.data(), copy.size());
  CHECK(view.has_value());
  CHECK(view->num_ranged_events() == 3);

  auto phases = [&](f32 t0, f32 t1, u64 want_hash) {
    bool enter = false, active = false, exit = false;
    clip->RangedEventsInRange(t0, t1, [&](const ClipRangedEvent& r, RangePhase p) {
      if (r.name_hash != want_hash) return;
      if (p == RangePhase::kEnter) enter = true;
      if (p == RangePhase::kActive) active = true;
      if (p == RangePhase::kExit) exit = true;
    });
    return std::make_tuple(enter, active, exit);
  };
  const u64 kAttack = HashName("Attack");
  // Enter step: begin crossed, sample inside -> enter + active, no exit.
  auto [e0, a0, x0] = phases(0.2f, 0.4f, kAttack);
  CHECK(e0 && a0 && !x0);
  // Middle step: purely active.
  auto [e1, a1, x1] = phases(0.4f, 0.6f, kAttack);
  CHECK(!e1 && a1 && !x1);
  // Exit step: end crossed, sample past the span -> exit only.
  auto [e2, a2, x2] = phases(0.6f, 0.8f, kAttack);
  CHECK(!e2 && !a2 && x2);

  // Loop wrap over the seam (0.9 -> 0.1): "Late" exits at the end, "Early"
  // enters just after the wrap.
  auto [le, la, lx] = phases(0.9f, 0.1f, HashName("Late"));
  CHECK(!le && !la && lx);
  auto [ee, ea, ex] = phases(0.9f, 0.1f, HashName("Early"));
  CHECK(ee && ea && !ex);
}

// Event sink that tallies routed events for the state machine tests.
struct EventLog {
  int point = 0, enter = 0, active = 0, exit = 0;
  static void OnPoint(void* u, const ClipEvent&) { ++static_cast<EventLog*>(u)->point; }
  static void OnRanged(void* u, const ClipRangedEvent&, RangePhase p) {
    auto* self = static_cast<EventLog*>(u);
    if (p == RangePhase::kEnter) ++self->enter;
    if (p == RangePhase::kActive) ++self->active;
    if (p == RangePhase::kExit) ++self->exit;
  }
};

void TestStateMachine() {
  OwnedClip a = MakeClipDur(0.0f, 1.0f), bClip = MakeClipDur(10.0f, 1.0f);

  // 1) Condition-driven transition fires; inertializer engages exactly at switch.
  {
    StateMachineBuilder sb(1);
    u16 A = sb.AddClipState(a.get());
    u16 B = sb.AddClipState(bClip.get());
    ConditionAtom cond = ConditionAtom::Greater(0, 0.5f);
    sb.AddTransition(A, B, TransitionDesc{0.2f, -1, -1, InterruptPolicy::kWaitForCompletion},
                     &cond, 1);
    StateMachine def = sb.Build();
    StateMachineInstance inst;
    inst.Init(def, A);
    PoseArena arena(1, def.max_registers());
    PoseView out = PoseView{};
    std::vector<Vec3> ot(1);
    std::vector<Quat> orr(1);
    std::vector<f32> os(1);
    out = PoseView{ot.data(), orr.data(), os.data(), 1};

    f32 p = 0.0f;
    PoseParams params{&p, 1};
    arena.Reset();
    inst.Update(0.05f, params, arena, out);
    CHECK(inst.state() == A);
    CHECK(!inst.inertializer().active());

    p = 1.0f;  // condition now true
    arena.Reset();
    inst.Update(0.05f, params, arena, out);
    CHECK(inst.state() == B);
    CHECK(inst.inertializer().active());  // engaged exactly at the switch
    CHECK(inst.transitioning());
  }

  // 2) Exit-time window respected: transition gated to phase >= 0.5.
  {
    StateMachineBuilder sb(1);
    u16 A = sb.AddClipState(a.get());
    u16 B = sb.AddClipState(bClip.get());
    ConditionAtom cond = ConditionAtom::Greater(0, 0.5f);
    sb.AddTransition(A, B, TransitionDesc{0.1f, 0.5f, 1.0f, InterruptPolicy::kWaitForCompletion},
                     &cond, 1);
    StateMachine def = sb.Build();
    StateMachineInstance inst;
    inst.Init(def, A);
    PoseArena arena(1, def.max_registers());
    std::vector<Vec3> ot(1);
    std::vector<Quat> orr(1);
    std::vector<f32> os(1);
    PoseView out{ot.data(), orr.data(), os.data(), 1};
    f32 p = 1.0f;  // condition always satisfied; only the exit window gates
    PoseParams params{&p, 1};
    f32 fire_time = -1.0f;
    for (int i = 0; i < 12 && fire_time < 0; ++i) {
      arena.Reset();
      inst.Update(0.1f, params, arena, out);
      if (inst.state() == B) fire_time = inst.state_time();  // fired this step
    }
    CHECK(fire_time >= 0.0f);
    // The machine was still in A until its phase reached 0.5, so the earliest
    // fire happened at source phase >= 0.5 (state_time of A was >= 0.5).
    // Only assert it did not fire in the first four 0.1s steps (phase < 0.5).
  }

  // Re-run the exit-window case but assert it stays in A while phase < 0.5.
  {
    StateMachineBuilder sb(1);
    u16 A = sb.AddClipState(a.get());
    u16 B = sb.AddClipState(bClip.get());
    ConditionAtom cond = ConditionAtom::Greater(0, 0.5f);
    sb.AddTransition(A, B, TransitionDesc{0.1f, 0.5f, 1.0f, InterruptPolicy::kWaitForCompletion},
                     &cond, 1);
    StateMachine def = sb.Build();
    StateMachineInstance inst;
    inst.Init(def, A);
    PoseArena arena(1, def.max_registers());
    std::vector<Vec3> ot(1);
    std::vector<Quat> orr(1);
    std::vector<f32> os(1);
    PoseView out{ot.data(), orr.data(), os.data(), 1};
    f32 p = 1.0f;
    PoseParams params{&p, 1};
    for (int i = 0; i < 4; ++i) {  // 4 * 0.1 = 0.4 < 0.5, gate closed
      arena.Reset();
      inst.Update(0.1f, params, arena, out);
      CHECK(inst.state() == A);
    }
    arena.Reset();
    inst.Update(0.1f, params, arena, out);  // now phase 0.5, gate opens
    CHECK(inst.state() == B);
  }

  // 3) Trigger consumed exactly once.
  {
    StateMachineBuilder sb(1);
    u16 A = sb.AddClipState(a.get());
    u16 B = sb.AddClipState(bClip.get());
    ConditionAtom cond = ConditionAtom::Trigger(0);
    sb.AddTransition(A, B, TransitionDesc{0.1f}, &cond, 1);
    StateMachine def = sb.Build();
    StateMachineInstance inst;
    inst.Init(def, A);
    PoseArena arena(1, def.max_registers());
    std::vector<Vec3> ot(1);
    std::vector<Quat> orr(1);
    std::vector<f32> os(1);
    PoseView out{ot.data(), orr.data(), os.data(), 1};
    PoseParams params{nullptr, 0};

    arena.Reset();
    inst.Update(0.05f, params, arena, out);
    CHECK(inst.state() == A);  // no trigger yet
    inst.SetTrigger(0);
    CHECK(inst.trigger(0));
    arena.Reset();
    inst.Update(0.05f, params, arena, out);
    CHECK(inst.state() == B);
    CHECK(!inst.trigger(0));  // consumed on fire, auto-reset
  }

  // 4) Interruption policy honored (blocking vs interruptible mid-blend).
  auto build_chain = [&](InterruptPolicy pol) {
    StateMachineBuilder sb(1);
    u16 A = sb.AddClipState(a.get());
    u16 B = sb.AddClipState(bClip.get());
    u16 C = sb.AddClipState(a.get());
    ConditionAtom c0 = ConditionAtom::Greater(0, 0.5f);
    ConditionAtom c1 = ConditionAtom::Greater(1, 0.5f);
    sb.AddTransition(A, B, TransitionDesc{0.5f}, &c0, 1);  // long blend
    sb.AddTransition(B, C, TransitionDesc{0.1f, -1, -1, pol}, &c1, 1);
    return sb.Build();
  };
  {
    StateMachine def = build_chain(InterruptPolicy::kWaitForCompletion);
    StateMachineInstance inst;
    inst.Init(def, 0);
    PoseArena arena(1, def.max_registers());
    std::vector<Vec3> ot(1);
    std::vector<Quat> orr(1);
    std::vector<f32> os(1);
    PoseView out{ot.data(), orr.data(), os.data(), 1};
    f32 buf[2] = {1.0f, 1.0f};  // both conditions true
    PoseParams params{buf, 2};
    arena.Reset();
    inst.Update(0.05f, params, arena, out);  // A -> B (starts a 0.5s blend)
    CHECK(inst.state() == 1);
    arena.Reset();
    inst.Update(0.05f, params, arena, out);  // mid-blend: B->C must wait
    CHECK(inst.state() == 1);
  }
  {
    StateMachine def = build_chain(InterruptPolicy::kInterruptible);
    StateMachineInstance inst;
    inst.Init(def, 0);
    PoseArena arena(1, def.max_registers());
    std::vector<Vec3> ot(1);
    std::vector<Quat> orr(1);
    std::vector<f32> os(1);
    PoseView out{ot.data(), orr.data(), os.data(), 1};
    f32 buf[2] = {1.0f, 1.0f};
    PoseParams params{buf, 2};
    arena.Reset();
    inst.Update(0.05f, params, arena, out);  // A -> B
    CHECK(inst.state() == 1);
    arena.Reset();
    inst.Update(0.05f, params, arena, out);  // mid-blend: B->C interrupts
    CHECK(inst.state() == 2);
  }

  // 5) Forced exit routing: a transition mid-range exits the leaving state's
  // open ranged event.
  {
    ClipBuilder ab(1, 31, 30.0f);
    for (u32 f = 0; f < 31; ++f) ab.SetSample(f, 0, Vec3{0, 0, 0}, Quat{}, 1);
    ab.AddRangedEvent("Guard", 0.05f, 1.0f);  // open across (almost) the whole clip
    OwnedClip guarded(ab.Build());

    StateMachineBuilder sb(1);
    u16 A = sb.AddClipState(guarded.get());
    u16 B = sb.AddClipState(bClip.get());
    ConditionAtom cond = ConditionAtom::Greater(0, 0.5f);
    sb.AddTransition(A, B, TransitionDesc{0.1f}, &cond, 1);
    StateMachine def = sb.Build();
    StateMachineInstance inst;
    inst.Init(def, A);
    PoseArena arena(1, def.max_registers());
    std::vector<Vec3> ot(1);
    std::vector<Quat> orr(1);
    std::vector<f32> os(1);
    PoseView out{ot.data(), orr.data(), os.data(), 1};
    EventLog log;
    EventCallback cb{&log, &EventLog::OnPoint, &EventLog::OnRanged};

    f32 p = 0.0f;
    PoseParams params{&p, 1};
    arena.Reset();
    inst.Update(0.1f, params, arena, out, &cb);  // inside the range, entered
    CHECK(log.enter == 1);
    CHECK(log.exit == 0);
    p = 1.0f;
    arena.Reset();
    inst.Update(0.1f, params, arena, out, &cb);  // fire A->B mid-range
    CHECK(inst.state() == B);
    CHECK(log.exit == 1);  // leaving state's open range force-exited
  }

  // 6) Transition-blended root motion: on the fire frame the delta tracks the
  // exiting (fast) state, then eases toward the entering (slow) state.
  {
    OwnedClip fast = MakeMoverClip(10.0f, 1.0f), slow = MakeMoverClip(2.0f, 1.0f);
    StateMachineBuilder sb(1);
    u16 A = sb.AddClipState(fast.get());
    u16 B = sb.AddClipState(slow.get());
    ConditionAtom cond = ConditionAtom::Greater(0, 0.5f);
    sb.AddTransition(A, B, TransitionDesc{0.4f}, &cond, 1);
    StateMachine def = sb.Build();
    StateMachineInstance inst;
    inst.Init(def, A);
    PoseArena arena(1, def.max_registers());
    std::vector<Vec3> ot(1);
    std::vector<Quat> orr(1);
    std::vector<f32> os(1);
    PoseView out{ot.data(), orr.data(), os.data(), 1};

    f32 p = 0.0f;
    PoseParams params{&p, 1};
    arena.Reset();
    inst.Update(0.05f, params, arena, out);
    CHECK_NEAR(inst.RootMotion().x, 10.0f * 0.05f, 1e-2f);  // pure fast state

    p = 1.0f;
    arena.Reset();
    inst.Update(0.05f, params, arena, out);  // fire: weight 0 -> exiting motion
    CHECK(inst.state() == B);
    CHECK_NEAR(inst.RootMotion().x, 10.0f * 0.05f, 1e-2f);

    // As the blend progresses the delta moves toward the slow state's motion.
    f32 fire_root = inst.RootMotion().x;
    for (int i = 0; i < 4; ++i) {
      arena.Reset();
      inst.Update(0.05f, params, arena, out);
    }
    CHECK(inst.RootMotion().x < fire_root);          // eased down from fast
    CHECK(inst.RootMotion().x >= 2.0f * 0.05f - 1e-2f);  // not below slow motion
  }
}

}  // namespace

int main() {
  TestClipRoundTrip();
  TestBlendKernels();
  TestProgram();
  TestInertializer();
  TestModelSpace();
  TestBlendSpace1D();
  TestBlendSpace2D();
  TestAdditiveRoundTrip();
  TestBoneMask();
  TestSyncGroup();
  TestProgramParams();
  TestRootMotionLoop();
  TestRangedEvents();
  TestStateMachine();
  if (failures == 0) {
    std::printf("kinematest: all passed\n");
    return 0;
  }
  std::printf("kinematest: %d failures\n", failures);
  return 1;
}
