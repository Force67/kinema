// kinema unit tests: synthetic data only, no game assets. Exits non-zero on
// the first failure so it slots into ctest.

#include <math.h>
#include <stdio.h>

#include "kinema/kinema.h"

namespace {

using namespace kinema;

int failures = 0;

#define CHECK(cond)                                                    \
  do {                                                                 \
    if (!(cond)) {                                                     \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

#define CHECK_NEAR(a, b, eps) CHECK(fabsf((a) - (b)) <= (eps))

Quat AxisAngle(f32 x, f32 y, f32 z, f32 angle) {
  f32 len = sqrtf(x * x + y * y + z * z);
  f32 s = sinf(angle * 0.5f) / (len > 0 ? len : 1.0f);
  return Quat{x * s, y * s, z * s, cosf(angle * 0.5f)};
}

f32 QuatError(const Quat& a, const Quat& b) {
  f32 dot = fabsf(a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w);
  return 1.0f - Min(dot, 1.0f);  // 0 = identical orientation
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
    *t = Vec3{50.0f * sinf(phase), 10.0f * static_cast<f32>(track),
              25.0f * cosf(phase * 0.7f)};
    *r = AxisAngle(0.2f, 1, 0.1f * static_cast<f32>(track), 2.0f * sinf(phase * 0.5f));
    *s = 1.0f + 0.25f * sinf(phase);
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
    u32 k = Min(static_cast<u32>(x), kFrames - 2);
    f32 a = x - static_cast<f32>(k);
    for (u32 track = 0; track < kTracks; ++track) {
      Vec3 t0, t1;
      Quat r0, r1;
      f32 s0, s1;
      truth(track, static_cast<f32>(k) / kRate, &t0, &r0, &s0);
      truth(track, static_cast<f32>(k + 1) / kRate, &t1, &r1, &s1);
      Vec3 t{t0.x + (t1.x - t0.x) * a, t0.y + (t1.y - t0.y) * a, t0.z + (t1.z - t0.z) * a};
      f32 s = s0 + (s1 - s0) * a;
      worst_t = Max(Max(Max(worst_t, fabsf(pose.translation[track].x - t.x)),
                        fabsf(pose.translation[track].y - t.y)),
                    fabsf(pose.translation[track].z - t.z));
      // Quat ground truth at key times only (nlerp between differs from the
      // codec's component lerp by < quantization for adjacent frames).
      if (a < 1e-4f) worst_q = Max(worst_q, QuatError(pose.rotation[track], r0));
      worst_s = Max(worst_s, fabsf(pose.scale[track] - s));
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
  Vector<u8> copy(clip.bytes());
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
  ops[0].kind = PoseOp::Kind::kSample; ops[0].dst = 0; ops[0].clip = ca.get(); ops[0].time = 0;
  ops[1].kind = PoseOp::Kind::kSample; ops[1].dst = 1; ops[1].clip = cb.get(); ops[1].time = 0;
  ops[2].kind = PoseOp::Kind::kBlend; ops[2].dst = 2; ops[2].a = 0; ops[2].b = 1; ops[2].alpha = 0.25f;
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
    *tr = Vec3{3.0f * sinf(p), 2.0f * static_cast<f32>(track), cosf(p)};
    *r = AxisAngle(0.3f, 1.0f, 0.2f, 0.8f * sinf(p));
    *s = 1.0f + 0.1f * sinf(p);
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
      worst_t = Max(Max(Max(worst_t, fabsf(recon.translation[t].x - truthp.translation[t].x)),
                        fabsf(recon.translation[t].y - truthp.translation[t].y)),
                    fabsf(recon.translation[t].z - truthp.translation[t].z));
      worst_q = Max(worst_q, QuatError(recon.rotation[t], truthp.rotation[t]));
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
  ops[0].kind = PoseOp::Kind::kSample; ops[0].dst = 0; ops[0].clip = c0.get(); ops[0].time_param = 0;
  ops[1].kind = PoseOp::Kind::kSample; ops[1].dst = 1; ops[1].clip = c1.get(); ops[1].time_param = 0;
  ops[2].kind = PoseOp::Kind::kBlend; ops[2].dst = 2; ops[2].a = 0; ops[2].b = 1; ops[2].alpha_param = 1;

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
  bsops[0].kind = PoseOp::Kind::kBlendSpace;
  bsops[0].dst = 0;
  bsops[0].a = 1;
  bsops[0].space = &bs;
  bsops[0].time_param = 2;
  bsops[0].coord_param = 3;
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
  u32 frames = static_cast<u32>(roundf(dur * 30.0f)) + 1;
  ClipBuilder b(tracks, frames, 30.0f);
  for (u32 f = 0; f < frames; ++f)
    for (u32 t = 0; t < tracks; ++t) b.SetSample(f, t, Vec3{val, 0, 0}, Quat{}, 1);
  return OwnedClip(b.Build());
}

// Constant-pose clip with a linear +x root ramp (speed vx over duration dur).
OwnedClip MakeMoverClip(f32 vx, f32 dur) {
  u32 frames = static_cast<u32>(roundf(dur * 30.0f)) + 1;
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
  Vector<u8> copy(clip.bytes());
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
    struct Phases {
      bool enter, active, exit;
    };
    return Phases{enter, active, exit};
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
    Vector<Vec3> ot(1);
    Vector<Quat> orr(1);
    Vector<f32> os(1);
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
    Vector<Vec3> ot(1);
    Vector<Quat> orr(1);
    Vector<f32> os(1);
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
    Vector<Vec3> ot(1);
    Vector<Quat> orr(1);
    Vector<f32> os(1);
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
    Vector<Vec3> ot(1);
    Vector<Quat> orr(1);
    Vector<f32> os(1);
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
    Vector<Vec3> ot(1);
    Vector<Quat> orr(1);
    Vector<f32> os(1);
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
    Vector<Vec3> ot(1);
    Vector<Quat> orr(1);
    Vector<f32> os(1);
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
    Vector<Vec3> ot(1);
    Vector<Quat> orr(1);
    Vector<f32> os(1);
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
    Vector<Vec3> ot(1);
    Vector<Quat> orr(1);
    Vector<f32> os(1);
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

// ---------------------------------------------------------------------------
// Wave 3: pose tooling.

u64 Fnv64(const Vector<u8>& bytes) {
  u64 h = 14695981039346656037ull;
  for (u8 b : bytes) {
    h ^= b;
    h *= 1099511628211ull;
  }
  return h;
}

// A branched skeleton: 0=root, 1=spine(0), 2=armL(1), 3=armR(1), 4=head(1),
// 5=handL(2). Parents strictly precede children.
Skeleton MakeBranchedSkeleton() {
  Skeleton s;
  s.parents = {-1, 0, 1, 1, 1, 2};
  const char* names[] = {"root", "spine", "armL", "armR", "head", "handL"};
  for (const char* n : names) s.name_hashes.push_back(HashName(n));
  s.bind_translation.assign(6, Vec3{});
  s.bind_rotation.assign(6, Quat{});
  s.bind_scale.assign(6, 1.0f);
  return s;
}

void TestLocalModelRoundTrip() {
  Skeleton s = MakeBranchedSkeleton();
  const u32 n = s.count();
  Vector<Vec3> lt(n);
  Vector<Quat> lr(n);
  Vector<f32> ls(n);
  PoseView local{lt.data(), lr.data(), ls.data(), n};
  for (u32 i = 0; i < n; ++i) {
    local.translation[i] = Vec3{0.5f + 0.3f * i, 1.0f - 0.1f * i, 0.2f * i};
    local.rotation[i] = AxisAngle(0.2f * i + 0.1f, 1.0f, 0.3f, 0.4f + 0.2f * i);
    local.scale[i] = 1.0f + 0.05f * i;
  }
  Vector<Vec3> mt(n);
  Vector<Quat> mr(n);
  Vector<f32> ms(n);
  PoseView model{mt.data(), mr.data(), ms.data(), n};
  LocalToModel(s, local, model);
  Vector<Vec3> bt(n);
  Vector<Quat> br(n);
  Vector<f32> bs(n);
  PoseView back{bt.data(), br.data(), bs.data(), n};
  ModelToLocal(s, model, back);
  for (u32 i = 0; i < n; ++i) {
    CHECK_NEAR(back.translation[i].x, local.translation[i].x, 1e-4f);
    CHECK_NEAR(back.translation[i].y, local.translation[i].y, 1e-4f);
    CHECK_NEAR(back.translation[i].z, local.translation[i].z, 1e-4f);
    CHECK_NEAR(QuatError(back.rotation[i], local.rotation[i]), 0.0f, 1e-5f);
    CHECK_NEAR(back.scale[i], local.scale[i], 1e-4f);
  }
  // ComputeModelSpace (raw-array form) must agree with LocalToModel.
  Vector<Vec3> ct(n);
  Vector<Quat> cr(n);
  Vector<f32> cs(n);
  ComputeModelSpace(s, local, ct.data(), cr.data(), cs.data());
  for (u32 i = 0; i < n; ++i) {
    CHECK_NEAR(ct[i].x, model.translation[i].x, 1e-5f);
    CHECK_NEAR(QuatError(cr[i], model.rotation[i]), 0.0f, 1e-6f);
  }
}

// A simple 4-bone arm chain: 0=root, 1=shoulder, 2=elbow, 3=wrist, straight
// along +x with unit segment lengths.
Skeleton MakeArm() {
  Skeleton s;
  s.parents = {-1, 0, 1, 2};
  s.name_hashes = {HashName("root"), HashName("shoulder"), HashName("elbow"), HashName("wrist")};
  s.bind_translation.assign(4, Vec3{});
  s.bind_rotation.assign(4, Quat{});
  s.bind_scale.assign(4, 1.0f);
  return s;
}

void SetArmPose(PoseView local) {
  local.translation[0] = Vec3{0, 0, 0};
  local.translation[1] = Vec3{0, 0, 0};  // shoulder at root
  local.translation[2] = Vec3{1, 0, 0};  // upper arm length 1
  local.translation[3] = Vec3{1, 0, 0};  // forearm length 1
  for (u32 i = 0; i < 4; ++i) {
    local.rotation[i] = Quat{};
    local.scale[i] = 1.0f;
  }
}

void TestTwoBoneIK() {
  Skeleton s = MakeArm();
  PoseArena arena(4, 2);
  auto solve_and_model = [&](Vec3 target, Vec3 pole, f32 soft, f32 weight, PoseView out_model) {
    PoseView local = arena.At(0);
    SetArmPose(local);
    PoseView model = arena.At(1);
    LocalToModel(s, local, model);
    TwoBoneIKSolve ik;
    ik.root_joint = 1;
    ik.mid_joint = 2;
    ik.end_joint = 3;
    ik.target = target;
    ik.pole = pole;
    ik.soft = soft;
    ik.weight = weight;
    SolveTwoBoneIK(s, model, local, ik);
    LocalToModel(s, local, out_model);
  };
  Vector<Vec3> mt(4);
  Vector<Quat> mr(4);
  Vector<f32> ms(4);
  PoseView out{mt.data(), mr.data(), ms.data(), 4};

  // Reachable target: wrist lands on it, elbow bends toward the pole (+z).
  solve_and_model(Vec3{1, 1, 0}, Vec3{0.5f, 0.5f, 2.0f}, 0.0f, 1.0f, out);
  CHECK_NEAR(out.translation[3].x, 1.0f, 2e-3f);
  CHECK_NEAR(out.translation[3].y, 1.0f, 2e-3f);
  CHECK_NEAR(out.translation[3].z, 0.0f, 2e-3f);
  CHECK(out.translation[2].z > 0.05f);  // elbow pushed toward +z pole

  // Pole on the other side flips the elbow.
  solve_and_model(Vec3{1, 1, 0}, Vec3{0.5f, 0.5f, -2.0f}, 0.0f, 1.0f, out);
  CHECK(out.translation[2].z < -0.05f);

  // Out-of-reach target clamps at full extension along the target direction.
  solve_and_model(Vec3{5, 0, 0}, Vec3{0, 0, 1}, 0.0f, 1.0f, out);
  f32 reach = sqrtf(out.translation[3].x * out.translation[3].x +
                    out.translation[3].y * out.translation[3].y +
                    out.translation[3].z * out.translation[3].z);
  CHECK_NEAR(reach, 2.0f, 5e-3f);  // both segments straight
  CHECK_NEAR(out.translation[3].x, 2.0f, 5e-3f);

  // weight = 0 leaves the FK pose; weight = 0.5 moves partway to the target.
  solve_and_model(Vec3{1, 1, 0}, Vec3{0, 0, 1}, 0.0f, 0.0f, out);
  f32 err0 = fabsf(out.translation[3].x - 1.0f) + fabsf(out.translation[3].y - 1.0f);
  CHECK(err0 > 0.9f);  // still near the straight FK pose (2,0,0)
  solve_and_model(Vec3{1, 1, 0}, Vec3{0, 0, 1}, 0.0f, 0.5f, out);
  f32 err5 = fabsf(out.translation[3].x - 1.0f) + fabsf(out.translation[3].y - 1.0f);
  CHECK(err5 < err0);  // moved toward the target
}

// Serial aim chain: 0=root, 1,2,3 stacked along +y, each parent of the next.
Skeleton MakeSpine() {
  Skeleton s;
  s.parents = {-1, 0, 1, 2};
  s.name_hashes = {HashName("root"), HashName("s0"), HashName("s1"), HashName("head")};
  s.bind_translation.assign(4, Vec3{});
  s.bind_rotation.assign(4, Quat{});
  s.bind_scale.assign(4, 1.0f);
  return s;
}

void TestLookAt() {
  // Single joint: aim local +z at a target, then clamp to a cone.
  Skeleton s;
  s.parents = {-1};
  s.name_hashes = {HashName("aim")};
  auto forward_of = [](const Quat& q) {
    return Vec3{2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x),
                1 - 2 * (q.x * q.x + q.y * q.y)};  // rotate {0,0,1}
  };
  {
    Vector<Vec3> lt(1), mt(1);
    Vector<Quat> lr(1), mr(1);
    Vector<f32> lsc(1), msc(1);
    PoseView local{lt.data(), lr.data(), lsc.data(), 1};
    local.translation[0] = Vec3{0, 0, 0};
    local.rotation[0] = Quat{};
    local.scale[0] = 1.0f;
    PoseView model{mt.data(), mr.data(), msc.data(), 1};
    LocalToModel(s, local, model);
    LookAtSolve la;
    la.joint = 0;
    la.target = Vec3{1, 0, 0};
    la.forward = Vec3{0, 0, 1};
    la.weight = 1.0f;
    SolveLookAt(s, model, local, la);
    LocalToModel(s, local, model);
    Vec3 fwd = forward_of(model.rotation[0]);
    CHECK_NEAR(fwd.x, 1.0f, 1e-3f);  // now points +x
    CHECK_NEAR(fwd.z, 0.0f, 1e-3f);
  }
  {
    Vector<Vec3> lt(1), mt(1);
    Vector<Quat> lr(1), mr(1);
    Vector<f32> lsc(1), msc(1);
    PoseView local{lt.data(), lr.data(), lsc.data(), 1};
    local.translation[0] = Vec3{0, 0, 0};
    local.rotation[0] = Quat{};
    local.scale[0] = 1.0f;
    PoseView model{mt.data(), mr.data(), msc.data(), 1};
    LocalToModel(s, local, model);
    LookAtSolve la;
    la.joint = 0;
    la.target = Vec3{1, 0, 0};  // wants 90 deg
    la.forward = Vec3{0, 0, 1};
    la.max_angle = 0.2f;  // but clamp to 0.2 rad
    la.weight = 1.0f;
    SolveLookAt(s, model, local, la);
    // Angle of the applied rotation about its axis is the clamp value.
    Quat q = local.rotation[0];
    f32 angle = 2.0f * acosf(Min(1.0f, fabsf(q.w)));
    CHECK_NEAR(angle, 0.2f, 1e-3f);
  }

  // N-joint chain: fractions summing to 1 rotate each joint by its share and
  // land the tip on the full aim.
  Skeleton sp = MakeSpine();
  const u32 n = 4;
  Vector<Vec3> lt(n), mt(n);
  Vector<Quat> lr(n), mr(n);
  Vector<f32> lsc(n), msc(n);
  PoseView local{lt.data(), lr.data(), lsc.data(), n};
  local.translation[0] = Vec3{0, 0, 0};
  local.translation[1] = Vec3{0, 0, 0};
  local.translation[2] = Vec3{0, 1, 0};
  local.translation[3] = Vec3{0, 1, 0};
  for (u32 i = 0; i < n; ++i) {
    local.rotation[i] = Quat{};
    local.scale[i] = 1.0f;
  }
  PoseView model{mt.data(), mr.data(), msc.data(), n};
  LocalToModel(sp, local, model);
  Vec3 tip_pos = model.translation[3];
  Vec3 dir{10.0f - tip_pos.x, 2.0f - tip_pos.y, 0.0f - tip_pos.z};
  f32 dl = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
  dir = Vec3{dir.x / dl, dir.y / dl, dir.z / dl};

  u32 joints[3] = {1, 2, 3};
  f32 fracs[3] = {0.5f, 0.3f, 0.2f};
  LookAtChainSolve lc;
  lc.joints = joints;
  lc.fractions = fracs;
  lc.count = 3;
  lc.target = Vec3{10, 2, 0};
  lc.forward = Vec3{0, 0, 1};
  lc.weight = 1.0f;
  SolveLookAtChain(sp, model, local, lc);
  // Each joint rotated by frac * total_angle (total = 90 deg here).
  f32 total = 1.57079633f;  // +z to +x
  for (u32 i = 0; i < 3; ++i) {
    Quat q = local.rotation[joints[i]];
    f32 angle = 2.0f * acosf(Min(1.0f, fabsf(q.w)));
    CHECK_NEAR(angle, fracs[i] * total, 5e-3f);
  }
  // Tip forward now points along the original aim direction.
  LocalToModel(sp, local, model);
  Quat q = model.rotation[3];
  Vec3 fwd{2 * (q.x * q.z + q.w * q.y), 2 * (q.y * q.z - q.w * q.x),
           1 - 2 * (q.x * q.x + q.y * q.y)};
  CHECK_NEAR(fwd.x, dir.x, 5e-3f);
  CHECK_NEAR(fwd.y, dir.y, 5e-3f);
  CHECK_NEAR(fwd.z, dir.z, 5e-3f);
}

void TestFootPlacement() {
  // Pelvis (root) with two legs: hipL/kneeL/ankleL and hipR/kneeR/ankleR.
  Skeleton s;
  s.parents = {-1, 0, 1, 2, 0, 4, 5};
  //           pel hipL kneeL ankL hipR kneeR ankR
  const char* names[] = {"pelvis", "hipL", "kneeL", "ankleL", "hipR", "kneeR", "ankleR"};
  for (const char* nm : names) s.name_hashes.push_back(HashName(nm));
  s.bind_translation.assign(7, Vec3{});
  s.bind_rotation.assign(7, Quat{});
  s.bind_scale.assign(7, 1.0f);
  const u32 n = 7;
  Vector<Vec3> lt(n), mt(n);
  Vector<Quat> lr(n);
  Vector<f32> ls(n), ms(n);
  Vector<Quat> mr(n);
  PoseView local{lt.data(), lr.data(), ls.data(), n};
  // Pelvis at height 2; legs hang straight down (-y), each segment length 1.
  local.translation[0] = Vec3{0, 2, 0};       // pelvis
  local.translation[1] = Vec3{-0.5f, 0, 0};   // hipL offset
  local.translation[2] = Vec3{0, -1, 0};      // kneeL
  local.translation[3] = Vec3{0, -1, 0};      // ankleL
  local.translation[4] = Vec3{0.5f, 0, 0};    // hipR offset
  local.translation[5] = Vec3{0, -1, 0};      // kneeR
  local.translation[6] = Vec3{0, -1, 0};      // ankleR
  for (u32 i = 0; i < n; ++i) {
    local.rotation[i] = Quat{};
    local.scale[i] = 1.0f;
  }
  PoseView model{mt.data(), mr.data(), ms.data(), n};

  FootLimb feet[2];
  feet[0].hip = 1;
  feet[0].knee = 2;
  feet[0].ankle = 3;
  feet[1].hip = 4;
  feet[1].knee = 5;
  feet[1].ankle = 6;
  // Ankles currently at y=0. Left ground at y=-0.3, right ground at y=-0.1.
  FootHit hits[2];
  hits[0].point = Vec3{-0.5f, -0.3f, 0};
  hits[0].normal = Vec3{0, 1, 0};
  hits[0].valid = true;
  hits[1].point = Vec3{0.5f, -0.1f, 0};
  hits[1].normal = Vec3{0, 1, 0};
  hits[1].valid = true;

  FootPlacementSolve fp;
  fp.pelvis = 0;
  fp.feet = feet;
  fp.hits = hits;
  fp.foot_count = 2;
  fp.up = Vec3{0, 1, 0};
  fp.ankle_height = 0.0f;
  fp.weight = 1.0f;
  f32 offset = SolveFootPlacement(s, local, model, fp);
  // Lowest-foot rule: pelvis drops by the deeper (left) contact, -0.3.
  CHECK_NEAR(offset, -0.3f, 1e-4f);
  // Both ankles end up on their ground contacts.
  LocalToModel(s, local, model);
  CHECK_NEAR(model.translation[3].y, -0.3f, 5e-3f);
  CHECK_NEAR(model.translation[6].y, -0.1f, 5e-3f);
}

// Name lookup for BuildMirrorTable.
struct NameList {
  const char** names;
};
StringView NameOf(void* user, u32 bone) {
  return static_cast<NameList*>(user)->names[bone];
}

void TestMirror() {
  // 0=root(center), 1=L, 2=R, plus 3=center2.
  MirrorTable table;
  table.Init(4, MirrorTable::Axis::kX);
  table.Pair(1, 2, MirrorTable::Axis::kX);
  // 0 and 3 stay self-paired (centerline).

  Vector<Vec3> st(4), dt(4), d2t(4);
  Vector<Quat> sr(4), dr(4), d2r(4);
  Vector<f32> ss(4), ds(4), d2s(4);
  PoseView src{st.data(), sr.data(), ss.data(), 4};
  for (u32 i = 0; i < 4; ++i) {
    src.translation[i] = Vec3{1.0f + i, 2.0f - i, 0.5f * i};
    src.rotation[i] = AxisAngle(0.2f, 0.3f, 1.0f, 0.4f + 0.1f * i);
    src.scale[i] = 1.0f + 0.1f * i;
  }
  PoseView dst{dt.data(), dr.data(), ds.data(), 4};
  MirrorPose(table, src, dst);
  // Asymmetric pose lands on the mirrored bone: dst[1] == flip(src[2]).
  CHECK_NEAR(dst.translation[1].x, -src.translation[2].x, 1e-5f);
  CHECK_NEAR(dst.translation[1].y, src.translation[2].y, 1e-5f);
  CHECK_NEAR(dst.rotation[1].x, src.rotation[2].x, 1e-5f);
  CHECK_NEAR(dst.rotation[1].y, -src.rotation[2].y, 1e-5f);

  // Mirror twice == identity.
  PoseView d2{d2t.data(), d2r.data(), d2s.data(), 4};
  MirrorPose(table, dst, d2);
  for (u32 i = 0; i < 4; ++i) {
    CHECK_NEAR(d2.translation[i].x, src.translation[i].x, 1e-5f);
    CHECK_NEAR(d2.translation[i].y, src.translation[i].y, 1e-5f);
    CHECK_NEAR(d2.translation[i].z, src.translation[i].z, 1e-5f);
    CHECK_NEAR(QuatError(d2.rotation[i], src.rotation[i]), 0.0f, 1e-6f);
    CHECK_NEAR(d2.scale[i], src.scale[i], 1e-6f);
  }

  // Auto-build from names, no engine naming baked in.
  const char* names[4] = {"Root", "L_Arm", "R_Arm", "Spine"};
  NameList nl{names};
  MirrorTable built;
  BuildMirrorTable(built, 4, &NameOf, &nl, "L_", "R_", MirrorTable::Axis::kX);
  CHECK(built.Partner(1) == 2);
  CHECK(built.Partner(2) == 1);
  CHECK(built.Partner(0) == 0);  // centerline
  CHECK(built.Partner(3) == 3);
}

void TestRetarget() {
  Skeleton src;
  src.parents = {-1, 0, 1};
  src.name_hashes = {HashName("a"), HashName("b"), HashName("c")};
  src.bind_translation = {Vec3{0, 0, 0}, Vec3{1, 0, 0}, Vec3{1, 0, 0}};
  src.bind_rotation.assign(3, Quat{});
  src.bind_scale.assign(3, 1.0f);

  u32 map_s[3] = {0, 1, 2};
  u32 map_t[3] = {0, 1, 2};

  // Identity: same skeleton -> pose passes through unchanged.
  {
    RetargetTable rt;
    rt.Build(src, src, map_s, map_t, 3);
    Vector<Vec3> st(3), tt(3);
    Vector<Quat> sr(3), tr(3);
    Vector<f32> ssc(3), tsc(3);
    PoseView sp{st.data(), sr.data(), ssc.data(), 3};
    for (u32 i = 0; i < 3; ++i) {
      sp.translation[i] = Vec3{static_cast<f32>(i), 0.5f, -0.3f};
      sp.rotation[i] = AxisAngle(0, 0, 1, 0.3f + 0.1f * i);
      sp.scale[i] = 1.0f;
    }
    PoseView tp{tt.data(), tr.data(), tsc.data(), 3};
    RetargetPose(rt, sp, tp);
    for (u32 i = 0; i < 3; ++i) {
      CHECK_NEAR(tp.translation[i].x, sp.translation[i].x, 1e-5f);
      CHECK_NEAR(QuatError(tp.rotation[i], sp.rotation[i]), 0.0f, 1e-6f);
    }
  }

  // Scaled target skeleton (2x bone offsets) -> translations scale by the
  // per-bone reference-length ratio, rotations pass through.
  {
    Skeleton tgt = src;
    for (u32 i = 0; i < 3; ++i) tgt.bind_translation[i] = Vec3{src.bind_translation[i].x * 2.0f,
                                                               src.bind_translation[i].y * 2.0f,
                                                               src.bind_translation[i].z * 2.0f};
    RetargetTable rt;
    rt.Build(src, tgt, map_s, map_t, 3);
    Vector<Vec3> st(3), tt(3);
    Vector<Quat> sr(3), tr(3);
    Vector<f32> ssc(3), tsc(3);
    PoseView sp{st.data(), sr.data(), ssc.data(), 3};
    for (u32 i = 0; i < 3; ++i) {
      sp.translation[i] = src.bind_translation[i];  // at bind
      sp.rotation[i] = AxisAngle(0, 1, 0, 0.5f);
      sp.scale[i] = 1.0f;
    }
    PoseView tp{tt.data(), tr.data(), tsc.data(), 3};
    RetargetPose(rt, sp, tp);
    // Bone 1 offset was (1,0,0); target proportion 2x -> (2,0,0).
    CHECK_NEAR(tp.translation[1].x, 2.0f, 1e-4f);
    CHECK_NEAR(QuatError(tp.rotation[1], sp.rotation[1]), 0.0f, 1e-6f);
  }
}

void TestCurves() {
  constexpr u32 kFrames = 31;
  constexpr f32 kRate = 30.0f;
  auto jaw = [](f32 t) { return 0.5f + 0.4f * sinf(t * 3.0f); };
  auto blink = [](f32 t) { return 0.2f * t; };
  ClipBuilder b(2, kFrames, kRate);
  u16 c_jaw = b.AddCurve("Jaw");
  u16 c_blink = b.AddCurve("Blink");
  for (u32 f = 0; f < kFrames; ++f) {
    f32 t = static_cast<f32>(f) / kRate;
    for (u32 tr = 0; tr < 2; ++tr) b.SetSample(f, tr, Vec3{}, Quat{}, 1.0f);
    b.SetCurveSample(f, c_jaw, jaw(t));
    b.SetCurveSample(f, c_blink, blink(t));
  }
  OwnedClip clip(b.Build());
  CHECK(static_cast<bool>(clip));
  CHECK(clip->num_curves() == 2);
  CHECK(clip->FindCurve(HashName("Jaw")) >= 0);
  CHECK(clip->FindCurve(HashName("Blink")) >= 0);
  CHECK(clip->FindCurve(HashName("Nope")) < 0);

  // Round-trip within quantization (piecewise-linear ground truth).
  f32 worst = 0;
  for (int i = 0; i <= 60; ++i) {
    f32 time = clip->duration() * static_cast<f32>(i) / 60.0f;
    f32 x = time * kRate;
    u32 k = Min(static_cast<u32>(x), kFrames - 2);
    f32 a = x - static_cast<f32>(k);
    f32 truth = jaw(static_cast<f32>(k) / kRate) * (1 - a) + jaw(static_cast<f32>(k + 1) / kRate) * a;
    worst = Max(worst, fabsf(clip->SampleCurve(HashName("Jaw"), time) - truth));
  }
  CHECK(worst < 2e-4f);  // 16-bit over the curve's range
  CHECK_NEAR(clip->SampleCurve(HashName("Missing"), 0.5f, -1.0f), -1.0f, 0.0f);  // fallback

  // Blob relocatability with the v3 curve block.
  Vector<u8> copy(clip.bytes());
  auto view = Clip::FromBlob(copy.data(), copy.size());
  CHECK(view.has_value());
  CHECK(view->num_curves() == 2);

  // v1/v2 clips load with zero curves.
  OwnedClip plain = MakeClipDur(0.0f, 1.0f);
  CHECK(plain->num_curves() == 0);
  ClipBuilder rb(1, 31, 30.0f);
  for (u32 f = 0; f < 31; ++f) rb.SetSample(f, 0, Vec3{}, Quat{}, 1.0f);
  rb.AddRangedEvent("Attack", 0.3f, 0.7f);  // v2
  OwnedClip ranged(rb.Build());
  CHECK(ranged->num_curves() == 0);
  CHECK(ranged->num_ranged_events() == 1);

  // Curves and ranged events coexist in one v3 clip.
  ClipBuilder cb(1, 31, 30.0f);
  for (u32 f = 0; f < 31; ++f) {
    cb.SetSample(f, 0, Vec3{}, Quat{}, 1.0f);
  }
  u16 w = cb.AddCurve("Weight");
  for (u32 f = 0; f < 31; ++f) cb.SetCurveSample(f, w, static_cast<f32>(f) / 30.0f);
  cb.AddRangedEvent("Guard", 0.1f, 0.9f);
  OwnedClip both(cb.Build());
  CHECK(both->num_curves() == 1);
  CHECK(both->num_ranged_events() == 1);
  CHECK_NEAR(both->SampleCurve(HashName("Weight"), 0.5f), 0.5f, 2e-3f);
}

// Curve-free golden blobs: their byte layout must never change when the format
// grows. Regenerate with tools/golden.cc if the layout intentionally changes.
constexpr u64 kGoldenV1 = 0x3fdff5e4d9fd7f10ull;
constexpr u64 kGoldenV2 = 0x180ba2330006ffc4ull;

Vector<u8> BuildGolden(bool with_ranged) {
  ClipBuilder b(6, 41, 30.0f);
  for (u32 f = 0; f < 41; ++f) {
    f32 t = static_cast<f32>(f) / 30.0f;
    for (u32 k = 0; k < 6; ++k) {
      Vec3 tr;
      Quat r;
      f32 s = 1.0f;
      if (k == 0) {
        tr = Vec3{1.0f, 2.0f, 3.0f};
        r = Quat{0.0f, 0.0f, 0.1986693f, 0.9800666f};
      } else {
        f32 p = t * (0.5f + 0.3f * static_cast<f32>(k));
        tr = Vec3{4.0f * (p - static_cast<f32>(static_cast<int>(p))), 2.0f * static_cast<f32>(k),
                  1.5f};
        f32 half = 0.5f * (0.3f * p);
        r = Quat{0.0f, 0.0f, half, 1.0f - 0.5f * half * half};
        s = 1.0f + 0.1f * (p - static_cast<f32>(static_cast<int>(p)));
      }
      b.SetSample(f, k, tr, r, s);
    }
  }
  b.AddEvent("FootLeft", 0.4f);
  b.AddEvent("FootRight", 1.1f);
  if (with_ranged) b.AddRangedEvent("Attack", 0.3f, 0.7f);
  b.AddRootKey(1.0f, Vec3{0.0f, 40.0f, 0.0f});
  return b.Build();
}

void TestGoldenByteCompat() {
  Vector<u8> v1 = BuildGolden(false);
  Vector<u8> v2 = BuildGolden(true);
  CHECK(Fnv64(v1) == kGoldenV1);  // curve-free v1 layout unchanged
  CHECK(Fnv64(v2) == kGoldenV2);  // curve-free v2 layout unchanged
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
  TestLocalModelRoundTrip();
  TestTwoBoneIK();
  TestLookAt();
  TestFootPlacement();
  TestMirror();
  TestRetarget();
  TestCurves();
  TestGoldenByteCompat();
  if (failures == 0) {
    printf("kinematest: all passed\n");
    return 0;
  }
  printf("kinematest: %d failures\n", failures);
  return 1;
}
