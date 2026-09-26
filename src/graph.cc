// Graph core: additive bake, bone masks, blend spaces and sync groups. These
// extend the compiled-program model - new builder-time structures and flat
// runtime kernels - without touching the clip blob format.

#include <assert.h>
#include <math.h>

#include "kinema/kinema.h"

namespace kinema {
namespace {

inline Quat Normalize(f32 x, f32 y, f32 z, f32 w) {
  f32 len2 = x * x + y * y + z * z + w * w;
  f32 inv = len2 > 1e-12f ? 1.0f / sqrtf(len2) : 0.0f;
  return Quat{x * inv, y * inv, z * inv, w * inv};
}

inline Quat Mul(const Quat& a, const Quat& b) {
  return Quat{a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
              a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
              a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
              a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline Quat Conjugate(const Quat& q) { return Quat{-q.x, -q.y, -q.z, q.w}; }

}  // namespace

// ---------------------------------------------------------------------------
// Additive bake

OwnedClip MakeAdditiveClip(const Clip& source, ConstPoseView reference) {
  const u32 tracks = source.num_tracks();
  const u32 frames = Max(source.num_frames(), 1u);
  const f32 rate = source.frame_rate();
  assert(reference.count == tracks);

  ClipBuilder builder(tracks, frames, rate);
  builder.SetAdditive(true);

  Vector<Vec3> st(tracks);
  Vector<Quat> sr(tracks);
  Vector<f32> ss(tracks);
  PoseView sp{st.data(), sr.data(), ss.data(), tracks};
  for (u32 f = 0; f < frames; ++f) {
    f32 time = rate > 0 ? static_cast<f32>(f) / rate : 0.0f;
    source.Sample(time, sp);
    for (u32 t = 0; t < tracks; ++t) {
      Vec3 dt{st[t].x - reference.translation[t].x, st[t].y - reference.translation[t].y,
              st[t].z - reference.translation[t].z};
      // delta rotation composed on the left of the reference: ref * d = src.
      Quat dr = Mul(Conjugate(reference.rotation[t]), sr[t]);
      dr = Normalize(dr.x, dr.y, dr.z, dr.w);
      f32 rs = reference.scale[t];
      f32 ds = fabsf(rs) > 1e-8f ? ss[t] / rs : ss[t];
      builder.SetSample(f, t, dt, dr, ds);
    }
  }
  return OwnedClip(builder.Build());
}

OwnedClip MakeAdditiveClipFromFirstFrame(const Clip& source) {
  const u32 tracks = source.num_tracks();
  Vector<Vec3> rt(tracks);
  Vector<Quat> rr(tracks);
  Vector<f32> rs(tracks);
  PoseView ref{rt.data(), rr.data(), rs.data(), tracks};
  source.Sample(0.0f, ref);
  return MakeAdditiveClip(source, ref);
}

// ---------------------------------------------------------------------------
// Bone mask

void BoneMask::Init(u32 bones, f32 fill) { weights_.assign(bones, fill); }
void BoneMask::Fill(f32 weight) {
  for (f32& w : weights_) w = weight;
}
void BoneMask::Set(u32 bone, f32 weight) {
  if (bone < weights_.size()) weights_[bone] = weight;
}

void BoneMask::SetChain(const Skeleton& skeleton, u32 bone, f32 weight, bool descendants) {
  if (bone >= weights_.size()) return;
  weights_[bone] = weight;
  if (!descendants) return;
  // Parents precede children, so a single forward sweep marks the whole subtree:
  // any joint whose parent is already in the set joins it.
  const u32 n = skeleton.count();
  Vector<u8> in(n, 0);
  in[bone] = 1;
  for (u32 i = 0; i < n && i < weights_.size(); ++i) {
    int p = skeleton.parents[i];
    if (p >= 0 && in[static_cast<u32>(p)]) {
      in[i] = 1;
      weights_[i] = weight;
    }
  }
}

// ---------------------------------------------------------------------------
// Blend spaces

BlendSpace& BlendSpace::Add(const Clip* clip, f32 x, f32 y) {
  clips_.push_back(clip);
  x_.push_back(x);
  y_.push_back(dim_ == Dim::k2D ? y : 0.0f);
  return *this;
}

void BlendSpace::Finalize() {
  if (dim_ != Dim::k1D || clips_.size() < 2) return;
  // Sort samples ascending by x so bracketing is a linear scan. Samples sharing
  // an x keep their Add order: up to 16 samples that is exactly libstdc++'s
  // std::sort order (a stable insertion sort at that size); past that
  // std::sort left them in unspecified order.
  Vector<u32> order(clips_.size());
  for (u32 i = 0; i < order.size(); ++i) order[i] = i;
  StableSort(order.data(), order.data() + order.size(),
             [this](u32 a, u32 b) { return x_[a] < x_[b]; });
  Vector<const Clip*> c(clips_.size());
  Vector<f32> nx(x_.size()), ny(y_.size());
  for (u32 i = 0; i < order.size(); ++i) {
    c[i] = clips_[order[i]];
    nx[i] = x_[order[i]];
    ny[i] = y_[order[i]];
  }
  clips_.swap(c);
  x_.swap(nx);
  y_.swap(ny);
}

namespace {

void SampleAtPhase(const Clip* clip, f32 phase, PoseView dst) {
  clip->Sample(Clamp(phase, 0.0f, 1.0f) * clip->duration(), dst);
}

}  // namespace

void EvalBlendSpace(const BlendSpace& space, f32 x, f32 y, f32 phase, PoseView dst,
                    PoseView scratch) {
  const u32 n = space.count();
  assert(n > 0);
  if (n == 1) {
    SampleAtPhase(space.clip(0), phase, dst);
    return;
  }

  if (space.dim() == BlendSpace::Dim::k1D) {
    // Samples are sorted by x (Finalize); bracket the coordinate and lerp the
    // two neighbours. Clamp to the end clips outside the range.
    if (x <= space.x(0)) {
      SampleAtPhase(space.clip(0), phase, dst);
      return;
    }
    if (x >= space.x(n - 1)) {
      SampleAtPhase(space.clip(n - 1), phase, dst);
      return;
    }
    u32 i = 0;
    while (i + 1 < n && x > space.x(i + 1)) ++i;
    f32 span = space.x(i + 1) - space.x(i);
    f32 w = span > 1e-8f ? (x - space.x(i)) / span : 0.0f;
    SampleAtPhase(space.clip(i), phase, dst);
    SampleAtPhase(space.clip(i + 1), phase, scratch);
    BlendPoses(dst, scratch, w, dst);
    return;
  }

  // 2D: gradient-band interpolation (Johansen). Chosen over a triangulation so
  // there is no persistent mesh to build/validate and it stays robust for
  // directional strafe layouts (including a centered idle sample); the query is
  // O(clips^2) over a handful of clips and allocates nothing. Each sample's
  // weight is min over the other samples of (1 - projection of the query onto
  // the sample->other direction); weights below zero are dropped, the rest are
  // normalized. Poses are accumulated incrementally so only one scratch
  // register is needed regardless of clip count.
  f32 total = 0.0f;
  for (u32 i = 0; i < n; ++i) {
    f32 wi = 1e30f;
    for (u32 j = 0; j < n; ++j) {
      if (j == i) continue;
      f32 ijx = space.x(j) - space.x(i);
      f32 ijy = space.y(j) - space.y(i);
      f32 isx = x - space.x(i);
      f32 isy = y - space.y(i);
      f32 denom = ijx * ijx + ijy * ijy;
      f32 t = denom > 1e-12f ? (isx * ijx + isy * ijy) / denom : 0.0f;
      wi = Min(wi, 1.0f - t);
    }
    wi = Max(wi, 0.0f);
    if (wi <= 0.0f) continue;
    SampleAtPhase(space.clip(i), phase, scratch);
    if (total <= 0.0f) {
      CopyPose(scratch, dst);
      total = wi;
    } else {
      // Running weighted mean: fold sample i in at its share of the new total.
      BlendPoses(dst, scratch, wi / (total + wi), dst);
      total += wi;
    }
  }
  if (total <= 0.0f) {
    // Query outside every band: fall back to the nearest sample.
    u32 best = 0;
    f32 bestd = 1e30f;
    for (u32 i = 0; i < n; ++i) {
      f32 dx = x - space.x(i), dy = y - space.y(i);
      f32 d = dx * dx + dy * dy;
      if (d < bestd) {
        bestd = d;
        best = i;
      }
    }
    SampleAtPhase(space.clip(best), phase, dst);
  }
}

// ---------------------------------------------------------------------------
// Sync groups

void SyncGroup::Clear() {
  tracks_.clear();
  markers_ = 0;
  phase_ = 0;
}

void SyncGroup::AddClipMarkers(const f32* marker_times, u32 marker_count, f32 duration) {
  assert(marker_count > 0);
  assert(tracks_.empty() || marker_count == markers_);
  Track tr;
  tr.markers.assign(marker_times, marker_times + marker_count);
  // Markers are bare floats, so tied ones are the same time (+0 against -0 at
  // most, equal as times): every correct sort yields the same marker track.
  StableSort(tr.markers.data(), tr.markers.data() + tr.markers.size(),
             [](f32 a, f32 b) { return a < b; });
  tr.duration = duration;
  markers_ = marker_count;
  tracks_.push_back(kinema::move(tr));
}

void SyncGroup::AddClip(const Clip& clip) {
  Vector<f32> times;
  const u32 ne = clip.num_events();
  times.reserve(ne);
  for (u32 i = 0; i < ne; ++i) times.push_back(clip.Event(i).time);
  assert(!times.empty());
  AddClipMarkers(times.data(), static_cast<u32>(times.size()), clip.duration());
}

void SyncGroup::Reset(f32 phase) {
  if (markers_ == 0) {
    phase_ = 0;
    return;
  }
  phase_ = fmodf(phase, static_cast<f32>(markers_));
  if (phase_ < 0) phase_ += static_cast<f32>(markers_);
}

f32 SyncGroup::MarkerTime(const Track& t, f32 g) const {
  const u32 K = markers_;
  if (K == 0) return 0.0f;
  u32 seg = static_cast<u32>(g) % K;
  f32 frac = g - floorf(g);
  f32 t0 = t.markers[seg];
  // The last segment wraps through the clip end back to the first marker.
  f32 t1 = (seg + 1 < K) ? t.markers[seg + 1] : t.duration + t.markers[0];
  f32 local = t0 + frac * (t1 - t0);
  if (t.duration > 1e-6f) {
    local = fmodf(local, t.duration);
    if (local < 0) local += t.duration;
  }
  return local;
}

void SyncGroup::Advance(f32 dt, u32 leader, f32 play_rate) {
  if (markers_ == 0 || leader >= tracks_.size()) return;
  const Track& L = tracks_[leader];
  const u32 K = markers_;
  f32 remaining = dt * play_rate;
  // Walk the leader's clock across marker segments, converting elapsed seconds
  // into phase using each segment's leader-local length. Followers need no
  // integration - their LocalTime is read back at the shared phase.
  int guard = 0;
  while (remaining > 1e-9f && guard++ < 4096) {
    u32 seg = static_cast<u32>(phase_) % K;
    f32 frac = phase_ - floorf(phase_);
    f32 t0 = L.markers[seg];
    f32 t1 = (seg + 1 < K) ? L.markers[seg + 1] : L.duration + L.markers[0];
    f32 seg_len = t1 - t0;
    if (seg_len <= 1e-8f) {  // degenerate segment: step over it
      phase_ = floorf(phase_) + 1.0f;
    } else {
      f32 time_left = seg_len * (1.0f - frac);
      if (remaining < time_left) {
        phase_ += remaining / seg_len;
        remaining = 0.0f;
      } else {
        remaining -= time_left;
        phase_ = floorf(phase_) + 1.0f;
      }
    }
    if (phase_ >= static_cast<f32>(K)) phase_ -= static_cast<f32>(K);
  }
}

f32 SyncGroup::LocalTime(u32 clip) const {
  if (clip >= tracks_.size() || markers_ == 0) return 0.0f;
  return MarkerTime(tracks_[clip], phase_);
}

}  // namespace kinema
