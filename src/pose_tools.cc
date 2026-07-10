// Pose tooling: model-space conversion plus the constraint kernels that build on
// it (two-bone IK, look-at, foot placement, mirroring, retargeting). Every
// solver reads a model-space pose the caller produced with LocalToModel and
// writes the result back to local space. All are flat, allocation-free kernels.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>

#include "kinema/kinema.h"

namespace kinema {
namespace {

inline Quat Normalize(const Quat& q) {
  f32 len2 = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  f32 inv = len2 > 1e-12f ? 1.0f / std::sqrt(len2) : 0.0f;
  return Quat{q.x * inv, q.y * inv, q.z * inv, q.w * inv};
}

inline Quat Mul(const Quat& a, const Quat& b) {
  return Quat{a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
              a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
              a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
              a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}

inline Quat Conjugate(const Quat& q) { return Quat{-q.x, -q.y, -q.z, q.w}; }

// Rotate a vector by a unit quaternion.
inline Vec3 Rotate(const Quat& q, const Vec3& v) {
  Vec3 u{q.x, q.y, q.z};
  f32 s = q.w;
  f32 uv = u.x * v.x + u.y * v.y + u.z * v.z;
  f32 uu = u.x * u.x + u.y * u.y + u.z * u.z;
  Vec3 cross{u.y * v.z - u.z * v.y, u.z * v.x - u.x * v.z, u.x * v.y - u.y * v.x};
  return Vec3{2.0f * uv * u.x + (s * s - uu) * v.x + 2.0f * s * cross.x,
              2.0f * uv * u.y + (s * s - uu) * v.y + 2.0f * s * cross.y,
              2.0f * uv * u.z + (s * s - uu) * v.z + 2.0f * s * cross.z};
}

inline Vec3 Sub(const Vec3& a, const Vec3& b) { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 Add(const Vec3& a, const Vec3& b) { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 Scale(const Vec3& a, f32 s) { return Vec3{a.x * s, a.y * s, a.z * s}; }
inline f32 Dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 Cross(const Vec3& a, const Vec3& b) {
  return Vec3{a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
inline f32 Length(const Vec3& a) { return std::sqrt(Dot(a, a)); }
inline Vec3 Normalize3(const Vec3& a) {
  f32 l = Length(a);
  return l > 1e-8f ? Scale(a, 1.0f / l) : Vec3{};
}

inline Quat AxisAngle(const Vec3& axis_unit, f32 angle) {
  f32 s = std::sin(angle * 0.5f);
  return Quat{axis_unit.x * s, axis_unit.y * s, axis_unit.z * s, std::cos(angle * 0.5f)};
}

// Axis*angle log/exp of unit quaternions (shortest arc).
inline Vec3 Log(const Quat& q_in) {
  Quat q = q_in.w < 0 ? Quat{-q_in.x, -q_in.y, -q_in.z, -q_in.w} : q_in;
  f32 len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z);
  if (len < 1e-8f) return Vec3{};
  f32 angle = 2.0f * std::atan2(len, q.w);
  f32 s = angle / len;
  return Vec3{q.x * s, q.y * s, q.z * s};
}

inline Quat Exp(const Vec3& v) {
  f32 angle = Length(v);
  if (angle < 1e-8f) return Quat{};
  f32 s = std::sin(angle * 0.5f) / angle;
  return Quat{v.x * s, v.y * s, v.z * s, std::cos(angle * 0.5f)};
}

// Shortest-arc rotation carrying unit vector a onto unit vector b.
inline Quat FromTo(const Vec3& a, const Vec3& b) {
  f32 d = Dot(a, b);
  if (d > 0.999999f) return Quat{};
  if (d < -0.999999f) {
    Vec3 axis = Cross(Vec3{1, 0, 0}, a);
    if (Length(axis) < 1e-6f) axis = Cross(Vec3{0, 1, 0}, a);
    return AxisAngle(Normalize3(axis), 3.14159265358979f);
  }
  Vec3 c = Cross(a, b);
  return Normalize(Quat{c.x, c.y, c.z, 1.0f + d});
}

// Clamp a rotation's magnitude to `max_angle` radians about its own axis.
inline Quat ClampAngle(const Quat& q, f32 max_angle) {
  Vec3 v = Log(q);
  f32 a = Length(v);
  if (a <= max_angle || a < 1e-8f) return q.w < 0 ? Quat{-q.x, -q.y, -q.z, -q.w} : q;
  return Exp(Scale(v, max_angle / a));
}

inline Quat Nlerp(const Quat& a, const Quat& b, f32 t) {
  f32 dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  f32 sign = dot >= 0 ? 1.0f : -1.0f;
  return Normalize(Quat{a.x + (b.x * sign - a.x) * t, a.y + (b.y * sign - a.y) * t,
                        a.z + (b.z * sign - a.z) * t, a.w + (b.w * sign - a.w) * t});
}

inline Quat ParentModelRot(const Skeleton& sk, ConstPoseView model, u32 bone) {
  int p = sk.parents[bone];
  return p < 0 ? Quat{} : model.rotation[p];
}

}  // namespace

// ---------------------------------------------------------------------------
// Model-space conversion

void LocalToModel(const Skeleton& skeleton, ConstPoseView local, PoseView model) {
  const u32 n = skeleton.count();
  assert(local.count == n && model.count == n);
  for (u32 i = 0; i < n; ++i) {
    int p = skeleton.parents[i];
    // Parents must precede children so the parent's model transform is ready.
    assert(p < static_cast<int>(i));
    if (p < 0) {
      model.translation[i] = local.translation[i];
      model.rotation[i] = local.rotation[i];
      model.scale[i] = local.scale[i];
      continue;
    }
    const Quat& pr = model.rotation[p];
    f32 ps = model.scale[p];
    Vec3 v = Rotate(pr, Scale(local.translation[i], ps));
    model.translation[i] = Add(model.translation[p], v);
    model.rotation[i] = Normalize(Mul(pr, local.rotation[i]));
    model.scale[i] = ps * local.scale[i];
  }
}

void ModelToLocal(const Skeleton& skeleton, ConstPoseView model, PoseView local) {
  const u32 n = skeleton.count();
  assert(model.count == n && local.count == n);
  for (u32 i = 0; i < n; ++i) {
    int p = skeleton.parents[i];
    if (p < 0) {
      local.translation[i] = model.translation[i];
      local.rotation[i] = model.rotation[i];
      local.scale[i] = model.scale[i];
      continue;
    }
    Quat prInv = Conjugate(model.rotation[p]);
    f32 ps = model.scale[p];
    f32 invPs = std::abs(ps) > 1e-8f ? 1.0f / ps : 0.0f;
    Vec3 d = Sub(model.translation[i], model.translation[p]);
    local.translation[i] = Scale(Rotate(prInv, d), invPs);
    local.rotation[i] = Normalize(Mul(prInv, model.rotation[i]));
    local.scale[i] = model.scale[i] * invPs;
  }
}

// ---------------------------------------------------------------------------
// Two-bone IK

void SolveTwoBoneIK(const Skeleton& skeleton, ConstPoseView model, PoseView local,
                    const TwoBoneIKSolve& s) {
  if (s.weight <= 0.0f) return;
  const u32 a = s.root_joint, b = s.mid_joint, c = s.end_joint;
  const Vec3 A = model.translation[a], B = model.translation[b], C = model.translation[c];
  const f32 l1 = Length(Sub(B, A)), l2 = Length(Sub(C, B));
  if (l1 < 1e-6f || l2 < 1e-6f) return;

  // Target reach, clamped to the limb's span. `soft` keeps a slack fraction at
  // full extension so the limb never fully locks.
  f32 max_reach = (l1 + l2) * (s.soft > 0.0f ? (1.0f - std::clamp(s.soft, 0.0f, 0.95f)) : 1.0f);
  f32 min_reach = std::abs(l1 - l2) + 1e-4f;
  Vec3 toT = Sub(s.target, A);
  f32 dist = Length(toT);
  Vec3 d = dist > 1e-6f ? Scale(toT, 1.0f / dist) : Normalize3(Sub(C, A));
  f32 reach = std::clamp(dist, min_reach, max_reach - 1e-5f);

  // Hinge axis: normal of the plane spanned by the aim direction and the pole,
  // so the knee bends toward the pole. Robust when the limb is straight (unlike a
  // cross of the two colinear bones). Falls back to any axis perpendicular to d.
  Vec3 n = Cross(d, Sub(s.pole, A));
  if (Length(n) < 1e-6f) {
    Vec3 t = std::abs(d.x) < 0.9f ? Vec3{1, 0, 0} : Vec3{0, 1, 0};
    n = Cross(d, t);
  }
  n = Normalize3(n);

  // Solve the triangle (A, B, C) with sides l1, l2 and base `reach`: the upper
  // bone direction is the aim rotated by the interior angle at A about the hinge.
  f32 cos_alpha =
      std::clamp((l1 * l1 + reach * reach - l2 * l2) / (2.0f * l1 * reach), -1.0f, 1.0f);
  f32 alpha = std::acos(cos_alpha);
  Vec3 upper_dir = Rotate(AxisAngle(n, alpha), d);  // bends toward the pole side
  Vec3 Bnew = Add(A, Scale(upper_dir, l1));
  Vec3 Cnew = Add(A, Scale(d, reach));

  // World rotations carrying each bone's current direction onto the solved one.
  Vec3 u0 = Normalize3(Sub(B, A)), u1 = Normalize3(Sub(Bnew, A));
  Quat qUpper = FromTo(u0, u1);
  Vec3 v_mid = Rotate(qUpper, Normalize3(Sub(C, B)));  // lower bone after the upper turn
  Vec3 v1 = Normalize3(Sub(Cnew, Bnew));
  Quat qLower = FromTo(v_mid, v1);

  Quat A_model_new = Mul(qUpper, model.rotation[a]);
  Quat B_model_new = Mul(qLower, Mul(qUpper, model.rotation[b]));
  Quat A_local_new = Normalize(Mul(Conjugate(ParentModelRot(skeleton, model, a)), A_model_new));
  Quat B_local_new = Normalize(Mul(Conjugate(A_model_new), B_model_new));

  f32 w = std::clamp(s.weight, 0.0f, 1.0f);
  local.rotation[a] = Nlerp(local.rotation[a], A_local_new, w);
  local.rotation[b] = Nlerp(local.rotation[b], B_local_new, w);
}

// ---------------------------------------------------------------------------
// Look-at / aim

void SolveLookAt(const Skeleton& skeleton, ConstPoseView model, PoseView local,
                 const LookAtSolve& s) {
  if (s.weight <= 0.0f) return;
  Vec3 dir = Normalize3(Sub(s.target, model.translation[s.joint]));
  if (Length(dir) < 1e-6f) return;
  Vec3 fwd = Normalize3(Rotate(model.rotation[s.joint], s.forward));
  Quat delta = ClampAngle(FromTo(fwd, dir), s.max_angle);
  Quat joint_new = Mul(delta, model.rotation[s.joint]);
  Quat local_new =
      Normalize(Mul(Conjugate(ParentModelRot(skeleton, model, s.joint)), joint_new));
  local.rotation[s.joint] = Nlerp(local.rotation[s.joint], local_new, std::clamp(s.weight, 0.0f, 1.0f));
}

void SolveLookAtChain(const Skeleton& skeleton, ConstPoseView model, PoseView local,
                      const LookAtChainSolve& s) {
  if (s.weight <= 0.0f || s.count == 0) return;
  const u32 tip = s.joints[s.count - 1];
  Vec3 dir = Normalize3(Sub(s.target, model.translation[tip]));
  if (Length(dir) < 1e-6f) return;
  Vec3 fwd = Normalize3(Rotate(model.rotation[tip], s.forward));
  Quat delta = ClampAngle(FromTo(fwd, dir), s.max_angle);
  Vec3 v = Log(delta);  // world axis * angle; distributed across the chain
  const f32 w = std::clamp(s.weight, 0.0f, 1.0f);
  for (u32 i = 0; i < s.count; ++i) {
    const u32 j = s.joints[i];
    f32 frac = (s.fractions ? s.fractions[i] : 1.0f / static_cast<f32>(s.count)) * w;
    // Express the world aim axis in this joint's parent frame; premultiplying
    // the local rotation there is a world-space rotation about that axis, so the
    // chain's contributions sum to the full aim at the tip (serial chains exact).
    Quat pInv = Conjugate(ParentModelRot(skeleton, model, j));
    Quat q = Exp(Rotate(pInv, Scale(v, frac)));
    local.rotation[j] = Normalize(Mul(q, local.rotation[j]));
  }
}

// ---------------------------------------------------------------------------
// Foot placement

f32 SolveFootPlacement(const Skeleton& skeleton, PoseView local, PoseView model_scratch,
                       const FootPlacementSolve& s) {
  LocalToModel(skeleton, local, model_scratch);
  ConstPoseView model = model_scratch;
  Vec3 up = Normalize3(s.up);
  if (Length(up) < 1e-6f) up = Vec3{0, 1, 0};

  // Lowest-foot rule: the pelvis sinks by the largest downward correction any
  // foot needs, so no leg has to overextend to reach its contact.
  f32 pelvis_offset = 0.0f;
  for (u32 i = 0; i < s.foot_count; ++i) {
    if (!s.hits[i].valid) continue;
    Vec3 ankle = model.translation[s.feet[i].ankle];
    Vec3 desired = Add(s.hits[i].point, Scale(up, s.ankle_height));
    f32 drop = Dot(Sub(desired, ankle), up);
    pelvis_offset = std::min(pelvis_offset, drop);
  }
  pelvis_offset = std::max(pelvis_offset, -std::abs(s.max_drop));

  // Sink the pelvis (and thus the whole model) along up. Applied to the pelvis's
  // local translation in its parent frame; the model scratch is shifted to match
  // so the following IK sees the lowered hips.
  Vec3 world_shift = Scale(up, pelvis_offset);
  int pp = skeleton.parents[s.pelvis];
  Quat pInv = Conjugate(pp < 0 ? Quat{} : model.rotation[pp]);
  f32 pscale = pp < 0 ? 1.0f : model.scale[pp];
  f32 inv_pscale = std::abs(pscale) > 1e-8f ? 1.0f / pscale : 1.0f;
  local.translation[s.pelvis] =
      Add(local.translation[s.pelvis], Scale(Rotate(pInv, world_shift), inv_pscale));
  for (u32 i = 0; i < model_scratch.count; ++i) {
    model_scratch.translation[i] = Add(model_scratch.translation[i], world_shift);
  }

  // Two-bone solve per valid foot. The ground contact target is independent of
  // the pelvis, so the lowered hips make the legs reach down to it.
  for (u32 i = 0; i < s.foot_count; ++i) {
    if (!s.hits[i].valid) continue;
    Vec3 desired = Add(s.hits[i].point, Scale(up, s.ankle_height));
    TwoBoneIKSolve ik;
    ik.root_joint = s.feet[i].hip;
    ik.mid_joint = s.feet[i].knee;
    ik.end_joint = s.feet[i].ankle;
    ik.target = desired;
    ik.pole = model.translation[s.feet[i].knee];  // keep the current bend direction
    ik.soft = s.soft;
    ik.weight = s.weight;
    SolveTwoBoneIK(skeleton, model, local, ik);
  }
  return pelvis_offset;
}

// ---------------------------------------------------------------------------
// Pose mirroring

void MirrorTable::Init(u32 bones, Axis default_axis) {
  partner_.resize(bones);
  axis_.assign(bones, default_axis);
  for (u32 i = 0; i < bones; ++i) partner_[i] = i;
}

void MirrorTable::Pair(u32 a, u32 b, Axis axis) {
  partner_[a] = b;
  partner_[b] = a;
  axis_[a] = axis;
  axis_[b] = axis;
}

void MirrorTable::SetSelf(u32 bone, Axis axis) {
  partner_[bone] = bone;
  axis_[bone] = axis;
}

namespace {

void MirrorTransform(MirrorTable::Axis axis, const Vec3& t, const Quat& q, Vec3* out_t,
                     Quat* out_q) {
  // Reflect across the plane whose normal is `axis`: negate the translation
  // component along the axis; negate the quaternion vector components orthogonal
  // to the axis (reflect the rotation axis, negate the angle).
  switch (axis) {
    case MirrorTable::Axis::kX:
      *out_t = Vec3{-t.x, t.y, t.z};
      *out_q = Quat{q.x, -q.y, -q.z, q.w};
      break;
    case MirrorTable::Axis::kY:
      *out_t = Vec3{t.x, -t.y, t.z};
      *out_q = Quat{-q.x, q.y, -q.z, q.w};
      break;
    case MirrorTable::Axis::kZ:
      *out_t = Vec3{t.x, t.y, -t.z};
      *out_q = Quat{-q.x, -q.y, q.z, q.w};
      break;
  }
}

}  // namespace

void MirrorPose(const MirrorTable& table, ConstPoseView src, PoseView dst) {
  const u32 n = table.size();
  assert(src.count == n && dst.count == n);
  for (u32 i = 0; i < n; ++i) {
    u32 j = table.Partner(i);
    MirrorTransform(table.axis(i), src.translation[j], src.rotation[j], &dst.translation[i],
                    &dst.rotation[i]);
    dst.scale[i] = src.scale[j];
  }
}

void BuildMirrorTable(MirrorTable& out, u32 bones, std::string_view (*name)(void* user, u32 bone),
                      void* user, std::string_view left_token, std::string_view right_token,
                      MirrorTable::Axis axis) {
  out.Init(bones, axis);
  for (u32 i = 0; i < bones; ++i) {
    if (out.Partner(i) != i) continue;  // already paired
    std::string ni(name(user, i));
    // Produce the mirrored name by swapping the first left<->right token.
    std::string mirrored;
    auto lpos = ni.find(std::string(left_token));
    auto rpos = ni.find(std::string(right_token));
    if (!left_token.empty() && lpos != std::string::npos) {
      mirrored = ni.substr(0, lpos) + std::string(right_token) + ni.substr(lpos + left_token.size());
    } else if (!right_token.empty() && rpos != std::string::npos) {
      mirrored = ni.substr(0, rpos) + std::string(left_token) + ni.substr(rpos + right_token.size());
    } else {
      continue;  // centerline bone: stays self-paired
    }
    for (u32 j = 0; j < bones; ++j) {
      if (j == i || out.Partner(j) != j) continue;
      if (std::string(name(user, j)) == mirrored) {
        out.Pair(i, j, axis);
        break;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Retargeting

void RetargetTable::Build(const Skeleton& src, const Skeleton& tgt, const u32* src_bone,
                          const u32* tgt_bone, u32 count) {
  maps_.clear();
  maps_.reserve(count);
  for (u32 k = 0; k < count; ++k) {
    RetargetTable::Map m;
    m.src = src_bone[k];
    m.tgt = tgt_bone[k];
    m.src_bind_t = src.bind_translation[m.src];
    m.tgt_bind_t = tgt.bind_translation[m.tgt];
    f32 sl = Length(m.src_bind_t), tl = Length(m.tgt_bind_t);
    m.ratio = sl > 1e-6f ? tl / sl : 1.0f;
    m.rot_fix = Mul(tgt.bind_rotation[m.tgt], Conjugate(src.bind_rotation[m.src]));
    maps_.push_back(m);
  }
}

void RetargetPose(const RetargetTable& table, ConstPoseView src_local, PoseView tgt_local) {
  const RetargetTable::Map* maps = table.maps();
  const u32 n = table.count();
  for (u32 k = 0; k < n; ++k) {
    const RetargetTable::Map& m = maps[k];
    tgt_local.rotation[m.tgt] = Normalize(Mul(m.rot_fix, src_local.rotation[m.src]));
    Vec3 d = Sub(src_local.translation[m.src], m.src_bind_t);
    tgt_local.translation[m.tgt] = Add(m.tgt_bind_t, Scale(d, m.ratio));
    tgt_local.scale[m.tgt] = src_local.scale[m.src];
  }
}

}  // namespace kinema
