#ifndef KINEMA_KINEMA_H_
#define KINEMA_KINEMA_H_

// kinema: a small, data-oriented skeletal animation runtime.
//
// Design (the short version):
//  - Clips are transcoded once into a relocatable binary blob: uniformly
//    sampled keys, 16-bit range-quantized, stored frame-major SoA so sampling
//    a pose is two contiguous row reads and one lerp - no per-bone branching,
//    no curve evaluation, no allocation. Constant tracks are detected at build
//    time and stored once at full precision (zero error for static bones).
//  - Poses are structure-of-arrays views over caller/arena memory. Blend
//    operations are flat kernels over whole arrays.
//  - Blend trees execute as a compiled flat program (PoseOp list) over pose
//    registers, so the authoring structure never appears in the hot path.
//  - Transitions use inertialization (capture the pose delta at the switch,
//    decay it) so a transition evaluates ONE graph, not two.
//
// The library is self-contained (no engine dependencies, no exceptions, no
// RTTI); physics engines integrate through adapters (see jolt_adapter.h).

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kinema {

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i16 = std::int16_t;
using f32 = float;

struct Vec3 {
  f32 x = 0, y = 0, z = 0;
};
struct Quat {
  f32 x = 0, y = 0, z = 0, w = 1;
};

// FNV-1a: stable name identity for bones and events across projects.
constexpr u64 HashName(std::string_view name) {
  u64 h = 14695981039346656037ull;
  for (char c : name) {
    h ^= static_cast<u8>(c);
    h *= 1099511628211ull;
  }
  return h;
}

// ---------------------------------------------------------------------------
// Poses: SoA views. `count` bones; arrays are caller-owned (see PoseArena).

struct PoseView {
  Vec3* translation = nullptr;
  Quat* rotation = nullptr;
  f32* scale = nullptr;
  u32 count = 0;
};

struct ConstPoseView {
  const Vec3* translation = nullptr;
  const Quat* rotation = nullptr;
  const f32* scale = nullptr;
  u32 count = 0;
  ConstPoseView() = default;
  ConstPoseView(const PoseView& p)
      : translation(p.translation), rotation(p.rotation), scale(p.scale), count(p.count) {}
};

// Fixed-capacity pose scratch for one evaluation frame. Acquire() hands out
// registers; Reset() recycles them. No per-frame heap traffic.
class PoseArena {
 public:
  PoseArena() = default;
  PoseArena(u32 bones, u32 max_poses) { Init(bones, max_poses); }
  void Init(u32 bones, u32 max_poses);
  PoseView Acquire();          // aborts via assert in debug when exhausted
  PoseView At(u32 index);      // register access without advancing
  void Reset() { used_ = 0; }
  u32 bones() const { return bones_; }

 private:
  std::vector<Vec3> translations_;
  std::vector<Quat> rotations_;
  std::vector<f32> scales_;
  u32 bones_ = 0, capacity_ = 0, used_ = 0;
};

// ---------------------------------------------------------------------------
// Skeleton: parents + hashed names + bind pose. Model-space helper included;
// matrix palettes are the host engine's business.

struct Skeleton {
  std::vector<i16> parents;  // -1 = root
  std::vector<u64> name_hashes;
  std::vector<Vec3> bind_translation;
  std::vector<Quat> bind_rotation;
  std::vector<f32> bind_scale;

  u32 count() const { return static_cast<u32>(parents.size()); }
  // Index of the bone with this name hash, or -1.
  int Find(u64 name_hash) const;
};

// Accumulates local TRS down the hierarchy (parents must precede children).
void ComputeModelSpace(const Skeleton& skeleton, ConstPoseView local, Vec3* out_translation,
                       Quat* out_rotation, f32* out_scale);

// ---------------------------------------------------------------------------
// Compressed clips.

struct ClipEvent {
  u64 name_hash = 0;
  f32 time = 0;
  const char* name = nullptr;  // points into the clip blob
};

// A ranged event ("notify state"): a named span [begin, end] in clip time.
struct ClipRangedEvent {
  u64 name_hash = 0;
  f32 begin = 0, end = 0;
  const char* name = nullptr;  // points into the clip blob
};

// Which edge of a ranged event a query step straddled.
enum class RangePhase : u8 { kEnter, kActive, kExit };

// True when a point `v` falls in the half-open step (t0, t1]; when t1 < t0 the
// step wraps through the clip end (looped playback). Shared by the point and
// ranged event queries.
inline bool StepContains(f32 v, f32 t0, f32 t1) {
  return t1 >= t0 ? (v > t0 && v <= t1) : (v > t0 || v <= t1);
}

// Non-owning validated view over a clip blob (see ClipBuilder::Build for the
// layout). Blobs are relocatable and mmap-friendly: load bytes, call FromBlob.
class Clip {
 public:
  static std::optional<Clip> FromBlob(const u8* data, size_t size);

  u32 num_tracks() const;
  u32 num_frames() const;
  f32 frame_rate() const;
  f32 duration() const;
  bool additive() const;

  // Samples the local pose at `time` (clamped). out.count == num_tracks().
  void Sample(f32 time, PoseView out) const;

  // Cumulative root displacement at `time` (zero at t=0, held past the last
  // key). Delta between two times, wrapping through the end when t1 < t0.
  Vec3 RootTranslation(f32 time) const;
  Vec3 RootDelta(f32 t0, f32 t1) const;
  // Root translation swept by advancing forward from `t0` for `dt` seconds with
  // looping over duration(). Unlike RootDelta (a single wrap), this accumulates
  // whole-loop displacement when dt exceeds the remaining/whole duration, so a
  // large timestep still yields the full travelled distance. Stored root keys
  // are translation-only (no rotation), so only translation is reported.
  Vec3 RootDeltaLooped(f32 t0, f32 dt) const;

  u32 num_events() const;
  ClipEvent Event(u32 index) const;
  // Invokes fn(const ClipEvent&) for events with time in (t0, t1]; when
  // t1 < t0 the range wraps through the clip end (looped playback).
  template <typename Fn>
  void EventsInRange(f32 t0, f32 t1, Fn&& fn) const {
    for (u32 i = 0; i < num_events(); ++i) {
      ClipEvent e = Event(i);
      bool hit = t1 >= t0 ? (e.time > t0 && e.time <= t1) : (e.time > t0 || e.time <= t1);
      if (hit) fn(e);
    }
  }

  // Ranged (notify-state) events. v1 clips have none (num_ranged_events()==0).
  u32 num_ranged_events() const;
  ClipRangedEvent RangedEvent(u32 index) const;
  // Reports, per ranged event, the phase(s) its span crosses over the step
  // (t0, t1] (loop-aware, same wrap rule as EventsInRange): kEnter when the
  // begin edge is crossed, kExit when the end edge is crossed, kActive when the
  // sample point t1 lies inside [begin, end) and the span is not exiting this
  // step. Assumes dt <= duration per call (single wrap), like EventsInRange.
  template <typename Fn>
  void RangedEventsInRange(f32 t0, f32 t1, Fn&& fn) const {
    for (u32 i = 0; i < num_ranged_events(); ++i) {
      ClipRangedEvent r = RangedEvent(i);
      bool enter = StepContains(r.begin, t0, t1);
      bool exit = StepContains(r.end, t0, t1);
      bool inside = r.begin <= t1 && t1 < r.end;
      if (enter) fn(r, RangePhase::kEnter);
      if (inside && !exit) fn(r, RangePhase::kActive);
      if (exit) fn(r, RangePhase::kExit);
    }
  }

  const u8* blob() const { return blob_; }
  size_t blob_size() const;

 private:
  const u8* blob_ = nullptr;
};

// Clip that owns its blob bytes.
class OwnedClip {
 public:
  OwnedClip() = default;
  explicit OwnedClip(std::vector<u8> blob);
  const Clip* get() const { return clip_ ? &*clip_ : nullptr; }
  const Clip* operator->() const { return get(); }
  explicit operator bool() const { return clip_.has_value(); }
  const std::vector<u8>& bytes() const { return blob_; }

 private:
  std::vector<u8> blob_;
  std::optional<Clip> clip_;
};

// Feed uniformly sampled full poses (track-ordered), then Build() quantizes,
// splits constant from animated tracks and packs the blob.
class ClipBuilder {
 public:
  ClipBuilder(u32 num_tracks, u32 num_frames, f32 frame_rate);
  void SetSample(u32 frame, u32 track, const Vec3& t, const Quat& r, f32 s);
  void SetAdditive(bool additive) { additive_ = additive; }
  void AddEvent(std::string_view name, f32 time);
  // Ranged event (notify state): a named span [begin, end] in clip seconds.
  // Adding any ranged event promotes the blob to v2; a clip with none stays
  // byte-identical to a v1 build (see Build).
  void AddRangedEvent(std::string_view name, f32 begin, f32 end);
  // Sparse cumulative root-motion keys (time, displacement-from-start).
  void AddRootKey(f32 time, const Vec3& translation);
  std::vector<u8> Build() const;

 private:
  struct RangedEntry {
    std::string name;
    f32 begin, end;
  };
  u32 tracks_, frames_;
  f32 rate_;
  bool additive_ = false;
  std::vector<Vec3> t_;  // [frame * tracks + track]
  std::vector<Quat> r_;
  std::vector<f32> s_;
  std::vector<std::pair<std::string, f32>> events_;
  std::vector<RangedEntry> ranged_;
  std::vector<std::pair<f32, Vec3>> root_keys_;
};

// ---------------------------------------------------------------------------
// Blend kernels. All operate over whole SoA arrays; rotation blends resolve
// quaternion double-cover per bone (branchless sign select) and renormalize.

void CopyPose(ConstPoseView src, PoseView dst);
// dst = nlerp(a, b, alpha)
void BlendPoses(ConstPoseView a, ConstPoseView b, f32 alpha, PoseView dst);
// dst = nlerp(a, b, alpha * mask[bone]); mask may be null (uniform).
void BlendPosesMasked(ConstPoseView a, ConstPoseView b, f32 alpha, const f32* mask, PoseView dst);
// dst = base (+) add, weighted: rotations compose, translations add, scales
// multiply. `add` is an additive pose (delta from its reference).
void ApplyAdditive(ConstPoseView base, ConstPoseView add, f32 weight, PoseView dst);

// ---------------------------------------------------------------------------
// Additive bake: turn a clip into an additive (delta) clip at build time.
//
// An additive clip stores, per frame, the source pose's delta from a reference
// pose: rotation delta = ref^-1 * src, translation delta = src - ref, scale
// delta = src / ref. Composed back over the reference with ApplyAdditive (or a
// kAdditive op) at weight 1 it reproduces the source (within quantization). The
// result is an ordinary clip blob with the additive flag set - no format
// change; the runtime samples it exactly like any other clip.
OwnedClip MakeAdditiveClip(const Clip& source, ConstPoseView reference);
// Reference = the source's own first frame (the common "additive from base
// pose" case for layered gestures on top of a neutral stance).
OwnedClip MakeAdditiveClipFromFirstFrame(const Clip& source);

// ---------------------------------------------------------------------------
// Bone masks: SoA per-bone blend weights for kBlendMasked / BlendPosesMasked.
// Built once (builder time), the op points at data(); the hot path just reads
// the array. Hierarchy-aware helpers fill a joint and optionally its whole
// subtree using the skeleton's parent table.
class BoneMask {
 public:
  BoneMask() = default;
  explicit BoneMask(u32 bones, f32 fill = 0.0f) { Init(bones, fill); }
  void Init(u32 bones, f32 fill = 0.0f);
  void Fill(f32 weight);
  void Set(u32 bone, f32 weight);
  // Set `weight` on `bone`; with `descendants`, on every joint below it too
  // (parents precede children, so a single forward sweep suffices).
  void SetChain(const Skeleton& skeleton, u32 bone, f32 weight, bool descendants = true);
  const f32* data() const { return weights_.data(); }
  u32 size() const { return static_cast<u32>(weights_.size()); }

 private:
  std::vector<f32> weights_;
};

// ---------------------------------------------------------------------------
// Blend spaces: parameterized blending over clips placed at coordinates. 1D
// (e.g. speed -> walk/jog/run) brackets the two neighbouring clips and lerps.
// 2D (e.g. direction x speed strafe set) uses gradient-band interpolation.
//
// Host-owned config, compiled once; a kBlendSpace op points at it. Every
// contributing clip is sampled at a shared normalized phase [0,1], so
// differently-timed locomotion clips stay phase-matched through the blend. The
// runtime query is allocation-free and O(clips) (clips are few).
class BlendSpace {
 public:
  enum class Dim : u8 { k1D = 0, k2D = 1 };
  BlendSpace() = default;
  explicit BlendSpace(Dim dim) : dim_(dim) {}
  BlendSpace& Add(const Clip* clip, f32 x, f32 y = 0.0f);  // 1D ignores y
  void Finalize();  // sorts 1D samples ascending by x (no-op for 2D)

  Dim dim() const { return dim_; }
  u32 count() const { return static_cast<u32>(clips_.size()); }
  const Clip* clip(u32 i) const { return clips_[i]; }
  f32 x(u32 i) const { return x_[i]; }
  f32 y(u32 i) const { return y_[i]; }

 private:
  Dim dim_ = Dim::k1D;
  std::vector<const Clip*> clips_;
  std::vector<f32> x_, y_;
};

// Evaluate a blend space into `dst` at coordinate (x[,y]) and normalized phase,
// using `scratch` as one temporary register. dst/scratch track counts must
// match the clips'. Exposed so hosts can drive it directly; kBlendSpace ops
// call it internally.
void EvalBlendSpace(const BlendSpace& space, f32 x, f32 y, f32 phase, PoseView dst,
                    PoseView scratch);

// ---------------------------------------------------------------------------
// Per-frame parameter block. Programs compile once; any op that carries a
// *_param index >= 0 reads that value from here each frame instead of from its
// own immediate, so times/alphas/coordinates change without rebuilding the
// program.
struct PoseParams {
  const f32* values = nullptr;
  u32 count = 0;
  f32 Get(int idx, f32 fallback) const {
    return (idx >= 0 && static_cast<u32>(idx) < count) ? values[idx] : fallback;
  }
};

// ---------------------------------------------------------------------------
// Compiled pose program: a flat op list over arena registers. Hosts compile
// their blend tree / state machine into this once per structural change and
// just patch times/alphas per frame (directly, or via a PoseParams block).

struct PoseOp {
  enum class Kind : u8 { kSample, kCopy, kBlend, kBlendMasked, kAdditive, kBlendSpace };
  Kind kind = Kind::kCopy;
  u8 dst = 0, a = 0, b = 0;
  const Clip* clip = nullptr;  // kSample
  f32 time = 0;                // kSample time / kBlendSpace normalized phase [0,1]
  f32 alpha = 0;               // kBlend*/kAdditive weight
  const f32* mask = nullptr;   // kBlendMasked, per-bone weights
  // Extensions below are append-only: existing designated-initializer call
  // sites keep compiling because they only name the fields above.
  const BlendSpace* space = nullptr;  // kBlendSpace; uses register `a` as scratch
  i16 time_param = -1;                // >=0: overrides `time`/phase from PoseParams
  i16 alpha_param = -1;               // >=0: overrides `alpha` from PoseParams
  i16 coord_param = -1;               // kBlendSpace: params[coord_param]=x, +1=y (2D)
};

// Executes ops in order against arena registers 0..N and returns the view of
// the last op's dst. The arena must hold max(dst,a,b)+1 registers. Pass a
// PoseParams block to resolve any *_param bindings; omit it to use immediates.
PoseView ExecuteProgram(const PoseOp* ops, size_t count, PoseArena& arena,
                        const PoseParams* params = nullptr);

// ---------------------------------------------------------------------------
// Inertialization: on a state switch, capture the offset between the pose the
// previous state would show and the new state's pose, then decay that offset
// smoothly while ONLY the new state is evaluated.

class Inertializer {
 public:
  void Init(u32 bones);
  // Capture offsets = from - to, start a blend of `duration` seconds.
  void Begin(ConstPoseView from, ConstPoseView to, f32 duration);
  // Applies the decayed offset onto pose; returns false once finished.
  bool Apply(PoseView pose, f32 dt);
  bool active() const { return remaining_ > 0; }

 private:
  std::vector<Vec3> dt_;
  std::vector<Vec3> dr_;  // rotation offsets, axis*angle
  std::vector<f32> ds_;
  f32 remaining_ = 0, duration_ = 0;
};

// ---------------------------------------------------------------------------
// Sync groups: keep phase-matched clips (walk <-> run) foot-synced across a
// blend. Each clip contributes a sorted marker track - its named events, e.g.
// footfalls - and all clips in a group must share the same marker count. A
// single global marker-phase advances on the current leader's clock; every
// clip's local sample time is read back at that phase, so followers are
// time-scaled to hit their k-th marker together with the leader. Feed the
// resulting LocalTime values into kSample ops (directly or via a PoseParams
// block) and one graph stays synced even as blend weights shift.
class SyncGroup {
 public:
  void Clear();
  // Markers = all of the clip's events, sorted by time.
  void AddClip(const Clip& clip);
  // Explicit markers (seconds, sorted internally); marker_count must match the
  // clips already in the group.
  void AddClipMarkers(const f32* marker_times, u32 marker_count, f32 duration);
  void Reset(f32 phase = 0.0f);
  // Advance the shared phase by dt seconds measured on `leader`'s clock.
  void Advance(f32 dt, u32 leader, f32 play_rate = 1.0f);

  f32 LocalTime(u32 clip) const;  // seconds into `clip` at the current phase
  f32 phase() const { return phase_; }
  u32 marker_count() const { return markers_; }
  u32 clip_count() const { return static_cast<u32>(tracks_.size()); }

 private:
  struct Track {
    std::vector<f32> markers;
    f32 duration = 0;
  };
  f32 MarkerTime(const Track& t, f32 g) const;
  std::vector<Track> tracks_;
  u32 markers_ = 0;
  f32 phase_ = 0;  // global marker-phase in [0, markers_)
};

// ---------------------------------------------------------------------------
// Animation state machine: a compiled, data-driven graph in the same flat
// idiom as the rest of kinema. States reference a pose source (a clip, a blend
// space or a caller-provided program fragment) that is compiled into one shared
// PoseOp template; transitions carry an AND-list of condition atoms over the
// live PoseParams block (plus edge triggers the machine owns and consumes), a
// blend duration, an optional normalized exit-time window and an interruption
// policy. Compile once (StateMachineBuilder -> StateMachine), then drive one
// StateMachineInstance per actor with Update(). Transitions are inertialized:
// only the target state's program is evaluated each frame while the captured
// pose offset decays - there is never a second graph walked in the hot path.

// One clause of a transition condition. Comparison atoms read a live value from
// the PoseParams block; trigger atoms test an edge flag the instance owns.
struct ConditionAtom {
  enum class Test : u8 { kLess, kGreater, kEqual, kNotEqual, kTrigger };
  Test test = Test::kGreater;
  i16 param = -1;  // PoseParams index (comparisons) or trigger id 0..63 (kTrigger)
  f32 value = 0;   // threshold (comparisons); kEqual/kNotEqual use a small epsilon

  static ConditionAtom Less(i16 p, f32 v) { return {Test::kLess, p, v}; }
  static ConditionAtom Greater(i16 p, f32 v) { return {Test::kGreater, p, v}; }
  static ConditionAtom Equal(i16 p, f32 v) { return {Test::kEqual, p, v}; }
  static ConditionAtom NotEqual(i16 p, f32 v) { return {Test::kNotEqual, p, v}; }
  static ConditionAtom Trigger(i16 id) { return {Test::kTrigger, id, 0}; }
};

// Whether a transition may fire while another transition is still blending.
enum class InterruptPolicy : u8 {
  kWaitForCompletion = 0,  // ignored until the active blend finishes
  kInterruptible = 1,      // fires mid-blend, re-inertializing from the current pose
};

struct TransitionDesc {
  f32 duration = 0.15f;  // inertialization / blend seconds
  f32 exit_min = -1.0f;  // normalized source phase [0,1] gate; <0 disables the gate
  f32 exit_max = -1.0f;
  InterruptPolicy policy = InterruptPolicy::kWaitForCompletion;
};

// Point + ranged event sink used by StateMachineInstance::Update. Plain function
// pointers keep it exception/RTTI/alloc free and usable across translation
// units (Update is compiled in the library).
struct EventCallback {
  void* user = nullptr;
  void (*point)(void* user, const ClipEvent&) = nullptr;
  void (*ranged)(void* user, const ClipRangedEvent&, RangePhase) = nullptr;
};

class StateMachineBuilder;

// Compiled, immutable state-machine definition. Shared across all actors of an
// archetype; carries no per-actor state.
class StateMachine {
 public:
  static constexpr u16 kAnyState = 0xFFFF;
  u32 bones() const { return bones_; }
  u16 state_count() const { return static_cast<u16>(states_.size()); }
  u32 max_registers() const { return max_regs_; }  // arena registers Update needs

 private:
  friend class StateMachineBuilder;
  friend class StateMachineInstance;
  struct State {
    u32 ops_begin = 0, ops_count = 0;
    i16 clock_op = -1;       // op whose .time carries the state clock (-1 = none)
    bool loop = true;
    bool phase_clock = false;  // true: clock maps to normalized phase [0,1]
    f32 loop_duration = 0;     // wrap period for the clock (clip seconds, or 1)
    f32 speed = 1.0f;
    const Clip* clip = nullptr;         // event source (may be null)
    const Clip* root_source = nullptr;  // root-motion source (may be null)
  };
  struct Transition {
    u16 from = 0, to = 0;
    u32 cond_begin = 0, cond_count = 0;
    f32 duration = 0.15f, exit_min = -1.0f, exit_max = -1.0f;
    InterruptPolicy policy = InterruptPolicy::kWaitForCompletion;
  };
  u32 bones_ = 0, max_regs_ = 1;
  std::vector<State> states_;
  std::vector<Transition> transitions_;
  std::vector<ConditionAtom> conds_;
  std::vector<PoseOp> ops_;  // template ops; states slice into this
};

// Builds a StateMachine. Each Add*State returns the new state's id (its index).
class StateMachineBuilder {
 public:
  explicit StateMachineBuilder(u32 bones) { bones_ = bones; }

  // A state that samples one clip on its own clock. loop wraps by the clip's
  // duration; speed scales the clock.
  u16 AddClipState(const Clip* clip, bool loop = true, f32 speed = 1.0f);
  // A state driven by a blend space; the state clock feeds the normalized phase
  // and params[coord_param] (and +1 in 2D) the coordinate. root_source (if any)
  // supplies root motion and clip events.
  u16 AddBlendSpaceState(const BlendSpace* space, i16 coord_param, bool loop = true,
                         f32 speed = 1.0f, const Clip* root_source = nullptr);
  // A state from a caller-provided program fragment (ops are copied in). If one
  // op should carry the state clock, pass its index in clock_op (its .time is
  // patched each frame) and the wrap period in loop_duration; pass -1/0 for a
  // static fragment. reg_count is the number of arena registers the fragment
  // uses; its result must be its last op's dst. clip supplies events/root.
  u16 AddProgramState(const PoseOp* ops, u32 count, u32 reg_count, i16 clock_op,
                      f32 loop_duration, bool loop = true, f32 speed = 1.0f,
                      const Clip* clip = nullptr, const Clip* root_source = nullptr);

  // from == StateMachine::kAnyState applies the transition from every state.
  // Conditions are ANDed; an empty list is always-true (guarded by exit-time).
  void AddTransition(u16 from, u16 to, const TransitionDesc& desc,
                     const ConditionAtom* conds = nullptr, u32 cond_count = 0);

  StateMachine Build() const;

 private:
  u32 bones_ = 0;
  StateMachine sm_;
};

// Per-actor runtime state. Small and cheap to hold; the only heap use is a
// one-time Init sizing of the capture buffers and op scratch (no per-frame
// allocation). Advance and evaluate with Update().
class StateMachineInstance {
 public:
  void Init(const StateMachine& def, u16 start_state = 0);

  // Raise an edge trigger (0..63). It stays raised until a transition whose
  // condition consumes it fires, then auto-resets.
  void SetTrigger(u16 id) {
    if (id < 64) triggers_ |= (1ull << id);
  }
  bool trigger(u16 id) const { return id < 64 && (triggers_ & (1ull << id)) != 0; }

  // Advance the machine by dt, evaluate conditions, fire at most one transition
  // (inertialized), and write this frame's pose into out through the program
  // model. arena must hold at least def.max_registers() registers; out is a
  // separate bones-sized view the pose is written to (and returned). An optional
  // event sink receives the active state's point/ranged events; on a transition
  // the exiting state's still-open ranges are force-reported as kExit.
  PoseView Update(f32 dt, const PoseParams& params, PoseArena& arena, PoseView out,
                  const EventCallback* events = nullptr);

  u16 state() const { return current_; }
  bool transitioning() const { return xfade_dur_ > 0; }
  f32 transition_alpha() const {
    return xfade_dur_ > 0 ? (xfade_t_ / xfade_dur_) : 1.0f;
  }
  f32 state_time() const { return time_; }
  // This frame's root translation delta (loop-aware; blended across an active
  // transition by transition progress). Valid after Update().
  Vec3 RootMotion() const { return root_; }
  const Inertializer& inertializer() const { return inert_; }

 private:
  PoseView RunState(u16 s, f32 clock, const PoseParams& params, PoseArena& arena);
  bool CondsPass(const StateMachine::Transition& tr, const PoseParams& params) const;
  f32 NormalizedPhase(u16 s, f32 t) const;
  int SelectTransition(const PoseParams& params) const;
  PoseView FromView();
  void EmitEvents(u16 s, f32 t0, f32 t1, const EventCallback* ev) const;
  void ForceExitOpenRanges(u16 s, f32 clock, const EventCallback* ev) const;

  const StateMachine* def_ = nullptr;
  u16 current_ = 0, from_ = 0;
  f32 time_ = 0, from_time_ = 0;
  f32 xfade_t_ = 0, xfade_dur_ = 0;
  u64 triggers_ = 0;
  Vec3 root_{};
  Inertializer inert_;
  std::vector<Vec3> from_t_;
  std::vector<Quat> from_r_;
  std::vector<f32> from_s_;
  std::vector<PoseOp> scratch_ops_;
};

}  // namespace kinema

#endif  // KINEMA_KINEMA_H_
