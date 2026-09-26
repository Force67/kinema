// Animation state machine: builder-time compilation into flat state/transition/
// condition arrays and one shared PoseOp template, plus a small per-actor
// runtime that advances the graph, fires inertialized transitions and drives
// the pose through the existing program model. No node trees are walked at
// runtime and nothing is allocated per frame (buffers are sized once at Init).

#include <assert.h>
#include <math.h>

#include "kinema/kinema.h"

namespace kinema {
namespace {

inline f32 Fract(f32 x) { return x - floorf(x); }
inline f32 Clamp01(f32 x) { return Clamp(x, 0.0f, 1.0f); }
inline Vec3 Lerp(const Vec3& a, const Vec3& b, f32 w) {
  return Vec3{a.x + (b.x - a.x) * w, a.y + (b.y - a.y) * w, a.z + (b.z - a.z) * w};
}

// Map a state clock into clip-seconds (for events / root motion). Phase-clock
// states advance in [0,1) cycles; scale by the reference clip's duration.
inline f32 ToClipTime(bool phase_clock, bool loop, f32 clock, f32 ref_dur) {
  if (phase_clock) return (loop ? Fract(clock) : Clamp01(clock)) * ref_dur;
  return (loop && ref_dur > 1e-6f) ? fmodf(clock, ref_dur) : clock;
}

}  // namespace

// ---------------------------------------------------------------------------
// Builder

u16 StateMachineBuilder::AddClipState(const Clip* clip, bool loop, f32 speed) {
  StateMachine::State st;
  st.ops_begin = static_cast<u32>(sm_.ops_.size());
  st.ops_count = 1;
  st.clock_op = 0;
  st.loop = loop;
  st.phase_clock = false;
  st.loop_duration = clip ? clip->duration() : 0.0f;
  st.speed = speed;
  st.clip = clip;
  st.root_source = clip;
  PoseOp op{};
  op.kind = PoseOp::Kind::kSample;
  op.dst = 0;
  op.clip = clip;
  sm_.ops_.push_back(op);
  sm_.max_regs_ = Max(sm_.max_regs_, 1u);
  sm_.states_.push_back(st);
  return static_cast<u16>(sm_.states_.size() - 1);
}

u16 StateMachineBuilder::AddBlendSpaceState(const BlendSpace* space, i16 coord_param, bool loop,
                                            f32 speed, const Clip* root_source) {
  StateMachine::State st;
  st.ops_begin = static_cast<u32>(sm_.ops_.size());
  st.ops_count = 1;
  st.clock_op = 0;
  st.loop = loop;
  st.phase_clock = true;  // clock feeds the normalized phase [0,1]
  st.loop_duration = 1.0f;
  st.speed = speed;
  st.clip = root_source;
  st.root_source = root_source;
  PoseOp op{};
  op.kind = PoseOp::Kind::kBlendSpace;
  op.dst = 0;
  op.a = 1;  // scratch register
  op.space = space;
  op.coord_param = coord_param;
  sm_.ops_.push_back(op);
  sm_.max_regs_ = Max(sm_.max_regs_, 2u);
  sm_.states_.push_back(st);
  return static_cast<u16>(sm_.states_.size() - 1);
}

u16 StateMachineBuilder::AddProgramState(const PoseOp* ops, u32 count, u32 reg_count, i16 clock_op,
                                         f32 loop_duration, bool loop, f32 speed, const Clip* clip,
                                         const Clip* root_source) {
  StateMachine::State st;
  st.ops_begin = static_cast<u32>(sm_.ops_.size());
  st.ops_count = count;
  st.clock_op = clock_op;
  st.loop = loop;
  st.phase_clock = false;
  st.loop_duration = loop_duration;
  st.speed = speed;
  st.clip = clip;
  st.root_source = root_source;
  for (u32 i = 0; i < count; ++i) sm_.ops_.push_back(ops[i]);
  sm_.max_regs_ = Max(sm_.max_regs_, Max(reg_count, 1u));
  sm_.states_.push_back(st);
  return static_cast<u16>(sm_.states_.size() - 1);
}

void StateMachineBuilder::AddTransition(u16 from, u16 to, const TransitionDesc& desc,
                                        const ConditionAtom* conds, u32 cond_count) {
  StateMachine::Transition tr;
  tr.from = from;
  tr.to = to;
  tr.cond_begin = static_cast<u32>(sm_.conds_.size());
  tr.cond_count = cond_count;
  tr.duration = desc.duration;
  tr.exit_min = desc.exit_min;
  tr.exit_max = desc.exit_max;
  tr.policy = desc.policy;
  for (u32 i = 0; i < cond_count; ++i) sm_.conds_.push_back(conds[i]);
  sm_.transitions_.push_back(tr);
}

StateMachine StateMachineBuilder::Build() const {
  StateMachine out = sm_;
  out.bones_ = bones_;
  return out;
}

// ---------------------------------------------------------------------------
// Instance

void StateMachineInstance::Init(const StateMachine& def, u16 start_state) {
  def_ = &def;
  current_ = from_ = start_state;
  time_ = from_time_ = 0.0f;
  xfade_t_ = xfade_dur_ = 0.0f;
  triggers_ = 0;
  root_ = Vec3{};
  inert_.Init(def.bones_);
  from_t_.assign(def.bones_, Vec3{});
  from_r_.assign(def.bones_, Quat{});
  from_s_.assign(def.bones_, 1.0f);
  u32 max_ops = 1;
  for (const auto& s : def.states_) max_ops = Max(max_ops, s.ops_count);
  scratch_ops_.assign(max_ops, PoseOp{});
}

PoseView StateMachineInstance::FromView() {
  return PoseView{from_t_.data(), from_r_.data(), from_s_.data(), def_->bones_};
}

PoseView StateMachineInstance::RunState(u16 s, f32 clock, const PoseParams& params,
                                        PoseArena& arena) {
  const StateMachine::State& st = def_->states_[s];
  for (u32 i = 0; i < st.ops_count; ++i) scratch_ops_[i] = def_->ops_[st.ops_begin + i];
  if (st.clock_op >= 0) {
    PoseOp& op = scratch_ops_[static_cast<u32>(st.clock_op)];
    op.time = st.phase_clock
                  ? (st.loop ? Fract(clock) : Clamp01(clock))
                  : (st.loop && st.loop_duration > 1e-6f ? fmodf(clock, st.loop_duration)
                                                      : clock);
    op.time_param = -1;  // the state clock overrides any param binding
  }
  return ExecuteProgram(scratch_ops_.data(), st.ops_count, arena, &params);
}

f32 StateMachineInstance::NormalizedPhase(u16 s, f32 t) const {
  const StateMachine::State& st = def_->states_[s];
  if (st.phase_clock) return st.loop ? Fract(t) : Clamp01(t);
  if (st.loop_duration <= 1e-6f) return 0.0f;
  f32 p = fmodf(t, st.loop_duration) / st.loop_duration;
  return p < 0 ? p + 1.0f : p;
}

bool StateMachineInstance::CondsPass(const StateMachine::Transition& tr,
                                     const PoseParams& params) const {
  for (u32 i = 0; i < tr.cond_count; ++i) {
    const ConditionAtom& c = def_->conds_[tr.cond_begin + i];
    if (c.test == ConditionAtom::Test::kTrigger) {
      if (c.param < 0 || c.param >= 64 || !(triggers_ & (1ull << c.param))) return false;
      continue;
    }
    f32 v = params.Get(c.param, 0.0f);
    switch (c.test) {
      case ConditionAtom::Test::kLess:
        if (!(v < c.value)) return false;
        break;
      case ConditionAtom::Test::kGreater:
        if (!(v > c.value)) return false;
        break;
      case ConditionAtom::Test::kEqual:
        if (!(fabsf(v - c.value) <= 1e-6f)) return false;
        break;
      case ConditionAtom::Test::kNotEqual:
        if (!(fabsf(v - c.value) > 1e-6f)) return false;
        break;
      case ConditionAtom::Test::kTrigger:
        break;  // handled above
    }
  }
  return true;
}

int StateMachineInstance::SelectTransition(const PoseParams& params) const {
  const bool mid = xfade_dur_ > 0 && xfade_t_ < xfade_dur_;
  for (size_t i = 0; i < def_->transitions_.size(); ++i) {
    const StateMachine::Transition& tr = def_->transitions_[i];
    if (tr.from != StateMachine::kAnyState && tr.from != current_) continue;
    if (tr.to == current_) continue;  // no self-transition
    if (mid && tr.policy == InterruptPolicy::kWaitForCompletion) continue;
    if (tr.exit_min >= 0.0f) {
      f32 phase = NormalizedPhase(current_, time_);
      if (!(phase >= tr.exit_min && phase <= tr.exit_max)) continue;
    }
    if (!CondsPass(tr, params)) continue;
    return static_cast<int>(i);
  }
  return -1;
}

void StateMachineInstance::EmitEvents(u16 s, f32 t0, f32 t1, const EventCallback* ev) const {
  if (!ev) return;
  const StateMachine::State& st = def_->states_[s];
  const Clip* c = st.clip;
  if (!c) return;
  const f32 dur = c->duration();
  const f32 a = ToClipTime(st.phase_clock, st.loop, t0, dur);
  const f32 b = ToClipTime(st.phase_clock, st.loop, t1, dur);
  if (ev->point) {
    void* u = ev->user;
    auto fn = ev->point;
    c->EventsInRange(a, b, [u, fn](const ClipEvent& e) { fn(u, e); });
  }
  if (ev->ranged) {
    void* u = ev->user;
    auto fn = ev->ranged;
    c->RangedEventsInRange(a, b, [u, fn](const ClipRangedEvent& r, RangePhase p) { fn(u, r, p); });
  }
}

void StateMachineInstance::ForceExitOpenRanges(u16 s, f32 clock, const EventCallback* ev) const {
  if (!ev || !ev->ranged) return;
  const StateMachine::State& st = def_->states_[s];
  const Clip* c = st.clip;
  if (!c) return;
  const f32 t = ToClipTime(st.phase_clock, st.loop, clock, c->duration());
  for (u32 i = 0; i < c->num_ranged_events(); ++i) {
    ClipRangedEvent r = c->RangedEvent(i);
    if (r.begin <= t && t < r.end) ev->ranged(ev->user, r, RangePhase::kExit);
  }
}

PoseView StateMachineInstance::Update(f32 dt, const PoseParams& params, PoseArena& arena,
                                      PoseView out, const EventCallback* events) {
  assert(def_ && out.count == def_->bones_);
  root_ = Vec3{};
  const bool blending_before = xfade_dur_ > 0 && xfade_t_ < xfade_dur_;
  const f32 cur_prev = time_;
  const f32 from_prev = from_time_;

  // Advance clocks. The from-state clock keeps running during a blend so its
  // root motion (and any residual pose) stays continuous.
  time_ += dt * def_->states_[current_].speed;
  if (blending_before) from_time_ += dt * def_->states_[from_].speed;

  const int ti = SelectTransition(params);

  auto root_step = [&](u16 s, f32 prev) -> Vec3 {
    const StateMachine::State& st = def_->states_[s];
    if (!st.root_source) return Vec3{};
    const f32 scale = st.phase_clock ? st.root_source->duration() : 1.0f;
    return st.root_source->RootDeltaLooped(prev * scale, dt * st.speed * scale);
  };

  if (ti >= 0) {
    const StateMachine::Transition& tr = def_->transitions_[ti];
    // Consume any edge triggers this transition tested.
    for (u32 i = 0; i < tr.cond_count; ++i) {
      const ConditionAtom& c = def_->conds_[tr.cond_begin + i];
      if (c.test == ConditionAtom::Test::kTrigger && c.param >= 0 && c.param < 64) {
        triggers_ &= ~(1ull << c.param);
      }
    }
    // The exiting state reports the segment it played, then force-exits any of
    // its ranges still open at the switch.
    EmitEvents(current_, cur_prev, time_, events);
    ForceExitOpenRanges(current_, time_, events);

    // Capture the current *visual* pose (old program + any residual offset) as
    // the inertialization source, so only the target graph is evaluated after.
    PoseView cp = RunState(current_, time_, params, arena);
    CopyPose(cp, FromView());
    if (inert_.active()) inert_.Apply(FromView(), 0.0f);

    const u16 exited = current_;
    const f32 exited_prev = cur_prev;
    from_ = exited;
    from_time_ = time_;
    current_ = tr.to;
    time_ = dt * def_->states_[current_].speed;  // new clock advances from 0
    xfade_dur_ = tr.duration;
    xfade_t_ = 0.0f;

    PoseView nw = RunState(current_, time_, params, arena);
    CopyPose(nw, out);
    inert_.Begin(FromView(), out, tr.duration);

    // Fresh blend: weight starts at 0, so this frame's root is the exited
    // state's motion, easing toward the new state's as the blend completes.
    Vec3 fromM = root_step(exited, exited_prev);
    Vec3 curM = root_step(current_, 0.0f);
    root_ = Lerp(fromM, curM, 0.0f);
    (void)curM;
  } else {
    EmitEvents(current_, cur_prev, time_, events);
    PoseView cp = RunState(current_, time_, params, arena);
    CopyPose(cp, out);
    if (blending_before) {
      Vec3 fromM = root_step(from_, from_prev);
      Vec3 curM = root_step(current_, cur_prev);
      f32 w = xfade_dur_ > 0 ? (xfade_t_ / xfade_dur_) : 1.0f;
      root_ = Lerp(fromM, curM, w);
    } else {
      root_ = root_step(current_, cur_prev);
    }
  }

  // Single inertialized graph: decay the captured offset onto the target pose.
  if (inert_.active()) inert_.Apply(out, dt);

  if (xfade_dur_ > 0) {
    xfade_t_ += dt;
    if (xfade_t_ >= xfade_dur_) {
      xfade_dur_ = 0.0f;
      xfade_t_ = 0.0f;
    }
  }
  return out;
}

}  // namespace kinema
