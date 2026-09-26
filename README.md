# kinema

A small, data-oriented skeletal animation runtime. Built for the *rx*
engine but deliberately self-contained: no engine types, no exceptions, no
RTTI, no dependencies beyond the C++ standard library. Pairs naturally with
[Jolt Physics](https://github.com/jrouwe/JoltPhysics) through a header-only
adapter.

## Why it's fast

- **Transcode once, stream forever.** Clips are baked into a relocatable
  binary blob: uniformly sampled keys, 16-bit range-quantized, stored
  frame-major SoA. Sampling a pose is two contiguous row reads and a lerp -
  no curve evaluation, no branches per bone, no allocation. Constant tracks
  are detected at build time and stored once at full precision.
- **Poses are SoA views** over arena memory; blend operations are flat
  kernels over whole arrays that autovectorize.
- **Blend trees run as compiled programs** (a flat `PoseOp` list over pose
  registers), so authoring structure never appears in the hot path.
- **Transitions are inertialized** - capture the pose offset at the switch
  and decay it C2-smoothly - so a transition evaluates one graph, not two.

## Graph core

Higher-level constructs compile into the same flat program model - new op
kinds and precomputed structures, never node-object trees walked at runtime:

- **Blend spaces** place clips at parameter coordinates. 1D (speed ->
  walk/jog/run) brackets and lerps; 2D (direction x speed strafe sets) uses
  gradient-band interpolation - no triangulation to build, allocation-free,
  O(clips) per query. A `kBlendSpace` op samples every contributor at a shared
  normalized phase, so differently-timed clips stay phase-matched.
- **Layers**: additive blending (`ApplyAdditive` / `kAdditive`) and per-bone
  masked blending (`BlendPosesMasked` / `kBlendMasked` over SoA `BoneMask`
  arrays). `MakeAdditiveClip` bakes a clip into an additive (delta) clip at
  build time - an ordinary blob with the additive flag, no format change.
- **Sync groups** keep phase-matched clips foot-synced across a blend:
  per-clip marker tracks (named events, e.g. footfalls), one shared
  marker-phase advanced on the leader's clock, followers time-scaled to hit
  their k-th marker together.
- **Per-frame parameters** (`PoseParams`): programs compile once per
  archetype; ops carrying a `*_param` index read live times/alphas/coords from
  a parameter block each frame, so nothing is reallocated to animate.
- **State machine** (`StateMachineBuilder` -> `StateMachine` +
  `StateMachineInstance`): a compiled, data-driven graph. States reference a
  pose source (clip / blend space / caller program fragment) baked into one
  shared `PoseOp` template; transitions carry an AND-list of condition atoms
  over the `PoseParams` block (plus edge triggers the instance owns and
  consumes), a blend duration, an optional normalized exit-time window and an
  interruption policy. `Update()` advances one instance per actor, patches the
  active state's clock into the template and evaluates it through the program
  model. Transitions are **inertialized** - it captures the current pose as the
  offset source and decays it, so still only one graph is walked.
- **Root motion**: `Clip::RootDeltaLooped(t0, dt)` sweeps the sparse cumulative
  root keys forward with loop wrap and multi-loop accumulation (`dt` may exceed
  the duration); keys are translation-only. The state machine reports the
  frame's delta (`RootMotion()`) blended across an active transition by its
  progress.
- **Ranged events** ("notify states"): `AddRangedEvent(name, begin, end)` plus
  a loop-aware `RangedEventsInRange` query that reports enter/active/exit per
  span over a step. The state machine routes them per frame and force-exits a
  leaving state's still-open ranges on a transition. Ranged events promote a
  clip to a v2 blob; a clip with none is byte-identical to a v1 build and the
  loader accepts both.
- **Float curve tracks**: named scalar channels (facial weights, material
  params, authored IK weights) baked alongside the bone tracks - uniformly
  sampled, 16-bit range-quantized. `ClipBuilder::AddCurve(name)` +
  `SetCurveSample(frame, curve, value)`; `Clip::SampleCurve(hash, t)` at
  runtime. Authoring a curve promotes the blob to v3; a curve-free clip stays
  byte-identical to a v1/v2 build (guarded by `tools/golden.cc`'s baked hashes)
  and the loader reads v1/v2/v3.

## Pose tooling

Constraints that run over a **model-space** pose and write the result back to
local. `LocalToModel` / `ModelToLocal` are the public conversion kernels (one
forward sweep, parents precede children); every solver below takes the
caller-produced model pose so a chain of solves shares one buffer. All are flat,
allocation-free kernels.

- **Two-bone IK** (`SolveTwoBoneIK`): arm/leg solve over a root->mid->end chain.
  Analytic triangle construction (law of cosines) drives the end joint onto a
  target with a pole/hint for the bend plane, a softness slack at full
  extension, and a weight that blends the solve over the FK pose.
- **Look-at / aim** (`SolveLookAt`, `SolveLookAtChain`): single-joint aim of a
  configurable forward axis with a cone clamp and weight, plus an N-joint
  distributed variant that spreads the aim across a chain by per-joint fractions
  (serial chains land the tip exactly when the fractions sum to one).
- **Foot placement** (`SolveFootPlacement`): the pure-math half of foot IK -
  kinema does no raycasts. The caller feeds per-foot hit points/normals; the
  helper computes each foot's IK target, drops the pelvis by the lowest-foot
  rule so no leg overextends, then runs the two-bone solves and returns the
  pelvis offset.
- **Pose mirroring** (`MirrorPose` + `MirrorTable`): reflects a local pose across
  the sagittal (X) plane - swap paired bones, negate translation along and
  quaternion components across the flip axis (the exact reflection of a
  rotation, so a symmetric rig round-trips). `BuildMirrorTable` auto-pairs from a
  caller-supplied name callback and left/right tokens, with no engine naming
  baked in.
- **Retargeting** (`RetargetTable` + `RetargetPose`): transfers a source local
  pose onto a target skeleton via a bone map and both bind poses - rotations
  carried through the bind-orientation difference, translations scaled by the
  per-bone reference-length ratio (proportion-aware). Basic by design: rotation
  copy + proportion scaling, not full IK retargeting.

## Use

```cmake
add_subdirectory(libs/kinema)
target_link_libraries(game PRIVATE kinema::kinema)
```

`-DKINEMA_USE_BASE=ON` builds kinema on equilibrium's `base` library instead of
the standard library, without exceptions; the API then takes and returns
`base::Vector`, `base::String`, `base::StringRef` and `base::Optional` (see
`include/kinema/config.h`). kinema reuses an existing `equilibrium::base`
target, or builds one from `-DKINEMA_EQUILIBRIUM_DIR=<checkout>`.

```cpp
#include <kinema/kinema.h>

// Import: feed uniformly sampled poses from your source format.
kinema::ClipBuilder builder(num_bones, num_frames, 30.0f);
for (frame, bone : source) builder.SetSample(frame, bone, t, r, s);
builder.AddEvent("FootLeft", 0.43f);
builder.AddRootKey(duration, total_displacement);
kinema::OwnedClip clip(builder.Build());   // blob is disk-cacheable as-is

// Runtime: registers + a compiled program per actor archetype. (Fields are set
// individually rather than with C++20 designated initializers so the code stays
// C++17 / MSVC-clean.)
kinema::PoseArena arena(num_bones, 4);
kinema::PoseOp program[3];
program[0].kind = kinema::PoseOp::Kind::kSample; program[0].dst = 0;
program[0].clip = clip.get(); program[0].time = t;
program[1].kind = kinema::PoseOp::Kind::kSample; program[1].dst = 1;
program[1].clip = other; program[1].time = t2;
program[2].kind = kinema::PoseOp::Kind::kBlend; program[2].dst = 2;
program[2].a = 0; program[2].b = 1; program[2].alpha = 0.3f;
kinema::PoseView pose = kinema::ExecuteProgram(program, 3, arena);

// Graph core: a blend space compiled into one op, driven by live params.
kinema::BlendSpace speed(kinema::BlendSpace::Dim::k1D);
speed.Add(walk.get(), 0.0f).Add(jog.get(), 3.0f).Add(run.get(), 6.0f);
speed.Finalize();
kinema::PoseOp locomotion[1];
locomotion[0].kind = kinema::PoseOp::Kind::kBlendSpace; locomotion[0].dst = 0;
locomotion[0].a = 1;  // scratch register
locomotion[0].space = &speed; locomotion[0].time_param = 0; locomotion[0].coord_param = 1;
float params[2] = {phase01, speed_mps};       // written per frame, no realloc
kinema::PoseParams pp{params, 2};
pose = kinema::ExecuteProgram(locomotion, 1, arena, &pp);

// Sync two locomotion clips on their footfall markers; feed LocalTime back in.
kinema::SyncGroup sync;
sync.AddClip(*walk.get());  // markers = the clip's events
sync.AddClip(*run.get());
sync.Advance(dt, /*leader=*/0);
params[0] = sync.LocalTime(0);  // per-clip, foot-aligned sample times

// State machine: compile once, drive one instance per actor.
kinema::StateMachineBuilder smb(num_bones);
kinema::u16 idle = smb.AddClipState(idle_clip.get());
kinema::u16 run = smb.AddClipState(run_clip.get());
kinema::ConditionAtom go = kinema::ConditionAtom::Greater(/*param=*/0, 3.0f);
kinema::TransitionDesc to_run;
to_run.duration = 0.2f;
smb.AddTransition(idle, run, to_run, &go, 1);   // speed>3 -> run
kinema::StateMachine machine = smb.Build();

kinema::StateMachineInstance actor;
actor.Init(machine, idle);
kinema::PoseArena sm_arena(num_bones, machine.max_registers());
// Per frame: conditions fire an inertialized transition; one graph is walked.
kinema::PoseView pose = actor.Update(dt, pp, sm_arena, out_pose);
character.Move(actor.RootMotion());   // loop-aware, transition-blended
```

Jolt integration (`kinema/jolt_adapter.h`, header-only, include where Jolt
headers are visible): `MakeSkeleton`, `SetRagdollPose` (hard keying),
`DriveRagdollPose` (motor-driven soft keying for physical hit reactions).

## Testing

`-DKINEMA_BUILD_TESTS=ON` builds `kinematest`, a synthetic round-trip suite
(no game data needed). In rx, `hkxinfo --kinema` cross-validates the
transcoder against the reference Havok spline sampler over real game clips
and reports compression + performance numbers.
