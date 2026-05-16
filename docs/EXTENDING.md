# Extending the simulator

Every extension point is a trait. Implement the trait, register the
instance with the matching registry on `SimulationState`, and it
participates in the next generation.

## Sensor

A sensor maps agent + world state to a neural input in `[0, 1]`. The
registry clamps return values, so out-of-range output is silently
truncated rather than crashing.

```rust
use biosim4_core::registry::{Sensor, SensorContext};

struct GrudgeSensor;

impl Sensor for GrudgeSensor {
    fn id(&self) -> &str { "grudge" }
    fn name(&self) -> &str { "grudge" }

    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // Read ctx.agent (the agent being evaluated, &Agent),
        // ctx.world (read-only World view),
        // ctx.sim_step (u32),
        // ctx.rng (per-agent forked RNG, deterministic by
        //          (rng_seed, generation, sim_step, agent_id) hash).
        0.5
    }
}

state.sensors.register(Box::new(GrudgeSensor));
```

The sensor takes effect on the next `commit_enabled()` (called inside
`spawn_new_generation`). Mid-generation enable/disable is supported
via `state.sensors.set_enabled("grudge", false)`; the change short-
circuits `evaluate` immediately and reshapes wiring at the next
generation boundary. See [ARCHITECTURE.md](ARCHITECTURE.md#registry-pendingcommit-lifecycle).

## Action

An action consumes a raw neural activation level (an arbitrary float)
and produces a side effect — typically queueing a move or death,
emitting a signal, or writing to a per-agent modulator.

```rust
use biosim4_core::registry::{Action, ActionContext};

struct Croak;

impl Action for Croak {
    fn id(&self) -> &str { "croak" }
    fn name(&self) -> &str { "croak" }

    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        // Squash level → probability; gate on responsiveness.
        let p = ((level.tanh() + 1.0) / 2.0) * ctx.responsiveness_adjusted;
        if p > 0.5 && ctx.rng.gen_bool(p) {
            ctx.signals.increment(0, ctx.agent.loc, ctx.world.grid);
        }
    }
}

state.actions.register(Box::new(Croak));
```

`biosim4_actions` exposes helpers for the common pipelines:
`level_to_prob`, `level_to_signed_prob`, `prob2bool`,
`prob2bool_responsive`, `fire_with_threshold`. Use them so a new
action matches the response curve of the built-ins.

Motor actions add to `ctx.move_x_urge` / `ctx.move_y_urge`. The
framework's `resolve_movement` collapses the accumulated urges into a
single step at the end of `step_one_agent`. Two opposing axis urges
cancel; orthogonal urges combine into a diagonal step.

## Challenge

A challenge evaluates each alive agent at generation end and returns
`(pass, fitness)`. The pass flag selects survivors; the fitness score
(0.0–1.0) biases parent selection.

```rust
use biosim4_core::registry::challenge::{
    Challenge, ChallengeOverlay, WorldMut,
};
use biosim4_core::agent::Agent;
use biosim4_core::world::World;
use serde_json::Value;

#[derive(Default)]
pub struct HermitChallenge { pub radius: f32 }

impl Challenge for HermitChallenge {
    fn id(&self) -> &str { "hermit" }
    fn name(&self) -> &str { "Hermit" }

    fn params_schema(&self) -> Value {
        serde_json::json!({ "radius": { "type": "number", "default": 5.0 } })
    }
    fn configure(&mut self, params: Value) {
        if let Some(r) = params.get("radius").and_then(|v| v.as_f64()) {
            self.radius = r as f32;
        }
    }

    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        // Pass if no neighbor within `radius` cells.
        // Fitness = normalized distance to nearest neighbor.
        (true, 1.0)
    }

    fn overlays(&self) -> Vec<ChallengeOverlay> { vec![] }

    fn on_generation_start(&mut self, _ctx: &mut WorldMut) {}
    fn on_sim_step(&mut self, _ctx: &mut WorldMut) {}
}

state.challenges.register(Box::new(HermitChallenge::default()));
state.challenges.set_single("hermit", Some(serde_json::json!({ "radius": 8.0 })));
```

`on_generation_start` and `on_sim_step` receive `&mut WorldMut`. Use
them to mutate `challenge_bits` on agents, spawn programmable
entities, write to signal layers, or queue deaths.

Multiple active challenges combine through `ChallengeComposition`:
`Any` (default), `All`, or `WeightedSum { weights, threshold }`.

## Breed

A breed is a curated bundle of sensor ids, action ids, and an
optional challenge configuration. Applying a breed disables every
sensor and action not in its set, then enables the listed ones.

```rust
use biosim4_core::registry::Breed;

let archer = Breed::from_static(
    "archer",
    "Archer",
    "Long-range scouting with kill_forward.",
    &[
        "loc_x", "loc_y", "longprobe_pop_fwd", "longprobe_bar_fwd",
        "last_move_dir_x", "last_move_dir_y", "age",
    ],
    &[
        "move_forward", "move_left", "move_right",
        "kill_forward", "set_longprobe_dist",
    ],
    None,
);

state.breeds.register(archer);
state.breeds.apply(
    "archer",
    &mut state.sensors,
    &mut state.actions,
    &mut state.challenges,
)?;
```

`Breed::apply` returns an error if any listed id is missing from the
registry, so breeds catch typos before the next generation runs.

## Programmable entity

A programmable is a non-evolved entity that lives in a grid cell and
runs deterministic Rust logic each step. Use it for predators,
herders, or any scripted creature that occupies space but does not
reproduce.

```rust
use biosim4_core::programmable::{
    OwnerTag, Program, ProgramContext, ProgramOutput, Programmable,
};

const HUNTER_OWNER: OwnerTag = 0xA001;

struct Hunter;

impl Program for Hunter {
    fn id(&self) -> &str { "hunter" }
    fn name(&self) -> &str { "Hunter" }

    fn step(
        &self,
        this: &mut Programmable,
        ctx: &mut ProgramContext,
        out: &mut ProgramOutput,
    ) {
        // Read ctx.world freely. Mutate only `this` and `out`.
        // out.move_to = Some(target);
        // out.kill_peep_at = Some(victim_loc);
    }
}
```

Then spawn from a challenge's `on_generation_start` hook:

```rust
fn on_generation_start(&mut self, ctx: &mut WorldMut) {
    let prog = ctx.programmable.register_or_get("hunter", || Box::new(Hunter));
    for _ in 0..self.count {
        let loc = ctx.grid.find_empty_location(ctx.rng);
        let _ = ctx.programmable.spawn(ctx.grid, prog, HUNTER_OWNER, loc, [220, 60, 60]);
    }
}
```

The pool is cleared automatically inside `reset_world` at every
generation rollover. Don't clean up manually.

`Program::step` runs in parallel across alive programmables. It must
read `ctx.world` and mutate only `this` and `out`. The framework
applies the requested effects sequentially after the parallel
section. See [`crates/biosim4-core/src/programmable/README.md`](../crates/biosim4-core/src/programmable/README.md)
for the full contract, including the merge order for a single
entity's outputs and the grid encoding for programmable cells.

## Registering built-ins

The catalogue crates each expose a single registration function:

```rust
let mut state = SimulationState::new(config);
biosim4_sensors::register_builtin_sensors(&mut state.sensors);
biosim4_actions::register_builtin_actions(&mut state.actions);
biosim4_challenges::register_builtin_challenges(&mut state.challenges);
biosim4_breeds::register_builtin_breeds(&mut state.breeds);
biosim4_core::initialize_generation_0(&mut state);
```

Register custom sensors/actions/challenges either before or after the
built-ins. Order only matters for registration index (which is
opaque) — `id` strings are the stable handle.
