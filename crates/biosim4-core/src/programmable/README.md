# Programmable peeps — developer guide

This module gives challenges a way to put **non-evolved, scripted entities** in
the world alongside the evolving peeps. They occupy grid cells, peeps can see
them through the forward-only `longprobe_alien_fwd` probe, and they step
every tick via a Rust trait. The Wanderers challenge in `biosim4-challenges`
is the smoke-test consumer; a predator challenge fits the same shape.

## When to use a programmable

Use a programmable when a challenge needs an entity that:

- lives in a grid cell and blocks movement like a peep,
- runs deterministic Rust logic, not a neural net,
- doesn't reproduce or carry genes across generations.

Use a peep instead when the entity should evolve. Use a grid barrier or
signal layer when it has no per-tick state.

## Core types

```rust
// crates/biosim4-core/src/programmable/mod.rs

pub type ProgrammableId = u32;   // 1..0x7FFF_FFFF
pub type ProgramId      = u16;   // index into the pool's program registry
pub type OwnerTag       = u32;   // free-form challenge-side discriminator

pub struct Programmable {
    pub id: ProgrammableId,
    pub loc: Coord,
    pub heading: Dir,
    pub alive: bool,
    pub program: ProgramId,
    pub owner: OwnerTag,
    pub state: [f32; 8],   // 8 free slots for cooldowns, counters, etc.
    pub color: [u8; 3],    // rendered verbatim by the grid renderer
}

pub trait Program: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn step(&self, this: &mut Programmable, ctx: &mut ProgramContext, out: &mut ProgramOutput);
    fn on_spawn(&self, _this: &mut Programmable, _ctx: &mut ProgramContext) {}
    fn on_despawn(&self, _this: &mut Programmable, _ctx: &mut ProgramContext) {}
}

pub struct ProgramContext<'a> {
    pub world: &'a World<'a>,   // read-only snapshot
    pub sim_step: u32,
    pub generation: u32,
    pub rng: &'a mut Rng,       // worker-local; no determinism guarantee
}

#[derive(Default)]
pub struct ProgramOutput {
    pub move_to: Option<Coord>,
    pub die: bool,
    pub kill_peep_at: Option<Coord>,
    pub signal_emit: Option<u8>,   // layer index
    pub set_color: Option<[u8; 3]>,
}
```

A `Programmable` is the entity. A `Program` impl is the **behavior shared by
all entities of a species**. Per-entity state lives in `Programmable.state`
or a side-table the impl owns and keys by `ProgrammableId`.

## Per-step pipeline

Each `sim_step::step_one` runs in this order:

1. `run_challenge_step_hooks(state)` — challenges run their `on_sim_step`
   (may spawn programmables, mutate `challenge_bits`, etc.).
2. **Peep step** (parallel, rayon): sensors evaluate, neural nets feed
   forward, actions queue moves and deaths. `longprobe_alien_fwd` walks
   the grid directly through the pool's grid encoding (no shared index
   to refresh, so the parallel section locks nothing).
3. Drain peep death queue, then peep move queue.
4. **Programmable step** (parallel, rayon): each alive programmable's
   `Program::step` runs in its own task. The framework merges all outputs
   sequentially after the parallel section.
5. Drain peep death queue again (for `kill_peep_at` requests).
6. Fade signals, regenerate food.

Step 4 is gated at the call site: if the pool is empty the entire stack
frame is skipped. Peep-only runs pay one boolean check per step for the
infrastructure.

### Parallel-safety contract

`Program::step` runs concurrently across alive programmables. The contract:

- **Read freely** from `ctx.world`, including `ctx.world.programmable` for
  other entities. Siblings reflect their state at the *start* of the step,
  not mid-merge.
- **Mutate only** `this` (the entity's own fields) and `out` (the requested
  effects).
- **Don't** mutate the grid, the population, or sibling programmables from
  inside `step`. Use `out.move_to` / `out.kill_peep_at` / `out.signal_emit`
  instead; the framework's sequential merge applies them.

This mirrors how peep actions queue moves and deaths rather than mutating
the grid directly.

### Merge order for a single entity

The framework applies one entity's outputs in this fixed order:

1. State write-back — `state`, `heading`, `color` from the mutated `this`.
2. `set_color` override, if present.
3. `die` — clears the grid cell and marks `alive = false`.
4. `kill_peep_at` — queues the peep on that cell for death.
5. `move_to` — attempts the move. The cell can be empty (move applies),
   a kill barrier (entity dies), a regular barrier (blocked), or occupied
   (blocked).
6. `signal_emit` — emits at the entity's possibly-updated `loc`.

Cross-entity merge order tracks the rayon collected order. Two
programmables requesting the same destination resolve to whichever lands
first; the loser stays put. This isn't reproducible across thread counts —
the codebase has already chosen speed over determinism.

## Writing a challenge that owns programmables

Two files: the `Program` impl and the `Challenge` that spawns it. The full
working example is `crates/biosim4-challenges/src/wanderers.rs`.

```rust
use biosim4_core::programmable::{
    OwnerTag, Program, ProgramContext, ProgramOutput, Programmable,
};
use biosim4_core::registry::challenge::{Challenge, WorldMut};
use biosim4_core::types::Coord;

const HUNTER_OWNER: OwnerTag = 0xA002;

struct Hunter;

impl Program for Hunter {
    fn id(&self) -> &str { "hunter" }
    fn name(&self) -> &str { "Hunter" }
    fn step(&self, this: &mut Programmable, ctx: &mut ProgramContext, out: &mut ProgramOutput) {
        // Read the world freely; write only to `this` and `out`.
        // out.move_to = Some(...);
        // out.kill_peep_at = Some(...);
    }
}

pub struct PredatorsChallenge { pub count: u16 }

impl Challenge for PredatorsChallenge {
    fn id(&self) -> &str { "predators" }
    fn name(&self) -> &str { "Predators" }
    // ... params_schema, configure, evaluate ...

    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        let prog = ctx.programmable.register_or_get("hunter", || Box::new(Hunter));
        for _ in 0..self.count {
            let loc = ctx.grid.find_empty_location(ctx.rng);
            let _ = ctx.programmable.spawn(ctx.grid, prog, HUNTER_OWNER, loc, [220, 60, 60]);
        }
    }
}
```

Then register the challenge in `biosim4-challenges/src/lib.rs::register_builtin_challenges`.

### What the pool guarantees

- `clear` runs automatically inside `reset_world` at every generation
  rollover. Don't clean up manually in `on_generation_start`.
- `spawn` returns `None` if the target cell isn't empty.
- `register_or_get` is idempotent. Safe to call every `on_generation_start`.
- `OwnerTag` lets a future hook clear only one challenge's entities via
  `clear_for_owner`, useful when multiple challenges share the pool.

### Heavier per-agent state

`state: [f32; 8]` covers cooldowns, target ids, patrol indices, last-seen
coords. For larger or non-float state — a waypoint list, a full struct —
keep a side-table on the `Program` impl keyed by `ProgrammableId`. The
trait is `Send + Sync`, so wrap the table in `Mutex` or `RwLock` if it
needs interior mutation, or rebuild it on each `on_generation_start`.

## Grid encoding

Cell values discriminate by exact-value checks plus bit 31:

| Range                       | Meaning                       |
| --------------------------- | ----------------------------- |
| `0`                         | `EMPTY`                       |
| `1..0x7FFF_FFFF` (bit 31=0) | live `AgentId`                |
| bit 31 = 1, except below    | `ProgrammableId` (lower bits) |
| `0xFFFF_FFFE`               | `KILL_BARRIER` sentinel       |
| `0xFFFF_FFFF`               | `BARRIER` sentinel            |

Use the `CellKind` enum and `cell_kind(cell)` helper instead of bit-checking
by hand. `encode_programmable(id)` and `programmable_id_of(cell)` round-trip
through the flag. `is_empty_at`, `is_occupied_at`, and `is_barrier_at` keep
working unchanged: programmables count as "occupied" because the cell value
isn't `EMPTY` or a barrier sentinel.

## Sensing programmables from peeps

The crate ships one generic sensor:

- **`longprobe_alien_fwd`** — forward long probe for the nearest live
  programmable. Walks `agent.long_probe_dist` cells along
  `agent.last_move_dir`, returning `(steps − 1) / long_probe_dist` if the
  probe finds a programmable cell, or `1.0` if it runs off the grid, hits
  a barrier, hits a peep (line-of-sight block), or finds nothing within
  range. Same shape as `longprobe_pop_fwd` — closer = lower reading.

It's *not* in the default breed's NN wiring. Adding it to every run would
cost one extra input weight per peep per step on peep-only runs. To enable
it, list `"longprobe_alien_fwd"` in your custom breed's sensor set
(`biosim4-breeds/src/lib.rs`). The probe reads the grid directly through
the pool's cell encoding (`grid::cell_kind`), so peeps see a fresh value
every step without any shared index to maintain.

To add a challenge-specific sensor (e.g. "distance to predators only"),
implement `Sensor` in your own crate and have your custom breed enable it.

## Lifecycle summary

| Event                     | What happens                                                |
| ------------------------- | ----------------------------------------------------------- |
| `SimulationState::new`    | Pool created, empty.                                        |
| `initialize_generation_0` | Pool cleared. Challenges may spawn in `on_generation_start`.|
| Every `step_one`          | Programs step in parallel after peeps.                      |
| `spawn_new_generation`    | Pool cleared inside `reset_world`. Challenges respawn.      |
| `Recreate` (resize)       | Pool cleared with the grid.                                 |

## Files in this module

| File         | Contents                                                        |
| ------------ | --------------------------------------------------------------- |
| `mod.rs`     | `Programmable`, `Program`, `ProgramContext`, `ProgramOutput`, `ProgrammablePool`, `step_all`. |
| `library.rs` | Reusable helpers (`has_line_of_sight`, `nearest_peep_in_los`, `move_towards`, `random_walk_step`). |

Related files outside the module:

| File                                            | Role                                                                 |
| ----------------------------------------------- | -------------------------------------------------------------------- |
| `core/src/grid.rs`                              | `CellKind`, `cell_kind`, `encode_programmable`, `PROGRAMMABLE_FLAG`. |
| `core/src/sim_step.rs`                          | Per-step orchestration; calls `step_all`.                            |
| `core/src/spawn.rs`                             | `reset_world` clears the pool at every rollover.                     |
| `sensors/src/programmable.rs`                   | `LongprobeAlienFwd` sensor.                                          |
| `sensors/src/helpers.rs`                        | `long_probe_alien_fwd` — the probe walk it delegates to.             |
| `challenges/src/wanderers.rs`                   | Reference consumer; ~160 lines including JSON schema.                |
| `bevy/src/grid_render.rs`                       | Paints programmable cells from their `color` field.                  |
| `bevy/src/ui/inspector.rs`                      | Inspector panel for selected programmables.                          |

## Tests

- `core/src/programmable/mod.rs` — unit tests covering spawn, despawn,
  clear, and `step_all` move and die paths.
- `core/src/programmable/library.rs` — unit tests for `has_line_of_sight`
  and `nearest_peep_in_los`.
- `core/tests/programmable_e2e.rs` — end-to-end tests covering generation
  rollover, programmable movement, and the `longprobe_alien_fwd` sensor.

Run all of them with:

```sh
cargo test --workspace --features biosim4-core/parallel
```
