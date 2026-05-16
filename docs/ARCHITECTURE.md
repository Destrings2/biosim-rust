# Architecture

biosim4-rs is a genetic neural-net artificial life simulator. Agents
evolve on a 2D toroidal-bounded grid; each agent runs a feed-forward
neural net compiled from a variable-length genome of 32-bit genes.
Sensors and actions are pluggable; survival challenges decide which
agents reproduce.

The workspace targets two frontends — a Bevy + egui GUI and a
headless CLI — over a single platform-agnostic engine.

---

## Workspace crates

| Crate | Role |
|---|---|
| `biosim4-core` | Engine. Genome, neural net, world state, registries, stepping, reproduction. No dependency on the catalogue crates. |
| `biosim4-sensors` | 40 built-in sensors. |
| `biosim4-actions` | 23 built-in actions. |
| `biosim4-challenges` | 27 built-in survival challenges plus two programmable-entity demos. |
| `biosim4-breeds` | Curated sensor/action/challenge presets (`default`, `navigator`, `forager`, `socialite`, `predator`, `scholar`, `minimal`). |
| `biosim4-native` | Headless CLI. Reads `SimConfig` JSON, runs N generations, prints stats. Enables `biosim4-core/parallel`. |
| `biosim4-bevy` | Bevy 0.18 + egui frontend. Live render, tool palette, parameter editor, fast-forward loop. |

Crate boundaries are deliberate: the engine never imports the
catalogue crates, so a new sensor never triggers a core rebuild. The
catalogue crates depend on `biosim4-core`, and the frontends depend on
all of the above.

---

## Module DAG (within `biosim4-core`)

Dependencies flow upward. No cycles.

```
analysis  sim_step  spawn
    └────────┬────────┘
          sim_state
    ┌─────────┼──────────────────────────────────┐
 sensors   actions   challenges   breeds   programmable
    └─────────┼──────────────────────────────────┘
           registry
    ┌─────────┼────────────────────────────────────┐
 population  world  signals_layer  food_layer  barriers
    └─────────┼────────────────────────────────────┘
            agent
       genome/{gene, ops, neural_net}
    ┌─────────┼────────────────────┐
   grid    sim_config     rng
              └──────┬──────────────┘
                   types
```

`SimulationState` owns every piece of mutable state (grid, population,
signals, food, programmable pool, registries, RNG, scratch buffers).
Its fields are public so the step engine can perform split borrows
(`&mut population` alongside `&grid`) without intermediate accessors.

---

## Generation lifecycle

```
initialize_generation_0(state)        ┐
  reset_world                          │  Called once at startup, then
  apply_feature_enables                │  again from spawn_new_generation
  challenges.on_generation_start       │  at every generation boundary.
                                       │  Replays the world from scratch.

for step in 0..steps_per_generation:
    step_one(state, step)              ┐  Per-step loop. See
      challenge_hooks.on_sim_step      │  docs/SIMULATION_LOOP.md for
      step_all_agents (Phase 1 + 2)    │  the call graph.
      population.drain_death_queue     │
      population.drain_move_queue      │
      programmable.step_all            │
      population.drain_death_queue     │  (programmable kill_peep_at)
      signals.fade(0)                  │
      food.regen                       │

spawn_new_generation(state)            ┐
  evaluate every alive agent           │  Survivor selection +
  build survivor pool (bootstrap       │  reproduction. Commits
    fallback at 10% if zero pass)      │  pending registry changes.
  sort by fitness ascending            │
  sensors.commit_enabled()             │
  actions.commit_enabled()             │
  preserve top-2 elites unchanged      │
  reproduce remaining via              │
    generate_child_genome              │
  reset_world + replace cohort         │
  state.generation += 1                │
  challenges.on_generation_start       │
```

The bootstrap fallback prevents stagnation on hard challenges where
zero agents pass in early generations. Elitism preserves the two
fittest genomes unchanged so a string of bad mutations cannot erase
hard-won progress.

---

## Cross-cutting patterns

### Registry pending/commit lifecycle

`SensorRegistry` and `ActionRegistry` keep two indices:

- the raw registration order (never changes after `register`), and
- the `active_map: Vec<u16>` mapping `enabled_idx → actual_idx`.

Genomes reference sensors and actions by `enabled_idx`. `set_enabled(id,
false)` queues a change; it does **not** rebuild `active_map`.

- Mid-generation: the registry's `evaluate` / `execute` short-circuits
  disabled entries to `0.0` or no-op. Existing agents are unaffected;
  their neural nets keep stable indices.
- At generation boundary: `commit_enabled()` rebuilds `active_map`.
  Newly compiled nets in `spawn_new_generation` wire against the
  updated `enabled_count`. Dead nets from the previous generation
  are never rewired.

`ChallengeRegistry` uses a different model — an explicit active list
plus a composition mode (`Any`, `All`, `WeightedSum`) — but exposes
the same JSON config surface.

### Deferred move and death queues

Agent actions never mutate the grid or population directly. They push
to two per-generation queues:

- `population.move_queue: Vec<(AgentId, Coord)>`
- `population.death_queue: Vec<AgentId>`

`step_one` drains the death queue first, then the move queue. Death
first guarantees a killed agent's slot is free before any move tries
to enter it. Immediate mutation would invalidate the `alive_ids`
snapshot being iterated and would conflict with the borrows held by
sensor and action code.

### Scratch buffers

`SimulationState.scratch.alive_ids` is snapshotted from
`population.alive_ids()` at the start of `step_all_agents`. Iterating
the snapshot lets the step engine mutate `population` mid-loop. The
buffer is reused across steps and carries no semantic state between
them.

### Raw-pointer Phase 1 / Phase 2 split in `step_one_agent`

Each agent step needs simultaneous mutable access to `agent.nnet` (to
update neuron outputs) and immutable access to `population` (for
sensors that scan neighbors). Safe references on the same
`Vec<Option<Agent>>` cannot express that split.

`step_one_agent` uses raw pointers to isolate the two domains:

- **Phase 1** (sensor eval + feed-forward) — `agent_ptr: *mut Agent`
  reaches only `agent.nnet`. Sensors receive `&Agent` (via
  `population.get(id)`) and read everything else.
- **Phase 2** (action execution) — `agent_ptr` is reused as `&mut
  Agent`. Population slots are index-stable (append-only `Vec`), so
  the pointer remains valid. The inline `SAFETY` comments in
  `sim_step.rs` document the aliasing analysis per pointer.

See [SIMULATION_LOOP.md](SIMULATION_LOOP.md) for the call graph.

### Genome modulo wiring

`Gene.source_num` and `Gene.sink_num` are raw 7-bit fields (0..127).
`create_wiring` remaps them modulo `sensor_count`, `action_count`, or
`max_neurons` at neural-net compile time, so a genome stays valid
against any registry configuration. Changing `enabled_count` shifts
the wiring semantics of every gene — `commit_enabled()` therefore
runs only at generation boundaries.

### Programmable entities

A challenge can place scripted, non-evolved entities (predators,
herders, wanderers) into the world via `state.programmable`. They
occupy grid cells, block movement, and step every tick through a
`Program` trait impl. Peeps perceive them via the `longprobe_alien_fwd`
sensor, which walks the agent's heading and reads programmable cells
directly off the grid (no shared index to refresh).

`Program::step` runs in parallel across alive programmables. It must
read freely from `ctx.world` but mutate only the entity's own fields
and a `ProgramOutput` request struct. The framework merges all
outputs sequentially after the parallel section, matching how peep
actions queue moves and deaths.

The full developer guide is in
[`crates/biosim4-core/src/programmable/README.md`](../crates/biosim4-core/src/programmable/README.md).

### Determinism contract

Determinism is conditional on thread count.

- `num_threads == 1` (or the `parallel` feature off) — fully
  reproducible at a fixed `rng_seed`. Every stochastic draw routes
  through `state.rng` or the per-agent Phase 1 hash. Same seed →
  same evolution byte-for-byte.
- `num_threads > 1` with `parallel` on — intentionally
  non-deterministic. Phase 2 workers seed thread-local RNGs from
  system entropy, and `rayon::fold + reduce` merges chunk-local
  queues in work-stealing order. Same seed → similar but not
  identical evolution. Trades roughly 3× throughput at 8 threads.

Phase 1 (sensors + neural feed-forward) uses a stateless
`(rng_seed, generation, sim_step, agent_id)` hash regardless of
thread count, so **per-agent sensor randomness is always
reproducible**. Only Phase 2 action draws diverge.

The executable contract lives in
[`crates/biosim4-core/tests/parallel_determinism.rs`](../crates/biosim4-core/tests/parallel_determinism.rs).

---

## Reference

- [`docs/SIMULATION_LOOP.md`](SIMULATION_LOOP.md) — per-step execution
  path, `step_one_agent` two-phase design, `feed_forward` invariant,
  generation transition.
- [`docs/CONFIG.md`](CONFIG.md) — `SimConfig` field reference.
- [`docs/EXTENDING.md`](EXTENDING.md) — sensor, action, challenge,
  breed, and programmable extension walkthrough.
- [`docs/BUILTINS.md`](BUILTINS.md) — catalogue of every built-in.
