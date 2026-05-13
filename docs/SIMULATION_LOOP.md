# Simulation Loop Internals

This document covers the execution path through one simulation step and one generation transition. For the broader architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Step Execution Call Graph

```
step_one(state, step)
│
├── state.sim_step = step
│
├── run_challenge_step_hooks(state)
│   └── challenges.on_sim_step(&mut WorldMut)
│       reads: grid, signals, population, rng, config
│       writes: grid, signals, population (via WorldMut)
│
├── step_all_agents(state)
│   ├── scratch.alive_ids ← clone of population.alive_ids   [no alloc]
│   └── for each id in scratch.alive_ids:
│       └── step_one_agent(state, id)
│           │
│           ├── Phase 1 — sensor eval + neural feed-forward
│           │   reads:  population (neighbors), grid, signals, config
│           │   writes: agent.nnet.neurons[*].output (via raw ptr)
│           │           scratch.action_accum, scratch.neuron_accum
│           │
│           └── Phase 2 — action execution
│               reads:  scratch.action_accum
│               writes: population.move_queue, population.death_queue
│                       signals (EmitSignal0)
│                       agent.{responsiveness, osc_period, long_probe_dist, age}
│
├── population.drain_death_queue(&mut grid)
│   writes: agent.alive = false, grid cell → EMPTY, alive_ids (retain)
│
├── population.drain_move_queue(&mut grid)
│   writes: grid (old cell → EMPTY, new cell → id), agent.{loc, last_move_dir, heading}
│
└── signals.fade(0)
    writes: signals.layers[0][x][y] = saturating_sub(1) for all cells
```

---

## `step_one_agent` Two-Phase Design

Each agent's tick is split into two phases to satisfy Rust's aliasing rules without sacrificing performance.

**Phase 1: sensor evaluation → neural feed-forward**

The neural net update writes into `agent.nnet.neurons[*].output` in-place. Sensors read from the rest of the agent (`loc`, `heading`, `age`, `osc_period`, `long_probe_dist`, `genome`, `responsiveness`, `last_move_dir`) and from the population at large (neighbor scanning). These two access paths — `&mut agent.nnet` and `&Agent`/`&Population` — are disjoint, so the implementation uses raw pointers to express the split.

The sensor evaluation callback is a closure passed into `feed_forward`:
```
|sensor_idx| → {
    let agent_ref = world.population.get(id).unwrap();
    let mut ctx = SensorContext { agent: agent_ref, world: &world, … };
    state.sensors.evaluate(sensor_idx, &mut ctx)
}
```

The implementation forks `sensor_rng` from `state.rng` before deriving any raw pointer, keeping the sensor RNG and the main RNG completely independent.

**Phase 2: action execution**

Phase 2 reads `action_accum` (written during Phase 1) as `&[f32]` via a raw pointer while `ActionContext` holds `&mut` references to `agent`, `move_queue`, `death_queue`, `signals`, and `rng`. These fields are disjoint from `action_accum` (which lives in `scratch`), so the aliasing is safe.

**Why the phases cannot collapse:** Phase 1 needs a stable `&Population` for neighbor-scanning sensors. Phase 2 needs `&mut Agent` (to update modulators) and `&mut Vec<...>` move/death queues (which are sub-fields of `Population`). Holding both `&Population` and `&mut Agent` through the same `population` field at the same time requires the raw pointer split.

---

## `feed_forward` Ordering Invariant

`NeuralNet.connections` is sorted: neuron→neuron connections come first, neuron/sensor→action connections come last. `feed_forward` exploits this ordering:

1. Walk connections in order, accumulating `weight * source_value` into `neuron_accum` or `action_accum`.
2. The first time an action-sink connection is encountered, apply `tanh()` to all driven neuron accumulators and write the results back into `neurons[i].output`. Un-driven neurons skip this step and keep their existing `output` value (which initializes to `0.5` and persists across steps), contributing a constant bias.
3. Continue accumulating into `action_accum`.

`tanh` applies exactly once per step per neuron, regardless of how many connections that neuron has. Applying `tanh` neuron-by-neuron inside the connection loop produces incorrect outputs (the tanh of a partial sum, not the full sum).

---

## Generation Transition (`spawn_new_generation`)

```
spawn_new_generation(state) -> survivor_count
│
├── world = state.world()          [read-only snapshot for challenge evaluation]
│
├── for each alive agent:
│   └── challenges.evaluate(agent, &world) -> (pass: bool, fitness: f32)
│
├── survivor_pool = agents where pass == true
│
├── if survivor_pool.is_empty() && population non-empty:
│   │   [bootstrap fallback — prevents selection stagnation on hard challenges]
│   └── take top 10% (min 2) by fitness regardless of pass flag
│
├── sort survivor_pool ascending by fitness
│       (so generate_child_genome bias works: higher index = fitter)
│
├── sensors.commit_enabled()       [apply pending enable/disable changes]
│   actions.commit_enabled()
│   wiring_cfg = state.wiring_config()
│
├── elites = top 2 survivors (from end of sorted pool)
│       [preserved unchanged to protect the best genome from mutation]
│
├── fill remaining population via generate_child_genome(parents, params, rng)
│   ├── sexual=true: slice-overlay crossover of two parents
│   │       (parent selection biased toward fitness via 1 - r² transform)
│   └── apply point mutations and insertion/deletion to child
│
├── world reset: population.clear(), grid.zero_fill(), create_barrier(), reapply_user_barriers()
│
├── place new agents on grid (compile nnet from genome + wiring_cfg)
│
├── state.generation += 1
│
└── challenges.on_generation_start(&mut WorldMut)
```

**Bootstrap fallback:** if zero agents pass the challenge (common in generation 0 on hard challenges like `location_sequence`), the algorithm takes the top 10% by raw fitness score as "soft" parents. Without this fallback, the algorithm would select all parents uniformly at random from the dead population, producing no selection gradient.

**Elitism:** the two fittest survivors are copied into the next generation unchanged. This prevents a sequence of harmful mutations from erasing a genome that was hard to evolve. On easy challenges where many agents pass, elitism has negligible effect. On hard challenges where only a handful pass, elitism guards the best-evolved genomes against loss.

---

## Alive-IDs Snapshot

At the start of `step_all_agents`:

```rust
state.scratch.alive_ids.clear();
state.scratch.alive_ids.extend_from_slice(state.population.alive_ids());
for i in 0..state.scratch.alive_ids.len() {
    let id = state.scratch.alive_ids[i];
    step_one_agent(state, id);
}
```

Two design choices here:

1. **Why snapshot into scratch instead of iterating `alive_ids` directly:** `step_one_agent` takes `&mut SimulationState`, which includes `population`. Holding a reference into `population.alive_ids` for the loop would conflict with the `&mut population` that action execution needs.

2. **Why indexed loop instead of `for id in &state.scratch.alive_ids`:** A range-for borrow would hold `&state.scratch` for the duration of the loop body, conflicting with `step_one_agent(state, ...)` taking `&mut state` (which includes `&mut state.scratch`). The index-based walk borrows `state.scratch.alive_ids[i]` for a single expression evaluation, then releases it before the `&mut state` is taken.

The snapshot is reused every step (no allocation); `alive_ids` in the main population is the live source of truth for death and move drain operations.

---

## `iter_alive_mut` Safety

`Population::iter_alive_mut` produces `&mut Agent` references from an `alive_ids` list without holding `&mut Population` for each element:

```rust
// SAFETY: alive_ids contains unique AgentId values. Each call to next()
// borrows a distinct slot of `agents`. The returned iterator holds the
// unique &mut borrow of the population for its full lifetime.
```

The raw pointer approach (`*const [AgentId]` + `*mut Vec<Option<Agent>>`) is necessary because the Rust iterator model cannot return `&mut` references into a container while also borrowing from the container's own index structure. `spawn` always pushes a new sequential ID, and `drain_death_queue` removes IDs before they can be duplicated, guaranteeing `alive_ids` values remain unique at all times.
