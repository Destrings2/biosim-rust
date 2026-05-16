# Built-in catalogue

Every sensor, action, challenge, and breed shipped with the workspace.
Source of truth lives in the catalogue crates — this page reflects
their `register_builtin_*` lists.

## Sensors (40)

Implemented in `biosim4-sensors`. Every sensor returns a value in
`[0, 1]`; the registry clamps the return.

### Location (5)

| ID | Description |
|---|---|
| `loc_x` | Normalized X position (0 = left, 1 = right). |
| `loc_y` | Normalized Y position (0 = bottom, 1 = top). |
| `boundary_dist_x` | Distance to nearest east/west wall, normalized by half-width. |
| `boundary_dist_y` | Distance to nearest north/south wall, normalized by half-height. |
| `boundary_dist` | Closest distance to any wall, normalized by the larger half-axis. |

### Movement (2)

| ID | Description |
|---|---|
| `last_move_dir_x` | Last-move X component, remapped to `[0, 1]` (0.5 = none). |
| `last_move_dir_y` | Last-move Y component, remapped to `[0, 1]` (0.5 = none). |

### Population density (3)

| ID | Description |
|---|---|
| `population` | Fraction of occupied cells in `population_sensor_radius`. |
| `population_fwd` | Inverse-distance signed projection along heading. 0.5 = symmetric. |
| `population_lr` | Same projection along the right-perpendicular axis. |

### Barrier probes (3)

| ID | Description |
|---|---|
| `barrier_fwd` | Bidirectional non-barrier distance along heading. >0.5 = more space ahead. |
| `barrier_lr` | Same shape, right-perpendicular axis. |
| `kill_barrier_fwd` | Distance to nearest `KILL_BARRIER` cell ahead. 0 = none in range. |

### Long-range probes (2)

| ID | Description |
|---|---|
| `longprobe_pop_fwd` | Distance to nearest occupied cell along heading, normalized by `long_probe_dist`. |
| `longprobe_bar_fwd` | Distance to nearest barrier along heading, normalized by `long_probe_dist`. |

### Genetic (1)

| ID | Description |
|---|---|
| `genetic_sim_fwd` | Jaro-Winkler genome similarity to the agent in the cell immediately ahead. |

### Internal (3)

| ID | Description |
|---|---|
| `osc1` | `(1 − cos(2π · phase)) / 2`, phase keyed to `sim_step % agent.osc_period`. |
| `age` | `agent.age / steps_per_generation`. |
| `random` | Uniform random in `[0, 1]` from the per-agent forked RNG. |

### Signals (9)

Three pheromone layers, each with base / forward / left-right
variants. Layers 1 and 2 are gated by `config.signal_layers`.

| ID | Description |
|---|---|
| `signal0`, `signal1`, `signal2` | Average density in `signal_sensor_radius`. |
| `signal{0,1,2}_fwd` | Inverse-distance signed projection along heading. |
| `signal{0,1,2}_lr` | Same projection along right-perpendicular axis. |

### Memory (4)

| ID | Description |
|---|---|
| `memory_0`..`memory_3` | Read back the four float scratch registers written by `write_memory_*`. |

### Challenge state (4)

| ID | Description |
|---|---|
| `challenge_bit_0`..`challenge_bit_3` | Low four bits of `agent.challenge_bits`. Meaning is challenge-defined (e.g. `tag` uses bit 0 for "am I it?", `quarantine` uses bit 0 for "am I infected?"). |

### Food / energy (4)

Gated by `config.enable_energy`.

| ID | Description |
|---|---|
| `energy_level` | Agent's energy in `[0, 1]`. |
| `food_here` | Food density at the agent's cell. |
| `food_fwd` | Food density along heading. |
| `food_lr` | Food density along right-perpendicular axis. |

### Programmable entities (1)

| ID | Description |
|---|---|
| `longprobe_alien_fwd` | Forward long probe for the nearest live programmable. Walks `agent.long_probe_dist` cells along `last_move_dir`; returns `(steps − 1) / long_probe_dist` when a programmable cell is hit, or `1.0` when the probe runs off the grid, hits a barrier, hits a peep (line-of-sight block), or finds nothing in range. Same shape as `longprobe_pop_fwd`. Not in the default breed; opt in via a custom breed that lists it. |

## Actions (23)

Implemented in `biosim4-actions`. Every motor action runs through the
shared squash → responsiveness-gate → draw pipeline; helpers
(`prob2bool_responsive`, `fire_with_threshold`) are re-exported for
custom actions.

### Directional movement (8)

| ID | Description |
|---|---|
| `move_east`, `move_west`, `move_north`, `move_south` | Cardinal moves, probabilistic. |
| `move_forward`, `move_reverse` | Along / against current heading. |
| `move_left`, `move_right` | 90° from heading. |

### Composite movement (4)

| ID | Description |
|---|---|
| `move_x`, `move_y` | Add raw `level` to axis urge accumulator. |
| `move_rl` | Right-perpendicular axis; sign of `level` chooses left vs right. |
| `move_random` | Uniform random over 8 directions, contributes `level · offset`. |

### Internal modulators (3)

These bypass the responsiveness gate — otherwise low responsiveness
would damp the signal an agent uses to raise itself out of that
state.

| ID | Description |
|---|---|
| `set_responsiveness` | Sets `agent.responsiveness` from `(tanh(level) + 1) / 2`. |
| `set_oscillator_period` | Sets `agent.osc_period` to `1 + (1.5 + e^{7·f01})` where `f01 = (tanh(level)+1)/2`. |
| `set_longprobe_dist` | Sets `agent.long_probe_dist` to `1 + 32·f01`. |

### Memory (4)

Also bypass the responsiveness gate (writes need a stable range).

| ID | Description |
|---|---|
| `write_memory_0`..`write_memory_3` | Writes `(tanh(level) + 1) / 2` to the matching register. |

### Interaction (4)

| ID | Description |
|---|---|
| `emit_signal0` | Deposits pheromone at agent location, layer 0. Threshold `0.5`. |
| `emit_signal1`, `emit_signal2` | Same for layers 1 and 2. Gated by `config.signal_layers`. |
| `kill_forward` | Queues death of the agent directly ahead. Requires `config.kill_enable`. Threshold `0.5`. |

## Challenges (27)

Implemented in `biosim4-challenges` across nine submodules. Activate
via `state.challenges.set_single("<id>", Some(params_json))` or
`apply_config` for multi-challenge compositions.

### Spatial (11)

End-of-generation evaluation based on agent position.

| ID | Description |
|---|---|
| `circle` | Inside a parameterized disc. |
| `right_half` | Right half of the grid. |
| `right_quarter` | Right quarter. |
| `left_eighth` | Left eighth. |
| `east_west_eighths` | Far-east or far-west eighth. |
| `center_weighted` | Inside center disc; fitness drops with distance from center. |
| `center_unweighted` | Inside center disc; binary fitness. |
| `corner` | In any corner region. |
| `corner_weighted` | Corner region with distance-weighted fitness. |
| `against_any_wall` | Touching any wall. |
| `near_barrier` | Within radius of a barrier centroid. |

### Social (3)

| ID | Description |
|---|---|
| `pairs` | Exactly one neighbor within radius. |
| `center_sparse` | Inside center disc with low neighbor density. |
| `string` | Connected linear chain of neighbors. |

### Migration (1)

| ID | Description |
|---|---|
| `migrate_distance` | Reward proportional to distance from birth location. |

### Sequential (2)

Use `challenge_bits` for in-generation progress tracking.

| ID | Description |
|---|---|
| `touch_any_wall` | Must visit a wall during the generation. |
| `location_sequence` | Must visit a list of waypoints in order. Bits 0..n track progress. |

### World-edge hazards (2)

| ID | Description |
|---|---|
| `radioactive_walls` | Probabilistic damage per step: kill probability falls off exponentially with distance from the currently-active wall (west, then east at mid-generation). Survivor = alive at end-of-generation. |
| `lethal_borders` | Instant-kill version: any agent sitting on the world's outer border row/column dies the same step. `grace_steps` (default 1) opens with a short safe window so peeps spawned on the border can step away. Counterpart to `against_any_wall`. |

### Altruism (2)

Group-fitness challenges from the reference simulator.

| ID | Description |
|---|---|
| `altruism` | Standard altruism setup. |
| `altruism_sacrifice` | Sacrifice variant. |

### Dynamic (3)

Use `on_sim_step` / `on_generation_start` for time-varying state.

| ID | Description |
|---|---|
| `sun_tracker` | Target zone rotates over the course of the generation. |
| `diaspora` | Spawn-time clustering with end-of-generation dispersal target. |
| `survivor` | Multi-stage survival across a sequence of hazards. |

### Tag / contagion (2)

Both use `challenge_bit_0` for agent self-awareness.

| ID | Description |
|---|---|
| `tag` | Contact-transferred "it" bit. Survivors = whoever isn't it at end-of-generation. |
| `quarantine` | Contagion spreads from a seed disc through contact. Survivors = uninfected. |

### Programmable demos (2)

Reference consumers of `ProgrammablePool`. Use them as templates for
custom challenges.

| ID | Description |
|---|---|
| `wanderers` | Spawns scripted wanderers that walk randomly. |
| `predators` | Spawns hunters that chase and kill peeps. |

## Breeds (7)

Implemented in `biosim4-breeds`. Each is a curated sensor + action
preset. Apply via `state.breeds.apply("<id>", &mut sensors, &mut actions, &mut challenges)`.

| ID | Description |
|---|---|
| `default` | Launch baseline. Every built-in sensor and action enabled except those gated by `enable_energy = false` and `signal_layers < 2/3`. Matches what `apply_feature_enables` produces from `SimConfig::default()`. |
| `minimal` | Smallest viable set — position, age, axis-aligned moves. Baseline for spatial challenges. |
| `navigator` | Position + boundary + barrier-probe sensors with full directional movement. No signals, no food, no memory. |
| `forager` | Food + energy sensors, movement, and memory registers. For energy-on food-foraging gameplay. |
| `socialite` | Three pheromone channels (local/fwd/lr), population probes, genetic similarity, `challenge_bit_0`. For `pairs`, `string`, `quarantine`, `tag`. |
| `predator` | Movement + `kill_forward`, long-range probes, genetic similarity. Requires `kill_enable`. |
| `scholar` | All four memory registers (read + write), oscillator, all four challenge bits. For sequential / waypoint challenges. |
