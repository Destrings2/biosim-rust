# Configuration reference

Every runtime parameter lives in [`SimConfig`](../crates/biosim4-core/src/sim_config.rs).
The struct is flat and `serde`-derived; the CLI accepts a JSON config
file, and the GUI edits the same fields in its Config tab.

Defaults below come from `SimConfig::default()`.

## Loading

```rust
let mut config = SimConfig::default();                 // baseline
config.patch_json(r#"{"size_x": 256, "population": 5000}"#)?;  // partial override
// or
let config = SimConfig::from_json(&fs::read_to_string("config.json")?)?;
```

`patch_json` only updates keys present in the patch.

CLI flags (`--generations`, `--threads`, `--seed`) override the
corresponding config fields after the file is loaded.

## World

| Field | Default | Notes |
|---|---|---|
| `size_x`, `size_y` | 128, 128 | Grid width and height in cells. Effectively immutable after `SimulationState::new`. |
| `population` | 3000 | Agents per generation. Also immutable post-init. |
| `num_threads` | 4 | Worker threads for parallel stepping. Set 1 for deterministic runs. CLI `--threads 0` resolves to all cores. |
| `rng_seed` | 12345 | Master RNG seed. `0` draws from system entropy. |
| `signal_layers` | 1 | Number of independent pheromone layers (1–3). Layers 2 and 3 also gate the matching `signal{1,2}*` sensors and `emit_signal{1,2}` actions via `apply_feature_enables`. |

## Evolution

| Field | Default | Notes |
|---|---|---|
| `steps_per_generation` | 300 | Step count before survivor evaluation. |
| `max_generations` | 200 | CLI stop point. The GUI ignores it. |
| `genome_initial_length_min` | 24 | Lower bound on gene count for generation-0 agents. |
| `genome_initial_length_max` | 24 | Upper bound on gene count for generation-0 agents. |
| `genome_max_length` | 300 | Hard cap after insertions. |
| `max_number_neurons` | 5 | Internal neurons per net. Gene `sink_num` is taken modulo this when wiring. |
| `point_mutation_rate` | 0.001 | Per-bit flip probability applied to each child gene. |
| `gene_insertion_deletion_rate` | 0.0 | Per-genome probability of an insertion/deletion event each generation. |
| `deletion_ratio` | 0.5 | Fraction of insertion/deletion events that delete rather than insert. |
| `sexual_reproduction` | false | When true, children use slice-overlay crossover from two parents. |
| `choose_parents_by_fitness` | true | When true, parent draws are biased toward higher-fitness survivors via a `1 − r²` transform. |
| `kill_enable` | false | Gates the `kill_forward` action. When false, the action no-ops. |

## Energy system

The energy subsystem is off by default. Enabling it gates the
`energy_level`, `food_here`, `food_fwd`, and `food_lr` sensors via
`apply_feature_enables`.

| Field | Default | Notes |
|---|---|---|
| `enable_energy` | false | Master switch for the food / energy subsystem. |
| `energy_per_step_cost` | 0.003 | Energy deducted from each agent per step. |
| `food_regen_rate` | 0.0005 | Food added per non-barrier cell per step. |
| `food_initial_density` | 0.3 | Fraction of non-barrier cells seeded with food at generation start. |

## Agent defaults

Set at agent spawn time. Actions like `set_responsiveness` and
`set_longprobe_dist` overwrite these per agent.

| Field | Default | Notes |
|---|---|---|
| `responsiveness` | 0.5 | Initial responsiveness gate. Multiplied through `response_curve` before scaling motor action probabilities. |
| `responsiveness_curve_k_factor` | 2.0 | Steepness of the responsiveness sigmoid. |
| `population_sensor_radius` | 2.5 | Cell radius for `population*` sensors. |
| `signal_sensor_radius` | 2.0 | Cell radius for `signal{0,1,2}*` sensors. |
| `long_probe_distance` | 16 | Initial `long_probe_dist` for the `longprobe_*` sensors. |
| `short_probe_barrier_distance` | 4 | Cell radius for `barrier_fwd` / `barrier_lr`. |

## Environment

| Field | Default | Notes |
|---|---|---|
| `barrier_type` | 0 | Procedural barrier layout. `0` = none, `1`–`7` = preset patterns. User-painted overrides in `SimulationState.user_barriers` layer on top each generation. |

## Analysis and output

| Field | Default | Notes |
|---|---|---|
| `genome_analysis_stride` | 25 | Stride between deep analysis passes (`--verbose` CLI only). |
| `display_sample_genomes` | 5 | Genomes printed per analysis pass. |
| `genome_comparison_method` | 0 | Diversity metric: `0` = Jaro-Winkler, `1` = Hamming bits, `2` = Hamming bytes. |
| `save_video` | true | CLI-only flag. Reserved — the native runner does not currently emit video. |
| `video_stride` | 25 | Reserved. |

## Worked example

Reproducible single-threaded run with stronger mutation pressure:

```json
{
  "size_x": 128,
  "size_y": 128,
  "population": 1000,
  "num_threads": 1,
  "rng_seed": 42,
  "steps_per_generation": 300,
  "max_generations": 500,
  "point_mutation_rate": 0.005,
  "gene_insertion_deletion_rate": 0.01,
  "sexual_reproduction": true,
  "barrier_type": 3
}
```

```sh
cargo run -p biosim4-native --release -- --config example.json
```
