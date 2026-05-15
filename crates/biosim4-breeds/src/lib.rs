//! Built-in [`Breed`] presets.
//!
//! Each breed is a curated bundle of sensor + action ids (and optionally a
//! challenge configuration) sized for a particular gameplay style. The aim
//! isn't to enumerate every reasonable combination — it's to give a new user
//! something useful to pick from in the UI without first reading the sensor
//! glossary.
//!
//! Call [`register_builtin_breeds`] after building a `SimulationState` and
//! registering the underlying sensors / actions / challenges:
//!
//! ```ignore
//! let mut state = biosim4_core::SimulationState::new(cfg);
//! biosim4_sensors::register_builtin_sensors(&mut state.sensors);
//! biosim4_actions::register_builtin_actions(&mut state.actions);
//! biosim4_challenges::register_builtin_challenges(&mut state.challenges);
//! biosim4_breeds::register_builtin_breeds(&mut state.breeds);
//! state.breeds.apply("forager", &mut state.sensors, &mut state.actions, &mut state.challenges)?;
//! ```

use biosim4_core::registry::{Breed, BreedRegistry, ChallengeConfig};

/// Register every built-in breed. `default` is registered first so it lands
/// at the top of the dropdown and matches the as-shipped runtime state of a
/// fresh `SimulationState`.
pub fn register_builtin_breeds(registry: &mut BreedRegistry) {
    registry.register(default());
    registry.register(minimal());
    registry.register(navigator());
    registry.register(forager());
    registry.register(socialite());
    registry.register(predator());
    registry.register(scholar());
}

// ── Minimal ─────────────────────────────────────────────────────────────────

/// Smallest viable sensor/action set for spatial challenges. Useful as a
/// baseline so you can prove convergence speed before adding richer sensors.
pub fn minimal() -> Breed {
    Breed::from_static(
        "minimal",
        "Minimal",
        "Tiniest workable sensor + action set — position + age in, axis-aligned moves out. Good baseline for spatial challenges.",
        &["loc_x", "loc_y", "age", "boundary_dist", "random"],
        &["move_x", "move_y", "move_random"],
        None,
    )
}

// ── Navigator ───────────────────────────────────────────────────────────────

/// Spatial / wayfinding agents — they sense barriers, walls, and direction,
/// but not pheromones or food. Pairs well with `circle`, `corner`,
/// `near_barrier`, `maze`, etc.
pub fn navigator() -> Breed {
    Breed::from_static(
        "navigator",
        "Navigator",
        "Position, boundary, and barrier-probe sensors with full directional movement. No signals, no food, no memory — pure pathfinding.",
        &[
            "loc_x",
            "loc_y",
            "boundary_dist",
            "boundary_dist_x",
            "boundary_dist_y",
            "last_move_dir_x",
            "last_move_dir_y",
            "barrier_fwd",
            "barrier_lr",
            "kill_barrier_fwd",
            "longprobe_bar_fwd",
            "osc1",
            "age",
        ],
        &[
            "move_forward",
            "move_reverse",
            "move_left",
            "move_right",
            "move_north",
            "move_south",
            "move_east",
            "move_west",
            "set_oscillator_period",
            "set_longprobe_dist",
        ],
        None,
    )
}

// ── Forager ─────────────────────────────────────────────────────────────────

/// Food + energy sensors plus movement. Designed for the energy-on
/// food-foraging gameplay.
pub fn forager() -> Breed {
    Breed::from_static(
        "forager",
        "Forager",
        "Senses food, energy, and barriers. Movement-heavy action set with memory registers so agents can stash a remembered food location.",
        &[
            "loc_x",
            "loc_y",
            "boundary_dist",
            "food_here",
            "food_fwd",
            "food_lr",
            "energy_level",
            "barrier_fwd",
            "barrier_lr",
            "last_move_dir_x",
            "last_move_dir_y",
            "memory_0",
            "memory_1",
            "age",
            "osc1",
        ],
        &[
            "move_forward",
            "move_reverse",
            "move_left",
            "move_right",
            "move_x",
            "move_y",
            "write_memory_0",
            "write_memory_1",
            "set_responsiveness",
        ],
        None,
    )
}

// ── Socialite ───────────────────────────────────────────────────────────────

/// Pheromone-aware agents — full signal sensor stack and signal-emit actions.
/// Designed for `pairs`, `string`, `quarantine`, `tag`, and other social /
/// coordination challenges. Includes the `challenge_bit_0` sensor so agents
/// know their own infected/it status.
pub fn socialite() -> Breed {
    Breed::from_static(
        "socialite",
        "Socialite",
        "Signal-rich profile: three pheromone channels (local/fwd/lr), population probes, genetic similarity, and the challenge-bit-0 self-awareness sensor.",
        &[
            "loc_x",
            "loc_y",
            "population",
            "population_fwd",
            "population_lr",
            "genetic_sim_fwd",
            "signal0",
            "signal0_fwd",
            "signal0_lr",
            "signal1",
            "signal1_fwd",
            "signal1_lr",
            "signal2",
            "signal2_fwd",
            "signal2_lr",
            "challenge_bit_0",
            "last_move_dir_x",
            "last_move_dir_y",
            "age",
        ],
        &[
            "move_forward",
            "move_reverse",
            "move_left",
            "move_right",
            "move_random",
            "emit_signal0",
            "emit_signal1",
            "emit_signal2",
            "set_responsiveness",
        ],
        None,
    )
}

// ── Predator ────────────────────────────────────────────────────────────────

/// Aggressive profile — fast movement and `kill_forward`. Pair with a
/// challenge that rewards aggression (or just watch what happens with the
/// default).
pub fn predator() -> Breed {
    Breed::from_static(
        "predator",
        "Predator",
        "Movement + kill_forward. Long-probe sensors for population spotting and barrier avoidance. Requires `kill_enable` in config.",
        &[
            "loc_x",
            "loc_y",
            "population_fwd",
            "population_lr",
            "longprobe_pop_fwd",
            "longprobe_bar_fwd",
            "genetic_sim_fwd",
            "barrier_fwd",
            "last_move_dir_x",
            "last_move_dir_y",
            "age",
            "osc1",
        ],
        &[
            "move_forward",
            "move_reverse",
            "move_left",
            "move_right",
            "move_random",
            "kill_forward",
            "set_oscillator_period",
            "set_longprobe_dist",
        ],
        None,
    )
}

// ── Scholar ─────────────────────────────────────────────────────────────────

/// Memory-heavy profile for sequential / waypoint-style challenges. Includes
/// `osc1` for clocking and all four memory registers so an agent can stash
/// recent state.
pub fn scholar() -> Breed {
    Breed::from_static(
        "scholar",
        "Scholar",
        "All four memory registers exposed for read+write, plus the oscillator and age sensors. Good for `location_sequence` and other multi-stage challenges.",
        &[
            "loc_x",
            "loc_y",
            "boundary_dist",
            "memory_0",
            "memory_1",
            "memory_2",
            "memory_3",
            "challenge_bit_0",
            "challenge_bit_1",
            "challenge_bit_2",
            "challenge_bit_3",
            "osc1",
            "age",
            "last_move_dir_x",
            "last_move_dir_y",
        ],
        &[
            "move_forward",
            "move_left",
            "move_right",
            "move_x",
            "move_y",
            "write_memory_0",
            "write_memory_1",
            "write_memory_2",
            "write_memory_3",
            "set_oscillator_period",
            "set_responsiveness",
        ],
        None,
    )
}

// ── Default ─────────────────────────────────────────────────────────────────

/// Mirrors the as-shipped runtime baseline.
///
/// The default [`SimConfig`](biosim4_core::sim_config::SimConfig) has
/// `signal_layers = 1` and `enable_energy = false`. At every generation
/// boundary, `apply_feature_enables` (in `biosim4_core::spawn`) turns OFF
/// the sensors/actions those features gate:
///
/// - signal-layer-1 trio  (`signal1`, `signal1_fwd`, `signal1_lr`) + `emit_signal1`
/// - signal-layer-2 trio  (`signal2`, `signal2_fwd`, `signal2_lr`) + `emit_signal2`
/// - food / energy quartet (`energy_level`, `food_here`, `food_fwd`, `food_lr`)
///
/// This breed enables everything EXCEPT those gated ids, so a user clicking
/// "Default → APPLY" gets the same enable mask they had at launch — without
/// having to also un-tick `enable_energy` or change `signal_layers`.
pub fn default() -> Breed {
    let sensors: Vec<String> = ALL_SENSOR_IDS
        .iter()
        .filter(|id| !FEATURE_GATED_SENSOR_IDS.contains(id))
        .map(|s| s.to_string())
        .collect();
    let actions: Vec<String> = ALL_ACTION_IDS
        .iter()
        .filter(|id| !FEATURE_GATED_ACTION_IDS.contains(id))
        .map(|s| s.to_string())
        .collect();
    Breed {
        id: "default".to_string(),
        name: "Default".to_string(),
        description: "Launch baseline — every built-in sensor and action enabled EXCEPT the ones gated by `enable_energy` and `signal_layers >= 2/3`. Matches what `apply_feature_enables` produces from the default `SimConfig`.".to_string(),
        sensors,
        actions,
        challenge: Option::<ChallengeConfig>::None,
    }
}

/// Sensor ids that `biosim4_core::spawn::apply_feature_enables` disables
/// under the default `SimConfig`. Keep in lockstep with that function.
const FEATURE_GATED_SENSOR_IDS: &[&str] = &[
    // gated by `enable_energy = false`
    "energy_level",
    "food_here",
    "food_fwd",
    "food_lr",
    // gated by `signal_layers < 2`
    "signal1",
    "signal1_fwd",
    "signal1_lr",
    // gated by `signal_layers < 3`
    "signal2",
    "signal2_fwd",
    "signal2_lr",
];

/// Action ids similarly gated. Keep in lockstep with `apply_feature_enables`.
const FEATURE_GATED_ACTION_IDS: &[&str] = &["emit_signal1", "emit_signal2"];

/// Snapshot of every built-in sensor id. Must stay in sync with
/// `biosim4_sensors::register_builtin_sensors` — the `default` breed
/// will reject at apply-time if a name drifts, which is the canary.
const ALL_SENSOR_IDS: &[&str] = &[
    "loc_x",
    "loc_y",
    "boundary_dist_x",
    "boundary_dist",
    "boundary_dist_y",
    "genetic_sim_fwd",
    "last_move_dir_x",
    "last_move_dir_y",
    "longprobe_pop_fwd",
    "longprobe_bar_fwd",
    "population",
    "population_fwd",
    "population_lr",
    "osc1",
    "age",
    "barrier_fwd",
    "barrier_lr",
    "kill_barrier_fwd",
    "random",
    "signal0",
    "signal0_fwd",
    "signal0_lr",
    "signal1",
    "signal1_fwd",
    "signal1_lr",
    "signal2",
    "signal2_fwd",
    "signal2_lr",
    "memory_0",
    "memory_1",
    "memory_2",
    "memory_3",
    "challenge_bit_0",
    "challenge_bit_1",
    "challenge_bit_2",
    "challenge_bit_3",
    "energy_level",
    "food_here",
    "food_fwd",
    "food_lr",
];

const ALL_ACTION_IDS: &[&str] = &[
    "move_x",
    "move_y",
    "move_forward",
    "move_rl",
    "move_random",
    "set_oscillator_period",
    "set_longprobe_dist",
    "set_responsiveness",
    "emit_signal0",
    "emit_signal1",
    "emit_signal2",
    "move_east",
    "move_west",
    "move_north",
    "move_south",
    "move_left",
    "move_right",
    "move_reverse",
    "kill_forward",
    "write_memory_0",
    "write_memory_1",
    "write_memory_2",
    "write_memory_3",
];
