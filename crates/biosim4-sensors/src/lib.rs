//! Built-in sensor implementations (40 sensors).
//!
//! Every sensor implements [`Sensor`] and returns a value in \[0.0, 1.0\].
//! The registry enforces the clamp at `evaluate()` time.
//!
//! Sensor structs are grouped into themed submodules; this crate's only
//! public surface is [`register_builtin_sensors`] (and the helper
//! free-functions in [`helpers`], which custom sensors can reuse).
//!
//! # Sensor catalogue
//!
//! **Location (5):** `loc_x`, `loc_y` — normalized position (0 = left/bottom,
//! 1 = right/top). `boundary_dist_x`, `boundary_dist_y` — normalized distance
//! to the nearest wall on that axis. `boundary_dist` — nearest wall overall.
//!
//! **Genetic (1):** `genetic_sim_fwd` — Jaro-Winkler similarity to the nearest
//! agent within `long_probe_dist` ahead.
//!
//! **Movement (2):** `last_move_dir_x`, `last_move_dir_y` — last move direction
//! components, normalized to \[0, 1\] (0.5 = stationary/center).
//!
//! **Population density (3):** `population` — fraction occupied within
//! `population_sensor_radius`. `population_fwd` / `population_lr` —
//! inverse-distance-weighted signed projection along the heading axis
//! and the right-perpendicular axis respectively, mapped to `[0, 1]`
//! with `0.5` as "symmetric / empty".
//!
//! **Barrier probes (2):** `barrier_fwd` — bidirectional non-barrier
//! distance along the heading axis; `0.5` means symmetric proximity,
//! `>0.5` means more clear space ahead. `barrier_lr` — same shape on the
//! right-perpendicular axis (left-vs-right asymmetry).
//!
//! **Long probes (2):** `longprobe_pop_fwd`, `longprobe_bar_fwd` — distance
//! along heading to nearest occupied cell or barrier, normalized by
//! `long_probe_dist` so higher = farther (`1.0` when nothing is in
//! range).
//!
//! **Internal (3):** `osc1` — oscillator `(1 − cos(2π · phase)) / 2`
//! keyed to the global step and the agent's `osc_period`. `age` —
//! `age / steps_per_generation`. `random` — uniform random in \[0, 1\]
//! via the per-agent forked RNG.
//!
//! **Signals (9):** `signal0..2`, `signal0..2_fwd`, `signal0..2_lr` — three
//! independent pheromone channels. The base sensor reads average density
//! in the neighborhood; `*_fwd` and `*_lr` are inverse-distance-weighted
//! projections of the signal magnitude along the heading axis and the
//! right-perpendicular axis respectively (same `[0, 1]` mapping with
//! `0.5` as symmetric).
//!
//! **Memory (4):** `memory_0..3` — read back the four float scratch registers
//! that `write_memory_N` actions write to.
//!
//! **Food / energy (4):** `energy_level`, `food_here`, `food_fwd`, `food_lr`.
//!
//! **Challenge state (4):** `challenge_bit_0..3` — read the low four bits of
//! `agent.challenge_bits`. The meaning is challenge-defined (e.g. `tag` uses
//! bit 0 for "am I it?"; `quarantine` uses bit 0 for "am I infected?").
//!
//! **Programmable entities (1):** `longprobe_alien_fwd` — forward long-probe
//! for the nearest live entry in [`biosim4_core::programmable`]. Same shape
//! as `longprobe_pop_fwd` but only fires on programmable cells; peeps and
//! barriers block the probe. Returns `1.0` when nothing is in range. The
//! label is deliberately generic — any challenge can register its own kind
//! of non-evolved entity. See [`biosim4_core::programmable`].

pub mod helpers;

mod barrier;
mod challenge_bits;
mod food;
mod genetic;
mod internal;
mod location;
mod memory;
mod movement;
mod population;
mod programmable;
mod signal;

use biosim4_core::registry::SensorRegistry;

pub fn register_builtin_sensors(registry: &mut SensorRegistry) {
    registry.register(Box::new(location::LocX));
    registry.register(Box::new(location::LocY));
    registry.register(Box::new(location::BoundaryDistX));
    registry.register(Box::new(location::BoundaryDist));
    registry.register(Box::new(location::BoundaryDistY));
    registry.register(Box::new(genetic::GeneticSimFwd));
    registry.register(Box::new(movement::LastMoveDirX));
    registry.register(Box::new(movement::LastMoveDirY));
    registry.register(Box::new(barrier::LongprobePopFwd));
    registry.register(Box::new(barrier::LongprobeBarFwd));
    registry.register(Box::new(population::PopulationSensor));
    registry.register(Box::new(population::PopulationFwd));
    registry.register(Box::new(population::PopulationLR));
    registry.register(Box::new(internal::Osc1));
    registry.register(Box::new(internal::Age));
    registry.register(Box::new(barrier::BarrierFwd));
    registry.register(Box::new(barrier::BarrierLR));
    registry.register(Box::new(barrier::KillBarrierFwd));
    registry.register(Box::new(internal::RandomSensor));
    registry.register(Box::new(signal::Signal0));
    registry.register(Box::new(signal::Signal0Fwd));
    registry.register(Box::new(signal::Signal0LR));
    registry.register(Box::new(signal::Signal1));
    registry.register(Box::new(signal::Signal1Fwd));
    registry.register(Box::new(signal::Signal1LR));
    registry.register(Box::new(signal::Signal2));
    registry.register(Box::new(signal::Signal2Fwd));
    registry.register(Box::new(signal::Signal2LR));
    registry.register(Box::new(memory::Memory0));
    registry.register(Box::new(memory::Memory1));
    registry.register(Box::new(memory::Memory2));
    registry.register(Box::new(memory::Memory3));
    registry.register(Box::new(challenge_bits::ChallengeBit0));
    registry.register(Box::new(challenge_bits::ChallengeBit1));
    registry.register(Box::new(challenge_bits::ChallengeBit2));
    registry.register(Box::new(challenge_bits::ChallengeBit3));
    registry.register(Box::new(food::EnergyLevel));
    registry.register(Box::new(food::FoodHere));
    registry.register(Box::new(food::FoodFwd));
    registry.register(Box::new(food::FoodLR));
    registry.register(Box::new(programmable::LongprobeAlienFwd));
}
