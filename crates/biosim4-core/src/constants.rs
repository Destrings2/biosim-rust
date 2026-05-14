//! Hard-coded tuning constants shared across the simulation.
//!
//! Anything that was previously a literal in sensor/genome/spawn code lives
//! here so it can be tweaked in one place and is greppable. Each constant is
//! grouped by subsystem and documented with the reasoning that picked its
//! current value.

// ── Sensor radii / probe distances ───────────────────────────────────────────

/// Radius (in cells) for population-density sensors that scan an axis-aligned
/// fan in the agent's heading. Matches the C++ reference.
pub const POPULATION_SENSOR_RADIUS: f32 = 2.5;

/// Short-probe distance (in cells) for `barrier_fwd` / `barrier_lr`. Tunes the
/// scan range of the obstacle-avoidance sensors.
pub const SHORT_PROBE_DIST: u32 = 4;

/// Short-probe distance for the genetic-similarity forward sensor.
pub const GENETIC_SIM_PROBE_DIST: i16 = 4;

/// Default oscillator period (steps) for newly-spawned agents. Mutations on
/// `set_oscillator_period` action vary this per-agent at runtime.
pub const DEFAULT_OSC_PERIOD: u32 = 34;

/// Default long-probe distance (cells) for newly-spawned agents.
pub const DEFAULT_LONG_PROBE_DIST: u32 = 16;

/// Radius (cells) for signal-density sensors.
pub const SIGNAL_SENSOR_RADIUS: f32 = 2.0;

/// Radius (cells) for food-density sensors.
pub const FOOD_SENSOR_RADIUS: f32 = 3.0;

// ── Genome / neural net ──────────────────────────────────────────────────────

/// Divisor used to map raw integer gene weights into the f32 connection
/// weights consumed by `feed_forward`. Picked to keep typical weights in a
/// useful ±4-ish range given the i16 raw weight domain. Matches C++.
pub const GENE_WEIGHT_SCALE: f32 = 8192.0;

/// Initial activation value for every neuron in a freshly-compiled
/// `NeuralNet`. Un-driven neurons keep this as a constant bias for the rest
/// of the generation.
pub const NEURON_INITIAL_OUTPUT: f32 = 0.5;
