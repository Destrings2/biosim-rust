//! Built-in sensor implementations (40 sensors).
//!
//! Every sensor implements [`Sensor`] and returns a value in \[0.0, 1.0\].
//! The registry enforces the clamp at `evaluate()` time.
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

pub mod helpers;

use biosim4_core::genome::genome_similarity;
use biosim4_core::registry::{Sensor, SensorContext, SensorRegistry};
use helpers::*;

pub fn register_builtin_sensors(registry: &mut SensorRegistry) {
    registry.register(Box::new(LocX));
    registry.register(Box::new(LocY));
    registry.register(Box::new(BoundaryDistX));
    registry.register(Box::new(BoundaryDist));
    registry.register(Box::new(BoundaryDistY));
    registry.register(Box::new(GeneticSimFwd));
    registry.register(Box::new(LastMoveDirX));
    registry.register(Box::new(LastMoveDirY));
    registry.register(Box::new(LongprobePopFwd));
    registry.register(Box::new(LongprobeBarFwd));
    registry.register(Box::new(PopulationSensor));
    registry.register(Box::new(PopulationFwd));
    registry.register(Box::new(PopulationLR));
    registry.register(Box::new(Osc1));
    registry.register(Box::new(Age));
    registry.register(Box::new(BarrierFwd));
    registry.register(Box::new(BarrierLR));
    registry.register(Box::new(KillBarrierFwd));
    registry.register(Box::new(RandomSensor));
    registry.register(Box::new(Signal0));
    registry.register(Box::new(Signal0Fwd));
    registry.register(Box::new(Signal0LR));
    registry.register(Box::new(Signal1));
    registry.register(Box::new(Signal1Fwd));
    registry.register(Box::new(Signal1LR));
    registry.register(Box::new(Signal2));
    registry.register(Box::new(Signal2Fwd));
    registry.register(Box::new(Signal2LR));
    registry.register(Box::new(Memory0));
    registry.register(Box::new(Memory1));
    registry.register(Box::new(Memory2));
    registry.register(Box::new(Memory3));
    registry.register(Box::new(ChallengeBit0));
    registry.register(Box::new(ChallengeBit1));
    registry.register(Box::new(ChallengeBit2));
    registry.register(Box::new(ChallengeBit3));
    registry.register(Box::new(EnergyLevel));
    registry.register(Box::new(FoodHere));
    registry.register(Box::new(FoodFwd));
    registry.register(Box::new(FoodLR));
}

// ─────────────────────────────────────────────────────────────────────────────
// Location sensors
// ─────────────────────────────────────────────────────────────────────────────

struct LocX;
impl Sensor for LocX {
    fn id(&self) -> &str {
        "loc_x"
    }
    fn name(&self) -> &str {
        "loc X"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.agent.loc.x as f32 / (ctx.world.size_x - 1) as f32
    }
}

struct LocY;
impl Sensor for LocY {
    fn id(&self) -> &str {
        "loc_y"
    }
    fn name(&self) -> &str {
        "loc Y"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.agent.loc.y as f32 / (ctx.world.size_y - 1) as f32
    }
}

struct BoundaryDistX;
impl Sensor for BoundaryDistX {
    fn id(&self) -> &str {
        "boundary_dist_x"
    }
    fn name(&self) -> &str {
        "boundary dist X"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // Distance from the agent to the nearer of the two east-west walls
        // (`min(x, size_x − 1 − x)`), normalized by half the grid width.
        let x = ctx.agent.loc.x as i32;
        let sx = ctx.world.size_x as i32;
        let min_dist = x.min((sx - x) - 1).max(0) as f32;
        (min_dist / (sx as f32 / 2.0)).clamp(0.0, 1.0)
    }
}

struct BoundaryDistY;
impl Sensor for BoundaryDistY {
    fn id(&self) -> &str {
        "boundary_dist_y"
    }
    fn name(&self) -> &str {
        "boundary dist Y"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let y = ctx.agent.loc.y as i32;
        let sy = ctx.world.size_y as i32;
        let min_dist = y.min((sy - y) - 1).max(0) as f32;
        (min_dist / (sy as f32 / 2.0)).clamp(0.0, 1.0)
    }
}

struct BoundaryDist;
impl Sensor for BoundaryDist {
    fn id(&self) -> &str {
        "boundary_dist"
    }
    fn name(&self) -> &str {
        "boundary dist"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // Closest distance to any wall, normalized by the *larger* of the
        // two half-grid spans so a perfectly centered agent reads 1.0 on
        // a square grid and slightly less than 1.0 on a long rectangle.
        let x = ctx.agent.loc.x as i32;
        let y = ctx.agent.loc.y as i32;
        let sx = ctx.world.size_x as i32;
        let sy = ctx.world.size_y as i32;
        let dx = x.min((sx - x) - 1).max(0);
        let dy = y.min((sy - y) - 1).max(0);
        let closest = dx.min(dy) as f32;
        let max_possible = ((sx / 2 - 1).max(sy / 2 - 1)).max(1) as f32;
        (closest / max_possible).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Movement sensors
// ─────────────────────────────────────────────────────────────────────────────

struct LastMoveDirX;
impl Sensor for LastMoveDirX {
    fn id(&self) -> &str {
        "last_move_dir_x"
    }
    fn name(&self) -> &str {
        "last move dir X"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let dx = ctx.agent.last_move_dir.as_normalized_coord().x;
        (dx as f32 + 1.0) / 2.0
    }
}

struct LastMoveDirY;
impl Sensor for LastMoveDirY {
    fn id(&self) -> &str {
        "last_move_dir_y"
    }
    fn name(&self) -> &str {
        "last move dir Y"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let dy = ctx.agent.last_move_dir.as_normalized_coord().y;
        (dy as f32 + 1.0) / 2.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Population sensors
// ─────────────────────────────────────────────────────────────────────────────

struct PopulationSensor;
impl Sensor for PopulationSensor {
    fn id(&self) -> &str {
        "population"
    }
    fn name(&self) -> &str {
        "population density"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let mut count = 0u32;
        let mut total = 0u32;
        biosim4_core::grid::visit_neighborhood(
            ctx.world.grid,
            ctx.agent.loc,
            biosim4_core::constants::POPULATION_SENSOR_RADIUS,
            |loc| {
                total += 1;
                if ctx.world.grid.is_occupied_at(loc) {
                    count += 1;
                }
            },
        );
        if total == 0 {
            return 0.0;
        }
        count as f32 / total as f32
    }
}

struct PopulationFwd;
impl Sensor for PopulationFwd {
    fn id(&self) -> &str {
        "population_fwd"
    }
    fn name(&self) -> &str {
        "population fwd"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        population_density_along_axis(
            ctx.agent.loc,
            ctx.agent.last_move_dir,
            biosim4_core::constants::POPULATION_SENSOR_RADIUS,
            ctx.world.grid,
            ctx.world.population,
        )
    }
}

struct PopulationLR;
impl Sensor for PopulationLR {
    fn id(&self) -> &str {
        "population_lr"
    }
    fn name(&self) -> &str {
        "population LR"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // Single bidirectional probe along the right-perpendicular axis.
        // Because the underlying density function returns a signed
        // `[0, 1]` reading on the chosen axis, this directly distinguishes
        // "more population on the right side" (>0.5) from "more on the
        // left" (<0.5).
        population_density_along_axis(
            ctx.agent.loc,
            ctx.agent.last_move_dir.rotate90cw(),
            biosim4_core::constants::POPULATION_SENSOR_RADIUS,
            ctx.world.grid,
            ctx.world.population,
        )
    }
}

struct GeneticSimFwd;
impl Sensor for GeneticSimFwd {
    fn id(&self) -> &str {
        "genetic_sim_fwd"
    }
    fn name(&self) -> &str {
        "genetic similarity fwd"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // Sample the single cell directly in front; no probe walk.
        let step = ctx.agent.last_move_dir.as_normalized_coord();
        let target =
            biosim4_core::types::Coord::new(ctx.agent.loc.x + step.x, ctx.agent.loc.y + step.y);
        if !ctx.world.grid.is_in_bounds(target) {
            return 0.0;
        }
        match ctx.world.population.get_at(ctx.world.grid, target) {
            Some(neighbor) if neighbor.alive => {
                genome_similarity(&ctx.agent.genome, &neighbor.genome, 0)
            }
            _ => 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Barrier / probe sensors
// ─────────────────────────────────────────────────────────────────────────────

struct BarrierFwd;
impl Sensor for BarrierFwd {
    fn id(&self) -> &str {
        "barrier_fwd"
    }
    fn name(&self) -> &str {
        "barrier fwd"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        short_probe_barrier_distance(
            ctx.agent.loc,
            ctx.agent.last_move_dir,
            biosim4_core::constants::SHORT_PROBE_DIST,
            ctx.world.grid,
        )
    }
}

struct BarrierLR;
impl Sensor for BarrierLR {
    fn id(&self) -> &str {
        "barrier_lr"
    }
    fn name(&self) -> &str {
        "barrier LR"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        short_probe_barrier_distance(
            ctx.agent.loc,
            ctx.agent.last_move_dir.rotate90cw(),
            biosim4_core::constants::SHORT_PROBE_DIST,
            ctx.world.grid,
        )
    }
}

/// "Distance to nearest kill barrier in the forward direction" — same
/// short-probe shape as `barrier_fwd` but only counts cells flagged with
/// `KILL_BARRIER`. Lets evolution learn to steer around hazards painted
/// by the user.
struct KillBarrierFwd;
impl Sensor for KillBarrierFwd {
    fn id(&self) -> &str {
        "kill_barrier_fwd"
    }
    fn name(&self) -> &str {
        "kill barrier fwd"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let step = ctx.agent.last_move_dir.as_normalized_coord();
        if step.x == 0 && step.y == 0 {
            return 0.0;
        }
        let max = biosim4_core::constants::GENETIC_SIM_PROBE_DIST;
        for i in 1..=max {
            let p = biosim4_core::types::Coord::new(
                ctx.agent.loc.x + step.x * i,
                ctx.agent.loc.y + step.y * i,
            );
            if !ctx.world.grid.is_in_bounds(p) {
                return 0.0;
            }
            if ctx.world.grid.is_kill_barrier_at(p) {
                // Closer kill barrier = stronger reading.
                return 1.0 - (i as f32 - 1.0) / max as f32;
            }
        }
        0.0
    }
}

struct LongprobePopFwd;
impl Sensor for LongprobePopFwd {
    fn id(&self) -> &str {
        "longprobe_pop_fwd"
    }
    fn name(&self) -> &str {
        "long probe pop fwd"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        long_probe_population_fwd(
            ctx.agent.loc,
            ctx.agent.last_move_dir,
            ctx.agent.long_probe_dist,
            ctx.world.grid,
        )
    }
}

struct LongprobeBarFwd;
impl Sensor for LongprobeBarFwd {
    fn id(&self) -> &str {
        "longprobe_bar_fwd"
    }
    fn name(&self) -> &str {
        "long probe barrier fwd"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        long_probe_barrier_fwd(
            ctx.agent.loc,
            ctx.agent.last_move_dir,
            ctx.agent.long_probe_dist,
            ctx.world.grid,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Oscillator / age / random
// ─────────────────────────────────────────────────────────────────────────────

struct Osc1;
impl Sensor for Osc1 {
    fn id(&self) -> &str {
        "osc1"
    }
    fn name(&self) -> &str {
        "oscillator 1"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // `(1 − cos(2π·phase)) / 2`: a unit-amplitude oscillation that
        // starts at 0 at step 0, peaks at 1 mid-period, and returns to 0
        // at the period boundary. Each agent shares the same global step
        // count but has its own `osc_period`, so different agents can
        // sample the same waveform at independent rates.
        let phase = (ctx.sim_step % ctx.agent.osc_period) as f32 / ctx.agent.osc_period as f32;
        let factor = -(std::f32::consts::TAU * phase).cos();
        ((factor + 1.0) / 2.0).clamp(0.0, 1.0)
    }
}

struct Age;
impl Sensor for Age {
    fn id(&self) -> &str {
        "age"
    }
    fn name(&self) -> &str {
        "age"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.agent.age as f32 / ctx.world.steps_per_generation as f32
    }
}

struct RandomSensor;
impl Sensor for RandomSensor {
    fn id(&self) -> &str {
        "random"
    }
    fn name(&self) -> &str {
        "random"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.rng.gen_f32()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Signal sensors
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! signal_sensors {
    ($s:ident, $sf:ident, $slr:ident, $layer:literal, $id:literal, $idf:literal, $idlr:literal, $name:literal, $namef:literal, $namelr:literal) => {
        struct $s;
        impl Sensor for $s {
            fn id(&self) -> &str {
                $id
            }
            fn name(&self) -> &str {
                $name
            }
            fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
                ctx.world.signals.get_density(
                    $layer,
                    ctx.agent.loc,
                    biosim4_core::constants::SIGNAL_SENSOR_RADIUS,
                    ctx.world.grid,
                )
            }
        }
        struct $sf;
        impl Sensor for $sf {
            fn id(&self) -> &str {
                $idf
            }
            fn name(&self) -> &str {
                $namef
            }
            fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
                signal_density_along_axis(
                    $layer,
                    ctx.agent.loc,
                    ctx.agent.last_move_dir,
                    biosim4_core::constants::SIGNAL_SENSOR_RADIUS,
                    ctx.world.grid,
                    ctx.world.signals,
                )
            }
        }
        struct $slr;
        impl Sensor for $slr {
            fn id(&self) -> &str {
                $idlr
            }
            fn name(&self) -> &str {
                $namelr
            }
            fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
                // Single bidirectional probe on the right-perpendicular
                // axis; the underlying density function already returns a
                // signed left-vs-right reading.
                signal_density_along_axis(
                    $layer,
                    ctx.agent.loc,
                    ctx.agent.last_move_dir.rotate90cw(),
                    biosim4_core::constants::SIGNAL_SENSOR_RADIUS,
                    ctx.world.grid,
                    ctx.world.signals,
                )
            }
        }
    };
}

signal_sensors!(
    Signal0,
    Signal0Fwd,
    Signal0LR,
    0,
    "signal0",
    "signal0_fwd",
    "signal0_lr",
    "signal layer 0",
    "signal 0 fwd",
    "signal 0 LR"
);
signal_sensors!(
    Signal1,
    Signal1Fwd,
    Signal1LR,
    1,
    "signal1",
    "signal1_fwd",
    "signal1_lr",
    "signal layer 1",
    "signal 1 fwd",
    "signal 1 LR"
);
signal_sensors!(
    Signal2,
    Signal2Fwd,
    Signal2LR,
    2,
    "signal2",
    "signal2_fwd",
    "signal2_lr",
    "signal layer 2",
    "signal 2 fwd",
    "signal 2 LR"
);

// ─────────────────────────────────────────────────────────────────────────────
// Memory sensors
// ─────────────────────────────────────────────────────────────────────────────

macro_rules! read_memory {
    ($name:ident, $id:literal, $label:literal, $reg:literal) => {
        struct $name;
        impl Sensor for $name {
            fn id(&self) -> &str {
                $id
            }
            fn name(&self) -> &str {
                $label
            }
            fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
                ctx.agent.memory[$reg]
            }
        }
    };
}

read_memory!(Memory0, "memory_0", "memory 0", 0);
read_memory!(Memory1, "memory_1", "memory 1", 1);
read_memory!(Memory2, "memory_2", "memory 2", 2);
read_memory!(Memory3, "memory_3", "memory 3", 3);

// ─────────────────────────────────────────────────────────────────────────────
// Challenge-bit sensors
// ─────────────────────────────────────────────────────────────────────────────
//
// Expose the low four bits of `agent.challenge_bits` (a per-agent u32 owned by
// the active challenge) as 0/1 sensors. The bit meanings are challenge-defined
// — e.g. `tag` uses bit 0 for "am I it?", `quarantine` uses bit 0 for "am I
// infected?", `location_sequence` uses bits 0..n for waypoint progress.

macro_rules! read_challenge_bit {
    ($name:ident, $id:literal, $label:literal, $bit:literal) => {
        struct $name;
        impl Sensor for $name {
            fn id(&self) -> &str {
                $id
            }
            fn name(&self) -> &str {
                $label
            }
            fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
                ((ctx.agent.challenge_bits >> $bit) & 1) as f32
            }
        }
    };
}

read_challenge_bit!(ChallengeBit0, "challenge_bit_0", "challenge bit 0", 0);
read_challenge_bit!(ChallengeBit1, "challenge_bit_1", "challenge bit 1", 1);
read_challenge_bit!(ChallengeBit2, "challenge_bit_2", "challenge bit 2", 2);
read_challenge_bit!(ChallengeBit3, "challenge_bit_3", "challenge bit 3", 3);

// ─────────────────────────────────────────────────────────────────────────────
// Energy / food sensors
// ─────────────────────────────────────────────────────────────────────────────

struct EnergyLevel;
impl Sensor for EnergyLevel {
    fn id(&self) -> &str {
        "energy_level"
    }
    fn name(&self) -> &str {
        "energy level"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.agent.energy.clamp(0.0, 1.0)
    }
}

struct FoodHere;
impl Sensor for FoodHere {
    fn id(&self) -> &str {
        "food_here"
    }
    fn name(&self) -> &str {
        "food here"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.world.food.get(ctx.agent.loc)
    }
}

struct FoodFwd;
impl Sensor for FoodFwd {
    fn id(&self) -> &str {
        "food_fwd"
    }
    fn name(&self) -> &str {
        "food fwd"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.world.food.get_density_fwd(
            ctx.agent.loc,
            ctx.agent.last_move_dir,
            biosim4_core::constants::FOOD_SENSOR_RADIUS,
            ctx.world.grid,
        )
    }
}

struct FoodLR;
impl Sensor for FoodLR {
    fn id(&self) -> &str {
        "food_lr"
    }
    fn name(&self) -> &str {
        "food LR"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let r = biosim4_core::constants::FOOD_SENSOR_RADIUS;
        lr_average(ctx.agent.last_move_dir, |d| {
            ctx.world.food.get_density_fwd(ctx.agent.loc, d, r, ctx.world.grid)
        })
    }
}
