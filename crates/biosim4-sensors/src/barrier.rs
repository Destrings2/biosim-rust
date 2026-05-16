//! Barrier and long-probe sensors.
//!
//! `barrier_fwd` / `barrier_lr` report bidirectional non-barrier distance
//! along the heading / right-perpendicular axes (`0.5` = symmetric).
//! `kill_barrier_fwd` is a forward-only probe that fires on `KILL_BARRIER`
//! cells specifically — useful for steering around hazards.
//! `longprobe_pop_fwd` / `longprobe_bar_fwd` walk forward until they hit
//! an occupied cell / barrier respectively, normalized by the agent's
//! `long_probe_dist` (`1.0` when nothing is in range).

use crate::helpers::{
    long_probe_barrier_fwd, long_probe_population_fwd, short_probe_barrier_distance,
};
use biosim4_core::registry::{Sensor, SensorContext};

pub(crate) struct BarrierFwd;
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

pub(crate) struct BarrierLR;
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
pub(crate) struct KillBarrierFwd;
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

pub(crate) struct LongprobePopFwd;
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

pub(crate) struct LongprobeBarFwd;
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
