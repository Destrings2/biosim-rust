//! Pheromone-signal sensors — three independent channels.
//!
//! For each layer there are three sensor variants:
//! - base reads average density in the neighborhood;
//! - `_fwd` is an inverse-distance-weighted signed projection along the
//!   heading axis (`0.5` symmetric, `>0.5` more in front);
//! - `_lr` is the same projection on the right-perpendicular axis
//!   (`0.5` symmetric, `>0.5` more on the right).

use crate::helpers::signal_density_along_axis;
use biosim4_core::registry::{Sensor, SensorContext};

macro_rules! signal_sensors {
    ($s:ident, $sf:ident, $slr:ident, $layer:literal, $id:literal, $idf:literal, $idlr:literal, $name:literal, $namef:literal, $namelr:literal) => {
        pub(crate) struct $s;
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
        pub(crate) struct $sf;
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
        pub(crate) struct $slr;
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
