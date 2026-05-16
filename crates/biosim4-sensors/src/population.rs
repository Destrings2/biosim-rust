//! Population-density sensors.
//!
//! `population` reports the fraction of occupied cells in the neighborhood.
//! `population_fwd` and `population_lr` are inverse-distance-weighted
//! signed projections along the heading and right-perpendicular axes
//! respectively, mapped to `[0, 1]` with `0.5` meaning symmetric / empty.

use crate::helpers::population_density_along_axis;
use biosim4_core::registry::{Sensor, SensorContext};

pub(crate) struct PopulationSensor;
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

pub(crate) struct PopulationFwd;
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

pub(crate) struct PopulationLR;
impl Sensor for PopulationLR {
    fn id(&self) -> &str {
        "population_lr"
    }
    fn name(&self) -> &str {
        "population LR"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // Single bidirectional probe along the right-perpendicular axis.
        // The underlying density function already returns a signed
        // `[0, 1]` reading on the chosen axis, so this directly
        // distinguishes "more population on the right side" (>0.5) from
        // "more on the left" (<0.5).
        population_density_along_axis(
            ctx.agent.loc,
            ctx.agent.last_move_dir.rotate90cw(),
            biosim4_core::constants::POPULATION_SENSOR_RADIUS,
            ctx.world.grid,
            ctx.world.population,
        )
    }
}
