//! Food & energy sensors.

use crate::helpers::lr_average;
use biosim4_core::registry::{Sensor, SensorContext};

pub(crate) struct EnergyLevel;
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

pub(crate) struct FoodHere;
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

pub(crate) struct FoodFwd;
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

pub(crate) struct FoodLR;
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
