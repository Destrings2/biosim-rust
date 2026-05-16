//! Genetic-similarity sensor.

use biosim4_core::genome::genome_similarity;
use biosim4_core::registry::{Sensor, SensorContext};

pub(crate) struct GeneticSimFwd;
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
