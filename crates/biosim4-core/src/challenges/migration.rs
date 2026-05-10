use crate::agent::Agent;
use crate::registry::challenge::Challenge;
use crate::world::World;
use serde_json::{json, Value};

/// Survive by migrating at least `min_distance` (normalized 0-1 of the
/// world diagonal) from your birth location.
pub struct MigrateDistanceChallenge {
    pub min_distance: f32,
}

impl Default for MigrateDistanceChallenge {
    fn default() -> Self { Self { min_distance: 0.30 } }
}

impl Challenge for MigrateDistanceChallenge {
    fn id(&self) -> &str { "migrate_distance" }
    fn name(&self) -> &str { "Migrate Distance" }
    fn description(&self) -> &str {
        "Travel at least `min_distance` (normalized 0-1) from your birth location."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "min_distance": { "type": "number", "minimum": 0.05, "maximum": 1.0, "default": 0.30 }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("min_distance") { self.min_distance = v.as_f64().ok_or("min_distance")? as f32; }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let dist = (agent.loc - agent.birth_loc).length();
        let max_dist = (world.size_x.max(world.size_y) as f32) * std::f32::consts::SQRT_2;
        let normalized = (dist / max_dist).clamp(0.0, 1.0);
        let pass = normalized >= self.min_distance;
        (pass, normalized)
    }
}
