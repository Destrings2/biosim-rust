//! Radioactive walls challenge.
//!
//! `radioactive_walls` — kills agents that enter configurable lethal border
//! zones via `on_sim_step`. The border width and lethality probability are
//! configurable. Agents that survive all steps in a non-lethal zone pass.

use crate::agent::Agent;
use crate::registry::challenge::{Challenge, WorldMut};
use crate::world::World;
use serde_json::{json, Value};

/// Each step, agents take probabilistic damage proportional to proximity to
/// the currently-active wall. The active wall flips between west (x=0) and
/// east (x=size_x-1) at the halfway mark of each generation, so the optimal
/// strategy is "go east, then go west" (or vice versa).
///
/// Damage model: kill probability per step = `intensity * exp(-dist/half_life)`,
/// where `dist` is the chebyshev distance to the active wall.
pub struct RadioactiveWallsChallenge {
    pub intensity: f32,
    pub half_life: f32,
}

impl Default for RadioactiveWallsChallenge {
    fn default() -> Self { Self { intensity: 0.5, half_life: 8.0 } }
}

impl Challenge for RadioactiveWallsChallenge {
    fn id(&self) -> &str { "radioactive_walls" }
    fn name(&self) -> &str { "Radioactive Walls" }
    fn description(&self) -> &str {
        "Active wall (west, then east) emits radiation. Per-step kill probability falls off exponentially with distance. Survivors at gen-end pass."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "intensity": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5,
                               "description": "Per-step kill probability AT the wall" },
                "half_life": { "type": "number", "minimum": 0.5, "maximum": 32.0, "default": 8.0,
                               "description": "Distance (cells) at which damage halves" }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("intensity") { self.intensity = v.as_f64().ok_or("intensity")? as f32; }
        if let Some(v) = p.get("half_life") { self.half_life = v.as_f64().ok_or("half_life")? as f32; }
        Ok(())
    }
    fn evaluate(&self, _agent: &Agent, _world: &World) -> (bool, f32) {
        // Damage is applied per-step; alive at gen-end == survived.
        (true, 1.0)
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        let half_gen = ctx.config.steps_per_generation / 2;
        let sx = ctx.config.size_x as i16;
        let active_wall_x: i16 = if ctx.step < half_gen { 0 } else { sx - 1 };

        let k = (2.0_f32).ln() / self.half_life.max(0.5);

        // Collect victim ids first so we can borrow population mutably afterward.
        let victims: Vec<u32> = ctx.population.iter_alive().filter_map(|a| {
            let dist = (a.loc.x as i32 - active_wall_x as i32).unsigned_abs() as f32;
            let p = self.intensity * (-k * dist).exp();
            if ctx.rng.gen_bool(p) { Some(a.id) } else { None }
        }).collect();

        for id in victims {
            ctx.population.queue_for_death(id);
        }
    }
}
