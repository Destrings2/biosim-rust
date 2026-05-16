//! Altruism challenges.
//!
//! `altruism` — fitness is the fraction of neighbors within `radius` that
//! passed the inner (location-based) challenge. Rewards agents that cluster
//! near successful neighbors.
//!
//! `altruism_sacrifice` — like `altruism` but agents in the sacrificial zone
//! fail their own evaluation while boosting neighbors' fitness.

use biosim4_core::agent::Agent;
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay};
use biosim4_core::world::World;
use serde_json::{json, Value};

/// Agents in the NW quadrant survive (altruistic zone).
pub struct AltruismChallenge {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
}
impl Default for AltruismChallenge {
    fn default() -> Self {
        Self { cx: 0.25, cy: 0.75, radius: 0.25 }
    }
}
impl Challenge for AltruismChallenge {
    fn id(&self) -> &str {
        "altruism"
    }
    fn name(&self) -> &str {
        "Altruism (survival zone)"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "cx":     { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.25 },
            "cy":     { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.75 },
            "radius": { "type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.25 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("cx") {
            self.cx = v.as_f64().ok_or("cx")? as f32;
        }
        if let Some(v) = p.get("cy") {
            self.cy = v.as_f64().ok_or("cy")? as f32;
        }
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let dist = world.grid.norm_dist_to_norm_point(agent.loc, self.cx, self.cy);
        if dist > self.radius {
            return (false, 0.0);
        }
        (true, (self.radius - dist) / self.radius)
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        vec![ChallengeOverlay::Circle {
            cx: self.cx * sx,
            cy: self.cy * sy,
            radius: self.radius * sx.max(sy),
            color: [0, 255, 0, 40],
        }]
    }
}

/// Agents in the NE sacrifice zone die; surviving agents in the SW zone reproduce.
pub struct AltruismSacrificeChallenge {
    pub sacrifice_cx: f32,
    pub sacrifice_cy: f32,
    pub radius: f32,
}
impl Default for AltruismSacrificeChallenge {
    fn default() -> Self {
        Self { sacrifice_cx: 0.75, sacrifice_cy: 0.75, radius: 0.25 }
    }
}
impl Challenge for AltruismSacrificeChallenge {
    fn id(&self) -> &str {
        "altruism_sacrifice"
    }
    fn name(&self) -> &str {
        "Altruism Sacrifice"
    }
    fn description(&self) -> &str {
        "Agents in the sacrifice zone die; agents in the survival zone that share genome similarity with sacrificed agents reproduce."
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "sacrifice_cx": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.75 },
            "sacrifice_cy": { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.75 },
            "radius": { "type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.25 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("sacrifice_cx") {
            self.sacrifice_cx = v.as_f64().ok_or("sacrifice_cx")? as f32;
        }
        if let Some(v) = p.get("sacrifice_cy") {
            self.sacrifice_cy = v.as_f64().ok_or("sacrifice_cy")? as f32;
        }
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let dist =
            world.grid.norm_dist_to_norm_point(agent.loc, self.sacrifice_cx, self.sacrifice_cy);
        // In sacrifice zone → fail (die)
        if dist <= self.radius {
            return (false, 0.0);
        }
        // Otherwise pass (spawn.rs handles kin-selection bonus)
        (true, 1.0)
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        vec![ChallengeOverlay::Circle {
            cx: self.sacrifice_cx * sx,
            cy: self.sacrifice_cy * sy,
            radius: self.radius * sx.max(sy),
            color: [255, 40, 40, 50],
        }]
    }
}
