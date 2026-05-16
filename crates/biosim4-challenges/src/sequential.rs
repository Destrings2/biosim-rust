//! Sequential challenges: require ordered behavior during the generation.
//!
//! `touch_any_wall` — uses `agent.challenge_bits` to record whether the agent
//! touched a border cell at any point during the generation. Set via the
//! `on_sim_step` hook; evaluated at generation end.
//!
//! `location_sequence` — requires visiting a sequence of zones in order within
//! the generation. Zone progress is tracked in `challenge_bits`.

use biosim4_core::agent::Agent;
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::world::World;
use serde_json::{json, Value};

/// Survive iff the agent touched any border cell during the generation.
/// Tracked via `agent.challenge_bits` bit 0.
pub struct TouchAnyWallChallenge;
impl Challenge for TouchAnyWallChallenge {
    fn id(&self) -> &str {
        "touch_any_wall"
    }
    fn name(&self) -> &str {
        "Touch Any Wall"
    }
    fn description(&self) -> &str {
        "Survive iff you touched any border cell during the generation."
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn configure(&mut self, _: Value) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, _world: &World) -> (bool, f32) {
        let pass = agent.challenge_bits & 1 != 0;
        (pass, if pass { 1.0 } else { 0.0 })
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        // Clear our bit at the start of each generation
        for a in ctx.population.iter_alive_mut() {
            a.challenge_bits &= !1;
        }
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        let sx = ctx.config.size_x as i16;
        let sy = ctx.config.size_y as i16;
        for a in ctx.population.iter_alive_mut() {
            let on_border = a.loc.x == 0 || a.loc.y == 0 || a.loc.x == sx - 1 || a.loc.y == sy - 1;
            if on_border {
                a.challenge_bits |= 1;
            }
        }
    }
}

/// Survive by visiting a sequence of waypoints in order. Each waypoint is a
/// disc of radius `radius` (normalized 0-1) at a normalized centre. Bit `i`
/// of `challenge_bits` is set when waypoint `i` is reached AND all earlier
/// waypoints have already been reached. Score = bits_set / waypoint_count.
pub struct LocationSequenceChallenge {
    pub radius: f32,
    pub waypoints: Vec<(f32, f32)>, // normalized centres
    pub min_visits: usize,          // must visit at least this many in order
}

impl Default for LocationSequenceChallenge {
    fn default() -> Self {
        // 4-corner tour: NW → NE → SE → SW
        Self {
            radius: 0.12,
            waypoints: vec![(0.15, 0.85), (0.85, 0.85), (0.85, 0.15), (0.15, 0.15)],
            min_visits: 3,
        }
    }
}

impl Challenge for LocationSequenceChallenge {
    fn id(&self) -> &str {
        "location_sequence"
    }
    fn name(&self) -> &str {
        "Location Sequence"
    }
    fn description(&self) -> &str {
        "Visit a sequence of waypoint discs in order (defaults: NW → NE → SE → SW). Survival requires reaching at least `min_visits` checkpoints in order."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "radius":     { "type": "number", "minimum": 0.05, "maximum": 0.3,  "default": 0.12 },
                "min_visits": { "type": "number", "minimum": 1.0,  "maximum": 8.0,  "default": 3.0 }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        if let Some(v) = p.get("min_visits") {
            self.min_visits = v.as_f64().ok_or("min_visits")? as usize;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, _world: &World) -> (bool, f32) {
        let count =
            (agent.challenge_bits & ((1u32 << self.waypoints.len()) - 1)).count_ones() as usize;
        let pass = count >= self.min_visits.min(self.waypoints.len());
        let score = count as f32 / self.waypoints.len() as f32;
        (pass, score)
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        let mask = (1u32 << self.waypoints.len()) - 1;
        for a in ctx.population.iter_alive_mut() {
            a.challenge_bits &= !mask;
        }
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        let sx = ctx.config.size_x as f32;
        let sy = ctx.config.size_y as f32;
        let r2 = (self.radius * sx.max(sy)).powi(2);
        let centres: Vec<(f32, f32)> =
            self.waypoints.iter().map(|(nx, ny)| (nx * (sx - 1.0), ny * (sy - 1.0))).collect();

        for a in ctx.population.iter_alive_mut() {
            // Find next checkpoint to claim (lowest unset bit within mask)
            let next = (0..centres.len()).find(|&i| a.challenge_bits & (1 << i) == 0);
            if let Some(i) = next {
                let (cx, cy) = centres[i];
                // Topology-aware: a waypoint at the wrap-opposite side
                // of the seam from the agent is still reachable via the
                // short path on TorusX/Sphere worlds.
                if ctx.grid.dist_sq_to_point(a.loc, cx, cy) <= r2 {
                    a.challenge_bits |= 1 << i;
                }
            }
        }
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        let r = self.radius * sx.max(sy);
        // Earlier waypoints render brighter so the visit order reads from the
        // overlay alone (no labels available in gizmo land).
        let n = self.waypoints.len().max(1);
        self.waypoints
            .iter()
            .enumerate()
            .map(|(i, (nx, ny))| {
                let t = 1.0 - (i as f32 / n as f32);
                let alpha = (40.0 + 60.0 * t) as u8;
                ChallengeOverlay::Circle {
                    cx: nx * (sx - 1.0),
                    cy: ny * (sy - 1.0),
                    radius: r,
                    color: [255, 220, 60, alpha],
                }
            })
            .collect()
    }
}
