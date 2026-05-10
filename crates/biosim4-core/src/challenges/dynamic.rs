//! Dynamic / time-varying challenges where the survival landscape itself
//! changes during the generation. Tests the population's ability to evolve
//! responsive (rather than static) behaviour.

use crate::agent::Agent;
use crate::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use crate::types::Coord;
use crate::world::World;
use serde_json::{json, Value};
use std::collections::HashSet;

// ── Sun Tracker ─────────────────────────────────────────────────────────

/// A "sun" disc rotates around the world centre over the course of the
/// generation. The challenge accumulates a per-agent "warmth" counter
/// (low 5 bits of `challenge_bits`, capped at 31) sampled ~32× across the
/// generation; agents survive iff they end inside the sun AND their warmth
/// reached `min_warmth`.
pub struct SunTrackerChallenge {
    pub radius: f32,         // sun-disc radius (normalized to max(size_x, size_y))
    pub orbit_radius: f32,   // distance from centre (normalized)
    pub revolutions: f32,    // full orbits per generation
    pub min_warmth: u32,     // required tracking ticks (out of 32)
}

impl Default for SunTrackerChallenge {
    fn default() -> Self {
        // Slow orbit + low warmth threshold so a typical 200-pop GA run can
        // bootstrap. Crank `revolutions` / `min_warmth` for a harder run.
        Self { radius: 0.20, orbit_radius: 0.25, revolutions: 0.25, min_warmth: 4 }
    }
}

fn sun_pos_at(c: &SunTrackerChallenge, step: u32, steps_per_gen: u32, size_x: u16, size_y: u16) -> (f32, f32) {
    let cx = (size_x - 1) as f32 * 0.5;
    let cy = (size_y - 1) as f32 * 0.5;
    let r  = c.orbit_radius * size_x.max(size_y) as f32;
    let phase = c.revolutions * (step as f32) / steps_per_gen.max(1) as f32;
    let angle = 2.0 * std::f32::consts::PI * phase;
    (cx + r * angle.cos(), cy + r * angle.sin())
}

impl Challenge for SunTrackerChallenge {
    fn id(&self) -> &str { "sun_tracker" }
    fn name(&self) -> &str { "Sun Tracker" }
    fn description(&self) -> &str {
        "A sun disc orbits the centre. Survive by being inside it at the final step AND tracking it for at least `min_warmth`/32 sampled ticks."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "radius":        { "type": "number", "minimum": 0.05, "maximum": 0.4,  "default": 0.15 },
                "orbit_radius":  { "type": "number", "minimum": 0.10, "maximum": 0.5,  "default": 0.30 },
                "revolutions":   { "type": "number", "minimum": 0.25, "maximum": 4.0,  "default": 1.0 },
                "min_warmth":    { "type": "number", "minimum": 0.0,  "maximum": 31.0, "default": 16.0 }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius")        { self.radius = v.as_f64().ok_or("radius")? as f32; }
        if let Some(v) = p.get("orbit_radius")  { self.orbit_radius = v.as_f64().ok_or("orbit_radius")? as f32; }
        if let Some(v) = p.get("revolutions")   { self.revolutions = v.as_f64().ok_or("revolutions")? as f32; }
        if let Some(v) = p.get("min_warmth")    { self.min_warmth = v.as_f64().ok_or("min_warmth")? as u32; }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        // Sun's final-step position
        let final_step = world.steps_per_generation.saturating_sub(1);
        let (sx, sy) = sun_pos_at(self, final_step, world.steps_per_generation, world.size_x, world.size_y);
        let dx = agent.loc.x as f32 - sx;
        let dy = agent.loc.y as f32 - sy;
        let dist = (dx * dx + dy * dy).sqrt();
        let r = self.radius * world.size_x.max(world.size_y) as f32;
        let in_sun = dist <= r;

        let warmth = agent.challenge_bits & 0x1F;
        let pass = in_sun && warmth >= self.min_warmth;
        let proximity = if dist < r * 2.0 { 1.0 - dist / (r * 2.0) } else { 0.0 };
        let score = 0.5 * proximity + 0.5 * (warmth as f32 / 31.0);
        (pass, score)
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        for a in ctx.population.iter_alive_mut() {
            a.challenge_bits &= !0x1F;
        }
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        // Sample warmth roughly 32 times across the generation
        let stride = (ctx.config.steps_per_generation / 32).max(1);
        if ctx.step % stride != 0 { return; }

        let (sx, sy) = sun_pos_at(self, ctx.step, ctx.config.steps_per_generation,
                                   ctx.config.size_x, ctx.config.size_y);
        let r = self.radius * ctx.config.size_x.max(ctx.config.size_y) as f32;
        let r2 = r * r;
        for a in ctx.population.iter_alive_mut() {
            let dx = a.loc.x as f32 - sx;
            let dy = a.loc.y as f32 - sy;
            if dx * dx + dy * dy <= r2 {
                let warmth = a.challenge_bits & 0x1F;
                if warmth < 31 {
                    a.challenge_bits = (a.challenge_bits & !0x1F) | (warmth + 1);
                }
            }
        }
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let (sx, sy) = sun_pos_at(self, world.step, world.steps_per_generation, world.size_x, world.size_y);
        let r = self.radius * world.size_x.max(world.size_y) as f32;
        vec![ChallengeOverlay::Circle {
            cx: sx,
            cy: sy,
            radius: r,
            color: [255, 200, 0, 80], // Translucent orange/yellow
        }]
    }
}

// ── Diaspora (anti-pairs) ────────────────────────────────────────────────

/// Survive iff your nearest alive neighbour is at least `min_distance`
/// (Euclidean, in cells) away. Selects for spreading-out / anti-flocking.
pub struct DiasporaChallenge {
    pub min_distance: f32,
}

impl Default for DiasporaChallenge {
    fn default() -> Self { Self { min_distance: 8.0 } }
}

impl Challenge for DiasporaChallenge {
    fn id(&self) -> &str { "diaspora" }
    fn name(&self) -> &str { "Diaspora (anti-flock)" }
    fn description(&self) -> &str {
        "Survive iff your nearest alive neighbour is at least `min_distance` cells away."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "min_distance": { "type": "number", "minimum": 2.0, "maximum": 32.0, "default": 8.0 }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("min_distance") { self.min_distance = v.as_f64().ok_or("min_distance")? as f32; }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let me = agent.loc;
        let mut nearest_sq = f32::INFINITY;
        for other in world.population.iter_alive() {
            if other.id == agent.id { continue; }
            let dx = (other.loc.x - me.x) as f32;
            let dy = (other.loc.y - me.y) as f32;
            let d2 = dx * dx + dy * dy;
            if d2 < nearest_sq { nearest_sq = d2; }
        }
        let nearest = nearest_sq.sqrt();
        let pass = nearest >= self.min_distance;
        let max = self.min_distance * 2.0;
        (pass, (nearest / max).clamp(0.0, 1.0))
    }
}

// ── Food Foraging ────────────────────────────────────────────────────────

/// Food pellets are placed at gen-start. Stepping onto a pellet consumes it
/// (incrementing the agent's eat counter, low 6 bits of `challenge_bits`).
/// Survive by eating at least `min_food`. Pellets do not respawn within a
/// generation, so this strongly selects for exploration.
pub struct FoodForagingChallenge {
    pub food_density: f32,
    pub min_food: u32,
    /// Pellet positions for the current generation. Reseeded in
    /// `on_generation_start`.
    pellets: HashSet<(i16, i16)>,
}

impl Default for FoodForagingChallenge {
    fn default() -> Self {
        Self { food_density: 0.05, min_food: 3, pellets: HashSet::new() }
    }
}

impl Challenge for FoodForagingChallenge {
    fn id(&self) -> &str { "food_foraging" }
    fn name(&self) -> &str { "Food Foraging" }
    fn description(&self) -> &str {
        "Pellets are scattered at gen-start; stepping onto one consumes it. Survive by eating at least `min_food`."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "food_density": { "type": "number", "minimum": 0.005, "maximum": 0.2,  "default": 0.05,
                                  "description": "Fraction of cells seeded with food" },
                "min_food":     { "type": "number", "minimum": 1.0,   "maximum": 63.0, "default": 3.0 }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("food_density") { self.food_density = v.as_f64().ok_or("food_density")? as f32; }
        if let Some(v) = p.get("min_food")     { self.min_food     = v.as_f64().ok_or("min_food")? as u32; }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, _world: &World) -> (bool, f32) {
        let eaten = agent.challenge_bits & 0x3F;
        let pass = eaten >= self.min_food;
        (pass, (eaten as f32 / 63.0).min(1.0))
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        // Reset eat counter for everybody
        for a in ctx.population.iter_alive_mut() {
            a.challenge_bits &= !0x3F;
        }
        // Reseed pellets at empty cells.
        self.pellets.clear();
        let sx = ctx.config.size_x;
        let sy = ctx.config.size_y;
        let n = ((sx as usize * sy as usize) as f32 * self.food_density) as usize;
        let mut placed = 0;
        let mut tries = 0;
        let cap = (n * 10).max(1);
        while placed < n && tries < cap {
            tries += 1;
            let x = ctx.rng.gen_range_u32(0, sx as u32) as i16;
            let y = ctx.rng.gen_range_u32(0, sy as u32) as i16;
            let loc = Coord::new(x, y);
            if ctx.grid.is_empty_at(loc) && self.pellets.insert((x, y)) {
                placed += 1;
            }
        }
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        if self.pellets.is_empty() { return; }
        let alive_ids: Vec<u32> = ctx.population.alive_ids().to_vec();
        for id in alive_ids {
            let loc = match ctx.population.get(id) {
                Some(a) if a.alive => a.loc,
                _ => continue,
            };
            let key = (loc.x, loc.y);
            if self.pellets.remove(&key) {
                if let Some(a) = ctx.population.get_mut(id) {
                    let eaten = a.challenge_bits & 0x3F;
                    if eaten < 63 {
                        a.challenge_bits = (a.challenge_bits & !0x3F) | (eaten + 1);
                    }
                }
            }
        }
    }
    fn overlays(&self, _world: &World) -> Vec<ChallengeOverlay> {
        let points = self.pellets.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
        vec![ChallengeOverlay::Points {
            points,
            color: [0, 255, 100, 255], // bright green for food
            size: 1.0,
        }]
    }
}

// ── Survivor (lethal predator pulse) ─────────────────────────────────────

/// At each step there's a small global per-agent kill probability: a "stress"
/// pulse. Agents in a circular safe-zone (centre, normalized radius `safe_radius`)
/// are immune. The safe-zone centre re-rolls every `period` steps, so agents
/// must keep finding it. Survivors at gen-end pass.
pub struct SurvivorChallenge {
    pub safe_radius: f32,
    pub period: u32,
    pub stress: f32,
    /// Current safe-zone centre in absolute coords (regenerated as needed).
    centre: (f32, f32),
}

impl Default for SurvivorChallenge {
    fn default() -> Self {
        Self { safe_radius: 0.18, period: 40, stress: 0.04, centre: (0.0, 0.0) }
    }
}

impl Challenge for SurvivorChallenge {
    fn id(&self) -> &str { "survivor" }
    fn name(&self) -> &str { "Survivor (shifting safe zone)" }
    fn description(&self) -> &str {
        "Each step every agent has a small kill probability unless inside the safe zone. The safe zone teleports every `period` steps. Survivors at gen-end pass."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "safe_radius": { "type": "number", "minimum": 0.05, "maximum": 0.4,  "default": 0.18 },
                "period":      { "type": "number", "minimum": 10.0, "maximum": 200.0, "default": 40.0 },
                "stress":      { "type": "number", "minimum": 0.0,  "maximum": 0.5,   "default": 0.04 }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("safe_radius") { self.safe_radius = v.as_f64().ok_or("safe_radius")? as f32; }
        if let Some(v) = p.get("period")      { self.period      = v.as_f64().ok_or("period")? as u32; }
        if let Some(v) = p.get("stress")      { self.stress      = v.as_f64().ok_or("stress")? as f32; }
        Ok(())
    }
    fn evaluate(&self, _agent: &Agent, _world: &World) -> (bool, f32) {
        // alive at gen-end == survived
        (true, 1.0)
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        // Pick the first centre.
        let sx = ctx.config.size_x as f32;
        let sy = ctx.config.size_y as f32;
        self.centre = (
            ctx.rng.gen_range_u32(0, sx as u32) as f32,
            ctx.rng.gen_range_u32(0, sy as u32) as f32,
        );
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        // Re-roll the centre every `period` steps.
        if self.period > 0 && ctx.step > 0 && ctx.step % self.period == 0 {
            let sx = ctx.config.size_x as u32;
            let sy = ctx.config.size_y as u32;
            self.centre = (
                ctx.rng.gen_range_u32(0, sx) as f32,
                ctx.rng.gen_range_u32(0, sy) as f32,
            );
        }
        let r = self.safe_radius * ctx.config.size_x.max(ctx.config.size_y) as f32;
        let r2 = r * r;
        let (cx, cy) = self.centre;
        let stress = self.stress;

        let victims: Vec<u32> = ctx.population.iter_alive().filter_map(|a| {
            let dx = a.loc.x as f32 - cx;
            let dy = a.loc.y as f32 - cy;
            let in_safe = dx * dx + dy * dy <= r2;
            if in_safe { return None; }
            if ctx.rng.gen_bool(stress) { Some(a.id) } else { None }
        }).collect();
        for id in victims { ctx.population.queue_for_death(id); }
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let r = self.safe_radius * world.size_x.max(world.size_y) as f32;
        vec![ChallengeOverlay::Circle {
            cx: self.centre.0,
            cy: self.centre.1,
            radius: r,
            color: [0, 150, 255, 80], // Translucent blue for safe zone
        }]
    }
}
