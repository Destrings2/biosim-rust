//! Dynamic / time-varying challenges where the survival landscape itself
//! changes during the generation. Tests the population's ability to evolve
//! responsive (rather than static) behaviour.

use biosim4_core::agent::Agent;
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::world::World;
use serde_json::{json, Value};

// ── Sun Tracker ─────────────────────────────────────────────────────────

/// Bit 0 of `challenge_bits`: in-sun-this-tick flag, read by the
/// `challenge_bit_0` sensor so peeps can sense their own warmth state.
const SUN_BIT_IN_SUN_NOW: u32 = 1 << 0;

/// Bits 8..15: accumulated warmth counter (0..=255). Lives above the
/// `challenge_bit_*` sensor window to keep the low bits reserved for
/// real-time signals.
const SUN_WARMTH_SHIFT: u32 = 8;
const SUN_WARMTH_MASK: u32 = 0xFF << SUN_WARMTH_SHIFT;
const SUN_MAX_WARMTH: u32 = 0xFF;

/// Warmth samples taken per generation. A perfect tracker reaches
/// `SUN_TARGET_SAMPLES`; the counter cannot saturate at any reasonable
/// `steps_per_generation`.
const SUN_TARGET_SAMPLES: u32 = 32;

/// A small sun disc circles the world centre once per generation.
/// Survival requires accumulating warmth above `min_warmth` across the
/// `SUN_TARGET_SAMPLES` sample ticks.
///
/// Geometry is tuned so that **stationary peeps can't pass**: the orbit
/// circumference and sun-disc width make any single fixed cell visible
/// to the sun for only a few sample ticks per gen, far below the
/// threshold. Only agents that actively follow the sun reach the bar.
/// This avoids the deceptive equilibrium of an earlier tuning where a
/// slow, wide sun let parkers accumulate enough warmth to survive
/// without ever tracking.
pub struct SunTrackerChallenge {
    pub radius: f32,       // sun-disc radius (normalized to max(size_x, size_y))
    pub orbit_radius: f32, // distance from centre (normalized)
    pub revolutions: f32,  // full orbits per generation
    pub min_warmth: u32,   // required warmth (0..=SUN_TARGET_SAMPLES)
}

impl Default for SunTrackerChallenge {
    fn default() -> Self {
        // Defaults sized so stationary peeps collect roughly four warmth
        // ticks per generation — half of `min_warmth`. Tracking peeps
        // approach the maximum.
        Self { radius: 0.12, orbit_radius: 0.30, revolutions: 1.0, min_warmth: 8 }
    }
}

#[inline]
fn warmth_of(bits: u32) -> u32 {
    (bits & SUN_WARMTH_MASK) >> SUN_WARMTH_SHIFT
}

#[inline]
fn with_warmth(bits: u32, w: u32) -> u32 {
    (bits & !SUN_WARMTH_MASK) | ((w.min(SUN_MAX_WARMTH)) << SUN_WARMTH_SHIFT)
}

fn sun_pos_at(
    c: &SunTrackerChallenge,
    step: u32,
    steps_per_gen: u32,
    size_x: u16,
    size_y: u16,
) -> (f32, f32) {
    let cx = (size_x - 1) as f32 * 0.5;
    let cy = (size_y - 1) as f32 * 0.5;
    let r = c.orbit_radius * size_x.max(size_y) as f32;
    let phase = c.revolutions * (step as f32) / steps_per_gen.max(1) as f32;
    let angle = 2.0 * std::f32::consts::PI * phase;
    (cx + r * angle.cos(), cy + r * angle.sin())
}

impl Challenge for SunTrackerChallenge {
    fn id(&self) -> &str {
        "sun_tracker"
    }
    fn name(&self) -> &str {
        "Sun Tracker"
    }
    fn description(&self) -> &str {
        "A small sun disc orbits the world centre. Bit 0 of `challenge_bits` reflects in-sun status each step (readable via the `challenge_bit_0` sensor). Roughly 32 sample ticks per generation increment the agent's warmth counter when it is in the disc. Survival requires warmth ≥ `min_warmth`. The sun's geometry makes stationary peeps fall short — only agents that actively follow the orbiting disc accumulate enough warmth to pass."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "radius":        { "type": "number", "minimum": 0.05, "maximum": 0.4,  "default": 0.12 },
                "orbit_radius":  { "type": "number", "minimum": 0.10, "maximum": 0.5,  "default": 0.30 },
                "revolutions":   { "type": "number", "minimum": 0.25, "maximum": 4.0,  "default": 1.0 },
                "min_warmth":    { "type": "number", "minimum": 1.0,  "maximum": 32.0, "default": 8.0 }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        if let Some(v) = p.get("orbit_radius") {
            self.orbit_radius = v.as_f64().ok_or("orbit_radius")? as f32;
        }
        if let Some(v) = p.get("revolutions") {
            self.revolutions = v.as_f64().ok_or("revolutions")? as f32;
        }
        if let Some(v) = p.get("min_warmth") {
            self.min_warmth = v.as_f64().ok_or("min_warmth")? as u32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, _world: &World) -> (bool, f32) {
        // Pure warmth-based selection. Earlier tunings added a
        // proximity-to-sun tiebreaker to give gen-0 a gradient, but
        // proximity also rewards parking near the sun's end position —
        // exactly the behaviour the new geometry is trying to eliminate.
        // With the orbit faster and the disc smaller, the warmth signal
        // alone is broad enough at gen 0 for tournament selection to
        // bootstrap.
        let warmth = warmth_of(agent.challenge_bits);
        let pass = warmth >= self.min_warmth;
        let score = (warmth as f32 / SUN_TARGET_SAMPLES as f32).clamp(0.0, 1.0);
        (pass, score)
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        // Reset the bits this challenge owns; leave the rest untouched.
        for a in ctx.population.iter_alive_mut() {
            a.challenge_bits &= !(SUN_BIT_IN_SUN_NOW | SUN_WARMTH_MASK);
        }
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        // Bit 0 updates every step so the sensor is always fresh;
        // warmth ticks on sample steps only.
        let stride = (ctx.config.steps_per_generation / SUN_TARGET_SAMPLES).max(1);
        let is_sample_step = ctx.step.is_multiple_of(stride);

        let (sx, sy) = sun_pos_at(
            self,
            ctx.step,
            ctx.config.steps_per_generation,
            ctx.config.size_x,
            ctx.config.size_y,
        );
        let r = self.radius * ctx.config.size_x.max(ctx.config.size_y) as f32;
        let r2 = r * r;
        for a in ctx.population.iter_alive_mut() {
            let in_sun = ctx.grid.dist_sq_to_point(a.loc, sx, sy) <= r2;
            if in_sun {
                a.challenge_bits |= SUN_BIT_IN_SUN_NOW;
                if is_sample_step {
                    let w = warmth_of(a.challenge_bits);
                    if w < SUN_MAX_WARMTH {
                        a.challenge_bits = with_warmth(a.challenge_bits, w + 1);
                    }
                }
            } else {
                a.challenge_bits &= !SUN_BIT_IN_SUN_NOW;
            }
        }
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let (sx, sy) =
            sun_pos_at(self, world.step, world.steps_per_generation, world.size_x, world.size_y);
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
    fn default() -> Self {
        Self { min_distance: 8.0 }
    }
}

impl Challenge for DiasporaChallenge {
    fn id(&self) -> &str {
        "diaspora"
    }
    fn name(&self) -> &str {
        "Diaspora (anti-flock)"
    }
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
        if let Some(v) = p.get("min_distance") {
            self.min_distance = v.as_f64().ok_or("min_distance")? as f32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let me = agent.loc;
        let mut nearest_sq = i32::MAX;
        for other in world.population.iter_alive() {
            if other.id == agent.id {
                continue;
            }
            // `grid.dist_sq` is topology-aware: on a torus, "anti-flock"
            // means anti-flock across the seam too — neighbours don't
            // get cheap distance from the cylinder wrap.
            let d2 = world.grid.dist_sq(me, other.loc);
            if d2 < nearest_sq {
                nearest_sq = d2;
            }
        }
        let nearest = (nearest_sq as f32).sqrt();
        let pass = nearest >= self.min_distance;
        let max = self.min_distance * 2.0;
        (pass, (nearest / max).clamp(0.0, 1.0))
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
    fn id(&self) -> &str {
        "survivor"
    }
    fn name(&self) -> &str {
        "Survivor (shifting safe zone)"
    }
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
        if let Some(v) = p.get("safe_radius") {
            self.safe_radius = v.as_f64().ok_or("safe_radius")? as f32;
        }
        if let Some(v) = p.get("period") {
            self.period = v.as_f64().ok_or("period")? as u32;
        }
        if let Some(v) = p.get("stress") {
            self.stress = v.as_f64().ok_or("stress")? as f32;
        }
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
        if self.period > 0 && ctx.step > 0 && ctx.step.is_multiple_of(self.period) {
            let sx = ctx.config.size_x as u32;
            let sy = ctx.config.size_y as u32;
            self.centre =
                (ctx.rng.gen_range_u32(0, sx) as f32, ctx.rng.gen_range_u32(0, sy) as f32);
        }
        let r = self.safe_radius * ctx.config.size_x.max(ctx.config.size_y) as f32;
        let r2 = r * r;
        let (cx, cy) = self.centre;
        let stress = self.stress;

        let victims: Vec<u32> = ctx
            .population
            .iter_alive()
            .filter_map(|a| {
                // Topology-aware distance keeps the safe zone shaped
                // correctly across wrap seams on a torus.
                let (dx, dy) = ctx.grid.delta_to_point(a.loc, cx, cy);
                let in_safe = dx * dx + dy * dy <= r2;
                if in_safe {
                    return None;
                }
                if ctx.rng.gen_bool(stress) {
                    Some(a.id)
                } else {
                    None
                }
            })
            .collect();
        for id in victims {
            ctx.population.queue_for_death(id);
        }
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
