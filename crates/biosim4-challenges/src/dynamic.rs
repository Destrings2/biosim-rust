//! Dynamic / time-varying challenges where the survival landscape itself
//! changes during the generation. Tests the population's ability to evolve
//! responsive (rather than static) behaviour.

use biosim4_core::agent::Agent;
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::world::World;
use serde_json::{json, Value};

// ── Sun Tracker ─────────────────────────────────────────────────────────

/// Bit 0 of `challenge_bits`: set on every step the agent is currently
/// inside the sun disc, cleared otherwise. This is what the
/// `challenge_bit_0` sensor reads, giving peeps a clean per-step "am I in
/// the sun right now?" signal they can wire into their NN.
const SUN_BIT_IN_SUN_NOW: u32 = 1 << 0;

/// Bits 8..15 of `challenge_bits`: accumulated warmth counter (0..=255).
/// Survival ranks agents by this counter — the user's "select the cells
/// that received the most warmth" semantics. Lives above the
/// `challenge_bit_0..3` sensor window so the low bits stay reserved for
/// real-time challenge signals.
const SUN_WARMTH_SHIFT: u32 = 8;
const SUN_WARMTH_MASK: u32 = 0xFF << SUN_WARMTH_SHIFT;
const SUN_MAX_WARMTH: u32 = 0xFF;

/// Number of warmth samples taken across a generation. Sampling stride
/// is `steps_per_generation / SUN_TARGET_SAMPLES`, so a perfect tracker
/// reaches warmth = `SUN_TARGET_SAMPLES`. Sized so the counter never
/// hits `SUN_MAX_WARMTH` under any reasonable `steps_per_generation`.
const SUN_TARGET_SAMPLES: u32 = 32;

/// A "sun" disc rotates around the world centre over the course of the
/// generation. Every step the challenge sets/clears bit 0 of each agent's
/// `challenge_bits` based on whether it's inside the disc; every
/// `steps_per_gen / 32` steps it also increments the agent's warmth
/// counter (bits 8..15) if the agent is in the sun on that tick.
///
/// Survival ranks agents by accumulated warmth (top-N selection by the
/// underlying GA): `pass = warmth >= min_warmth`, with the score equal
/// to `warmth / SUN_TARGET_SAMPLES` so the fitness curve stays
/// monotonic with respect to "how much sunlight did I catch this gen".
/// There is *no* final-position requirement — the old "must end inside
/// the sun" clause was the main reason the GA stalled around 30%
/// survival.
pub struct SunTrackerChallenge {
    pub radius: f32,       // sun-disc radius (normalized to max(size_x, size_y))
    pub orbit_radius: f32, // distance from centre (normalized)
    pub revolutions: f32,  // full orbits per generation
    pub min_warmth: u32,   // required warmth (0..=SUN_TARGET_SAMPLES)
}

impl Default for SunTrackerChallenge {
    fn default() -> Self {
        // Slow orbit + ~third-of-a-generation warmth threshold so a typical
        // 200-pop GA can bootstrap without survival rates collapsing in the
        // first few gens. Crank `revolutions` / `min_warmth` for harder runs.
        Self { radius: 0.20, orbit_radius: 0.25, revolutions: 0.25, min_warmth: 12 }
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
        "A sun disc orbits the centre. Each step bit 0 of `challenge_bits` is set when the agent is inside the disc (readable via the `challenge_bit_0` sensor); roughly 32 times across the generation the warmth counter ticks up for in-sun agents. Survivors are the agents whose warmth reached `min_warmth` — selection is purely by accumulated sunlight, with no end-of-generation position requirement."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "radius":        { "type": "number", "minimum": 0.05, "maximum": 0.4,  "default": 0.20 },
                "orbit_radius":  { "type": "number", "minimum": 0.10, "maximum": 0.5,  "default": 0.25 },
                "revolutions":   { "type": "number", "minimum": 0.25, "maximum": 4.0,  "default": 0.25 },
                "min_warmth":    { "type": "number", "minimum": 1.0,  "maximum": 32.0, "default": 12.0 }
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
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        // Survival is pure warmth-based selection: top-N by accumulated
        // tracking ticks. No final-step position filter — that was the
        // reason the old version stalled around 30% survival.
        let warmth = warmth_of(agent.challenge_bits);
        let pass = warmth >= self.min_warmth;

        // The fitness score that the GA sorts on combines warmth (the
        // dominant term) with a small proximity-to-sun bonus that breaks
        // ties when many agents have warmth = 0 (gen-0 random pop). Without
        // this gradient, the bootstrap fallback picks parents uniformly at
        // random from the top 10% and evolution can't gain traction.
        // Proximity weight is < 1/SUN_TARGET_SAMPLES so a 1-unit warmth
        // difference always beats any proximity difference — the gradient
        // is a tiebreaker, not a competing signal.
        let warmth_term = warmth as f32 / SUN_TARGET_SAMPLES as f32;
        let (sx, sy) =
            sun_pos_at(self, world.step, world.steps_per_generation, world.size_x, world.size_y);
        let dx = agent.loc.x as f32 - sx;
        let dy = agent.loc.y as f32 - sy;
        let diag = ((world.size_x as f32).powi(2) + (world.size_y as f32).powi(2)).sqrt();
        let proximity = 1.0 - ((dx * dx + dy * dy).sqrt() / diag).clamp(0.0, 1.0);
        let proximity_weight = 1.0 / (SUN_TARGET_SAMPLES as f32 + 1.0);
        let score = (warmth_term + proximity * proximity_weight).clamp(0.0, 1.0);
        (pass, score)
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        // Clear every bit we own — both the per-step in-sun flag and the
        // warmth counter — so the new gen starts cold.
        for a in ctx.population.iter_alive_mut() {
            a.challenge_bits &= !(SUN_BIT_IN_SUN_NOW | SUN_WARMTH_MASK);
        }
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        // One pass per step over the alive population: update bit 0 for
        // everyone (so peeps reading `challenge_bit_0` see fresh state),
        // and on sample ticks also bump the warmth counter.
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
            let dx = a.loc.x as f32 - sx;
            let dy = a.loc.y as f32 - sy;
            let in_sun = dx * dx + dy * dy <= r2;
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
        let mut nearest_sq = f32::INFINITY;
        for other in world.population.iter_alive() {
            if other.id == agent.id {
                continue;
            }
            let dx = (other.loc.x - me.x) as f32;
            let dy = (other.loc.y - me.y) as f32;
            let d2 = dx * dx + dy * dy;
            if d2 < nearest_sq {
                nearest_sq = d2;
            }
        }
        let nearest = nearest_sq.sqrt();
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
                let dx = a.loc.x as f32 - cx;
                let dy = a.loc.y as f32 - cy;
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
