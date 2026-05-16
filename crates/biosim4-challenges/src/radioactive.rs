//! World-edge hazard challenges.
//!
//! - `radioactive_walls` — kills agents probabilistically based on distance
//!   to the currently-active wall (west then east, flipped at gen midpoint).
//! - `lethal_borders` — kills any agent that steps onto the world's outer
//!   border row/column. Deterministic, instant.
//!
//! Both apply damage per-step in `on_sim_step` and report "alive at gen
//! end == pass" from `evaluate`.

use biosim4_core::agent::Agent;
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::world::World;
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
    fn default() -> Self {
        Self { intensity: 0.5, half_life: 8.0 }
    }
}

impl Challenge for RadioactiveWallsChallenge {
    fn id(&self) -> &str {
        "radioactive_walls"
    }
    fn name(&self) -> &str {
        "Radioactive Walls"
    }
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
        if let Some(v) = p.get("intensity") {
            self.intensity = v.as_f64().ok_or("intensity")? as f32;
        }
        if let Some(v) = p.get("half_life") {
            self.half_life = v.as_f64().ok_or("half_life")? as f32;
        }
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
        let victims: Vec<u32> = ctx
            .population
            .iter_alive()
            .filter_map(|a| {
                let dist = (a.loc.x as i32 - active_wall_x as i32).unsigned_abs() as f32;
                let p = self.intensity * (-k * dist).exp();
                if ctx.rng.gen_bool(p) {
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
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        let half_gen = world.steps_per_generation / 2;
        let active_is_west = world.step < half_gen;

        // Three concentric bands at half_life, 2*half_life, 3*half_life from
        // the active wall. The gizmo renderer outlines each rect, so this
        // shows up as three vertical lines marking the kill-prob isolines.
        let h = sy;
        let bands: [(f32, u8); 3] =
            [(self.half_life, 90), (2.0 * self.half_life, 60), (3.0 * self.half_life, 35)];
        let mut out = Vec::with_capacity(4);
        for (w, alpha) in bands {
            let w = w.min(sx);
            // Active wall (vivid red)
            let (x_active, _) = if active_is_west { (0.0, 0.0) } else { (sx - w, 0.0) };
            out.push(ChallengeOverlay::Rectangle {
                x: x_active,
                y: 0.0,
                w,
                h,
                color: [255, 40, 40, alpha],
            });
            // Inactive wall (dim hint that it will activate later)
            let (x_inactive, _) = if active_is_west { (sx - w, 0.0) } else { (0.0, 0.0) };
            out.push(ChallengeOverlay::Rectangle {
                x: x_inactive,
                y: 0.0,
                w,
                h,
                color: [200, 80, 80, alpha / 3],
            });
        }
        out
    }
}

// ── Lethal Borders ──────────────────────────────────────────────────────

/// Instant-kill version of "stay off the wall": any agent sitting on the
/// world's outer border row or column (x = 0 / size_x − 1 or y = 0 /
/// size_y − 1) is queued for death the same step. Selection pressure is
/// "learn to stay inside the grid".
///
/// `grace_steps` skips the kill check for the first N steps of the
/// generation so peeps unlucky enough to spawn at the border get a
/// chance to step away. The default is 1 — peeps spawn, take one move,
/// and then any peep still at a border dies.
pub struct LethalBordersChallenge {
    /// Number of opening steps during which the border is *not* lethal.
    /// Prevents spawn-at-border peeps from being killed on step 0 before
    /// they ever get to act, which would inject pure environmental
    /// noise into the fitness signal.
    pub grace_steps: u32,
}

impl Default for LethalBordersChallenge {
    fn default() -> Self {
        Self { grace_steps: 1 }
    }
}

impl Challenge for LethalBordersChallenge {
    fn id(&self) -> &str {
        "lethal_borders"
    }
    fn name(&self) -> &str {
        "Lethal Borders"
    }
    fn description(&self) -> &str {
        "Touching the world's outer border (x = 0 / size_x − 1, y = 0 / size_y − 1) kills the agent the same step. `grace_steps` opens the generation with a short safe window so peeps that spawn on a border can step away before the kill check begins. Survivors at gen-end pass — counterpart to `against_any_wall`, which rewards exactly the opposite behaviour."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "grace_steps": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 64,
                    "default": 1,
                    "description": "Steps from gen start with no kill check — keeps spawn-at-border peeps from dying on step 0"
                }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("grace_steps") {
            self.grace_steps = v.as_u64().ok_or("grace_steps")? as u32;
        }
        Ok(())
    }
    fn evaluate(&self, _agent: &Agent, _world: &World) -> (bool, f32) {
        // Kill is applied per-step; alive at gen-end == survived.
        (true, 1.0)
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        if ctx.step < self.grace_steps {
            return;
        }
        // Snapshot the victim ids before mutating — `queue_for_death`
        // takes `&mut population` and we're already iterating it.
        let victims: Vec<u32> = ctx
            .population
            .iter_alive()
            .filter(|a| ctx.grid.is_border(a.loc))
            .map(|a| a.id)
            .collect();
        for id in victims {
            ctx.population.queue_for_death(id);
        }
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        // Paint the four border strips so the kill zone is visible in
        // the canvas. Single-cell wide; the gizmo renderer outlines each
        // rect, so this shows up as a red frame hugging the edge.
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        let color = [255, 40, 40, 110];
        vec![
            // top
            ChallengeOverlay::Rectangle { x: 0.0, y: 0.0, w: sx, h: 1.0, color },
            // bottom
            ChallengeOverlay::Rectangle { x: 0.0, y: sy - 1.0, w: sx, h: 1.0, color },
            // left
            ChallengeOverlay::Rectangle { x: 0.0, y: 0.0, w: 1.0, h: sy, color },
            // right
            ChallengeOverlay::Rectangle { x: sx - 1.0, y: 0.0, w: 1.0, h: sy, color },
        ]
    }
}
