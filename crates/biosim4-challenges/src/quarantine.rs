//! Quarantine — contagion challenge with a seed zone.
//!
//! Bit 0 of `challenge_bits` means "infected". On generation start, any agent
//! within the seed disc is infected. Each step, every infected agent has a
//! per-neighbor `transmit_prob` chance to infect each orthogonally-adjacent
//! uninfected neighbour. Survival = uninfected at gen-end.
//!
//! Agents read their own status via `challenge_bit_0`, so they can learn to
//! flee crowds when healthy and stop spreading (e.g. immobilise) when sick.

use biosim4_core::agent::Agent;
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::types::Coord;
use biosim4_core::world::World;
use serde_json::{json, Value};

const BIT_INFECTED: u32 = 1 << 0;

pub struct QuarantineChallenge {
    /// Normalized centre of the initial-infection disc.
    pub seed_cx: f32,
    pub seed_cy: f32,
    /// Normalized radius (relative to max(size_x, size_y)).
    pub seed_radius: f32,
    /// Per-step per-contact transmission probability.
    pub transmit_prob: f32,
}

impl Default for QuarantineChallenge {
    fn default() -> Self {
        Self { seed_cx: 0.5, seed_cy: 0.5, seed_radius: 0.12, transmit_prob: 0.20 }
    }
}

impl Challenge for QuarantineChallenge {
    fn id(&self) -> &str {
        "quarantine"
    }
    fn name(&self) -> &str {
        "Quarantine"
    }
    fn description(&self) -> &str {
        "Agents inside the seed disc start infected; infection spreads to adjacent uninfected agents probabilistically each step. Survive iff uninfected at gen-end."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "seed_cx":       { "type": "number", "minimum": 0.0,  "maximum": 1.0,  "default": 0.5 },
                "seed_cy":       { "type": "number", "minimum": 0.0,  "maximum": 1.0,  "default": 0.5 },
                "seed_radius":   { "type": "number", "minimum": 0.02, "maximum": 0.5,  "default": 0.12 },
                "transmit_prob": { "type": "number", "minimum": 0.0,  "maximum": 1.0,  "default": 0.20 }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("seed_cx") {
            self.seed_cx = v.as_f64().ok_or("seed_cx")? as f32;
        }
        if let Some(v) = p.get("seed_cy") {
            self.seed_cy = v.as_f64().ok_or("seed_cy")? as f32;
        }
        if let Some(v) = p.get("seed_radius") {
            self.seed_radius = v.as_f64().ok_or("seed_radius")? as f32;
        }
        if let Some(v) = p.get("transmit_prob") {
            self.transmit_prob = (v.as_f64().ok_or("transmit_prob")? as f32).clamp(0.0, 1.0);
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, _world: &World) -> (bool, f32) {
        let infected = agent.challenge_bits & BIT_INFECTED != 0;
        if infected {
            (false, 0.0)
        } else {
            (true, 1.0)
        }
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        let sx = ctx.config.size_x as f32;
        let sy = ctx.config.size_y as f32;
        let cx = self.seed_cx * sx;
        let cy = self.seed_cy * sy;
        let r2 = (self.seed_radius * sx.max(sy)).powi(2);

        for a in ctx.population.iter_alive_mut() {
            a.challenge_bits &= !BIT_INFECTED;
            // Topology-aware: seed disc wraps across the seam on torus
            // worlds — same behavioural intent (one connected disc),
            // just measured along the shortest path.
            if ctx.grid.dist_sq_to_point(a.loc, cx, cy) <= r2 {
                a.challenge_bits |= BIT_INFECTED;
            }
        }
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        // Snapshot infected positions; mutate new infections afterward.
        let infected: Vec<Coord> = ctx
            .population
            .iter_alive()
            .filter(|a| a.challenge_bits & BIT_INFECTED != 0)
            .map(|a| a.loc)
            .collect();

        let dirs: [(i16, i16); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        let p = self.transmit_prob;
        let mut to_infect: std::collections::HashSet<u32> = std::collections::HashSet::new();

        for loc in infected {
            for (dx, dy) in dirs {
                let nb = Coord::new(loc.x + dx, loc.y + dy);
                if !ctx.grid.is_occupied_at(nb) {
                    continue;
                }
                let nb_id = ctx.grid.at(nb);
                if to_infect.contains(&nb_id) {
                    continue;
                }
                let Some(nb_agent) = ctx.population.get(nb_id) else { continue };
                if nb_agent.challenge_bits & BIT_INFECTED != 0 {
                    continue;
                }
                if ctx.rng.gen_bool(p) {
                    to_infect.insert(nb_id);
                }
            }
        }

        for id in to_infect {
            if let Some(a) = ctx.population.get_mut(id) {
                a.challenge_bits |= BIT_INFECTED;
            }
        }
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        // Static seed disc (orange tint) + live points for each currently
        // infected agent so the spread is visible at a glance.
        let mut out = vec![ChallengeOverlay::Circle {
            cx: self.seed_cx * sx,
            cy: self.seed_cy * sy,
            radius: self.seed_radius * sx.max(sy),
            color: [255, 140, 40, 50],
        }];
        let points: Vec<(f32, f32)> = world
            .population
            .iter_alive()
            .filter(|a| a.challenge_bits & BIT_INFECTED != 0)
            .map(|a| (a.loc.x as f32 + 0.5, a.loc.y as f32 + 0.5))
            .collect();
        if !points.is_empty() {
            out.push(ChallengeOverlay::Points { points, color: [255, 140, 40, 220], size: 1.4 });
        }
        out
    }
}
