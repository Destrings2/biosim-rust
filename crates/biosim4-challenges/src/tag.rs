//! Tag — a contact-transfer "you're it!" challenge.
//!
//! On generation start, a fraction of the population is marked as "it"
//! (bit 0 of `challenge_bits`). Each step, every "it" agent tries to pass
//! the bit to an orthogonally-adjacent non-it neighbor. The former "it" gets
//! a few steps of cooldown (bits 1-3 used as a counter) so the bit doesn't
//! ping-pong immediately. Survivors at gen-end are agents that are not "it"
//! at the final step.
//!
//! The `challenge_bit_0` sensor lets evolved nets read their own status, so
//! agents can learn distinct behaviors for "I'm it, find someone" vs "I'm
//! safe, avoid the it crowd".

use biosim4_core::agent::Agent;
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::types::Coord;
use biosim4_core::world::World;
use serde_json::{json, Value};

const BIT_IT: u32 = 1 << 0;
const COOLDOWN_SHIFT: u32 = 1;
const COOLDOWN_MASK: u32 = 0b111 << COOLDOWN_SHIFT; // bits 1-3, max value 7

pub struct TagChallenge {
    pub it_fraction: f32,
    pub cooldown_steps: u32,
}

impl Default for TagChallenge {
    fn default() -> Self {
        Self { it_fraction: 0.10, cooldown_steps: 5 }
    }
}

fn cooldown(bits: u32) -> u32 {
    (bits & COOLDOWN_MASK) >> COOLDOWN_SHIFT
}

fn set_cooldown(bits: u32, cd: u32) -> u32 {
    (bits & !COOLDOWN_MASK) | ((cd.min(7)) << COOLDOWN_SHIFT)
}

impl Challenge for TagChallenge {
    fn id(&self) -> &str {
        "tag"
    }
    fn name(&self) -> &str {
        "Tag (you're it!)"
    }
    fn description(&self) -> &str {
        "Some agents start as 'it'. On contact, the bit transfers to a neighbor and the former 'it' enters a cooldown. Survive iff you are NOT 'it' at gen-end."
    }
    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "it_fraction":    { "type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.10,
                                    "description": "Fraction of the population marked 'it' at gen-start" },
                "cooldown_steps": { "type": "number", "minimum": 0.0,  "maximum": 7.0, "default": 5.0,
                                    "description": "Steps of immunity after losing the it bit" }
            }
        })
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("it_fraction") {
            self.it_fraction = (v.as_f64().ok_or("it_fraction")? as f32).clamp(0.01, 1.0);
        }
        if let Some(v) = p.get("cooldown_steps") {
            self.cooldown_steps = (v.as_f64().ok_or("cooldown_steps")? as u32).min(7);
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, _world: &World) -> (bool, f32) {
        let is_it = agent.challenge_bits & BIT_IT != 0;
        if is_it {
            (false, 0.0)
        } else {
            (true, 1.0)
        }
    }
    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        let n = ctx.population.alive_count();
        let target = ((n as f32) * self.it_fraction).round() as usize;
        // Clear the bits we own on every agent.
        for a in ctx.population.iter_alive_mut() {
            a.challenge_bits &= !(BIT_IT | COOLDOWN_MASK);
        }
        // Reservoir-style sampling: pick the first `target` ids after a
        // Fisher–Yates-ish shuffle done by random swaps. The id list is
        // already in `alive_ids`.
        let ids: Vec<u32> = ctx.population.alive_ids().to_vec();
        let mut shuffled = ids;
        for i in (1..shuffled.len()).rev() {
            let j = ctx.rng.gen_range_u32(0, (i + 1) as u32) as usize;
            shuffled.swap(i, j);
        }
        for &id in shuffled.iter().take(target.min(shuffled.len())) {
            if let Some(a) = ctx.population.get_mut(id) {
                a.challenge_bits |= BIT_IT;
            }
        }
    }
    fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        // Snapshot the (id, loc) of all current "it" agents.
        let it_agents: Vec<(u32, Coord)> = ctx
            .population
            .iter_alive()
            .filter(|a| a.challenge_bits & BIT_IT != 0)
            .map(|a| (a.id, a.loc))
            .collect();

        // Decrement cooldowns on every alive agent (saturating at 0).
        for a in ctx.population.iter_alive_mut() {
            let cd = cooldown(a.challenge_bits);
            if cd > 0 {
                a.challenge_bits = set_cooldown(a.challenge_bits, cd - 1);
            }
        }

        // Resolve transfers. For each "it", look at 4-neighbours; if any has
        // no cooldown and is not already "it", pass the bit there.
        let dirs: [(i16, i16); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        // Track which agents *already received* the bit this step so an "it"
        // can't ricochet through the chain in one step.
        let mut new_it: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut clear_it: Vec<u32> = Vec::new();

        for (it_id, loc) in it_agents {
            // Randomized neighbor order so the bit doesn't always head east.
            let mut order = [0, 1, 2, 3];
            for i in (1..4).rev() {
                let j = ctx.rng.gen_range_u32(0, (i + 1) as u32) as usize;
                order.swap(i, j);
            }
            for &k in &order {
                let (dx, dy) = dirs[k];
                let nb = Coord::new(loc.x + dx, loc.y + dy);
                if !ctx.grid.is_occupied_at(nb) {
                    continue;
                }
                let nb_id = ctx.grid.at(nb);
                if new_it.contains(&nb_id) {
                    continue;
                }
                let Some(nb_agent) = ctx.population.get(nb_id) else { continue };
                if nb_agent.challenge_bits & BIT_IT != 0 {
                    continue;
                }
                if cooldown(nb_agent.challenge_bits) > 0 {
                    continue;
                }
                // Transfer.
                new_it.insert(nb_id);
                clear_it.push(it_id);
                break;
            }
        }

        let cd = self.cooldown_steps;
        for id in clear_it {
            if let Some(a) = ctx.population.get_mut(id) {
                a.challenge_bits &= !BIT_IT;
                a.challenge_bits = set_cooldown(a.challenge_bits, cd);
            }
        }
        for id in new_it {
            if let Some(a) = ctx.population.get_mut(id) {
                a.challenge_bits |= BIT_IT;
            }
        }
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        // Visualise current "it" agents as red dots so the user can track
        // who's holding the hot potato. No static region to draw.
        let points: Vec<(f32, f32)> = world
            .population
            .iter_alive()
            .filter(|a| a.challenge_bits & BIT_IT != 0)
            .map(|a| (a.loc.x as f32 + 0.5, a.loc.y as f32 + 0.5))
            .collect();
        if points.is_empty() {
            return Vec::new();
        }
        vec![ChallengeOverlay::Points { points, color: [255, 60, 60, 220], size: 1.6 }]
    }
}
