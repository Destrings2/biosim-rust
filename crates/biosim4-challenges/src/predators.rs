//! Challenge where programmable predators hunt peeps.
//!
//! Predators use line-of-sight (LOS) to track targets and must consume agents to avoid starvation.
//! After eating, a predator enters a full state and stops hunting until fullness expires.

use biosim4_core::agent::Agent;
use biosim4_core::programmable::library::{actions, sensors};
use biosim4_core::programmable::{OwnerTag, Program, ProgramContext, ProgramOutput, Programmable};
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay, WorldMut};
use biosim4_core::types::Coord;
use biosim4_core::world::World;
use serde_json::{json, Value};

const PREDATORS_TAG: OwnerTag = 0xA002;

/// Per-entity state slots.
const STATE_STARVATION_TIMER: usize = 0;
const STATE_CHASING_TARGET_X: usize = 1;
const STATE_CHASING_TARGET_Y: usize = 2;
const STATE_IS_CHASING: usize = 3;
/// Counts down from `fullness_duration` after a meal. Predator hunts only when this is 0.
const STATE_FULLNESS_TIMER: usize = 4;

struct Predator {
    view_distance: u16,
    /// Steps before an unfed predator dies.
    max_starvation_time: u16,
    /// Steps a predator stays full (and won't hunt) after eating.
    fullness_duration: u16,
}

impl Program for Predator {
    fn id(&self) -> &str {
        "predator"
    }

    fn name(&self) -> &str {
        "Predator"
    }

    fn on_spawn(&self, this: &mut Programmable, _rng: &mut biosim4_core::rng::Rng) {
        this.state[STATE_STARVATION_TIMER] = self.max_starvation_time as f32;
        this.state[STATE_IS_CHASING] = 0.0;
        this.state[STATE_FULLNESS_TIMER] = 0.0;
    }

    fn step(&self, this: &mut Programmable, ctx: &mut ProgramContext, out: &mut ProgramOutput) {
        // Tick fullness down. While full, wander and skip hunting.
        if this.state[STATE_FULLNESS_TIMER] > 0.0 {
            this.state[STATE_FULLNESS_TIMER] -= 1.0;
            self.wander(this, ctx, out);
            self.set_color_full(this, out);
            return;
        }

        // Tick starvation. Die if time runs out.
        this.state[STATE_STARVATION_TIMER] -= 1.0;
        if this.state[STATE_STARVATION_TIMER] <= 0.0 {
            out.die = true;
            return;
        }

        if let Some((target_loc, _agent_id)) = sensors::nearest_peep_in_los(ctx, this.loc, self.view_distance) {
            this.state[STATE_IS_CHASING] = 1.0;
            this.state[STATE_CHASING_TARGET_X] = target_loc.x as f32;
            this.state[STATE_CHASING_TARGET_Y] = target_loc.y as f32;

            let dx = (target_loc.x - this.loc.x).abs();
            let dy = (target_loc.y - this.loc.y).abs();

            if dx <= 1 && dy <= 1 {
                // Consume the adjacent peep.
                out.kill_peep_at = Some(target_loc);
                // Claim the cell; peep death is processed first in the merge phase.
                out.move_to = Some(target_loc);

                // Reset starvation and enter fullness cooldown.
                this.state[STATE_STARVATION_TIMER] = self.max_starvation_time as f32;
                this.state[STATE_FULLNESS_TIMER] = self.fullness_duration as f32;
                this.state[STATE_IS_CHASING] = 0.0;
            } else {
                if let Some(next_loc) = actions::move_towards(this.loc, target_loc) {
                    out.move_to = Some(next_loc);
                    this.heading = (next_loc - this.loc).as_dir();
                }
            }
        } else {
            // No visible target — continue to last known location or wander.
            if this.state[STATE_IS_CHASING] > 0.0 {
                let target_loc = Coord::new(
                    this.state[STATE_CHASING_TARGET_X] as i16,
                    this.state[STATE_CHASING_TARGET_Y] as i16,
                );
                if this.loc == target_loc {
                    // Stop pursuit if the last known location is empty.
                    this.state[STATE_IS_CHASING] = 0.0;
                } else if let Some(next_loc) = actions::move_towards(this.loc, target_loc) {
                    out.move_to = Some(next_loc);
                    this.heading = (next_loc - this.loc).as_dir();
                }
            } else {
                self.wander(this, ctx, out);
            }
        }

        self.set_color_hungry(this, out);
    }
}

impl Predator {
    /// Random 8-directional step.
    fn wander(&self, this: &mut Programmable, ctx: &mut ProgramContext, out: &mut ProgramOutput) {
        let roll = ctx.rng.gen_range_u32(0, 9) as i16;
        let (dx, dy) = match roll {
            0 => (0, 0),
            1 => (-1, 0),
            2 => (1, 0),
            3 => (0, -1),
            4 => (0, 1),
            5 => (-1, -1),
            6 => (1, -1),
            7 => (-1, 1),
            _ => (1, 1),
        };
        if dx != 0 || dy != 0 {
            let dest = Coord::new(this.loc.x + dx, this.loc.y + dy);
            out.move_to = Some(dest);
            this.heading = Coord::new(dx, dy).as_dir();
        }
    }

    /// Color while full: green fading to yellow as fullness drains.
    fn set_color_full(&self, this: &Programmable, out: &mut ProgramOutput) {
        let fraction = this.state[STATE_FULLNESS_TIMER] / (self.fullness_duration as f32).max(1.0);
        let green = (100.0 + 155.0 * fraction) as u8;
        out.set_color = Some([50, green, 0]);
    }

    /// Color while hungry: bright red fading to near-black as starvation grows.
    fn set_color_hungry(&self, this: &Programmable, out: &mut ProgramOutput) {
        let fraction = this.state[STATE_STARVATION_TIMER] / (self.max_starvation_time as f32).max(1.0);
        let red = (50.0 + 205.0 * fraction) as u8;
        out.set_color = Some([red, 0, 0]);
    }
}

pub struct PredatorsChallenge {
    pub count: u16,
    pub view_distance: u16,
    pub max_starvation_time: u16,
    /// Steps a predator stays satiated after a meal before hunting again.
    pub fullness_duration: u16,
    pub initial_color: [u8; 3],
}

impl Default for PredatorsChallenge {
    fn default() -> Self {
        Self {
            count: 4,
            view_distance: 12,
            max_starvation_time: 150,
            fullness_duration: 50,
            initial_color: [255, 0, 0],
        }
    }
}

impl Challenge for PredatorsChallenge {
    fn id(&self) -> &str {
        "predators"
    }

    fn name(&self) -> &str {
        "Predators"
    }

    fn description(&self) -> &str {
        "Spawns predators to hunt agents (peeps) using line-of-sight (LOS). Predators die if they fail to eat within the limit. After eating they enter a fullness cooldown before hunting again."
    }

    fn params_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "count": {
                    "type": "integer",
                    "title": "count",
                    "minimum": 0,
                    "maximum": 256,
                    "default": 4
                },
                "view_distance": {
                    "type": "integer",
                    "title": "view distance",
                    "minimum": 1,
                    "maximum": 64,
                    "default": 12
                },
                "max_starvation_time": {
                    "type": "integer",
                    "title": "max starvation time",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 150
                },
                "fullness_duration": {
                    "type": "integer",
                    "title": "fullness duration",
                    "minimum": 0,
                    "maximum": 1000,
                    "default": 50
                }
            }
        })
    }

    fn configure(&mut self, params: Value) -> Result<(), String> {
        if let Some(c) = params.get("count").and_then(|v| v.as_u64()) {
            self.count = c.min(u16::MAX as u64) as u16;
        }
        if let Some(vd) = params.get("view_distance").and_then(|v| v.as_u64()) {
            self.view_distance = vd.min(u16::MAX as u64) as u16;
        }
        if let Some(mst) = params.get("max_starvation_time").and_then(|v| v.as_u64()) {
            self.max_starvation_time = mst.min(u16::MAX as u64) as u16;
        }
        if let Some(fd) = params.get("fullness_duration").and_then(|v| v.as_u64()) {
            self.fullness_duration = fd.min(u16::MAX as u64) as u16;
        }
        Ok(())
    }

    fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        let view_distance = self.view_distance;
        let max_starvation_time = self.max_starvation_time;
        let fullness_duration = self.fullness_duration;

        let prog = ctx.programmable.register_or_get("predator", || {
            Box::new(Predator {
                view_distance,
                max_starvation_time,
                fullness_duration,
            })
        });

        for _ in 0..self.count {
            let loc = ctx.grid.find_empty_location(ctx.rng);
            let _ = ctx.programmable.spawn(ctx.grid, prog, PREDATORS_TAG, loc, self.initial_color);
        }
    }

    fn evaluate(&self, _agent: &Agent, _world: &World) -> (bool, f32) {
        // Agents killed by predators are removed from the world before evaluation.
        (true, 1.0)
    }

    fn overlays(&self, _world: &World) -> Vec<ChallengeOverlay> {
        Vec::new()
    }
}
