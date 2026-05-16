//! `kill_forward` action — request death of the agent directly ahead.

use crate::util::{level_to_signed_prob, prob2bool, ACTION_MIN, ACTION_RANGE, KILL_THRESHOLD};
use biosim4_core::registry::{Action, ActionContext};
use biosim4_core::types::Coord;

pub(crate) struct KillForward;
impl Action for KillForward {
    fn id(&self) -> &str {
        "kill_forward"
    }
    fn name(&self) -> &str {
        "kill forward"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if !ctx.config_kill_enable {
            return;
        }
        // Same squash-and-scale as the signal-emit pipeline, then a kill-
        // specific remap of the probability through `ACTION_MIN` /
        // `ACTION_RANGE` before the draw.
        let p = level_to_signed_prob(level) * ctx.responsiveness_adjusted;
        if p <= KILL_THRESHOLD {
            return;
        }
        let p_kill = ((p - ACTION_MIN) / ACTION_RANGE).clamp(0.0, 1.0);
        if !prob2bool(p_kill, ctx.rng) {
            return;
        }
        let step = ctx.agent.last_move_dir.as_normalized_coord();
        let target = Coord::new(ctx.agent.loc.x + step.x, ctx.agent.loc.y + step.y);
        if ctx.world.grid.is_occupied_at(target) {
            let victim_id = ctx.world.grid.at(target);
            ctx.death_queue.push(victim_id);
        }
    }
}
