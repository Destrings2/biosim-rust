//! Pheromone-signal emission actions.
//!
//! Each emit_signal* action runs the standard threshold pipeline (sign-
//! preserving squash · responsiveness, gated at `> 0.5`, drawn). On fire
//! it deposits a burst at the agent's current cell on the corresponding
//! signal layer.

use crate::util::{fire_with_threshold, EMIT_THRESHOLD};
use biosim4_core::registry::{Action, ActionContext};

pub(crate) struct EmitSignal0;
impl Action for EmitSignal0 {
    fn id(&self) -> &str {
        "emit_signal0"
    }
    fn name(&self) -> &str {
        "emit signal 0"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if fire_with_threshold(level, EMIT_THRESHOLD, ctx) {
            ctx.signals.increment(0, ctx.agent.loc, ctx.world.grid);
        }
    }
}

pub(crate) struct EmitSignal1;
impl Action for EmitSignal1 {
    fn id(&self) -> &str {
        "emit_signal1"
    }
    fn name(&self) -> &str {
        "emit signal 1"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if fire_with_threshold(level, EMIT_THRESHOLD, ctx) {
            ctx.signals.increment(1, ctx.agent.loc, ctx.world.grid);
        }
    }
}

pub(crate) struct EmitSignal2;
impl Action for EmitSignal2 {
    fn id(&self) -> &str {
        "emit_signal2"
    }
    fn name(&self) -> &str {
        "emit signal 2"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        if fire_with_threshold(level, EMIT_THRESHOLD, ctx) {
            ctx.signals.increment(2, ctx.agent.loc, ctx.world.grid);
        }
    }
}
