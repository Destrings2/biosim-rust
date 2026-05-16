//! Internal-state modulator actions.
//!
//! These three actions update internal agent state (responsiveness,
//! oscillator period, long-probe distance) rather than producing motor
//! output. They consume `level` directly and never multiply by
//! `responsiveness_adjusted` — otherwise a low responsiveness would
//! dampen the very signal an agent uses to raise itself out of that
//! state, making it a one-way trap.

use biosim4_core::registry::{Action, ActionContext};

pub(crate) struct SetResponsiveness;
impl Action for SetResponsiveness {
    fn id(&self) -> &str {
        "set_responsiveness"
    }
    fn name(&self) -> &str {
        "set responsiveness"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        ctx.agent.responsiveness = ((level.tanh() + 1.0) / 2.0).clamp(0.0, 1.0);
    }
}

pub(crate) struct SetOscillatorPeriod;
impl Action for SetOscillatorPeriod {
    fn id(&self) -> &str {
        "set_oscillator_period"
    }
    fn name(&self) -> &str {
        "set oscillator period"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let f01 = (level.tanh() + 1.0) / 2.0;
        ctx.agent.osc_period = (1.5 + (7.0 * f01).exp()) as u32 + 1;
    }
}

pub(crate) struct SetLongprobeDist;
impl Action for SetLongprobeDist {
    fn id(&self) -> &str {
        "set_longprobe_dist"
    }
    fn name(&self) -> &str {
        "set longprobe dist"
    }
    fn execute(&self, level: f32, ctx: &mut ActionContext) {
        let f01 = (level.tanh() + 1.0) / 2.0;
        ctx.agent.long_probe_dist = (1.0 + f01 * 32.0) as u32;
    }
}
