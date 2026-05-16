//! Per-agent internal sensors: oscillator, age, and the stochastic
//! `random` channel.

use biosim4_core::registry::{Sensor, SensorContext};

pub(crate) struct Osc1;
impl Sensor for Osc1 {
    fn id(&self) -> &str {
        "osc1"
    }
    fn name(&self) -> &str {
        "oscillator 1"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // `(1 − cos(2π·phase)) / 2`: a unit-amplitude oscillation that
        // starts at 0 at step 0, peaks at 1 mid-period, and returns to 0
        // at the period boundary. Each agent shares the same global step
        // count but has its own `osc_period`, so different agents can
        // sample the same waveform at independent rates.
        let phase = (ctx.sim_step % ctx.agent.osc_period) as f32 / ctx.agent.osc_period as f32;
        let factor = -(std::f32::consts::TAU * phase).cos();
        ((factor + 1.0) / 2.0).clamp(0.0, 1.0)
    }
}

pub(crate) struct Age;
impl Sensor for Age {
    fn id(&self) -> &str {
        "age"
    }
    fn name(&self) -> &str {
        "age"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.agent.age as f32 / ctx.world.steps_per_generation as f32
    }
}

pub(crate) struct RandomSensor;
impl Sensor for RandomSensor {
    fn id(&self) -> &str {
        "random"
    }
    fn name(&self) -> &str {
        "random"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.rng.gen_f32()
    }
}
