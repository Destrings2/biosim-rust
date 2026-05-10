use crate::agent::Agent;
use crate::rng::Rng;
use crate::world::World;

/// Context passed to every sensor during evaluation.
pub struct SensorContext<'a> {
    pub agent: &'a Agent,
    pub world: &'a World<'a>,
    pub sim_step: u32,
    pub rng: &'a mut Rng,
}

/// A pluggable sensor that reads environment/agent state and returns 0.0..1.0.
pub trait Sensor: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    /// Must return a value in [0.0, 1.0].
    fn evaluate(&self, ctx: &mut SensorContext) -> f32;
}

/// Ordered registry — position in the vec equals the genome index (after modulo).
pub struct SensorRegistry {
    sensors: Vec<Box<dyn Sensor>>,
}

impl SensorRegistry {
    pub fn new() -> Self { Self { sensors: Vec::new() } }

    pub fn register(&mut self, sensor: Box<dyn Sensor>) {
        self.sensors.push(sensor);
    }

    pub fn count(&self) -> u16 { self.sensors.len() as u16 }

    pub fn evaluate(&self, idx: u16, ctx: &mut SensorContext) -> f32 {
        let val = self.sensors[idx as usize].evaluate(ctx);
        val.clamp(0.0, 1.0)
    }

    pub fn name(&self, idx: u16) -> &str {
        self.sensors[idx as usize].name()
    }

    pub fn id(&self, idx: u16) -> &str {
        self.sensors[idx as usize].id()
    }

    pub fn iter(&self) -> impl Iterator<Item = (u16, &dyn Sensor)> {
        self.sensors.iter().enumerate().map(|(i, s)| (i as u16, s.as_ref()))
    }
}

impl Default for SensorRegistry {
    fn default() -> Self { Self::new() }
}
