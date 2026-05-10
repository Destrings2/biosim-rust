use crate::agent::{Agent, AgentId};
use crate::rng::Rng;
use crate::signals_layer::Signals;
use crate::types::Coord;
use crate::world::World;

/// Mutable context passed to every action during execution.
pub struct ActionContext<'a> {
    /// The acting agent — for immediate writes (responsiveness, osc_period, etc.)
    pub agent: &'a mut Agent,
    /// Read-only world snapshot (grid, population, etc.)
    pub world: &'a World<'a>,
    /// Deferred movement: applied at end-of-step.
    pub move_queue: &'a mut Vec<(AgentId, Coord)>,
    /// Deferred death: applied at end-of-step.
    pub death_queue: &'a mut Vec<AgentId>,
    /// Signal layers — can be mutated immediately (thread-safe in parallel mode via AtomicU8).
    pub signals: &'a mut Signals,
    pub rng: &'a mut Rng,
    pub config_kill_enable: bool,
}

/// A pluggable action that receives a raw activation level and mutates world state.
pub trait Action: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    /// `level` is the raw neural output for this action (arbitrary float range).
    fn execute(&self, level: f32, ctx: &mut ActionContext);
}

/// Ordered registry — position in the vec equals the genome index (after modulo).
pub struct ActionRegistry {
    actions: Vec<Box<dyn Action>>,
}

impl ActionRegistry {
    pub fn new() -> Self { Self { actions: Vec::new() } }

    pub fn register(&mut self, action: Box<dyn Action>) {
        self.actions.push(action);
    }

    pub fn count(&self) -> u16 { self.actions.len() as u16 }

    pub fn execute(&self, idx: u16, level: f32, ctx: &mut ActionContext) {
        self.actions[idx as usize].execute(level, ctx);
    }

    pub fn name(&self, idx: u16) -> &str {
        self.actions[idx as usize].name()
    }

    pub fn id(&self, idx: u16) -> &str {
        self.actions[idx as usize].id()
    }

    pub fn iter(&self) -> impl Iterator<Item = (u16, &dyn Action)> {
        self.actions.iter().enumerate().map(|(i, a)| (i as u16, a.as_ref()))
    }
}

impl Default for ActionRegistry {
    fn default() -> Self { Self::new() }
}
