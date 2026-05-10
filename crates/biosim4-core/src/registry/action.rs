use std::collections::HashSet;
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

/// Ordered registry — position in the vec is the *registration* index, but
/// only **enabled** actions participate in genome wiring.
///
/// The lifecycle mirrors `SensorRegistry`: pending disables via `set_enabled`,
/// committed at generation boundaries via `commit_enabled()`.
pub struct ActionRegistry {
    actions: Vec<Box<dyn Action>>,
    /// Stable IDs of actions that are disabled (pending or committed).
    disabled: HashSet<String>,
    /// Dense map: `active_map[enabled_idx] = actual_idx`.
    active_map: Vec<u16>,
}

impl ActionRegistry {
    pub fn new() -> Self {
        Self { actions: Vec::new(), disabled: HashSet::new(), active_map: Vec::new() }
    }

    pub fn register(&mut self, action: Box<dyn Action>) {
        self.actions.push(action);
        self.rebuild_active_map();
    }

    // ── Enable / disable ─────────────────────────────────────────────────

    /// Mark an action enabled or disabled by its stable ID.
    /// The change is *pending* until the next `commit_enabled()` call.
    /// Mid-generation the disabled action is immediately silenced (skipped).
    pub fn set_enabled(&mut self, id: &str, enabled: bool) {
        if enabled {
            self.disabled.remove(id);
        } else {
            self.disabled.insert(id.to_string());
        }
        // Do NOT rebuild active_map here — see SensorRegistry for rationale.
    }

    /// Returns `true` if the action is currently enabled (not pending-disabled).
    pub fn is_enabled(&self, id: &str) -> bool {
        !self.disabled.contains(id)
    }

    /// Commit pending changes. Call **before** `wiring_config()` in
    /// `spawn_new_generation` so new nnets wire against the updated active set.
    pub fn commit_enabled(&mut self) {
        self.rebuild_active_map();
    }

    fn rebuild_active_map(&mut self) {
        self.active_map = self.actions
            .iter()
            .enumerate()
            .filter(|(_, a)| !self.disabled.contains(a.id()))
            .map(|(i, _)| i as u16)
            .collect();
    }

    // ── Counts ────────────────────────────────────────────────────────────

    /// Total number of registered actions.
    pub fn count(&self) -> u16 { self.actions.len() as u16 }

    /// Number of *active* (committed-enabled) actions.
    pub fn enabled_count(&self) -> u16 { self.active_map.len() as u16 }

    // ── Execution ─────────────────────────────────────────────────────────

    /// Execute action at `enabled_idx` — an index into the *active* (dense)
    /// set. Silently skips pending-disabled actions mid-generation.
    pub fn execute(&self, enabled_idx: u16, level: f32, ctx: &mut ActionContext) {
        let actual_idx = self.active_map[enabled_idx as usize];
        let a = &self.actions[actual_idx as usize];
        // Honour pending mid-generation disables immediately.
        if self.disabled.contains(a.id()) {
            return;
        }
        a.execute(level, ctx);
    }

    // ── Introspection ─────────────────────────────────────────────────────

    pub fn name(&self, idx: u16) -> &str {
        self.actions[idx as usize].name()
    }

    pub fn id(&self, idx: u16) -> &str {
        self.actions[idx as usize].id()
    }

    /// Iterate all registered actions with `(registration_index, action, is_enabled)`.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &dyn Action, bool)> {
        self.actions.iter().enumerate().map(|(i, a)| {
            let enabled = !self.disabled.contains(a.id());
            (i as u16, a.as_ref(), enabled)
        })
    }
}

impl Default for ActionRegistry {
    fn default() -> Self { Self::new() }
}
