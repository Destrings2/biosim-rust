//! Action trait, context, and registry.
//!
//! The lifecycle mirrors [`SensorRegistry`](super::sensor::SensorRegistry):
//! `set_enabled` marks changes pending; `commit_enabled` rebuilds `active_map`
//! at generation boundaries. During a generation, disabled actions are silently
//! skipped by `execute()` rather than shifting any enabled indices.
//!
//! `ActionContext` provides mutable access to the acting agent, the deferred
//! move and death queues, the signal layer, and the RNG. Actions must not
//! modify the grid or population directly — moves and deaths are queued and
//! applied at end-of-step in `drain_move_queue` / `drain_death_queue`.

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
    /// Signal layers — atomic cells, increments are thread-safe via `&Signals`.
    pub signals: &'a Signals,
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
    /// Per-action pending-disabled mask, indexed by registration index.
    /// Hot-path lookup for `execute()` — mirrors `disabled` so we don't
    /// hash a string on every action invocation.
    disabled_mask: Vec<bool>,
    /// Dense map: `active_map[enabled_idx] = actual_idx`.
    active_map: Vec<u16>,
}

impl ActionRegistry {
    pub fn new() -> Self {
        Self {
            actions: Vec::new(),
            disabled: HashSet::new(),
            disabled_mask: Vec::new(),
            active_map: Vec::new(),
        }
    }

    pub fn register(&mut self, action: Box<dyn Action>) {
        self.actions.push(action);
        self.rebuild_state();
    }

    // ── Enable / disable ─────────────────────────────────────────────────

    /// Mark an action enabled or disabled by its stable ID.
    /// The change is *pending* until the next `commit_enabled()` call:
    /// `active_map` stays put (genome wiring is stable within a generation)
    /// but `execute()` immediately skips the disabled action via
    /// `disabled_mask`.
    pub fn set_enabled(&mut self, id: &str, enabled: bool) {
        if enabled {
            self.disabled.remove(id);
        } else {
            self.disabled.insert(id.to_string());
        }
        self.rebuild_disabled_mask();
    }

    /// Returns `true` if the action is currently enabled (not pending-disabled).
    pub fn is_enabled(&self, id: &str) -> bool {
        !self.disabled.contains(id)
    }

    /// Commit pending changes. Call **before** `wiring_config()` in
    /// `spawn_new_generation` so new nnets wire against the updated active set.
    pub fn commit_enabled(&mut self) {
        self.rebuild_state();
    }

    fn rebuild_disabled_mask(&mut self) {
        self.disabled_mask = self.actions
            .iter()
            .map(|a| self.disabled.contains(a.id()))
            .collect();
    }

    fn rebuild_state(&mut self) {
        self.rebuild_disabled_mask();
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
        let actual_idx = self.active_map[enabled_idx as usize] as usize;
        if self.disabled_mask[actual_idx] {
            return;
        }
        self.actions[actual_idx].execute(level, ctx);
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
