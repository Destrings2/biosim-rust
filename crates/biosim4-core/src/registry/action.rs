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

use crate::agent::{Agent, AgentId};
use crate::rng::Rng;
use crate::signals_layer::Signals;
use crate::types::Coord;
use crate::world::World;
use std::collections::HashSet;

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
    /// Per-agent responsiveness scale in `[0.0, 1.0]`, precomputed once per
    /// step as `response_curve(agent.responsiveness, k)`. Motor actions
    /// multiply their squashed level by this before drawing a probabilistic
    /// bool; internal-state actions (`set_*`, memory writes) ignore it.
    pub responsiveness_adjusted: f32,
    /// Combined X-axis movement urge (east positive). Movement actions add
    /// their signed contribution here instead of queueing a move directly;
    /// [`resolve_movement`] runs once after all actions and converts the
    /// pair `(move_x_urge, move_y_urge)` into a single grid step.
    pub move_x_urge: f32,
    /// Combined Y-axis movement urge (north positive).
    pub move_y_urge: f32,
}

/// A pluggable action that translates neural output into world state changes.
///
/// Implement this trait to create a custom action. Register the action with
/// `state.actions.register(Box::new(my_action))` before the first generation.
///
/// # Implementing
///
/// - `id` must return a unique, stable ASCII string used for enable/disable
///   lookup and JSON persistence.
/// - `execute` receives the raw accumulated neural output for this action
///   slot. Apply a threshold or probability transform (see `prob2bool` and
///   `response_curve` in the `actions` module) before acting. Actions must
///   not modify the grid or population directly — push to `ctx.move_queue` or
///   `ctx.death_queue` instead.
pub trait Action: Send + Sync {
    /// Stable machine identifier. Must be unique across all registered actions.
    fn id(&self) -> &str;
    /// Human-readable display name.
    fn name(&self) -> &str;
    /// Execute the action at the given neural activation `level`.
    ///
    /// `level` is the **raw** (unbounded) accumulated neural output —
    /// motor actions squash it via `tanh`, multiply by
    /// `ctx.responsiveness_adjusted`, then convert to a probability or
    /// direction. Internal-state actions (modulators, memory writes) read
    /// `level` directly and ignore responsiveness.
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

    /// Add an action to the registry and enable it immediately.
    ///
    /// Call this before `initialize_generation_0` or at a generation boundary
    /// followed by `commit_enabled()` so the new action participates in wiring.
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
        self.disabled_mask = self.actions.iter().map(|a| self.disabled.contains(a.id())).collect();
    }

    fn rebuild_state(&mut self) {
        self.rebuild_disabled_mask();
        self.active_map = self
            .actions
            .iter()
            .enumerate()
            .filter(|(_, a)| !self.disabled.contains(a.id()))
            .map(|(i, _)| i as u16)
            .collect();
    }

    // ── Counts ────────────────────────────────────────────────────────────

    /// Total number of registered actions.
    pub fn count(&self) -> u16 {
        self.actions.len() as u16
    }

    /// Number of *active* (committed-enabled) actions.
    pub fn enabled_count(&self) -> u16 {
        self.active_map.len() as u16
    }

    // ── Execution ─────────────────────────────────────────────────────────

    /// Execute action at `enabled_idx` — an index into the *active* (dense)
    /// set. Silently skips pending-disabled actions mid-generation.
    ///
    /// `level` is forwarded verbatim. Motor actions are responsible for
    /// squashing it (`tanh`) and multiplying by `ctx.responsiveness_adjusted`;
    /// internal-state actions ignore responsiveness entirely.
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

    /// Look up the *enabled* (dense `active_map`) index for an action by its
    /// stable string id. Returns `None` if the action isn't registered or is
    /// currently disabled. Used by `sim_step` to dispatch state-update
    /// actions out of the normal iteration order (e.g. `set_responsiveness`
    /// must run before any motor action so all gated actions in the same
    /// step see the freshly-updated responsiveness).
    pub fn enabled_index(&self, id: &str) -> Option<u16> {
        self.active_map
            .iter()
            .enumerate()
            .find(|(_, &reg_idx)| self.actions[reg_idx as usize].id() == id)
            .map(|(enabled_idx, _)| enabled_idx as u16)
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
    fn default() -> Self {
        Self::new()
    }
}

/// Non-linear responsiveness curve: scales motor-action levels by a factor in
/// `[0, 1]` so that low-responsiveness agents react sluggishly and
/// high-responsiveness agents react sharply.
///
/// Computes `f(r) = (2 − r)^(−2k) − 2^(−2k) · (1 − r)` (positive base form,
/// so `powf` stays inside the IEEE-754 real domain — a negative base with a
/// fractional exponent returns NaN). Endpoints: `f(0) = 0`, `f(1) = 1`.
/// With the default `k = 2.0`, `f(0.5) ≈ 0.166`.
#[inline]
pub fn response_curve(r: f32, k: f32) -> f32 {
    let r = r.clamp(0.0, 1.0);
    let two_k = 2.0 * k;
    ((2.0 - r).powf(-two_k) - 2.0_f32.powf(-two_k) * (1.0 - r)).clamp(0.0, 1.0)
}

/// Combined-urge movement resolution. Reads `move_x_urge` / `move_y_urge`
/// accumulated by each movement action's `execute`, squashes them with
/// `tanh`, scales by `responsiveness_adjusted`, and draws one bool per axis
/// to produce a single grid step `(dx, dy) ∈ {−1, 0, 1}²`. If the resulting
/// target is in-bounds and empty, a single `(agent_id, target)` entry is
/// pushed onto `move_queue`.
///
/// Resets the urge accumulators to `0.0` so the same context can be reused
/// for another resolution cycle (helpful in tests).
pub fn resolve_movement(ctx: &mut ActionContext) {
    let move_x = ctx.move_x_urge.tanh() * ctx.responsiveness_adjusted;
    let move_y = ctx.move_y_urge.tanh() * ctx.responsiveness_adjusted;
    ctx.move_x_urge = 0.0;
    ctx.move_y_urge = 0.0;

    let fire_x = ctx.rng.gen_bool(move_x.abs());
    let fire_y = ctx.rng.gen_bool(move_y.abs());
    let dx: i16 = if fire_x {
        if move_x < 0.0 {
            -1
        } else {
            1
        }
    } else {
        0
    };
    let dy: i16 = if fire_y {
        if move_y < 0.0 {
            -1
        } else {
            1
        }
    } else {
        0
    };
    if dx == 0 && dy == 0 {
        return;
    }
    // Route the candidate through `grid.wrap`: on bounded axes a step off
    // the edge returns `None` (move dropped, just like the old
    // `is_in_bounds` check); on a wrapping axis it returns the canonical
    // in-bounds coord so the queued move lands on the wrapped cell
    // rather than being filtered out.
    let raw = Coord::new(ctx.agent.loc.x + dx, ctx.agent.loc.y + dy);
    let Some(new_loc) = ctx.world.grid.wrap(raw) else {
        return;
    };
    // Queue moves into empty cells AND into kill-barrier cells. The latter
    // are not "free to step onto" — `Population::apply_moves` recognises
    // the kill-barrier case and converts the move into the agent's death
    // (clears the source cell, marks `alive = false`, prunes `alive_ids`).
    // If we filtered on `is_empty_at` alone, agents stepping into hazards
    // would have their moves silently dropped instead of dying, which
    // makes the kill-barrier tool look broken.
    if ctx.world.grid.is_empty_at(new_loc) || ctx.world.grid.is_kill_barrier_at(new_loc) {
        ctx.move_queue.push((ctx.agent.id, new_loc));
    }
}
