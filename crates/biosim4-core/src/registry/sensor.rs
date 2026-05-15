//! Sensor trait, context, and registry.
//!
//! # Enabled/disabled lifecycle
//!
//! Every sensor has a stable string ID and a registration index (its position
//! in the `sensors` vec). Genome genes reference sensors by *enabled index* —
//! an index into the `active_map`, not the registration index.
//!
//! `active_map[enabled_idx] = actual_idx` is a dense vector rebuilt by
//! `commit_enabled()`. It contains only the registration indices of currently
//! enabled sensors. `enabled_count()` returns `active_map.len()` and is the
//! `sensor_count` value used in `WiringConfig` when compiling new neural nets.
//!
//! **`set_enabled(id, false)`** marks a sensor pending-disabled. The
//! `active_map` is NOT rebuilt yet; existing neural nets keep their wiring
//! indices stable for the rest of the current generation. The disabled sensor
//! immediately returns `0.0` from `evaluate()`, making its gene connections
//! inert without shifting any indices.
//!
//! **`commit_enabled()`** is called at generation boundaries (inside
//! `spawn_new_generation`) before `wiring_config()`. It rebuilds `active_map`,
//! so new neural nets are compiled against the updated enabled set.
//!
//! The invariant: within any generation, a given `enabled_idx` always maps to
//! the same sensor. Across generation boundaries, the mapping may change.

use crate::agent::Agent;
use crate::rng::Rng;
use crate::world::World;
use std::collections::HashSet;

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

/// Ordered registry — position in the vec is the *registration* index, but
/// only **enabled** sensors participate in genome wiring.
///
/// ## Enabled / disabled lifecycle
///
/// - `set_enabled(id, false)` marks a sensor as *pending-disabled*.
/// - `commit_enabled()` — called at every generation boundary (inside
///   `spawn_new_generation`) — rebuilds the `active_map`, which is the dense
///   vector of `(enabled_index → actual_index)` that genome genes resolve
///   against.  From that point, new neural nets are wired against
///   `enabled_count()` rather than the full registration count.
/// - **During the current generation** disabled sensors return `0.0`
///   immediately (their gene connections become dead weight), but the wiring
///   indices of existing nets are not disturbed.
pub struct SensorRegistry {
    sensors: Vec<Box<dyn Sensor>>,
    /// Stable IDs of sensors that are disabled (pending or committed).
    /// Source of truth for ID-keyed lookups (`is_enabled`, `register`).
    disabled: HashSet<String>,
    /// Per-sensor pending-disabled mask, indexed by registration index
    /// (`actual_idx`). Mirrors `disabled` and is the hot-path lookup used
    /// by `evaluate()` so we don't hash a string on every sensor read.
    /// Rebuilt by `set_enabled` and `register`, both rare events.
    disabled_mask: Vec<bool>,
    /// Dense map: `active_map[enabled_idx] = actual_idx`.
    /// Rebuilt by `commit_enabled()` and on each `register()`.
    active_map: Vec<u16>,
}

impl SensorRegistry {
    pub fn new() -> Self {
        Self {
            sensors: Vec::new(),
            disabled: HashSet::new(),
            disabled_mask: Vec::new(),
            active_map: Vec::new(),
        }
    }

    pub fn register(&mut self, sensor: Box<dyn Sensor>) {
        self.sensors.push(sensor);
        self.rebuild_state();
    }

    // ── Enable / disable ─────────────────────────────────────────────────

    /// Mark a sensor enabled or disabled by its stable ID.
    /// The change is *pending* until the next `commit_enabled()` call:
    /// `active_map` (and therefore `enabled_count()`) stays put, so genome
    /// wiring is stable within a generation, but `evaluate()` immediately
    /// returns `0.0` for the disabled sensor via `disabled_mask`.
    pub fn set_enabled(&mut self, id: &str, enabled: bool) {
        if enabled {
            self.disabled.remove(id);
        } else {
            self.disabled.insert(id.to_string());
        }
        self.rebuild_disabled_mask();
    }

    /// Returns `true` if the sensor is currently enabled (not pending-disabled).
    pub fn is_enabled(&self, id: &str) -> bool {
        !self.disabled.contains(id)
    }

    /// Commit pending enable/disable changes.  Call this **before**
    /// `wiring_config()` / `create_wiring()` in `spawn_new_generation` so that
    /// new neural nets are wired against the updated active set.
    pub fn commit_enabled(&mut self) {
        self.rebuild_state();
    }

    fn rebuild_disabled_mask(&mut self) {
        self.disabled_mask = self.sensors.iter().map(|s| self.disabled.contains(s.id())).collect();
    }

    fn rebuild_state(&mut self) {
        self.rebuild_disabled_mask();
        self.active_map = self
            .sensors
            .iter()
            .enumerate()
            .filter(|(_, s)| !self.disabled.contains(s.id()))
            .map(|(i, _)| i as u16)
            .collect();
    }

    // ── Counts ────────────────────────────────────────────────────────────

    /// Total number of registered sensors (independent of enable/disable state).
    pub fn count(&self) -> u16 {
        self.sensors.len() as u16
    }

    /// Number of *active* (committed-enabled) sensors.
    /// Use this as `sensor_count` in `WiringConfig` so new nnets only have
    /// genes that reference actually-enabled sensors.
    pub fn enabled_count(&self) -> u16 {
        self.active_map.len() as u16
    }

    // ── Evaluation ────────────────────────────────────────────────────────

    /// Evaluate sensor at `enabled_idx` — an index into the *active* (dense)
    /// set, as produced by `create_wiring` using `enabled_count`.
    ///
    /// If the sensor was disabled mid-generation (pending) the call returns
    /// `0.0` immediately rather than querying the sensor implementation.
    pub fn evaluate(&self, enabled_idx: u16, ctx: &mut SensorContext) -> f32 {
        let actual_idx = self.active_map[enabled_idx as usize] as usize;
        if self.disabled_mask[actual_idx] {
            return 0.0;
        }
        self.sensors[actual_idx].evaluate(ctx).clamp(0.0, 1.0)
    }

    // ── Introspection ─────────────────────────────────────────────────────

    pub fn name(&self, idx: u16) -> &str {
        self.sensors[idx as usize].name()
    }

    pub fn id(&self, idx: u16) -> &str {
        self.sensors[idx as usize].id()
    }

    /// Iterate all registered sensors with `(registration_index, sensor, is_enabled)`.
    pub fn iter(&self) -> impl Iterator<Item = (u16, &dyn Sensor, bool)> {
        self.sensors.iter().enumerate().map(|(i, s)| {
            let enabled = !self.disabled.contains(s.id());
            (i as u16, s.as_ref(), enabled)
        })
    }
}

impl Default for SensorRegistry {
    fn default() -> Self {
        Self::new()
    }
}
