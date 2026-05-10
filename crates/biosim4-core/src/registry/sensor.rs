use std::collections::HashSet;
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
    disabled: HashSet<String>,
    /// Dense map: `active_map[enabled_idx] = actual_idx`.
    /// Rebuilt by `commit_enabled()` and on each `register()`.
    active_map: Vec<u16>,
}

impl SensorRegistry {
    pub fn new() -> Self {
        Self { sensors: Vec::new(), disabled: HashSet::new(), active_map: Vec::new() }
    }

    pub fn register(&mut self, sensor: Box<dyn Sensor>) {
        self.sensors.push(sensor);
        self.rebuild_active_map();
    }

    // ── Enable / disable ─────────────────────────────────────────────────

    /// Mark a sensor enabled or disabled by its stable ID.
    /// The change is *pending* until the next `commit_enabled()` call.
    /// Mid-generation the disabled sensor immediately returns `0.0`.
    pub fn set_enabled(&mut self, id: &str, enabled: bool) {
        if enabled {
            self.disabled.remove(id);
        } else {
            self.disabled.insert(id.to_string());
        }
        // Do NOT rebuild active_map here: genome wiring must stay stable
        // within a generation. commit_enabled() applies the change at the
        // next generation boundary.
    }

    /// Returns `true` if the sensor is currently enabled (not pending-disabled).
    pub fn is_enabled(&self, id: &str) -> bool {
        !self.disabled.contains(id)
    }

    /// Commit pending enable/disable changes.  Call this **before**
    /// `wiring_config()` / `create_wiring()` in `spawn_new_generation` so that
    /// new neural nets are wired against the updated active set.
    pub fn commit_enabled(&mut self) {
        self.rebuild_active_map();
    }

    fn rebuild_active_map(&mut self) {
        self.active_map = self.sensors
            .iter()
            .enumerate()
            .filter(|(_, s)| !self.disabled.contains(s.id()))
            .map(|(i, _)| i as u16)
            .collect();
    }

    // ── Counts ────────────────────────────────────────────────────────────

    /// Total number of registered sensors (independent of enable/disable state).
    pub fn count(&self) -> u16 { self.sensors.len() as u16 }

    /// Number of *active* (committed-enabled) sensors.
    /// Use this as `sensor_count` in `WiringConfig` so new nnets only have
    /// genes that reference actually-enabled sensors.
    pub fn enabled_count(&self) -> u16 { self.active_map.len() as u16 }

    // ── Evaluation ────────────────────────────────────────────────────────

    /// Evaluate sensor at `enabled_idx` — an index into the *active* (dense)
    /// set, as produced by `create_wiring` using `enabled_count`.
    ///
    /// If the sensor was disabled mid-generation (pending) the call returns
    /// `0.0` immediately rather than querying the sensor implementation.
    pub fn evaluate(&self, enabled_idx: u16, ctx: &mut SensorContext) -> f32 {
        let actual_idx = self.active_map[enabled_idx as usize];
        let s = &self.sensors[actual_idx as usize];
        // Honour pending mid-generation disables immediately.
        if self.disabled.contains(s.id()) {
            return 0.0;
        }
        s.evaluate(ctx).clamp(0.0, 1.0)
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
    fn default() -> Self { Self::new() }
}
