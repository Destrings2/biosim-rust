//! Breed — a named preset that toggles a curated subset of sensors and
//! actions and (optionally) installs a challenge configuration.
//!
//! Built-in registries (`biosim4-sensors`, `biosim4-actions`,
//! `biosim4-challenges`) register *everything*; the user then chooses which
//! to leave enabled. Breeds are the shortcut: instead of clicking 40 sensor
//! checkboxes, "Apply Forager" enables the food + energy + movement subset.
//!
//! # Lifecycle
//!
//! 1. Caller registers built-in breeds (or custom ones) once at startup via
//!    [`BreedRegistry::register`].
//! 2. User picks a breed in the UI and triggers [`Breed::apply`] (typically
//!    via a `SimCommand::ApplyBreed(id)` in the bevy front-end).
//! 3. The breed:
//!    - disables every sensor / action in the registry,
//!    - re-enables only the ones it names,
//!    - calls `commit_enabled()` on both — wiring uses the new set on the
//!      next generation boundary,
//!    - (if `challenge` is `Some`) forwards the embedded
//!      [`ChallengeConfig`] to the challenge registry.
//!
//! Breeds DO NOT auto-restart the run. The new wiring kicks in at the next
//! generation boundary, just like a manual sensor toggle.

use super::{ActionRegistry, ChallengeConfig, ChallengeRegistry, SensorRegistry};
use serde::{Deserialize, Serialize};

/// A named bundle of sensor / action / challenge selections.
///
/// The string lists are checked against the relevant registry at `apply` time;
/// an unknown id returns `Err(...)` and leaves the registries untouched.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Breed {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Sensor ids to enable. Everything not listed is disabled.
    pub sensors: Vec<String>,
    /// Action ids to enable. Everything not listed is disabled.
    pub actions: Vec<String>,
    /// Optional challenge configuration applied alongside the registry set.
    /// `None` leaves the active challenge set untouched.
    #[serde(default)]
    pub challenge: Option<ChallengeConfig>,
}

impl Breed {
    /// Convenience constructor for static-string breeds defined in code.
    pub fn from_static(
        id: &str,
        name: &str,
        description: &str,
        sensors: &[&str],
        actions: &[&str],
        challenge: Option<ChallengeConfig>,
    ) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            description: description.to_string(),
            sensors: sensors.iter().map(|s| s.to_string()).collect(),
            actions: actions.iter().map(|a| a.to_string()).collect(),
            challenge,
        }
    }

    /// Apply this breed: rewrite the sensor + action enable sets (pending
    /// state only — `active_map` is left untouched). Returns an error and
    /// leaves both registries unchanged if any id is unknown.
    ///
    /// # Why we don't `commit_enabled` here
    ///
    /// Committing mid-generation rebuilds `active_map`. The currently-alive
    /// agents are wired against the OLD active set: their gene `source_num`
    /// values are `% old_enabled_count`. If we shrink/reorder `active_map`,
    /// those source_nums silently re-index into different sensors than the
    /// genome intended — and may hit feature-gated entries the underlying
    /// simulation can't service (e.g. a `signal2` sensor when `signal_layers
    /// = 1`, causing an out-of-bounds in `Signals::get`).
    ///
    /// Instead we follow the same contract as a manual sensor toggle:
    /// enable/disable is *pending* until the next generation boundary, where
    /// `spawn_new_generation` calls `apply_feature_enables` (gates against
    /// `signal_layers`/`enable_energy`) before `commit_enabled` and the new
    /// wiring. Mid-generation evaluations short-circuit via `disabled_mask`
    /// and return `0.0`.
    pub fn apply(
        &self,
        sensors: &mut SensorRegistry,
        actions: &mut ActionRegistry,
        challenges: &mut ChallengeRegistry,
    ) -> Result<(), String> {
        // Validate up-front so we can fail without partially-applying.
        for id in &self.sensors {
            if !sensors.iter().any(|(_, s, _)| s.id() == id) {
                return Err(format!("Breed `{}` references unknown sensor `{}`", self.id, id));
            }
        }
        for id in &self.actions {
            if !actions.iter().any(|(_, a, _)| a.id() == id) {
                return Err(format!("Breed `{}` references unknown action `{}`", self.id, id));
            }
        }

        // Sensors: snapshot the full id list so we don't hold a borrow while
        // mutating the registry. `set_enabled` is pending until next gen.
        let all_sensor_ids: Vec<String> =
            sensors.iter().map(|(_, s, _)| s.id().to_string()).collect();
        for id in &all_sensor_ids {
            sensors.set_enabled(id, self.sensors.iter().any(|s| s == id));
        }

        let all_action_ids: Vec<String> =
            actions.iter().map(|(_, a, _)| a.id().to_string()).collect();
        for id in &all_action_ids {
            actions.set_enabled(id, self.actions.iter().any(|a| a == id));
        }

        if let Some(cfg) = &self.challenge {
            challenges.apply_config(cfg.clone())?;
        }
        Ok(())
    }
}

/// Holds the catalogue of named breeds available to the UI / API.
///
/// This is intentionally separate from [`SensorRegistry`] /
/// [`ActionRegistry`] / [`ChallengeRegistry`]: a breed doesn't *own* any
/// runtime objects — it's a recipe that mutates the others. So callers can
/// freely add or remove breeds without disturbing the simulation state.
#[derive(Default)]
pub struct BreedRegistry {
    breeds: Vec<Breed>,
}

impl BreedRegistry {
    pub fn new() -> Self {
        Self { breeds: Vec::new() }
    }

    pub fn register(&mut self, breed: Breed) {
        // De-duplicate by id: upsert semantics, so callers can override a
        // built-in breed with a custom one without first calling `remove`.
        if let Some(pos) = self.breeds.iter().position(|b| b.id == breed.id) {
            self.breeds[pos] = breed;
        } else {
            self.breeds.push(breed);
        }
    }

    pub fn remove(&mut self, id: &str) -> bool {
        if let Some(pos) = self.breeds.iter().position(|b| b.id == id) {
            self.breeds.remove(pos);
            true
        } else {
            false
        }
    }

    pub fn get(&self, id: &str) -> Option<&Breed> {
        self.breeds.iter().find(|b| b.id == id)
    }

    pub fn list(&self) -> &[Breed] {
        &self.breeds
    }

    pub fn count(&self) -> usize {
        self.breeds.len()
    }

    /// Apply the breed with `id` against the given registries. See
    /// [`Breed::apply`] for the lifecycle.
    pub fn apply(
        &self,
        id: &str,
        sensors: &mut SensorRegistry,
        actions: &mut ActionRegistry,
        challenges: &mut ChallengeRegistry,
    ) -> Result<(), String> {
        let breed = self.get(id).ok_or_else(|| format!("Unknown breed `{id}`"))?;
        breed.apply(sensors, actions, challenges)
    }

    /// Compact JSON summary suitable for the UI / API. Returns an array of
    /// `{ id, name, description, sensor_count, action_count, has_challenge }`.
    pub fn schema_list(&self) -> serde_json::Value {
        let arr: Vec<serde_json::Value> = self
            .breeds
            .iter()
            .map(|b| {
                serde_json::json!({
                    "id": b.id,
                    "name": b.name,
                    "description": b.description,
                    "sensor_count": b.sensors.len(),
                    "action_count": b.actions.len(),
                    "has_challenge": b.challenge.is_some(),
                })
            })
            .collect();
        serde_json::Value::Array(arr)
    }
}
