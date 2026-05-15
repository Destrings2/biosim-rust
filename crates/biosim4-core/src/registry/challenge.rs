//! Challenge trait, registry, and composition logic.
//!
//! A challenge evaluates each alive agent once per generation and returns
//! `(pass: bool, fitness: f32)`. The `pass` flag determines whether the agent
//! enters the survivor pool. The `fitness` score (0.0..1.0) is used to bias
//! parent selection and, as a fallback, to rank agents when no one passes.
//!
//! # `ChallengeComposition`
//!
//! Multiple active challenges are combined by the registry before returning:
//!
//! - `Any` — passes if at least one challenge passes; fitness = max score.
//! - `All` — passes only if all challenges pass; fitness = min score.
//! - `WeightedSum { weights, threshold }` — fitness = weighted average of
//!   all scores; passes if that average ≥ threshold.
//!
//! The default composition is `Any`.
//!
//! # `WorldMut`
//!
//! Challenge hooks (`on_sim_step`, `on_generation_start`) receive a
//! [`WorldMut`] rather than the read-only [`World`] so they can mutate
//! agent `challenge_bits`, queue deaths, or write signals. Sensors and the
//! `evaluate` method always receive `&World` (read-only).
//!
//! # `ChallengeConfig` JSON format
//!
//! ```json
//! {
//!   "active": ["circle", "right_half"],
//!   "composition": { "type": "Any" },
//!   "params": {
//!     "circle": { "cx": 0.5, "cy": 0.5, "radius": 0.25, "weighted": false }
//!   }
//! }
//! ```

use crate::agent::Agent;
use crate::grid::Grid;
use crate::population::Population;
use crate::signals_layer::Signals;
use crate::world::World;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ChallengeOverlay {
    #[serde(rename = "circle")]
    Circle { cx: f32, cy: f32, radius: f32, color: [u8; 4] },
    #[serde(rename = "rectangle")]
    Rectangle { x: f32, y: f32, w: f32, h: f32, color: [u8; 4] },
    #[serde(rename = "points")]
    Points { points: Vec<(f32, f32)>, color: [u8; 4], size: f32 },
}

/// Mutable world reference for on_sim_step / on_generation_start side effects.
/// Challenges can iterate alive agents, mutate their `challenge_bits`, and
/// queue deaths via `population.queue_for_death(...)`.
pub struct WorldMut<'a> {
    pub grid: &'a mut Grid,
    pub signals: &'a mut Signals,
    pub population: &'a mut Population,
    pub rng: &'a mut crate::rng::Rng,
    pub step: u32,
    pub generation: u32,
    pub config: &'a crate::sim_config::SimConfig,
}

/// A survival challenge. Evaluated once per agent per generation.
pub trait Challenge: Send + Sync {
    fn id(&self) -> &str;
    fn name(&self) -> &str;
    fn description(&self) -> &str {
        self.name()
    }

    /// JSON Schema (draft-07 object) describing configurable params.
    fn params_schema(&self) -> Value;

    /// Apply a JSON params object. Return Err with a message if invalid.
    fn configure(&mut self, params: Value) -> Result<(), String>;

    /// Evaluate whether this agent passes and return a fitness score 0.0..1.0.
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32);

    /// Called once per simulation step (single-threaded). Default: no-op.
    fn on_sim_step(&mut self, _ctx: &mut WorldMut) {}

    /// Called once at the start of each generation. Default: no-op.
    fn on_generation_start(&mut self, _ctx: &mut WorldMut) {}

    /// Return any visual overlays for this challenge. Default: empty.
    fn overlays(&self, _world: &World) -> Vec<ChallengeOverlay> {
        Vec::new()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum ChallengeComposition {
    #[default]
    Any,
    All,
    WeightedSum {
        weights: Vec<f32>,
        threshold: f32,
    },
}

/// Frontend-facing challenge configuration.
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct ChallengeConfig {
    pub active: Vec<String>,
    pub composition: ChallengeComposition,
    pub params: std::collections::HashMap<String, Value>,
}

/// Registry of all challenges, both built-in and custom.
pub struct ChallengeRegistry {
    challenges: Vec<Box<dyn Challenge>>,
    active: Vec<usize>,
    composition: ChallengeComposition,
}

impl ChallengeRegistry {
    pub fn new() -> Self {
        Self { challenges: Vec::new(), active: Vec::new(), composition: ChallengeComposition::Any }
    }

    pub fn register(&mut self, challenge: Box<dyn Challenge>) {
        self.challenges.push(challenge);
    }

    /// Replace an existing challenge with the same `id` in-place, preserving its
    /// position in the registry (so any `active` indices stay valid). Returns
    /// `true` if a replacement happened, `false` if no challenge with that id
    /// existed (in which case the caller should `register` instead).
    pub fn replace_by_id(&mut self, id: &str, challenge: Box<dyn Challenge>) -> bool {
        if let Some(pos) = self.challenges.iter().position(|c| c.id() == id) {
            self.challenges[pos] = challenge;
            true
        } else {
            false
        }
    }

    /// Insert if absent, replace if present. Convenience for upsert-style
    /// flows (e.g., the WASM `register_js_challenge` endpoint).
    pub fn upsert_by_id(&mut self, id: &str, challenge: Box<dyn Challenge>) {
        if let Some(pos) = self.challenges.iter().position(|c| c.id() == id) {
            self.challenges[pos] = challenge;
        } else {
            self.challenges.push(challenge);
        }
    }

    /// Remove a challenge by id. Also drops it from the `active` list.
    /// Returns true if removed.
    pub fn remove_by_id(&mut self, id: &str) -> bool {
        let Some(pos) = self.challenges.iter().position(|c| c.id() == id) else { return false };
        self.challenges.remove(pos);
        // Rebuild `active` since indices have shifted.
        self.active.retain(|&i| i != pos);
        for i in self.active.iter_mut() {
            if *i > pos {
                *i -= 1;
            }
        }
        true
    }

    /// Apply a ChallengeConfig: set active set, composition, and per-challenge params.
    pub fn apply_config(&mut self, cfg: ChallengeConfig) -> Result<(), String> {
        self.composition = cfg.composition;
        self.active.clear();
        for id in &cfg.active {
            let pos = self
                .challenges
                .iter()
                .position(|c| c.id() == id)
                .ok_or_else(|| format!("Unknown challenge id: {id}"))?;
            if let Some(params) = cfg.params.get(id) {
                self.challenges[pos].configure(params.clone())?;
            }
            self.active.push(pos);
        }
        Ok(())
    }

    /// Set a single challenge active by id, with optional params.
    pub fn set_single(&mut self, id: &str, params: Option<Value>) -> Result<(), String> {
        let pos = self
            .challenges
            .iter()
            .position(|c| c.id() == id)
            .ok_or_else(|| format!("Unknown challenge id: {id}"))?;
        if let Some(p) = params {
            self.challenges[pos].configure(p)?;
        }
        self.active = vec![pos];
        Ok(())
    }

    /// Evaluate all active challenges for an agent and compose the result.
    pub fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        if self.active.is_empty() {
            return (true, 1.0); // no challenge = everyone survives
        }
        let results: Vec<(bool, f32)> =
            self.active.iter().map(|&i| self.challenges[i].evaluate(agent, world)).collect();

        match &self.composition {
            ChallengeComposition::Any => {
                let passed = results.iter().any(|(p, _)| *p);
                let score = results.iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
                (passed, score)
            }
            ChallengeComposition::All => {
                let passed = results.iter().all(|(p, _)| *p);
                let score = results.iter().map(|(_, s)| *s).fold(1.0f32, f32::min);
                (passed, score)
            }
            ChallengeComposition::WeightedSum { weights, threshold } => {
                let score: f32 =
                    results.iter().zip(weights.iter()).map(|((_, s), w)| s * w).sum::<f32>()
                        / weights.iter().sum::<f32>().max(1e-6);
                (score >= *threshold, score)
            }
        }
    }

    pub fn on_sim_step(&mut self, ctx: &mut WorldMut) {
        for &i in &self.active {
            self.challenges[i].on_sim_step(ctx);
        }
    }

    pub fn on_generation_start(&mut self, ctx: &mut WorldMut) {
        for &i in &self.active {
            self.challenges[i].on_generation_start(ctx);
        }
    }

    /// Return overlays from all active challenges.
    pub fn get_overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let mut overlays = Vec::new();
        for &i in &self.active {
            overlays.extend(self.challenges[i].overlays(world));
        }
        overlays
    }

    /// Return JSON schema list for all registered challenges.
    pub fn schema_list(&self) -> Value {
        let list: Vec<Value> = self
            .challenges
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": c.id(),
                    "name": c.name(),
                    "description": c.description(),
                    "schema": c.params_schema(),
                })
            })
            .collect();
        Value::Array(list)
    }
}

impl Default for ChallengeRegistry {
    fn default() -> Self {
        Self::new()
    }
}
