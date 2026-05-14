//! Master simulation state container.
//!
//! [`SimulationState`] owns every piece of mutable simulation state. Its
//! public fields are intentionally public so that `sim_step` can perform
//! split borrows (e.g., `&mut population` alongside `&grid`) without needing
//! wrapper methods for every combination.
//!
//! # Key methods
//!
//! - `new(config)` — initializes all registries with built-in sensors,
//!   actions, and challenges, then calls `initialize_generation_0`.
//! - `world()` — constructs a [`World`] read-only snapshot on demand.
//!   `World` does not own any data; it borrows from `self`.
//! - `wiring_config()` — derives `{sensor_count, action_count, max_neurons}`
//!   from the committed active registry sets. Call this after
//!   `commit_enabled()` so the counts reflect any pending changes.
//! - `reapply_user_barriers()` — stamps `user_barriers` overrides on top of
//!   whatever `create_barrier` produced. Must be called after every
//!   `grid.zero_fill() + create_barrier(...)` sequence to preserve
//!   manually-painted walls across generation resets.
//!
//! # `StepScratch`
//!
//! Holds three reusable buffers that carry no semantic state between steps.
//! They are public so `sim_step` can split-borrow them alongside `population`
//! and other fields. See the `StepScratch` doc comment for details.

use crate::barriers::create_barrier;
use crate::grid::Grid;
use crate::population::Population;
use crate::registry::action::ActionRegistry;
use crate::registry::breed::BreedRegistry;
use crate::registry::challenge::{ChallengeConfig, ChallengeRegistry};
use crate::registry::sensor::SensorRegistry;
use crate::rng::Rng;
use crate::sim_config::SimConfig;
use crate::food_layer::FoodLayer;
use crate::signals_layer::Signals;
use crate::world::World;
use crate::genome::neural_net::WiringConfig;
use crate::agent::AgentId;
use crate::types::Coord;
use std::collections::HashMap;

/// Reusable scratch buffers — allocated once, cleared and reused each step.
/// Avoids ~600K Vec allocations per generation at typical sizes (1000 agents
/// × 200 steps × 3 buffers).
#[derive(Default)]
pub struct StepScratch {
    /// Snapshot of `population.alive_ids` taken at the start of `step_all_agents`,
    /// so we can iterate while mutating the population (drains, kills, etc.).
    pub alive_ids: Vec<AgentId>,
    /// Per-agent action accumulator scratch reused by `feed_forward`.
    /// Sized to `alive_ids.len()` at the start of each step. Phase 1 writes
    /// to slot `i`; Phase 2 reads from slot `i`. This enables Phase 1 to run
    /// in parallel — each thread touches a unique slot.
    pub per_agent_action_levels: Vec<Vec<f32>>,
    /// Per-agent neuron accumulator scratch reused by `feed_forward`.
    /// Same indexing as `per_agent_action_levels`.
    pub per_agent_neuron_accum: Vec<Vec<f32>>,
}

pub struct SimulationState {
    pub config: SimConfig,
    pub grid: Grid,
    pub signals: Signals,
    pub food: FoodLayer,
    pub population: Population,
    pub generation: u32,
    pub sim_step: u32,
    pub sensors: SensorRegistry,
    pub actions: ActionRegistry,
    pub challenges: ChallengeRegistry,
    pub breeds: BreedRegistry,
    pub rng: Rng,
    /// Scratch buffers reused each step. Not part of the simulation state proper
    /// — they hold no semantic information between steps. Public so `sim_step`
    /// can split-borrow them alongside `population` etc.
    pub scratch: StepScratch,
    /// User-applied overrides on top of the procedural `barrier_type` layout.
    /// `true` = force barrier, `false` = force cleared. Re-applied at the end
    /// of `initialize_generation_0` and `spawn_new_generation` so manually
    /// painted walls survive across generations.
    pub user_barriers: HashMap<(i16, i16), bool>,
}

impl SimulationState {
    /// Re-apply the user's manual barrier overrides on top of whatever
    /// procedural pattern `create_barrier` produced. Call this after every
    /// `grid.zero_fill() + create_barrier(...)` sequence.
    pub fn reapply_user_barriers(&mut self) {
        let sx = self.config.size_x as i16;
        let sy = self.config.size_y as i16;
        for (&(x, y), &on) in &self.user_barriers {
            if x < 0 || y < 0 || x >= sx || y >= sy { continue; }
            let loc = Coord::new(x, y);
            // Don't stamp over an agent slot (shouldn't happen at gen-boundary
            // since the grid is freshly zero-filled, but defensive).
            let cell = self.grid.at(loc);
            if cell != crate::grid::EMPTY && cell != crate::grid::BARRIER { continue; }
            if on {
                self.grid.set(loc, crate::grid::BARRIER);
            } else {
                self.grid.set(loc, crate::grid::EMPTY);
            }
        }
    }
}

impl SimulationState {
    pub fn new(config: SimConfig) -> Self {
        use crate::sensors::register_builtin_sensors;
        use crate::actions::register_builtin_actions;

        let mut sensors = SensorRegistry::new();
        register_builtin_sensors(&mut sensors);

        let mut actions = ActionRegistry::new();
        register_builtin_actions(&mut actions);

        let mut challenges = ChallengeRegistry::new();
        crate::challenges::register_builtin_challenges(&mut challenges);

        let breeds = BreedRegistry::new();

        let mut grid = Grid::new(config.size_x, config.size_y);
        create_barrier(&mut grid, config.barrier_type);

        let signals = Signals::new(config.signal_layers, config.size_x, config.size_y);
        let food = FoodLayer::new(config.size_x, config.size_y);
        let population = Population::new(config.population);
        let rng = if config.rng_seed == 0 {
            Rng::from_entropy()
        } else {
            Rng::seeded(config.rng_seed)
        };

        let mut state = Self {
            config,
            grid,
            signals,
            food,
            population,
            generation: 0,
            sim_step: 0,
            sensors,
            actions,
            challenges,
            breeds,
            rng,
            scratch: StepScratch::default(),
            user_barriers: HashMap::new(),
        };

        crate::spawn::initialize_generation_0(&mut state);
        state
    }

    pub fn new_from_json(json: &str) -> Result<Self, String> {
        let config = SimConfig::from_json(json).map_err(|e| e.to_string())?;
        Ok(Self::new(config))
    }

    pub fn set_challenge(&mut self, json: &str) -> Result<(), String> {
        let cfg: ChallengeConfig = serde_json::from_str(json).map_err(|e| e.to_string())?;
        self.challenges.apply_config(cfg)
    }

    pub fn world(&self) -> World<'_> {
        World {
            grid: &self.grid,
            signals: &self.signals,
            food: &self.food,
            population: &self.population,
            size_x: self.config.size_x,
            size_y: self.config.size_y,
            steps_per_generation: self.config.steps_per_generation,
            generation: self.generation,
            step: self.sim_step,
        }
    }

    pub fn wiring_config(&self) -> WiringConfig {
        WiringConfig {
            sensor_count: self.sensors.enabled_count(),
            action_count: self.actions.enabled_count(),
            max_neurons: self.config.max_number_neurons,
        }
    }
}
