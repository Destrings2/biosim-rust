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

use crate::agent::AgentId;
use crate::barriers::create_barrier;
use crate::food_layer::FoodLayer;
use crate::genome::neural_net::WiringConfig;
use crate::grid::Grid;
use crate::population::Population;
use crate::programmable::ProgrammablePool;
use crate::registry::action::ActionRegistry;
use crate::registry::challenge::{ChallengeConfig, ChallengeRegistry};
use crate::registry::sensor::SensorRegistry;
use crate::rng::Rng;
use crate::signals_layer::Signals;
use crate::sim_config::SimConfig;
use crate::types::Coord;
use crate::world::World;
use std::collections::HashMap;

/// Reusable scratch buffers — allocated once, cleared and reused each step.
#[derive(Default)]
pub struct StepScratch {
    /// Snapshot of `population.alive_ids` taken at the start of
    /// `step_all_agents`, so we can iterate while mutating the population
    /// (drains, kills, etc.).
    pub alive_ids: Vec<AgentId>,
}

/// Root container for all mutable simulation state.
///
/// All fields are `pub` so the step engine can perform split borrows — for
/// example, holding `&mut population` alongside `&grid` — without intermediate
/// accessor methods for every combination.
///
/// # Initialization
///
/// Use [`SimulationState::new`] to construct a state with built-in sensors
/// and actions registered. Register built-in challenges separately via
/// `biosim4_challenges::register_builtin_challenges(&mut state.challenges)`.
///
/// # Stepping
///
/// Call [`step_generation`](crate::step_generation) (one full generation) or
/// [`step_one`](crate::step_one) (one step, for incremental frontends).
/// After each generation call [`spawn_new_generation`](crate::spawn_new_generation)
/// to select survivors and populate the next generation.
pub struct SimulationState {
    pub config: SimConfig,
    pub grid: Grid,
    pub signals: Signals,
    pub food: FoodLayer,
    pub population: Population,
    /// Non-evolved, challenge-owned entities that live in the world.
    /// See [`crate::programmable`].
    pub programmable: ProgrammablePool,
    pub generation: u32,
    pub sim_step: u32,
    pub sensors: SensorRegistry,
    pub actions: ActionRegistry,
    pub challenges: ChallengeRegistry,
    /// Curated presets that bundle sensor/action subsets with an optional
    /// challenge configuration. See [`crate::registry::Breed`].
    pub breeds: crate::registry::BreedRegistry,
    pub rng: Rng,
    /// Scratch buffers reused each step. Not part of the simulation state proper
    /// — they hold no semantic information between steps. Public so `sim_step`
    /// can split-borrow them alongside `population` etc.
    pub scratch: StepScratch,
    /// User-applied overrides on top of the procedural `barrier_type` layout.
    /// See [`BarrierTile`] for the variants. Re-applied at the end of
    /// `initialize_generation_0` and `spawn_new_generation` so manually
    /// painted cells survive across generations.
    pub user_barriers: HashMap<(i16, i16), BarrierTile>,
}

/// User-painted cell override.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BarrierTile {
    /// Force this cell empty (clears a procedural barrier).
    Clear,
    /// Force this cell to be a static wall.
    Wall,
    /// Force this cell to be a kill barrier — agents moving into it die.
    Kill,
}

impl SimulationState {
    /// Re-apply the user's manual barrier overrides on top of whatever
    /// procedural pattern `create_barrier` produced. Call this after every
    /// `grid.zero_fill() + create_barrier(...)` sequence.
    ///
    /// Each painted Wall/Kill cell is also recorded in `grid.barrier_centers`
    /// so the `near_barrier` challenge (and anything else iterating centers)
    /// reacts to user-drawn barriers, not just the procedural cluster
    /// centroids that `create_barrier` populates.
    pub fn reapply_user_barriers(&mut self) {
        let sx = self.config.size_x as i16;
        let sy = self.config.size_y as i16;
        for (&(x, y), &tile) in &self.user_barriers {
            if x < 0 || y < 0 || x >= sx || y >= sy {
                continue;
            }
            let loc = Coord::new(x, y);
            // Don't stamp over an agent slot (shouldn't happen at gen-boundary
            // since the grid is freshly zero-filled, but defensive).
            let cell = self.grid.at(loc);
            if cell != crate::grid::EMPTY
                && cell != crate::grid::BARRIER
                && cell != crate::grid::KILL_BARRIER
            {
                continue;
            }
            self.grid.set(
                loc,
                match tile {
                    BarrierTile::Clear => crate::grid::EMPTY,
                    BarrierTile::Wall => crate::grid::BARRIER,
                    BarrierTile::Kill => crate::grid::KILL_BARRIER,
                },
            );
            if matches!(tile, BarrierTile::Wall | BarrierTile::Kill)
                && !self.grid.barrier_centers.contains(&loc)
            {
                self.grid.barrier_centers.push(loc);
            }
        }
    }
}

impl SimulationState {
    /// Create a `SimulationState` from a config, register all built-in sensors
    /// and actions.
    ///
    /// Built-in sensors, actions, and challenges live in sibling crates and
    /// are **not** registered here — callers register them after construction
    /// and then call `initialize_generation_0` to spawn the starting cohort:
    ///
    /// ```ignore
    /// let mut state = SimulationState::new(cfg);
    /// biosim4_sensors::register_builtin_sensors(&mut state.sensors);
    /// biosim4_actions::register_builtin_actions(&mut state.actions);
    /// biosim4_challenges::register_builtin_challenges(&mut state.challenges);
    /// biosim4_core::initialize_generation_0(&mut state);
    /// ```
    pub fn new(config: SimConfig) -> Self {
        let sensors = SensorRegistry::new();
        let actions = ActionRegistry::new();
        let challenges = ChallengeRegistry::new();
        let breeds = crate::registry::BreedRegistry::new();

        let mut grid = Grid::with_topology(config.size_x, config.size_y, config.topology);
        create_barrier(&mut grid, config.barrier_type);

        let signals = Signals::new(config.signal_layers, config.size_x, config.size_y);
        let food = FoodLayer::new(config.size_x, config.size_y);
        let population = Population::new(config.population);
        let rng =
            if config.rng_seed == 0 { Rng::from_entropy() } else { Rng::seeded(config.rng_seed) };

        Self {
            config,
            grid,
            signals,
            food,
            population,
            programmable: ProgrammablePool::new(),
            generation: 0,
            sim_step: 0,
            sensors,
            actions,
            challenges,
            breeds,
            rng,
            scratch: StepScratch::default(),
            user_barriers: HashMap::new(),
        }
    }

    /// Deserialize a [`SimConfig`] from JSON and call [`Self::new`].
    pub fn new_from_json(json: &str) -> Result<Self, String> {
        let config = SimConfig::from_json(json).map_err(|e| e.to_string())?;
        Ok(Self::new(config))
    }

    /// Apply a JSON-encoded [`ChallengeConfig`] to the challenge registry.
    ///
    /// Activates the named challenges, sets their composition rule, and
    /// forwards any per-challenge params to each challenge's `configure` method.
    pub fn set_challenge(&mut self, json: &str) -> Result<(), String> {
        let cfg: ChallengeConfig = serde_json::from_str(json).map_err(|e| e.to_string())?;
        self.challenges.apply_config(cfg)
    }

    /// Apply a breed by id. Performs the split-borrow against the three
    /// registries internally so callers don't have to.
    pub fn apply_breed(&mut self, id: &str) -> Result<(), String> {
        // Look up first; copy out the breed (cheap — small struct of strings)
        // so we can drop the immutable borrow before mutating siblings.
        let breed = self.breeds.get(id).ok_or_else(|| format!("Unknown breed `{id}`"))?.clone();
        breed.apply(&mut self.sensors, &mut self.actions, &mut self.challenges)
    }

    /// Construct a read-only [`World`] view of the current simulation state.
    ///
    /// The `World` borrows from `self` and is cheap to create — it copies
    /// only references and a few scalar fields.
    pub fn world(&self) -> World<'_> {
        World {
            grid: &self.grid,
            signals: &self.signals,
            food: &self.food,
            population: &self.population,
            programmable: &self.programmable,
            size_x: self.config.size_x,
            size_y: self.config.size_y,
            steps_per_generation: self.config.steps_per_generation,
            generation: self.generation,
            step: self.sim_step,
        }
    }

    /// Build a [`WiringConfig`] from the current committed sensor/action counts.
    ///
    /// Call this after `sensors.commit_enabled()` and `actions.commit_enabled()`
    /// so that new neural networks are compiled against the active set.
    pub fn wiring_config(&self) -> WiringConfig {
        WiringConfig {
            sensor_count: self.sensors.enabled_count(),
            action_count: self.actions.enabled_count(),
            max_neurons: self.config.max_number_neurons,
        }
    }
}
