//! Evolutionary simulation engine for biosim4-rs.
//!
//! Provides the state machine, agent representation, genome encoding,
//! pluggable sensor/action/challenge system, and stepping logic. Built-in
//! sensors, actions, and challenges live in sibling crates
//! (`biosim4-sensors`, `biosim4-actions`, `biosim4-challenges`) so adding
//! one doesn't trigger a core rebuild.
//!
//! # Main loop
//!
//! ```rust,no_run
//! use biosim4_core::{SimConfig, SimulationState, step_generation, spawn_new_generation};
//!
//! let config = SimConfig::default();
//! let mut state = SimulationState::new(config);
//! // biosim4_sensors::register_builtin_sensors(&mut state.sensors);
//! // biosim4_actions::register_builtin_actions(&mut state.actions);
//! // biosim4_challenges::register_builtin_challenges(&mut state.challenges);
//!
//! loop {
//!     step_generation(&mut state);
//!     let survivors = spawn_new_generation(&mut state);
//!     if state.generation >= state.config.max_generations {
//!         break;
//!     }
//! }
//! ```
//!
//! # Extension points
//!
//! - **Custom sensor**: implement [`registry::sensor::Sensor`], then call
//!   `state.sensors.register(Box::new(my_sensor))`.
//! - **Custom action**: implement [`registry::action::Action`], then call
//!   `state.actions.register(Box::new(my_action))`.
//! - **Custom challenge**: implement [`registry::challenge::Challenge`], then call
//!   `state.challenges.register(Box::new(my_challenge))`.
//!
//! # Feature flags
//!
//! - `parallel` — enables rayon-based multi-threaded stepping when
//!   `config.num_threads > 1`. Omit for reproducible single-thread runs.

pub mod agent;
pub mod analysis;
pub mod barriers;
pub mod constants;
pub mod food_layer;
pub mod genome;
pub mod grid;
pub mod population;
pub mod registry;
pub mod rng;
pub mod signals_layer;
pub mod sim_config;
pub mod sim_state;
pub mod sim_step;
pub mod spawn;
pub mod types;
pub mod world;

pub use analysis::{collect_epoch_stats, print_epoch_stats, EpochStats};
pub use sim_config::SimConfig;
pub use sim_state::SimulationState;
pub use sim_step::{step_generation, step_one};
pub use spawn::{initialize_generation_0, spawn_new_generation};
pub use types::{Coord, Dir};
