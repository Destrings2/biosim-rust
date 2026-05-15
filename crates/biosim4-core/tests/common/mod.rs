//! Shared test helpers. Loaded via `mod common;` in each integration test that
//! needs a fully-populated `SimulationState`.

use biosim4_core::sim_config::SimConfig;
use biosim4_core::sim_state::SimulationState;
use biosim4_core::spawn::initialize_generation_0;

/// Build a `SimulationState` with every built-in sensor, action, and challenge
/// registered and generation 0 populated. This is the equivalent of the old
/// auto-registration-and-init flow inside `SimulationState::new`, kept out of
/// the constructor so adding a sensor/action/challenge doesn't trigger a core
/// rebuild — but tests want the ready-to-step bundle.
#[allow(dead_code)] // each #[test] file pulls in `common` but uses different subsets
pub fn new_state(config: SimConfig) -> SimulationState {
    let mut state = SimulationState::new(config);
    biosim4_sensors::register_builtin_sensors(&mut state.sensors);
    biosim4_actions::register_builtin_actions(&mut state.actions);
    biosim4_challenges::register_builtin_challenges(&mut state.challenges);
    initialize_generation_0(&mut state);
    state
}
