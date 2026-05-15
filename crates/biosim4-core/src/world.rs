//! Read-only world view passed to sensors and challenge evaluators.
//!
//! [`World`] holds immutable references into `SimulationState` fields. It
//! exists to give sensors and challenge `evaluate` methods a coherent,
//! borrow-safe view of the simulation without exposing mutation. The lifetime
//! `'a` is borrowed from `SimulationState`, so a `World` cannot outlive the
//! state that produced it.
//!
//! `SimulationState::world()` creates a `World` on demand; it is cheap (just
//! reference copies). Challenge hooks that need mutation receive `WorldMut`
//! (from `crate::registry::challenge`) instead.

use crate::food_layer::FoodLayer;
use crate::grid::Grid;
use crate::population::Population;
use crate::programmable::ProgrammablePool;
use crate::signals_layer::Signals;

/// Read-only view of the simulation world, passed to sensors and challenge evaluators.
pub struct World<'a> {
    pub grid: &'a Grid,
    pub signals: &'a Signals,
    pub food: &'a FoodLayer,
    pub population: &'a Population,
    /// Non-evolved, challenge-owned entities (predators, herders, …). See
    /// [`crate::programmable`]. Empty pool when no challenge spawns any.
    pub programmable: &'a ProgrammablePool,
    pub size_x: u16,
    pub size_y: u16,
    pub steps_per_generation: u32,
    pub generation: u32,
    pub step: u32,
}

impl<'a> World<'a> {
    /// Construct a `World` from borrowed references and scalar metadata.
    ///
    /// Prefer [`SimulationState::world`](crate::sim_state::SimulationState::world)
    /// for the common case — call `new` only when assembling a `World` from
    /// individually borrowed fields.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        grid: &'a Grid,
        signals: &'a Signals,
        food: &'a FoodLayer,
        population: &'a Population,
        programmable: &'a ProgrammablePool,
        steps_per_generation: u32,
        generation: u32,
        step: u32,
    ) -> Self {
        let size_x = grid.size_x;
        let size_y = grid.size_y;
        Self {
            grid,
            signals,
            food,
            population,
            programmable,
            size_x,
            size_y,
            steps_per_generation,
            generation,
            step,
        }
    }
}
