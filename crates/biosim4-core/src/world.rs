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

use crate::grid::Grid;
use crate::population::Population;
use crate::signals_layer::Signals;

/// Read-only view of the simulation world, passed to sensors and challenge evaluators.
pub struct World<'a> {
    pub grid: &'a Grid,
    pub signals: &'a Signals,
    pub population: &'a Population,
    pub size_x: u16,
    pub size_y: u16,
    pub steps_per_generation: u32,
    pub generation: u32,
    pub step: u32,
}

impl<'a> World<'a> {
    pub fn new(
        grid: &'a Grid,
        signals: &'a Signals,
        population: &'a Population,
        steps_per_generation: u32,
        generation: u32,
        step: u32,
    ) -> Self {
        let size_x = grid.size_x;
        let size_y = grid.size_y;
        Self { grid, signals, population, size_x, size_y, steps_per_generation, generation, step }
    }
}
