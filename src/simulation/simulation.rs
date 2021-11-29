use crate::Parameters;
use crate::simulation::grid::Grid;
use crate::simulation::peeps::Peeps;
use crate::simulation::signal::Signal;

pub struct Simulation<'a> {
    pub grid: Grid,
    pub peeps: Peeps,
    pub signals: Signal,
    pub parameters: &'a Parameters,
    pub simulation_step: u32,
}

// impl<'a> Simulation<'a>{
//     pub fn new(parameters: &'a Parameters) -> Simulation {
//
//     }
// }