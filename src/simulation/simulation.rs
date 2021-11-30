use std::cell::{RefCell, RefMut};
use std::rc::Rc;
use crate::Parameters;
use crate::population::brain::sensor_actions::action_implementation::get_action_dispatch;
use crate::population::brain::sensor_actions::ENABLED_ACTIONS;
use crate::population::individual::Individual;
use crate::simulation::peeps::Peeps;

pub struct Simulation<'a> {
    pub peeps: Peeps<'a>,
    pub parameters: &'a Parameters,
    pub simulation_step: u32,
}

impl<'a> Simulation<'a>{
    pub fn initialize(parameters: &'a Parameters) -> Self {
        let peeps = Peeps::new(parameters);
        Simulation {
            peeps,
            parameters,
            simulation_step: 0,
        }
    }

    pub fn run_simulation_step(&mut self) {
        self.peeps.simulate_all(self.parameters, self.simulation_step);
        self.simulation_step += 1;
    }

    pub fn run_simulation(&mut self, steps: u32) {
        for _ in 0..steps {
            self.run_simulation_step();
        }
    }
}