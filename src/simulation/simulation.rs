use std::cell::RefCell;
use crate::Parameters;
use crate::population::brain::sensor_actions::action_implementation::get_action_dispatch;
use crate::population::brain::sensor_actions::ENABLED_ACTIONS;
use crate::population::individual::Individual;
use crate::simulation::peeps::Peeps;

pub struct Simulation<'a> {
    pub peeps: RefCell<Peeps<'a>>,
    pub parameters: &'a Parameters,
    pub simulation_step: u32,
}

impl<'a> Simulation<'a>{
    pub fn initialize(parameters: &'a Parameters) -> Self {
        let peeps = Peeps::new(parameters);
        Simulation {
            peeps: RefCell::new(peeps),
            parameters,
            simulation_step: 0,
        }
    }

    pub fn simulate_individual(&self,  individual_ref: &RefCell<Individual>) {
        individual_ref.borrow_mut().age += 1;
        let action_levels = individual_ref.borrow_mut().feed_forward(&self);
        for (i, action) in ENABLED_ACTIONS.iter().enumerate() {
            let action_executor = get_action_dispatch(action);
            let level = action_levels[i];
            action_executor(
                individual_ref,
                &self.peeps,
                &self.parameters,
                level
            );
        }
    }

    pub fn run_simulation_step(&mut self) {
        let peeps = self.peeps.borrow_mut();
        let alive_individuals = peeps.get_alive_individuals();
        for &individual in alive_individuals.iter() {
            let borrowed_individual = individual;
            self.simulate_individual(borrowed_individual);
        }
        self.simulation_step += 1;
    }

    pub fn run_simulation(&mut self, steps: u32) {
        for _ in 0..steps {
            self.run_simulation_step();
        }
    }
}