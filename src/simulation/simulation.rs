use crate::Parameters;
use crate::simulation::peeps::Peeps;

pub struct Simulation<'a> {
    pub peeps: Peeps<'a>,
    pub parameters: &'a Parameters,
    pub simulation_step: u32,
}

impl<'a> Simulation<'a>{
    pub fn initialize(parameters: &'a Parameters) -> Self {
        return Simulation {
            peeps: Peeps::new(parameters),
            parameters,
            simulation_step: 0,
        };
    }

    pub fn run_simulation_step(&mut self) {
        self.peeps.simulate_all(&self.parameters, self.simulation_step);
        self.simulation_step += 1;
    }

    pub fn run_simulation(&mut self, generations: u32, steps: u32) {
        for _ in 0..generations {
            for _ in 0..steps {
                self.run_simulation_step();
            }
            self.peeps.end_generation();
        }
    }
}