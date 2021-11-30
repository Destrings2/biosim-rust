use crate::Parameters;
use crate::population::brain::sensor_actions::action_implementation::get_action_dispatch;
use crate::population::brain::sensor_actions::ENABLED_ACTIONS;
use crate::population::individual::Individual;
use crate::simulation::grid::Grid;
use crate::simulation::peeps::Peeps;
use crate::simulation::signals::Signals;

pub struct Simulation<'a> {
    pub peeps: Peeps,
    pub parameters: &'a Parameters,
    pub simulation_step: u32,
}

impl<'a> Simulation<'a>{
    pub fn simulate_individual(&mut self, individual: &mut Individual) {
        individual.age += 1;
        let action_levels = individual.feed_forward(&self);
        for (i, action) in ENABLED_ACTIONS.iter().enumerate() {
            let action_executor = get_action_dispatch(action);
            let level = action_levels[i];
            action_executor(
                individual,
                self,
                level,
            );
        }
    }
}