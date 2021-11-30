#![allow(dead_code)]
use crate::population::genome::similarity::SimilarityMetric;
use crate::simulation::parameters::Parameters;
use crate::simulation::simulation::Simulation;

mod simulation;
mod population;

fn main() {
    let params = Parameters::defaults();
    let mut sim = Simulation::initialize(&params);
    sim.run_simulation_step()
}