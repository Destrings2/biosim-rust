use crate::population::brain::NeuralNet;
use crate::population::genome::{Genome, make_random_genome};
use crate::simulation::parameters::Parameters;

mod simulation;
mod population;

fn main() {
    let genome = make_random_genome(16);
    let net = NeuralNet::new(&genome, 2);
    let igraph = net.to_graph_string();
    dbg!("{}", igraph);
}