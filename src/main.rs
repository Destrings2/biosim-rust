#![allow(dead_code)]
use crate::population::brain::NeuralNet;
use crate::population::genome::{Genome, genome_to_hex, make_random_genome};
use crate::population::genome::similarity::genome_similarity;
use crate::population::genome::similarity::SimilarityMetric;
use crate::simulation::parameters::Parameters;

mod simulation;
mod population;

fn main() {
    let genome = make_random_genome(16);
    let net = NeuralNet::new(&genome, 4);
    let igraph = net.to_mathematica_string();
    println!("{}", igraph);
}