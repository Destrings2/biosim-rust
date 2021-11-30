use std::mem::size_of;
use strsim::generic_jaro_winkler;
use crate::Genome;
use crate::population::genome::gene::Gene;

pub enum SimilarityMetric {
    JaroWinkler,
    Hamming,
}

pub fn genome_jaro_winkler(genome1: &Genome, genome2: &Genome) -> f64 {
    generic_jaro_winkler(genome1, genome2)
}

pub fn genome_hamming_genes(genome1: &Genome, genome2: &Genome) -> f32 {
    if genome1.len() != genome2.len() {
        panic!("Genomes must be of equal length");
    }

    let bytes_per_element= size_of::<Gene>();

    let mut gene_difference = 0;
    for i in 0..genome1.len() {
        gene_difference += if genome1[i] != genome2[i] {1} else {0};
    }

    return (gene_difference as f32) /  (genome1.len() as f32 * bytes_per_element as f32);
}

pub fn genome_hamming_bits(genome1: &Genome, genome2: &Genome) -> f32 {
    if genome1.len() != genome2.len() {
        panic!("Genomes must be of equal length");
    }

    let bytes_per_element= size_of::<Gene>();

    let mut bit_difference = 0;
    for i in 0..genome1.len() {
        bit_difference += (genome1[i].encoding ^ genome2[i].encoding).count_ones();
        bit_difference += (genome1[i].weight ^ genome2[i].weight).count_ones();
    }

    return 1.0 - f32::min(1.0, (2.0 * bit_difference as f32) / (genome1.len() as f32 * bytes_per_element as f32 * 8.0));
}