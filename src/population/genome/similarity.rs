use std::mem::size_of;
use strsim::generic_jaro_winkler;
use crate::Genome;
use crate::population::genome::gene::Gene;

pub enum SimilarityMetric {
    JaroWinkler,
    HammingGenes,
    HammingBits,
}

fn genome_jaro_winkler(genome1: &Genome, genome2: &Genome) -> f32 {
    generic_jaro_winkler(genome1, genome2) as f32
}

fn genome_hamming_genes(genome1: &Genome, genome2: &Genome) -> f32 {
    if genome1.len() != genome2.len() {
        panic!("Genomes must be of equal length");
    }

    let mut equal_genes = 0;
    for i in 0..genome1.len() {
        equal_genes += if genome1[i] == genome2[i] {1} else {0};
    }

    return equal_genes as f32 / genome1.len() as f32
}

fn genome_hamming_bits(genome1: &Genome, genome2: &Genome) -> f32 {
    if genome1.len() != genome2.len() {
        panic!("Genomes must be of equal length");
    }

    let bytes_per_element= size_of::<Gene>();
    let total_bytes = genome1.len() * bytes_per_element;
    let total_bits = total_bytes * 8;

    let mut bit_difference = 0;
    for i in 0..genome1.len() {
        let encoding_difference = (genome1[i].encoding ^ genome2[i].encoding).count_ones();
        let weight_difference = (genome1[i].weight ^ genome2[i].weight).count_ones();
        bit_difference += encoding_difference + weight_difference;
    }

    return 1.0 - f32::min(1.0, (2.0 * bit_difference as f32) / total_bits as f32)
}


pub fn genome_similarity(genome1: &Genome, genome2: &Genome, metric: SimilarityMetric) -> f32 {
    match metric {
        SimilarityMetric::JaroWinkler => genome_jaro_winkler(genome1, genome2),
        SimilarityMetric::HammingGenes => genome_hamming_genes(genome1, genome2),
        SimilarityMetric::HammingBits => genome_hamming_bits(genome1, genome2),
    }
}