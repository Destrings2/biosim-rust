use rand::Rng;
use crate::Parameters;
use crate::population::genome::{make_random_genome, Genome};
use crate::population::genome::gene::Gene;

pub fn random_bit_flip(genome: &mut Genome) {
    let mut rng = rand::thread_rng();
    let element_index = rng.gen_range(0..genome.len());
    let bit_index = rng.gen_range(0..16u8);
    let bit = genome[element_index].get_bit(bit_index);
    genome[element_index].set_bit(bit_index, !bit);
}

pub fn crop_length(genome: &mut Genome, length: usize) {
    if genome.len() > length && length > 0 {
        let truncate_back: bool = rand::random();
        if truncate_back {
            genome.truncate(length);
        } else {
            genome.drain(length..);
        }
    }
}

pub fn random_insertion_deletion(genome: &mut Genome, p: &Parameters) {
   let mut rng = rand::thread_rng();
    if rng.gen_range(0.0..1.0) < p.gene_insertion_deletion_rate {
        if rng.gen_range(0.0..1.0) < p.delete_ration {
            if genome.len() > 1 {
                let index = rng.gen_range(0..genome.len());
                genome.remove(index);
            }
        } else if genome.len() < p.max_genome_length {
            genome.push(Gene::make_random_gene());
        }
    }
}

pub fn apply_point_mutation_to_genome(genome: &mut Genome, p: &Parameters) {
    let mut rng = rand::thread_rng();
    for i in 0..genome.len() {
        if rng.gen_range(0.0..1.0) < p.point_mutation_rate {
            random_bit_flip(genome);
        }
    }
}