use crate::population::gene::Gene;

// An individual's genome is a set of Genes, see [`Gene`]. Each
// gene is equivalent to one connection in a neural net. An individual's
// neural net is derived from its set of genes
pub type Genome = Vec<Gene>;

// Returns by value a single genome with random genes.
pub fn make_random_genome(num_genes: usize) -> Genome {
    let mut genome = Vec::with_capacity(num_genes);
    for _ in 0..num_genes {
        genome.push(Gene::make_random_gene());
    }
    return genome;
}