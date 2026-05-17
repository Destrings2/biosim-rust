//! Global simulation configuration.
//!
//! [`SimConfig`] is a flat, fully JSON-serializable struct. The WASM frontend
//! serializes it for handoff; `from_json` / `patch_json` handle deserialization.
//!
//! # Field groups
//!
//! - **World**: `size_x`, `size_y`, `population`, `num_threads`,
//!   `rng_seed` (0 = from entropy), `signal_layers`.
//! - **Evolution**: `steps_per_generation`, `max_generations`,
//!   `genome_initial_length_{min,max}`, `genome_max_length`, `max_number_neurons`,
//!   `point_mutation_rate`, `gene_insertion_deletion_rate`, `deletion_ratio`,
//!   `sexual_reproduction`, `tournament_size`, `elitism_count`,
//!   `adaptive_mutation`, `mutation_rate_jitter`, `kill_enable`,
//!   `bloat_penalty_weight`.
//! - **Agent defaults**: `responsiveness`, `responsiveness_curve_k_factor`,
//!   `population_sensor_radius`, `signal_sensor_radius`,
//!   `long_probe_distance`, `short_probe_barrier_distance`.
//! - **Environment**: `barrier_type` (0 = none, 1–7 = preset layouts).
//! - **Analysis/output**: `genome_analysis_stride`, `display_sample_genomes`,
//!   `genome_comparison_method` (0 = Jaro-Winkler, 1 = Hamming bits,
//!   2 = Hamming bytes), `save_video`, `video_stride`.
//!
//! `patch_json` applies a partial JSON object — only the keys present in the
//! patch are updated. This is how the frontend applies per-session overrides.

use serde::{Deserialize, Serialize};

use crate::topology::Topology;

/// Where to place each new agent at the generation boundary.
///
/// The grid is wiped between generations, so the default behaviour
/// (`Random`) re-randomises every spatial relationship the population
/// built up. The other modes inherit a position from the parents so
/// lineages remain geographically coherent — this turns the grid into a
/// niching substrate (cellular / spatially structured GA) and pairs
/// with the genome-based speciation pipeline to produce parapatric
/// species (genetic + geographic isolation).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OffspringPlacementMode {
    /// Uniform random empty cell across the whole grid. Historical
    /// default; preserves the gen-to-gen RNG trace byte-for-byte.
    #[default]
    Random,
    /// Place the child within `offspring_placement_radius` of parent A's
    /// last position (parent A is already the structural primary in
    /// `uniform_crossover`). Asexual / elite / fallback paths use the
    /// sole parent's position. Falls back to a global random pick when
    /// the local disk is full.
    NearPrimaryParent,
    /// Place the child within `offspring_placement_radius` of the
    /// (wrap-aware) midpoint of parents A and B. Asexual / interspecies /
    /// elite / extinction-fallback paths degrade to `NearPrimaryParent`.
    MidpointOfParents,
}

/// Full simulation configuration. JSON-serializable so frontends can read and write it.
///
/// World dimensions (`size_x`, `size_y`, `population`) are effectively
/// immutable after `SimulationState::new` — changing them mid-run without
/// reinitializing produces undefined grid state.
///
/// Use [`SimConfig::from_json`] to deserialize a full config, or
/// [`SimConfig::patch_json`] to apply partial overrides on top of an existing
/// instance.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimConfig {
    // ── World ──────────────────────────────────────────────────────────────
    /// Width of the simulation grid in cells.
    pub size_x: u16,
    /// Height of the simulation grid in cells.
    pub size_y: u16,
    /// Number of agents in each generation.
    pub population: u32,
    /// Number of worker threads for the parallel step pipeline.
    /// Set to 1 (or disable the `parallel` feature) for deterministic output.
    pub num_threads: u32,
    /// RNG seed. Use 0 to draw from system entropy (non-deterministic).
    pub rng_seed: u64,
    /// Number of independent pheromone signal layers (1–3).
    pub signal_layers: u8,

    // ── Evolution ──────────────────────────────────────────────────────────
    /// Number of simulation steps per generation.
    pub steps_per_generation: u32,
    /// Total number of generations to run. The native runner stops at this value.
    pub max_generations: u32,
    /// Minimum genome length (in genes) for generation 0 agents.
    pub genome_initial_length_min: u16,
    /// Maximum genome length (in genes) for generation 0 agents.
    pub genome_initial_length_max: u16,
    /// Hard upper limit on genome length after mutations.
    pub genome_max_length: u16,
    /// Maximum number of internal neurons per neural network.
    pub max_number_neurons: u16,
    /// Per-gene bit-flip probability each generation. Default `0.05`
    /// yields ~1 flip per child on the 24-gene starter genome — the
    /// lower bound of useful search pressure.
    pub point_mutation_rate: f32,
    /// Per-genome insertion-or-deletion probability each generation.
    /// Default `0.01` lets genome length grow new structure without
    /// destabilising the size distribution. `0.0` pins length to the
    /// random initial value forever.
    pub gene_insertion_deletion_rate: f32,
    /// Fraction of indel events that delete (vs. insert) a gene.
    pub deletion_ratio: f32,
    /// Use two-parent uniform crossover instead of asexual cloning.
    /// Default `true`: uniform crossover preserves individual genes
    /// with probability ½, so sexual reproduction dominates cloning.
    pub sexual_reproduction: bool,
    /// Tournament size `k` for parent selection. Each child draws `k`
    /// uniform candidates from the parent pool and reproduces from the
    /// fittest. `k = 1` means no fitness pressure; `k = 3` is the
    /// default; `k = 5` is the practical ceiling for 1500-pop runs
    /// before diversity collapses.
    pub tournament_size: u32,
    /// Top-fitness parents copied unchanged into the next generation.
    /// Default `2`. Raise to protect more genomes from mutation noise
    /// when many agents pass; lowers variation.
    pub elitism_count: u32,
    /// Let each lineage evolve its own per-individual mutation rate
    /// (Evolution-Strategies self-adaptation). Off by default — the
    /// fixed-rate path is the convergence-tuning baseline.
    pub adaptive_mutation: bool,
    /// Jitter scale `τ` for adaptive-mutation rate inheritance.
    /// Default `0.2` keeps the per-generation change inside ±10 %.
    /// Consulted only when [`adaptive_mutation`] is `true`.
    ///
    /// [`adaptive_mutation`]: Self::adaptive_mutation
    pub mutation_rate_jitter: f32,
    /// Allow the `kill_forward` action to kill nearby agents.
    pub kill_enable: bool,
    /// Parsimony pressure on dead-end gene count. Each agent's raw
    /// fitness is reduced by `weight × dead_norm²` (where
    /// `dead_norm = dead_genes / max(genome_len, 1)`) before parent
    /// ranking, so genomes carrying connections that `create_wiring`
    /// culled (and that therefore contribute no behaviour) sort below
    /// clean genomes of equal effective fitness.
    ///
    /// The **quadratic** curve makes moderate bloat (`dead_norm ≤ 0.3`,
    /// the normal operating range for healthy lineages) nearly free,
    /// while extreme bloat (`dead_norm ≥ 0.8`) still pays close to the
    /// full weight. This is the second iteration of the penalty: a
    /// linear curve was too aggressive on exploring lineages because
    /// new wirings frequently come with newly-dead old chains.
    ///
    /// Default `0.0` (off). Under the quadratic curve, suggested tuning
    /// range is `0.05 – 0.15`; values above `0.3` start to over-prune
    /// at high bloat. The penalty only re-orders the parent pool —
    /// challenge `pass`/fail is unaffected.
    #[serde(default)]
    pub bloat_penalty_weight: f32,

    // ── Offspring placement ────────────────────────────────────────────────
    /// Where to place each new agent on the grid at generation boundary.
    /// Default `Random` (current behaviour: uniform empty cell). Other
    /// modes inherit a position from the parents to create *spatial
    /// niching* (cellular GA / parapatric speciation): lineages that
    /// thrive in a region stay in that region, complementing the
    /// (genetic) speciation pipeline.
    #[serde(default)]
    pub offspring_placement_mode: OffspringPlacementMode,
    /// Chebyshev (L∞) radius around the inherited seed within which new
    /// agents may be placed. Ignored when `offspring_placement_mode` is
    /// `Random`. `0` collapses to "same cell" → almost always falls
    /// back to global random (parent's cell is the only candidate and
    /// is empty post-reset, but the next sibling can't reuse it).
    /// Useful range `3..16`; defaults to `6` which gives a 13×13 disk.
    #[serde(default = "default_offspring_placement_radius")]
    pub offspring_placement_radius: u32,

    // ── Speciation ─────────────────────────────────────────────────────────
    /// Bucket population into species by genome distance and reproduce
    /// within species. Default `false`.
    #[serde(default)]
    pub enable_speciation: bool,
    /// Compatibility distance threshold (1 − similarity). Lower = more
    /// species; higher = fewer. Adaptive τ adjusts at runtime to keep the
    /// species count near `species_count_target`.
    #[serde(default = "default_compatibility_threshold")]
    pub compatibility_threshold: f32,
    /// Target number of species. Adaptive τ adjusts toward this.
    #[serde(default = "default_species_count_target")]
    pub species_count_target: u32,
    /// Acceptable band for target species count.
    #[serde(default = "default_species_count_target_tolerance")]
    pub species_count_target_tolerance: u32,
    /// τ adjustment step per generation when out of band.
    #[serde(default = "default_compatibility_threshold_step")]
    pub compatibility_threshold_step: f32,
    /// Generations a species can go without improving before it is denied
    /// offspring. The two top species are immune.
    #[serde(default = "default_stagnation_limit")]
    pub stagnation_limit: u32,
    /// Minimum members for a species to copy its top genome unchanged.
    /// Default `2`: a species with two or more members preserves its
    /// best. Solo species (1 member) skip elitism to keep their slot
    /// available for fresh exploration.
    #[serde(default = "default_species_elitism_min")]
    pub species_elitism_min: u32,
    /// Probability of drawing the second parent from a different species
    /// during sexual reproduction.
    #[serde(default = "default_interspecies_mating_rate")]
    pub interspecies_mating_rate: f32,
    /// Similarity metric used when placing parents into species. Independent
    /// of [`genome_comparison_method`]: that knob drives the analysis
    /// dashboard and `genetic_diversity`, while this one drives the
    /// speciation pipeline only.
    ///
    /// Values:
    /// - `0` = Jaro-Winkler on raw genome bytes
    /// - `1` = Hamming bits on raw genome bytes
    /// - `2` = Hamming bytes on raw genome bytes
    /// - `3` = Network topology (Jaccard on culled-NN edge set with coarse
    ///   weight bucketing). **Default.** Buckets agents by behaviourally
    ///   meaningful niche — same wiring + similar weight magnitudes cluster
    ///   together. Bit-based methods cluster by gene byte-packing instead,
    ///   which is rarely what's wanted for niching.
    ///
    /// [`genome_comparison_method`]: Self::genome_comparison_method
    #[serde(default = "default_speciation_similarity_method")]
    pub speciation_similarity_method: u8,

    // ── Energy system ──────────────────────────────────────────────────────
    /// Enable the energy and food subsystems.
    pub enable_energy: bool,
    /// Energy deducted from each agent per step.
    pub energy_per_step_cost: f32,
    /// Food added per non-barrier cell per step when the energy system is on.
    pub food_regen_rate: f32,
    /// Fraction of non-barrier cells initialized with food at generation start.
    pub food_initial_density: f32,

    // ── Agent defaults ─────────────────────────────────────────────────────
    /// Initial `responsiveness` value for new agents.
    pub responsiveness: f32,
    /// Steepness of the sigmoid applied to raw neural output. Higher values
    /// produce a sharper threshold response.
    pub responsiveness_curve_k_factor: f32,
    /// Radius (in cells) for the population-density sensors.
    pub population_sensor_radius: f32,
    /// Radius (in cells) for the pheromone signal sensors.
    pub signal_sensor_radius: f32,
    /// Initial `long_probe_dist` for new agents (cells).
    pub long_probe_distance: u32,
    /// Distance for short-range barrier probes (cells).
    pub short_probe_barrier_distance: u32,

    // ── Environment ────────────────────────────────────────────────────────
    /// Procedural barrier layout: 0 = none, 1–7 = preset patterns.
    /// User-painted overrides in `SimulationState::user_barriers` layer on top.
    pub barrier_type: u8,
    /// World topology — controls whether the edges wrap. See
    /// [`Topology`] for the variants. Defaults to `Plane` (bounded
    /// rectangle, historical behaviour). Changing this is structural:
    /// `SimulationState::new` consumes it at grid-construction time.
    #[serde(default)]
    pub topology: Topology,

    // ── Analysis / output ──────────────────────────────────────────────────
    /// Collect genome analysis statistics every N generations.
    pub genome_analysis_stride: u32,
    /// Number of sample genomes to print each analysis stride.
    pub display_sample_genomes: u32,
    /// Genome similarity algorithm: 0 = Jaro-Winkler, 1 = Hamming bits, 2 = Hamming bytes.
    pub genome_comparison_method: u8,
    /// Save a rendered video of each generation (native runner only).
    pub save_video: bool,
    /// Write one video frame every N simulation steps.
    pub video_stride: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            size_x: 128,
            size_y: 128,
            population: 3000,
            num_threads: 4,
            rng_seed: 12345,
            signal_layers: 1,
            steps_per_generation: 300,
            max_generations: 200,
            genome_initial_length_min: 24,
            genome_initial_length_max: 24,
            genome_max_length: 300,
            max_number_neurons: 5,
            point_mutation_rate: 0.05,
            gene_insertion_deletion_rate: 0.01,
            deletion_ratio: 0.5,
            sexual_reproduction: true,
            tournament_size: 3,
            elitism_count: 2,
            adaptive_mutation: false,
            mutation_rate_jitter: 0.2,
            kill_enable: false,
            bloat_penalty_weight: 0.0,
            offspring_placement_mode: OffspringPlacementMode::Random,
            offspring_placement_radius: default_offspring_placement_radius(),
            enable_speciation: false,
            compatibility_threshold: default_compatibility_threshold(),
            species_count_target: default_species_count_target(),
            species_count_target_tolerance: default_species_count_target_tolerance(),
            compatibility_threshold_step: default_compatibility_threshold_step(),
            stagnation_limit: default_stagnation_limit(),
            species_elitism_min: default_species_elitism_min(),
            interspecies_mating_rate: default_interspecies_mating_rate(),
            speciation_similarity_method: default_speciation_similarity_method(),
            enable_energy: false,
            energy_per_step_cost: 0.003,
            food_regen_rate: 0.0005,
            food_initial_density: 0.3,
            responsiveness: 0.5,
            responsiveness_curve_k_factor: 2.0,
            population_sensor_radius: 2.5,
            signal_sensor_radius: 2.0,
            long_probe_distance: 16,
            short_probe_barrier_distance: 4,
            barrier_type: 0,
            topology: Topology::Plane,
            genome_analysis_stride: 25,
            display_sample_genomes: 5,
            genome_comparison_method: 0,
            save_video: true,
            video_stride: 25,
        }
    }
}

impl SimConfig {
    /// Deserialize a complete `SimConfig` from a JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Apply a partial JSON patch (only provided keys are updated).
    pub fn patch_json(&mut self, patch: &str) -> Result<(), serde_json::Error> {
        let v: serde_json::Value = serde_json::from_str(patch)?;
        let mut current = serde_json::to_value(&*self)?;
        if let (serde_json::Value::Object(cur), serde_json::Value::Object(p)) = (&mut current, v) {
            for (k, val) in p {
                cur.insert(k, val);
            }
        }
        *self = serde_json::from_value(current)?;
        Ok(())
    }
}

// ── Serde Defaults ───────────────────────────────────────────────────────
fn default_compatibility_threshold() -> f32 {
    0.30
}
fn default_species_count_target() -> u32 {
    15
}
fn default_species_count_target_tolerance() -> u32 {
    5
}
fn default_compatibility_threshold_step() -> f32 {
    0.02
}
fn default_stagnation_limit() -> u32 {
    15
}
fn default_species_elitism_min() -> u32 {
    2
}
fn default_interspecies_mating_rate() -> f32 {
    0.001
}
fn default_speciation_similarity_method() -> u8 {
    3
}
fn default_offspring_placement_radius() -> u32 {
    6
}
