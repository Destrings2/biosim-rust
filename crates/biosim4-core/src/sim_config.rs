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
//!   `adaptive_mutation`, `mutation_rate_jitter`, `kill_enable`.
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
    #[serde(default = "default_species_elitism_min")]
    pub species_elitism_min: u32,
    /// Probability of drawing the second parent from a different species
    /// during sexual reproduction.
    #[serde(default = "default_interspecies_mating_rate")]
    pub interspecies_mating_rate: f32,

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
            enable_speciation: false,
            compatibility_threshold: default_compatibility_threshold(),
            species_count_target: default_species_count_target(),
            species_count_target_tolerance: default_species_count_target_tolerance(),
            compatibility_threshold_step: default_compatibility_threshold_step(),
            stagnation_limit: default_stagnation_limit(),
            species_elitism_min: default_species_elitism_min(),
            interspecies_mating_rate: default_interspecies_mating_rate(),
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
fn default_compatibility_threshold() -> f32 { 0.30 }
fn default_species_count_target() -> u32 { 15 }
fn default_species_count_target_tolerance() -> u32 { 5 }
fn default_compatibility_threshold_step() -> f32 { 0.02 }
fn default_stagnation_limit() -> u32 { 15 }
fn default_species_elitism_min() -> u32 { 5 }
fn default_interspecies_mating_rate() -> f32 { 0.001 }
