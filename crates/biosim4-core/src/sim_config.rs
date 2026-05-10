use serde::{Deserialize, Serialize};

/// Full simulation configuration. Serializable so the WASM frontend can set/get it as JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimConfig {
    // World dimensions — immutable after sim start
    pub size_x: u16,
    pub size_y: u16,
    pub population: u32,
    pub num_threads: u32,
    pub deterministic: bool,
    pub rng_seed: u64,
    pub signal_layers: u8,

    // Evolution
    pub steps_per_generation: u32,
    pub max_generations: u32,
    pub genome_initial_length_min: u16,
    pub genome_initial_length_max: u16,
    pub genome_max_length: u16,
    pub max_number_neurons: u16,
    pub point_mutation_rate: f32,
    pub gene_insertion_deletion_rate: f32,
    pub deletion_ratio: f32,
    pub sexual_reproduction: bool,
    pub choose_parents_by_fitness: bool,
    pub kill_enable: bool,

    // Agent defaults
    pub responsiveness: f32,
    pub responsiveness_curve_k_factor: f32,
    pub population_sensor_radius: f32,
    pub signal_sensor_radius: f32,
    pub long_probe_distance: u32,
    pub short_probe_barrier_distance: u32,

    // Environment
    pub barrier_type: u8,

    // Analysis / output
    pub genome_analysis_stride: u32,
    pub display_sample_genomes: u32,
    pub genome_comparison_method: u8,  // 0=jaro-winkler, 1=hamming-bits, 2=hamming-bytes
    pub save_video: bool,
    pub video_stride: u32,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            size_x: 128,
            size_y: 128,
            population: 3000,
            num_threads: 4,
            deterministic: false,
            rng_seed: 12345,
            signal_layers: 1,
            steps_per_generation: 300,
            max_generations: 200,
            genome_initial_length_min: 24,
            genome_initial_length_max: 24,
            genome_max_length: 300,
            max_number_neurons: 5,
            point_mutation_rate: 0.001,
            gene_insertion_deletion_rate: 0.0,
            deletion_ratio: 0.5,
            sexual_reproduction: false,
            choose_parents_by_fitness: true,
            kill_enable: false,
            responsiveness: 0.5,
            responsiveness_curve_k_factor: 2.0,
            population_sensor_radius: 2.5,
            signal_sensor_radius: 2.0,
            long_probe_distance: 16,
            short_probe_barrier_distance: 4,
            barrier_type: 0,
            genome_analysis_stride: 25,
            display_sample_genomes: 5,
            genome_comparison_method: 0,
            save_video: true,
            video_stride: 25,
        }
    }
}

impl SimConfig {
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Apply a partial JSON patch (only provided keys are updated).
    pub fn patch_json(&mut self, patch: &str) -> Result<(), serde_json::Error> {
        let v: serde_json::Value = serde_json::from_str(patch)?;
        let mut current = serde_json::to_value(&*self)?;
        if let (serde_json::Value::Object(cur), serde_json::Value::Object(p)) =
            (&mut current, v)
        {
            for (k, val) in p { cur.insert(k, val); }
        }
        *self = serde_json::from_value(current)?;
        Ok(())
    }
}
