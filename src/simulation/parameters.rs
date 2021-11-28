mod parameter_defaults;

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use serde::{Serialize, Deserialize};

//<editor-fold desc="Parameter struct">
// To add a parameter, add it to the `Parameters` struct below.
// Then, add a function returning its default value to the `parameter_defaults` module.
// Finally, use the serde default attribute to point to the function.
#[derive(Serialize, Deserialize, Debug)]
pub struct Parameters {
    #[serde(default = "parameter_defaults::size_y")]
    pub size_y: u16,

    #[serde(default = "parameter_defaults::population")]
    pub population: u16,

    #[serde(default = "parameter_defaults::steps_per_generation")]
    pub steps_per_generation: u16,

    #[serde(default = "parameter_defaults::max_generations")]
    pub max_generations: u32,

    #[serde(default = "parameter_defaults::num_threads")]
    pub num_threads: u8,

    #[serde(default = "parameter_defaults::signal_layers")]
    pub signal_layers: u8,

    #[serde(default = "parameter_defaults::max_genome_length")]
    pub max_genome_length: u16,

    #[serde(default = "parameter_defaults::max_number_neurons")]
    pub max_number_neurons: u16,

    #[serde(default = "parameter_defaults::point_mutation_rate")]
    pub point_mutation_rate: f64,

    #[serde(default = "parameter_defaults::gene_insertion_deletion_rate")]
    pub gene_insertion_deletion_rate: f64,

    #[serde(default = "parameter_defaults::delete_ration")]
    pub delete_ration: f64,

    #[serde(default = "parameter_defaults::sexual_reproduction")]
    pub sexual_reproduction: bool,

    #[serde(default = "parameter_defaults::kill_enabled")]
    pub kill_enabled: bool,

    #[serde(default = "parameter_defaults::choose_parents_by_fitness")]
    pub choose_parents_by_fitness: bool,

    #[serde(default = "parameter_defaults::population_sensor_radius")]
    pub population_sensor_radius: f32,

    #[serde(default = "parameter_defaults::signal_sensor_radius")]
    pub signal_sensor_radius: u16,

    #[serde(default = "parameter_defaults::responsiveness")]
    pub responsiveness: f32,

    #[serde(default = "parameter_defaults::responsiveness_curve_k_factor")]
    pub responsiveness_curve_k_factor: u16,

    #[serde(default = "parameter_defaults::long_probe_distance")]
    pub long_probe_distance: u32,

    #[serde(default = "parameter_defaults::valence_saturation_magnitude")]
    pub valence_saturation_magnitude: f32,
}
//</editor-fold>

impl Parameters {
    pub fn read_from_reader(reader: &mut BufReader<File>) -> Result<Parameters, Box<dyn Error>> {
        let parameters: Parameters = serde_yaml::from_reader(reader)?;
        Ok(parameters)
    }

    pub fn read_from_file(file_name: &str) -> Result<Parameters, Box<dyn Error>> {
        let file = File::open(file_name)?;
        let mut reader = BufReader::new(file);
        return Parameters::read_from_reader(&mut reader);
    }

    pub fn defaults() -> Parameters {
        let params : Parameters = serde_yaml::from_str("default: true").unwrap();
        return params
    }
}

//<editor-fold desc="Unit tests">
#[cfg(test)]
mod test {
    use crate::simulation::parameters::Parameters;
    use super::parameter_defaults::kill_enabled;
    use super::parameter_defaults::size_y;

    #[test]
    fn test_parameter_read() {
        let parameters = super::Parameters::read_from_file("src/simulation/parameters.yaml").unwrap();
        assert_eq!(parameters.population, 128);
        assert_eq!(parameters.steps_per_generation, 100);
        assert_eq!(parameters.max_generations, 100);
        assert_eq!(parameters.num_threads, 4);
        assert_eq!(parameters.signal_layers, 2);
        assert_eq!(parameters.max_genome_length, 100);
        assert_eq!(parameters.max_number_neurons, 100);
    }

    #[test]
    fn test_defaults() {
        let params : Parameters = serde_yaml::from_str("default: true").unwrap();
        assert_eq!(params.size_y, size_y());
        assert_eq!(params.kill_enabled, kill_enabled());
        assert_eq!(params.population, 100);

    }
}
//</editor-fold>