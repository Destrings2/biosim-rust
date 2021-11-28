use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Parameters {
    population: u16,
    steps_per_generation: u16,
    max_generations: u32,
    num_threads: u8,
    signal_layers: u8,
    max_genome_length: u16,
    max_number_neurons: u16,
}

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
}

#[cfg(test)]
mod test {
    #[test]
    fn test_parameter_read() {
        let parameters = super::Parameters::read_from_file("src/simulation/parameters.yaml").unwrap();
        assert_eq!(parameters.population, 100);
        assert_eq!(parameters.steps_per_generation, 100);
        assert_eq!(parameters.max_generations, 100);
        assert_eq!(parameters.num_threads, 4);
        assert_eq!(parameters.signal_layers, 2);
        assert_eq!(parameters.max_genome_length, 100);
        assert_eq!(parameters.max_number_neurons, 100);
    }
}