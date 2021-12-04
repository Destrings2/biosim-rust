pub mod survival_criteria;

use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;
use crate::Parameters;
use crate::population::genome::{Genome, make_random_genome};
use crate::population::genome::mutations::breed_from_parents;
use crate::population::individual::Individual;
use crate::simulation::grid::EMPTY_CELL;
use crate::simulation::peeps::survival_criteria::{Challenges, get_challenge_function};
use crate::simulation::probability_to_bool;
use crate::simulation::signals::Signals;
use crate::simulation::types::Coord;
use crate::simulation::world::World;

pub type MoveQueue = HashMap<u16, Vec<(f32, f32)>>;
pub type DeathQueue = Vec<u16>;

pub struct Peeps<'a> {
    pub world: World,
    pub signals: Signals,
    pub population: Vec<Individual>,
    pub death_queue: DeathQueue,
    // An individual can have multiple urges to move in a given direction. We need to keep track of them
    // and process them to get the overall direction of the movement urge.
    pub move_queue: MoveQueue,
    pub parameters: &'a Parameters
}

impl<'a> Peeps<'a> {
    pub fn new(p: &'a Parameters) -> Peeps<'a> {
        let mut population: Vec<Individual> = Vec::with_capacity(p.population as usize);
        population.push(Individual::new(0, Coord(-1, -1), make_random_genome(1), &p));

        let signals = Signals::new(1, p.size_x, p.size_y);
        let move_queue = HashMap::new();
        let death_queue = Vec::new();
        let mut world = World::new(p.size_x, p.size_y);
        let mut rng = rand::thread_rng();

        for i in 1..=p.population {
            let empty_coord = world.find_random_empty_location();
            let genome_size = rng.gen_range(1..=p.max_genome_length);
            let individual = Individual::new(i, empty_coord, make_random_genome(genome_size), &p);
            world.set_at_coord(empty_coord, individual.index);
            population.insert(i as usize, individual);
        }
        return Peeps {
            world,
            signals,
            population,
            move_queue,
            death_queue,
            parameters: p
        };
    }

    pub fn queue_for_death(death_queue: &mut DeathQueue, id: u16) {
        death_queue.push(id);
    }

    pub fn drain_death_queue(&mut self) {
        for id in self.death_queue.drain(..) {
            let individual: &mut Individual = self.population.get_mut(id as usize).unwrap();
            individual.alive = false;
        }
    }

    pub fn queue_for_move(move_queue: &mut MoveQueue, peep_index: u16, move_data: (f32, f32)) {
        move_queue.entry(peep_index).or_insert(Vec::new()).push(move_data);
    }

    pub fn drain_move_queue(&mut self) {
        for (id, urges) in self.move_queue.drain() {
            let individual: &mut Individual = self.population.get_mut(id as usize).unwrap();
            // sum the urges
            let mut sum_urges = (0.0, 0.0);
            for urge in urges {
                sum_urges.0 += urge.0;
                sum_urges.1 += urge.1;
            }

            // Normalize the urges
            sum_urges.0 = f32::tanh(sum_urges.0);
            sum_urges.1 = f32::tanh(sum_urges.1);

            //adjust to response
            let response = Individual::response_curve(individual.responsiveness,
                                                      self.parameters.responsiveness_curve_k_factor as f32);
            sum_urges.0 *= response;
            sum_urges.1 *= response;

            //Convert to direction
            let move_x = probability_to_bool(sum_urges.0);
            let move_y = probability_to_bool(sum_urges.1);
            let sign_x = if sum_urges.0 > 0.0 { 1 } else { -1 };
            let sign_y = if sum_urges.1 > 0.0 { 1 } else { -1 };
            let coord = individual.location  + Coord(sign_x * move_x as i16, sign_y * move_y as i16);
            if self.world.is_in_bounds( coord) && self.world.is_empty_at(coord) {
                self.world.set_at_coord(coord, id);
                self.world.set_at_coord(individual.location, EMPTY_CELL);
                individual.location = coord;
            }
        }
    }

    pub fn individual_at(population: &'a Vec<Individual>, world: &World, coord: Coord) -> Option<&'a Individual> {
        return population.get(world.at_coord(coord) as usize)
    }

    pub fn simulate_all(&mut self, parameters: &Parameters, simulation_step: u32) {
        //Collect all the genomes
        let mut genomes_copy: Vec<Genome> = self.population.iter().skip(1).map(|i| i.genome.clone()).collect::<Vec<_>>();
        for individual in self.population.iter_mut().skip(1) {
            individual.simulate(&mut genomes_copy, &mut self.world, &mut self.signals, &parameters,
                                &mut self.death_queue, &mut self.move_queue, simulation_step);
        }

        self.drain_move_queue();
        self.drain_death_queue();
    }

    pub fn end_generation(&mut self) {
        let challenge = get_challenge_function(Challenges::Circle);
        // Get all the genomes from individuals that survived the challenge
        let genomes: Vec<Genome> = self.population.iter().skip(1)
            .filter(|&i| {
                challenge(i, &self.world, &self.signals, &self.parameters, vec![50])
            })
            .map(|i| i.genome.clone())
            .collect();

        self.new_generation(&genomes);
    }

    pub fn new_generation(&mut self, genomes: &Vec<Genome>) {
        self.world.zero_fill();
        self.population.clear();
        self.population.push(Individual::new(0, Coord(-1, -1), make_random_genome(1), self.parameters));


        let mut rng = rand::thread_rng();
        for i in 1..=self.parameters.population {
            let random_father = genomes.choose(&mut rng);
            let random_mother = genomes.choose(&mut rng);
            let child_location = self.world.find_random_empty_location();

            // If any of the parents is None, child is random
            let child = if let (Some(father), Some(mother)) = (random_father, random_mother) {
                breed_from_parents(father, mother, &self.parameters)
            } else {
                let genome_size = rng.gen_range(1..=self.parameters.max_genome_length);
                make_random_genome(genome_size)
            };

            self.population.insert(i as usize, Individual::new(i, child_location, child, &self.parameters));
        }
    }

    pub fn get_population_locations(&self) -> Vec<(f64,f64)> {
        self.population.iter().map(|i| (i.location.0 as f64, i.location.1 as f64)).collect()
    }
}