use std::cell::{RefCell};
use std::collections::HashMap;
use rand::Rng;
use crate::Parameters;
use crate::population::genome::{Genome, make_random_genome};
use crate::population::brain::sensor_actions::action_implementation::get_action_dispatch;
use crate::population::brain::sensor_actions::ENABLED_ACTIONS;
use crate::population::individual::Individual;
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
        let mut population = Vec::new();
        let signals = Signals::new(1, p.size_x, p.size_y);
        let move_queue = HashMap::new();
        let death_queue = Vec::new();
        let mut world = World::new(p.size_x, p.size_y);
        let mut rng = rand::thread_rng();

        for i in 0..p.population {
            let empty_coord = world.find_random_empty_location();
            let genome_size = rng.gen_range(1..=p.max_genome_length);
            let individual = Individual::new(i, empty_coord, make_random_genome(genome_size), &p);
            world.set_at_coord(empty_coord, individual.index);
            population.push(individual);
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
                individual.location = coord;
            }
        }
    }

    pub fn individual_at(population: &'a Vec<Individual>, world: &World, coord: Coord) -> Option<&'a Individual> {
        return population.get(world.at_coord(coord) as usize)
    }

    pub fn simulate_all(&mut self, parameters: &Parameters, simulation_step: u32) {
        //Collect all the genomes
        let mut genomes_copy: Vec<Genome> = self.population.iter().map(|i| i.genome.clone()).collect::<Vec<_>>();
        for individual in self.population.iter_mut() {
            individual.simulate(&mut genomes_copy, &mut self.world, &mut self.signals, &parameters,
                                &mut self.death_queue, &mut self.move_queue, simulation_step);
        }
    }
}