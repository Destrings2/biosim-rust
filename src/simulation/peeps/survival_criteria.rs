use crate::Parameters;
use crate::population::individual::Individual;
use crate::simulation::signals::Signals;
use crate::simulation::world::World;

pub enum Challenges {
    Circle,
}

pub fn get_challenge_function(challenge: Challenges) -> fn(&Individual, &World, &Signals, &Parameters, Vec<i16>) -> bool {
    match challenge {
        Challenges::Circle => circle_challenge,
    }
}

pub fn circle_challenge(individual: &Individual, world: &World, _signals: &Signals, _parameters: &Parameters,
                        arguments: Vec<i16>) -> bool {
    if !individual.alive {
        return false;
    }

    let radius = arguments[0];

    let center_x = (world.width / 2) as i16;
    let center_y = (world.height / 2) as i16;

    // The individuals within a distance of the world's center survive.
    let distance_from_center = (center_x - individual.location.0).pow(2) + (center_y - individual.location.1).pow(2);
    if distance_from_center < radius {
        return true;
    } else {
        return false;
    }
}