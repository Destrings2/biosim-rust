use rand::Rng;

pub mod types;
pub mod parameters;
pub mod grid;
pub mod peeps;
pub mod simulation;
pub mod signals;
pub mod world;

// Generates a random number, and returns true if it falls within the probability
pub fn probability_to_bool(probability: f32) -> bool {
    let mut rng = rand::thread_rng();
    let random_number = rng.gen_range(0.0..1.0f32);
    random_number < probability
}