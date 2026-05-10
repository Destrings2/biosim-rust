pub mod sensor;
pub mod action;
pub mod challenge;
pub mod breed;

pub use sensor::{Sensor, SensorContext, SensorRegistry};
pub use action::{Action, ActionContext, ActionRegistry};
pub use challenge::{Challenge, ChallengeRegistry, ChallengeComposition, ChallengeConfig};
pub use breed::{Breed, BreedId, BreedRegistry};
