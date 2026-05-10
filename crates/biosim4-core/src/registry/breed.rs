use crate::agent::PropValue;
use serde::{Deserialize, Serialize};

pub type BreedId = u16;
pub const DEFAULT_BREED: BreedId = 0;

/// A breed defines a class of agents with shared default sensors, actions, color, and properties.
/// Inspired by NetLogo breed declarations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Breed {
    pub id: BreedId,
    pub name: String,
    /// Default RGB color for agents of this breed (overridden by genome-derived color if desired).
    pub default_color: [u8; 3],
    /// IDs of sensors this breed uses. Empty = use all registered sensors.
    pub sensor_ids: Vec<String>,
    /// IDs of actions this breed can execute. Empty = use all registered actions.
    pub action_ids: Vec<String>,
    /// Default values for breed-specific properties (turtle variables).
    pub default_props: Vec<(String, PropValue)>,
}

impl Breed {
    pub fn default_breed() -> Self {
        Breed {
            id: DEFAULT_BREED,
            name: "default".to_string(),
            default_color: [200, 200, 200],
            sensor_ids: vec![],
            action_ids: vec![],
            default_props: vec![],
        }
    }
}

pub struct BreedRegistry {
    breeds: Vec<Breed>,
}

impl BreedRegistry {
    pub fn new() -> Self {
        Self { breeds: vec![Breed::default_breed()] }
    }

    pub fn register(&mut self, mut breed: Breed) -> BreedId {
        let id = self.breeds.len() as BreedId;
        breed.id = id;
        self.breeds.push(breed);
        id
    }

    pub fn get(&self, id: BreedId) -> &Breed {
        self.breeds.get(id as usize).unwrap_or(&self.breeds[0])
    }

    pub fn count(&self) -> usize { self.breeds.len() }
}

impl Default for BreedRegistry {
    fn default() -> Self { Self::new() }
}
