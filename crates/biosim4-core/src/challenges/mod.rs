//! Built-in challenge implementations (26 challenges across 7 submodules).
//!
//! - `spatial` — challenges based on agent location at generation end:
//!   `circle`, `right_half`, `right_quarter`, `left_eighth`,
//!   `east_west_eighths`, `center_weighted`, `center_unweighted`,
//!   `corner`, `corner_weighted`, `against_any_wall`, `near_barrier`.
//! - `social` — challenges based on proximity to other agents:
//!   `pairs`, `center_sparse`, `string`.
//! - `migration` — `migrate_distance`: rewards traveling far from birth location.
//! - `sequential` — challenges requiring ordered behavior during the generation:
//!   `touch_any_wall` (uses `challenge_bits`), `location_sequence`.
//! - `radioactive` — `radioactive_walls`: lethal border zones.
//! - `altruism` — `altruism`, `altruism_sacrifice`: proximity-based group fitness.
//! - `dynamic` — time-varying challenges with `on_sim_step` and
//!   `on_generation_start` hooks: `sun_tracker`, `diaspora`,
//!   `food_foraging`, `survivor`.
//!
//! All built-in challenges are registered by `register_builtin_challenges`.

mod spatial;
mod migration;
mod social;
mod sequential;
mod radioactive;
mod altruism;
mod dynamic;

pub use spatial::*;
pub use migration::*;
pub use social::*;
pub use sequential::*;
pub use radioactive::*;
pub use altruism::*;
pub use dynamic::*;

use crate::registry::ChallengeRegistry;

pub fn register_builtin_challenges(registry: &mut ChallengeRegistry) {
    // Spatial
    registry.register(Box::new(CircleChallenge::default()));
    registry.register(Box::new(RightHalfChallenge));
    registry.register(Box::new(RightQuarterChallenge));
    registry.register(Box::new(LeftEighthChallenge));
    registry.register(Box::new(EastWestEighthsChallenge));
    registry.register(Box::new(CenterWeightedChallenge::default()));
    registry.register(Box::new(CenterUnweightedChallenge::default()));
    registry.register(Box::new(CornerChallenge::default()));
    registry.register(Box::new(CornerWeightedChallenge::default()));
    registry.register(Box::new(AgainstAnyWallChallenge));
    registry.register(Box::new(NearBarrierChallenge::default()));
    // Social
    registry.register(Box::new(PairsChallenge));
    registry.register(Box::new(CenterSparseChallenge::default()));
    registry.register(Box::new(StringChallenge::default()));
    // Migration
    registry.register(Box::new(MigrateDistanceChallenge::default()));
    // Sequential
    registry.register(Box::new(TouchAnyWallChallenge));
    registry.register(Box::new(LocationSequenceChallenge::default()));
    // Radioactive
    registry.register(Box::new(RadioactiveWallsChallenge::default()));
    // Altruism
    registry.register(Box::new(AltruismChallenge::default()));
    registry.register(Box::new(AltruismSacrificeChallenge::default()));
    // Dynamic / time-varying
    registry.register(Box::new(SunTrackerChallenge::default()));
    registry.register(Box::new(DiasporaChallenge::default()));
    registry.register(Box::new(SurvivorChallenge::default()));
}
