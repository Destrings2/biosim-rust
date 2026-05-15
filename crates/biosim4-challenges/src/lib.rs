//! Built-in challenge implementations across 9 submodules.
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
//! - `tag` — `tag`: contact-transferred "you're it!" bit (uses
//!   `challenge_bit_0` sensor so agents know their own status).
//! - `quarantine` — `quarantine`: contagion spreads from a seed disc through
//!   contact (uses `challenge_bit_0` sensor).
//!
//! All built-in challenges are registered by `register_builtin_challenges`.

mod altruism;
mod dynamic;
mod migration;
mod quarantine;
mod radioactive;
mod sequential;
mod social;
mod spatial;
mod tag;
mod wanderers;

pub use altruism::*;
pub use dynamic::*;
pub use migration::*;
pub use quarantine::*;
pub use radioactive::*;
pub use sequential::*;
pub use social::*;
pub use spatial::*;
pub use tag::*;
pub use wanderers::*;

use biosim4_core::registry::ChallengeRegistry;

/// Register all 23 built-in challenges with `registry`.
///
/// Call this immediately after [`SimulationState::new`](biosim4_core::SimulationState::new),
/// before activating any challenge. After registration, use
/// [`ChallengeRegistry::set_single`] or [`ChallengeRegistry::apply_config`]
/// to choose which challenge(s) are active.
///
/// # Registered challenges
///
/// **Spatial** (11): `circle`, `right_half`, `right_quarter`, `left_eighth`,
/// `east_west_eighths`, `center_weighted`, `center_unweighted`, `corner`,
/// `corner_weighted`, `against_any_wall`, `near_barrier`.
///
/// **Social** (3): `pairs`, `center_sparse`, `string`.
///
/// **Migration** (1): `migrate_distance`.
///
/// **Sequential** (2): `touch_any_wall`, `location_sequence`.
///
/// **Radioactive** (1): `radioactive_walls`.
///
/// **Altruism** (2): `altruism`, `altruism_sacrifice`.
///
/// **Dynamic** (3): `sun_tracker`, `diaspora`, `survivor`.
///
/// **Tag / contagion** (2): `tag`, `quarantine`. Both use the
/// `challenge_bit_0` sensor for agent self-awareness.
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
    // Tag / contagion
    registry.register(Box::new(TagChallenge::default()));
    registry.register(Box::new(QuarantineChallenge::default()));
    // Programmable-agent demo (smoke test for ProgrammablePool).
    registry.register(Box::new(WanderersChallenge::default()));
}
