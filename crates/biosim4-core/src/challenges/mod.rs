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
    registry.register(Box::new(FoodForagingChallenge::default()));
    registry.register(Box::new(SurvivorChallenge::default()));
}
