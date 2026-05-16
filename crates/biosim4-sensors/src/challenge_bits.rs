//! Challenge-bit read sensors.
//!
//! Expose the low four bits of `agent.challenge_bits` (a per-agent u32
//! owned by the active challenge) as 0/1 sensors. The bit meanings are
//! challenge-defined — e.g. `tag` uses bit 0 for "am I it?", `quarantine`
//! uses bit 0 for "am I infected?", `location_sequence` uses bits 0..n
//! for waypoint progress.

use biosim4_core::registry::{Sensor, SensorContext};

macro_rules! read_challenge_bit {
    ($name:ident, $id:literal, $label:literal, $bit:literal) => {
        pub(crate) struct $name;
        impl Sensor for $name {
            fn id(&self) -> &str {
                $id
            }
            fn name(&self) -> &str {
                $label
            }
            fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
                ((ctx.agent.challenge_bits >> $bit) & 1) as f32
            }
        }
    };
}

read_challenge_bit!(ChallengeBit0, "challenge_bit_0", "challenge bit 0", 0);
read_challenge_bit!(ChallengeBit1, "challenge_bit_1", "challenge bit 1", 1);
read_challenge_bit!(ChallengeBit2, "challenge_bit_2", "challenge bit 2", 2);
read_challenge_bit!(ChallengeBit3, "challenge_bit_3", "challenge bit 3", 3);
