//! Memory-register read sensors. Pair with the `write_memory_N` actions
//! to give peeps four `f32` scratch registers per agent.

use biosim4_core::registry::{Sensor, SensorContext};

macro_rules! read_memory {
    ($name:ident, $id:literal, $label:literal, $reg:literal) => {
        pub(crate) struct $name;
        impl Sensor for $name {
            fn id(&self) -> &str {
                $id
            }
            fn name(&self) -> &str {
                $label
            }
            fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
                ctx.agent.memory[$reg]
            }
        }
    };
}

read_memory!(Memory0, "memory_0", "memory 0", 0);
read_memory!(Memory1, "memory_1", "memory 1", 1);
read_memory!(Memory2, "memory_2", "memory 2", 2);
read_memory!(Memory3, "memory_3", "memory 3", 3);
