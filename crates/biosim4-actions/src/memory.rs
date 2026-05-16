//! Memory-register write actions.
//!
//! Memory writes are internal-state updates, not motor outputs — they
//! consume `level` directly without the responsiveness gate (otherwise
//! low responsiveness would compress the writable range toward 0.5 and
//! break the agent's ability to flip stored bits).

use biosim4_core::registry::{Action, ActionContext};

macro_rules! write_memory {
    ($name:ident, $id:literal, $label:literal, $reg:literal) => {
        pub(crate) struct $name;
        impl Action for $name {
            fn id(&self) -> &str {
                $id
            }
            fn name(&self) -> &str {
                $label
            }
            fn execute(&self, level: f32, ctx: &mut ActionContext) {
                ctx.agent.memory[$reg] = (level.tanh() + 1.0) / 2.0;
            }
        }
    };
}

write_memory!(WriteMemory0, "write_memory_0", "write memory 0", 0);
write_memory!(WriteMemory1, "write_memory_1", "write memory 1", 1);
write_memory!(WriteMemory2, "write_memory_2", "write memory 2", 2);
write_memory!(WriteMemory3, "write_memory_3", "write memory 3", 3);
