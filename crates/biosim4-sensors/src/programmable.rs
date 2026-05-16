//! Programmable-entity sensors.

use crate::helpers::long_probe_alien_fwd;
use biosim4_core::registry::{Sensor, SensorContext};

/// Forward long-probe for programmable ("alien") entities. Walks along
/// the agent's `last_move_dir` for up to `long_probe_dist` empty cells.
/// Returns `(steps − 1) / long_probe_dist` when a programmable cell is
/// the first non-empty thing the probe sees, or `1.0` when the probe
/// runs off the grid, hits a barrier, hits a peep, or finds nothing
/// within range. The label is deliberately generic ("alien" rather than
/// "predator") because the pool is generic: any challenge can register
/// its own kind of non-evolved entity.
pub(crate) struct LongprobeAlienFwd;
impl Sensor for LongprobeAlienFwd {
    fn id(&self) -> &str {
        "longprobe_alien_fwd"
    }
    fn name(&self) -> &str {
        "long probe alien fwd"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        long_probe_alien_fwd(
            ctx.agent.loc,
            ctx.agent.last_move_dir,
            ctx.agent.long_probe_dist,
            ctx.world.grid,
        )
    }
}
