//! Last-move-direction sensors. Decompose `agent.last_move_dir` into a
//! `[0, 1]` reading per axis, with `0.5` meaning "stationary / center".

use biosim4_core::registry::{Sensor, SensorContext};

pub(crate) struct LastMoveDirX;
impl Sensor for LastMoveDirX {
    fn id(&self) -> &str {
        "last_move_dir_x"
    }
    fn name(&self) -> &str {
        "last move dir X"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let dx = ctx.agent.last_move_dir.as_normalized_coord().x;
        (dx as f32 + 1.0) / 2.0
    }
}

pub(crate) struct LastMoveDirY;
impl Sensor for LastMoveDirY {
    fn id(&self) -> &str {
        "last_move_dir_y"
    }
    fn name(&self) -> &str {
        "last move dir Y"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let dy = ctx.agent.last_move_dir.as_normalized_coord().y;
        (dy as f32 + 1.0) / 2.0
    }
}
