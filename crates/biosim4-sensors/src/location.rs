//! Location & boundary sensors.
//!
//! `loc_x` / `loc_y` report the agent's normalized position; the
//! `boundary_dist_*` family report distance to the nearest wall on each
//! axis (or overall) as a `[0, 1]` value.

use biosim4_core::registry::{Sensor, SensorContext};

pub(crate) struct LocX;
impl Sensor for LocX {
    fn id(&self) -> &str {
        "loc_x"
    }
    fn name(&self) -> &str {
        "loc X"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.agent.loc.x as f32 / (ctx.world.size_x - 1) as f32
    }
}

pub(crate) struct LocY;
impl Sensor for LocY {
    fn id(&self) -> &str {
        "loc_y"
    }
    fn name(&self) -> &str {
        "loc Y"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        ctx.agent.loc.y as f32 / (ctx.world.size_y - 1) as f32
    }
}

pub(crate) struct BoundaryDistX;
impl Sensor for BoundaryDistX {
    fn id(&self) -> &str {
        "boundary_dist_x"
    }
    fn name(&self) -> &str {
        "boundary dist X"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // Distance from the agent to the nearer of the two east-west walls
        // (`min(x, size_x − 1 − x)`), normalized by half the grid width.
        let x = ctx.agent.loc.x as i32;
        let sx = ctx.world.size_x as i32;
        let min_dist = x.min((sx - x) - 1).max(0) as f32;
        (min_dist / (sx as f32 / 2.0)).clamp(0.0, 1.0)
    }
}

pub(crate) struct BoundaryDistY;
impl Sensor for BoundaryDistY {
    fn id(&self) -> &str {
        "boundary_dist_y"
    }
    fn name(&self) -> &str {
        "boundary dist Y"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        let y = ctx.agent.loc.y as i32;
        let sy = ctx.world.size_y as i32;
        let min_dist = y.min((sy - y) - 1).max(0) as f32;
        (min_dist / (sy as f32 / 2.0)).clamp(0.0, 1.0)
    }
}

pub(crate) struct BoundaryDist;
impl Sensor for BoundaryDist {
    fn id(&self) -> &str {
        "boundary_dist"
    }
    fn name(&self) -> &str {
        "boundary dist"
    }
    fn evaluate(&self, ctx: &mut SensorContext) -> f32 {
        // Closest distance to any wall, normalized by the *larger* of the
        // two half-grid spans so a perfectly centered agent reads 1.0 on
        // a square grid and slightly less than 1.0 on a long rectangle.
        let x = ctx.agent.loc.x as i32;
        let y = ctx.agent.loc.y as i32;
        let sx = ctx.world.size_x as i32;
        let sy = ctx.world.size_y as i32;
        let dx = x.min((sx - x) - 1).max(0);
        let dy = y.min((sy - y) - 1).max(0);
        let closest = dx.min(dy) as f32;
        let max_possible = ((sx / 2 - 1).max(sy / 2 - 1)).max(1) as f32;
        (closest / max_possible).clamp(0.0, 1.0)
    }
}
