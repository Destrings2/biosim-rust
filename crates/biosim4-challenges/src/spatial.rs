//! Spatial challenges: evaluate agent position at generation end.
//!
//! Predicates: `circle` (configurable center/radius/weighted), `right_half`,
//! `right_quarter`, `left_eighth`, `east_west_eighths`, `center_weighted`,
//! `center_unweighted`, `corner`, `corner_weighted`, `against_any_wall`,
//! `near_barrier`. Most return overlays (circles or rectangles) for
//! visualization.

use biosim4_core::agent::Agent;
use biosim4_core::registry::challenge::{Challenge, ChallengeOverlay};
use biosim4_core::world::World;
use serde_json::{json, Value};

/// Helper: normalized agent location (0..1 range).
fn nloc(agent: &Agent, world: &World) -> (f32, f32) {
    (agent.loc.x as f32 / (world.size_x - 1) as f32, agent.loc.y as f32 / (world.size_y - 1) as f32)
}

// ── Circle ─────────────────────────────────────────────────────────────────

pub struct CircleChallenge {
    pub cx: f32,
    pub cy: f32,
    pub radius: f32,
    pub weighted: bool,
}
impl Default for CircleChallenge {
    fn default() -> Self {
        CircleChallenge { cx: 0.25, cy: 0.75, radius: 0.25, weighted: true }
    }
}
impl Challenge for CircleChallenge {
    fn id(&self) -> &str {
        "circle"
    }
    fn name(&self) -> &str {
        "Circle Safe Zone"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "cx":     { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.25 },
            "cy":     { "type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.75 },
            "radius": { "type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.25 },
            "weighted": { "type": "boolean", "default": true }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("cx") {
            self.cx = v.as_f64().ok_or("cx")? as f32;
        }
        if let Some(v) = p.get("cy") {
            self.cy = v.as_f64().ok_or("cy")? as f32;
        }
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        if let Some(v) = p.get("weighted") {
            self.weighted = v.as_bool().ok_or("weighted")?;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        // Wrap-aware: on a torus the safe disc straddles the seam and
        // an agent on the opposite side of the wrap is still inside.
        let dist = world.grid.norm_dist_to_norm_point(agent.loc, self.cx, self.cy);
        if dist > self.radius {
            return (false, 0.0);
        }
        let score = if self.weighted { (self.radius - dist) / self.radius } else { 1.0 };
        (true, score)
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        vec![ChallengeOverlay::Circle {
            cx: self.cx * sx,
            cy: self.cy * sy,
            radius: self.radius * sx.max(sy),
            color: [0, 255, 0, 40],
        }]
    }
}

// ── Right Half ────────────────────────────────────────────────────────────

pub struct RightHalfChallenge;
impl Challenge for RightHalfChallenge {
    fn id(&self) -> &str {
        "right_half"
    }
    fn name(&self) -> &str {
        "Right Half"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn configure(&mut self, _: Value) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let pass = agent.loc.x as u16 > world.size_x / 2;
        (pass, if pass { 1.0 } else { 0.0 })
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        vec![ChallengeOverlay::Rectangle {
            x: sx / 2.0,
            y: 0.0,
            w: sx / 2.0,
            h: sy,
            color: [0, 255, 0, 40],
        }]
    }
}

// ── Right Quarter ─────────────────────────────────────────────────────────

pub struct RightQuarterChallenge;
impl Challenge for RightQuarterChallenge {
    fn id(&self) -> &str {
        "right_quarter"
    }
    fn name(&self) -> &str {
        "Right Quarter"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn configure(&mut self, _: Value) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let pass = agent.loc.x as u16 > world.size_x / 2 + world.size_x / 4;
        (pass, if pass { 1.0 } else { 0.0 })
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        let start_x = sx * 0.75;
        vec![ChallengeOverlay::Rectangle {
            x: start_x,
            y: 0.0,
            w: sx - start_x,
            h: sy,
            color: [0, 255, 0, 40],
        }]
    }
}

// ── Left Eighth ───────────────────────────────────────────────────────────

pub struct LeftEighthChallenge;
impl Challenge for LeftEighthChallenge {
    fn id(&self) -> &str {
        "left_eighth"
    }
    fn name(&self) -> &str {
        "Left Eighth"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn configure(&mut self, _: Value) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let pass = (agent.loc.x as u16) < world.size_x / 8;
        (pass, if pass { 1.0 } else { 0.0 })
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        vec![ChallengeOverlay::Rectangle {
            x: 0.0,
            y: 0.0,
            w: sx / 8.0,
            h: sy,
            color: [0, 255, 0, 40],
        }]
    }
}

// ── East/West Eighths ─────────────────────────────────────────────────────

pub struct EastWestEighthsChallenge;
impl Challenge for EastWestEighthsChallenge {
    fn id(&self) -> &str {
        "east_west_eighths"
    }
    fn name(&self) -> &str {
        "East/West Eighths"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn configure(&mut self, _: Value) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let x = agent.loc.x as u16;
        let pass = x < world.size_x / 8 || x >= world.size_x - world.size_x / 8;
        (pass, if pass { 1.0 } else { 0.0 })
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        let w = sx / 8.0;
        vec![
            ChallengeOverlay::Rectangle { x: 0.0, y: 0.0, w, h: sy, color: [0, 255, 0, 40] },
            ChallengeOverlay::Rectangle { x: sx - w, y: 0.0, w, h: sy, color: [0, 255, 0, 40] },
        ]
    }
}

// ── Center Weighted ───────────────────────────────────────────────────────

pub struct CenterWeightedChallenge {
    pub radius: f32,
}
impl Default for CenterWeightedChallenge {
    fn default() -> Self {
        Self { radius: 0.33 }
    }
}
impl Challenge for CenterWeightedChallenge {
    fn id(&self) -> &str {
        "center_weighted"
    }
    fn name(&self) -> &str {
        "Center (weighted)"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "radius": { "type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.33 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let dist = world.grid.norm_dist_to_norm_point(agent.loc, 0.5, 0.5);
        if dist > self.radius {
            return (false, 0.0);
        }
        (true, (self.radius - dist) / self.radius)
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        vec![ChallengeOverlay::Circle {
            cx: sx * 0.5,
            cy: sy * 0.5,
            radius: self.radius * sx.max(sy),
            color: [0, 255, 0, 40],
        }]
    }
}

// ── Center Unweighted ────────────────────────────────────────────────────

pub struct CenterUnweightedChallenge {
    pub radius: f32,
}
impl Default for CenterUnweightedChallenge {
    fn default() -> Self {
        Self { radius: 0.33 }
    }
}
impl Challenge for CenterUnweightedChallenge {
    fn id(&self) -> &str {
        "center_unweighted"
    }
    fn name(&self) -> &str {
        "Center (unweighted)"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "radius": { "type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.33 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let dist = world.grid.norm_dist_to_norm_point(agent.loc, 0.5, 0.5);
        let pass = dist <= self.radius;
        (pass, if pass { 1.0 } else { 0.0 })
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        vec![ChallengeOverlay::Circle {
            cx: sx * 0.5,
            cy: sy * 0.5,
            radius: self.radius * sx.max(sy),
            color: [0, 255, 0, 40],
        }]
    }
}

// ── Corner ────────────────────────────────────────────────────────────────

pub struct CornerChallenge {
    pub radius: f32,
}
impl Default for CornerChallenge {
    fn default() -> Self {
        Self { radius: 0.125 }
    }
}
impl Challenge for CornerChallenge {
    fn id(&self) -> &str {
        "corner"
    }
    fn name(&self) -> &str {
        "Corner"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "radius": { "type": "number", "minimum": 0.01, "maximum": 0.4, "default": 0.125 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let (nx, ny) = nloc(agent, world);
        let corners = [(0.0f32, 0.0f32), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
        let min_dist = corners
            .iter()
            .map(|&(cx, cy)| ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt())
            .fold(f32::MAX, f32::min);
        let pass = min_dist <= self.radius;
        (pass, if pass { 1.0 } else { 0.0 })
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        let r = self.radius * sx.max(sy);
        vec![
            ChallengeOverlay::Circle { cx: 0.0, cy: 0.0, radius: r, color: [0, 255, 0, 40] },
            ChallengeOverlay::Circle { cx: 0.0, cy: sy, radius: r, color: [0, 255, 0, 40] },
            ChallengeOverlay::Circle { cx: sx, cy: 0.0, radius: r, color: [0, 255, 0, 40] },
            ChallengeOverlay::Circle { cx: sx, cy: sy, radius: r, color: [0, 255, 0, 40] },
        ]
    }
}

// ── Corner Weighted ───────────────────────────────────────────────────────

pub struct CornerWeightedChallenge {
    pub radius: f32,
}
impl Default for CornerWeightedChallenge {
    fn default() -> Self {
        Self { radius: 0.25 }
    }
}
impl Challenge for CornerWeightedChallenge {
    fn id(&self) -> &str {
        "corner_weighted"
    }
    fn name(&self) -> &str {
        "Corner (weighted)"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "radius": { "type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.25 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let (nx, ny) = nloc(agent, world);
        let corners = [(0.0f32, 0.0f32), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)];
        let min_dist = corners
            .iter()
            .map(|&(cx, cy)| ((nx - cx).powi(2) + (ny - cy).powi(2)).sqrt())
            .fold(f32::MAX, f32::min);
        if min_dist > self.radius {
            return (false, 0.0);
        }
        (true, (self.radius - min_dist) / self.radius)
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let sx = world.size_x as f32;
        let sy = world.size_y as f32;
        let r = self.radius * sx.max(sy);
        vec![
            ChallengeOverlay::Circle { cx: 0.0, cy: 0.0, radius: r, color: [0, 255, 0, 40] },
            ChallengeOverlay::Circle { cx: 0.0, cy: sy, radius: r, color: [0, 255, 0, 40] },
            ChallengeOverlay::Circle { cx: sx, cy: 0.0, radius: r, color: [0, 255, 0, 40] },
            ChallengeOverlay::Circle { cx: sx, cy: sy, radius: r, color: [0, 255, 0, 40] },
        ]
    }
}

// ── Against Any Wall ──────────────────────────────────────────────────────

pub struct AgainstAnyWallChallenge;
impl Challenge for AgainstAnyWallChallenge {
    fn id(&self) -> &str {
        "against_any_wall"
    }
    fn name(&self) -> &str {
        "Against Any Wall"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {} })
    }
    fn configure(&mut self, _: Value) -> Result<(), String> {
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let pass = world.grid.is_border(agent.loc);
        (pass, if pass { 1.0 } else { 0.0 })
    }
}

// ── Near Barrier ─────────────────────────────────────────────────────────

pub struct NearBarrierChallenge {
    pub radius: f32,
}
impl Default for NearBarrierChallenge {
    fn default() -> Self {
        Self { radius: 0.5 }
    }
}
impl Challenge for NearBarrierChallenge {
    fn id(&self) -> &str {
        "near_barrier"
    }
    fn name(&self) -> &str {
        "Near Barrier"
    }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "radius": { "type": "number", "minimum": 0.01, "maximum": 1.0, "default": 0.5 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius") {
            self.radius = v.as_f64().ok_or("radius")? as f32;
        }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        if world.grid.barrier_centers.is_empty() {
            return (false, 0.0);
        }
        let radius_px = self.radius * world.size_x as f32;
        // Walk barrier centres in pixel space using `grid.dist` so the
        // distance respects topology wrap. On the bounded `Plane` this
        // is identical to the previous normalized-then-Euclidean math;
        // on a torus an agent next to a barrier across the seam is
        // correctly counted as adjacent.
        let min_dist_px = world
            .grid
            .barrier_centers
            .iter()
            .map(|bc| world.grid.dist(agent.loc, *bc))
            .fold(f32::MAX, f32::min);
        if min_dist_px > radius_px {
            return (false, 0.0);
        }
        let score = (radius_px - min_dist_px) / radius_px;
        (true, score.clamp(0.0, 1.0))
    }
    fn overlays(&self, world: &World) -> Vec<ChallengeOverlay> {
        let r = self.radius * world.size_x as f32;
        world
            .grid
            .barrier_centers
            .iter()
            .map(|bc| ChallengeOverlay::Circle {
                cx: bc.x as f32,
                cy: bc.y as f32,
                radius: r,
                color: [0, 255, 0, 40],
            })
            .collect()
    }
}
