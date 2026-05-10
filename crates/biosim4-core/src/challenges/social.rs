use crate::agent::Agent;
use crate::grid::visit_neighborhood;
use crate::registry::challenge::{Challenge, WorldMut};
use crate::world::World;
use serde_json::{json, Value};

// ── Pairs ─────────────────────────────────────────────────────────────────

/// Agents survive if they form stable pairs: exactly 1 neighbor, and that neighbor has exactly 1 neighbor.
pub struct PairsChallenge;
impl Challenge for PairsChallenge {
    fn id(&self) -> &str { "pairs" }
    fn name(&self) -> &str { "Pairs" }
    fn params_schema(&self) -> Value { json!({ "type": "object", "properties": {} }) }
    fn configure(&mut self, _: Value) -> Result<(), String> { Ok(()) }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let count = neighbor_count(agent.loc, 1.5, world);
        if count != 1 { return (false, 0.0); }
        // Find the one neighbor and check it also has exactly 1 neighbor
        let step = crate::types::Coord::new(0,0); // placeholder
        let mut partner_ok = false;
        visit_neighborhood(world.grid, agent.loc, 1.5, |nloc| {
            if nloc == agent.loc { return; }
            if world.grid.is_occupied_at(nloc) {
                let nc = neighbor_count(nloc, 1.5, world);
                if nc == 1 { partner_ok = true; }
            }
        });
        (partner_ok, if partner_ok { 1.0 } else { 0.0 })
    }
}

// ── Center Sparse ─────────────────────────────────────────────────────────

/// Agents in the center zone survive if they have a moderate number of neighbors (not too crowded, not too isolated).
pub struct CenterSparseChallenge { pub radius: f32, pub min_neighbors: u32, pub max_neighbors: u32 }
impl Default for CenterSparseChallenge {
    fn default() -> Self { Self { radius: 0.25, min_neighbors: 5, max_neighbors: 8 } }
}
impl Challenge for CenterSparseChallenge {
    fn id(&self) -> &str { "center_sparse" }
    fn name(&self) -> &str { "Center Sparse" }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "radius": { "type": "number", "minimum": 0.01, "maximum": 0.5, "default": 0.25 },
            "min_neighbors": { "type": "integer", "minimum": 0, "maximum": 20, "default": 5 },
            "max_neighbors": { "type": "integer", "minimum": 1, "maximum": 50, "default": 8 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("radius") { self.radius = v.as_f64().ok_or("radius")? as f32; }
        if let Some(v) = p.get("min_neighbors") { self.min_neighbors = v.as_u64().ok_or("min_neighbors")? as u32; }
        if let Some(v) = p.get("max_neighbors") { self.max_neighbors = v.as_u64().ok_or("max_neighbors")? as u32; }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let nx = agent.loc.x as f32 / (world.size_x - 1) as f32;
        let ny = agent.loc.y as f32 / (world.size_y - 1) as f32;
        let dx = nx - 0.5;
        let dy = ny - 0.5;
        // Must be in outer zone (not too central)
        if (dx * dx + dy * dy).sqrt() < self.radius / 2.0 { return (false, 0.0); }
        if (dx * dx + dy * dy).sqrt() > self.radius { return (false, 0.0); }
        let count = neighbor_count(agent.loc, 1.5, world);
        let pass = count >= self.min_neighbors && count <= self.max_neighbors;
        (pass, if pass { 1.0 } else { 0.0 })
    }
}

// ── String ────────────────────────────────────────────────────────────────

/// Agents survive if they form a "string" — 2 neighbors (chain topology).
pub struct StringChallenge { pub min_neighbors: u32, pub max_neighbors: u32 }
impl Default for StringChallenge {
    fn default() -> Self { Self { min_neighbors: 2, max_neighbors: 2 } }
}
impl Challenge for StringChallenge {
    fn id(&self) -> &str { "string" }
    fn name(&self) -> &str { "String Formation" }
    fn params_schema(&self) -> Value {
        json!({ "type": "object", "properties": {
            "min_neighbors": { "type": "integer", "minimum": 1, "maximum": 8, "default": 2 },
            "max_neighbors": { "type": "integer", "minimum": 1, "maximum": 8, "default": 2 }
        }})
    }
    fn configure(&mut self, p: Value) -> Result<(), String> {
        if let Some(v) = p.get("min_neighbors") { self.min_neighbors = v.as_u64().ok_or("min_neighbors")? as u32; }
        if let Some(v) = p.get("max_neighbors") { self.max_neighbors = v.as_u64().ok_or("max_neighbors")? as u32; }
        Ok(())
    }
    fn evaluate(&self, agent: &Agent, world: &World) -> (bool, f32) {
        let count = neighbor_count(agent.loc, 1.5, world);
        let pass = count >= self.min_neighbors && count <= self.max_neighbors;
        (pass, if pass { 1.0 } else { 0.0 })
    }
}

// ── Helper ────────────────────────────────────────────────────────────────

fn neighbor_count(loc: crate::types::Coord, radius: f32, world: &World) -> u32 {
    let mut count = 0u32;
    visit_neighborhood(world.grid, loc, radius, |nloc| {
        if nloc != loc && world.grid.is_occupied_at(nloc) { count += 1; }
    });
    count
}
