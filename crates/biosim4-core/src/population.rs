//! Agent storage and deferred action queues.
//!
//! # Slot stability
//!
//! Agents are stored in a `Vec<Option<Agent>>`. Slot 0 is permanently reserved
//! (`INVALID_AGENT = 0`), so a zero-valued grid cell unambiguously means empty.
//! `spawn()` always appends — slots are never reused or relocated. An `AgentId`
//! returned by `spawn()` remains valid for the lifetime of the population.
//!
//! # Deferred queues
//!
//! Actions do not mutate the grid or population directly during agent stepping.
//! Instead they push to `move_queue` and `death_queue`, which are drained at
//! end-of-step by `drain_death_queue` (runs first) and `drain_move_queue`.
//!
//! Death runs before move so that a cell freed by a killed agent can be entered
//! by a moving agent in the same step. `drain_move_queue` silently skips any
//! agent that was killed in the same drain cycle.

use crate::agent::{Agent, AgentId};
use crate::grid::Grid;
use crate::types::Coord;

/// Manages all agents and deferred action queues.
/// Slot 0 is reserved (INVALID_AGENT = 0). Agents occupy indices 1..capacity.
pub struct Population {
    /// Index-stable storage. `None` = dead / unoccupied slot.
    agents: Vec<Option<Agent>>,
    /// Indices of alive agents for fast iteration.
    alive_ids: Vec<AgentId>,
    /// Deferred death queue — drained at end-of-step.
    pub death_queue: Vec<AgentId>,
    /// Deferred move queue — drained at end-of-step.
    pub move_queue: Vec<(AgentId, Coord)>,
}

impl Population {
    pub fn new(capacity: u32) -> Self {
        let mut agents = Vec::with_capacity(capacity as usize + 1);
        agents.push(None); // slot 0 reserved
        Self {
            agents,
            alive_ids: Vec::with_capacity(capacity as usize),
            death_queue: Vec::new(),
            move_queue: Vec::new(),
        }
    }

    /// Reset, clearing all agents.
    pub fn clear(&mut self) {
        self.agents.truncate(1);
        self.alive_ids.clear();
        self.death_queue.clear();
        self.move_queue.clear();
    }

    /// The ID that the next call to `spawn()` will assign.
    pub fn next_id(&self) -> AgentId { self.agents.len() as AgentId }

    /// Add a new agent. Returns its assigned ID.
    pub fn spawn(&mut self, agent: Agent) -> AgentId {
        let id = self.agents.len() as AgentId;
        self.alive_ids.push(id);
        self.agents.push(Some(agent));
        id
    }

    pub fn get(&self, id: AgentId) -> Option<&Agent> {
        self.agents.get(id as usize)?.as_ref()
    }

    pub fn get_mut(&mut self, id: AgentId) -> Option<&mut Agent> {
        self.agents.get_mut(id as usize)?.as_mut()
    }

    pub fn get_at(&self, grid: &Grid, loc: Coord) -> Option<&Agent> {
        let id = grid.at(loc);
        if id == crate::grid::EMPTY || id == crate::grid::BARRIER { return None; }
        self.get(id)
    }

    /// Number of currently alive agents.
    pub fn alive_count(&self) -> usize { self.alive_ids.len() }

    pub fn alive_ids(&self) -> &[AgentId] { &self.alive_ids }

    /// Rebuild the `alive_ids` cache from scratch by scanning every slot.
    /// Used by alternate stepping backends (e.g. the GPU fast-forward path)
    /// that mutate `agent.alive` directly without going through
    /// `queue_for_death`/`drain_death_queue`. O(capacity).
    pub fn rebuild_alive_ids(&mut self) {
        self.alive_ids.clear();
        for a in self.agents.iter().skip(1).flatten() {
            if a.alive {
                self.alive_ids.push(a.id);
            }
        }
    }

    /// Iterate over all alive agents.
    pub fn iter_alive(&self) -> impl Iterator<Item = &Agent> {
        self.alive_ids.iter().filter_map(|&id| self.get(id))
    }

    /// Iterate alive agents mutably. Walks `agents` slots directly and
    /// filters dead/empty slots. Slot order matches insertion order, which
    /// matches the order of `alive_ids` because slots are append-only.
    ///
    /// Used by challenge `on_generation_start` / `on_sim_step` hooks, which
    /// run once per step (not per agent), so the cost of scanning the few
    /// dead slots that may exist is negligible vs. the per-agent fold body.
    pub fn iter_alive_mut(&mut self) -> impl Iterator<Item = &mut Agent> {
        self.agents
            .iter_mut()
            .skip(1)
            .filter_map(|s| s.as_mut())
            .filter(|a| a.alive)
    }

    // ── Deferred queues ───────────────────────────────────────────────

    /// Queue an agent for end-of-step death. Duplicate entries are harmless —
    /// `drain_death_queue` deduplicates via the `alive` flag.
    pub fn queue_for_death(&mut self, id: AgentId) {
        self.death_queue.push(id);
    }

    pub fn queue_for_move(&mut self, id: AgentId, new_loc: Coord) {
        self.move_queue.push((id, new_loc));
    }

    /// Apply all queued deaths. Clears corresponding grid cells. Idempotent
    /// on duplicate IDs: each slot is flipped to `alive = false` and skipped
    /// thereafter. Single linear pass over `alive_ids` rather than O(N×D).
    pub fn drain_death_queue(&mut self, grid: &mut Grid) {
        if self.death_queue.is_empty() { return; }
        let q = std::mem::take(&mut self.death_queue);
        self.apply_deaths(grid, q);
    }

    /// Variant of `drain_death_queue` that consumes an externally-owned death
    /// list, skipping the round-trip through `self.death_queue`. Used by the
    /// parallel step pipeline, where `step_all_agents` already produces the
    /// merged death list as the rayon fold result.
    pub fn drain_death_queue_from(&mut self, grid: &mut Grid, deaths: Vec<AgentId>) {
        if deaths.is_empty() && self.death_queue.is_empty() { return; }
        // Apply any leftover queued deaths first (e.g. from `queue_for_death`
        // calls outside the step pipeline), then the externally-supplied list.
        let mut combined = std::mem::take(&mut self.death_queue);
        combined.extend(deaths);
        self.apply_deaths(grid, combined);
    }

    fn apply_deaths(&mut self, grid: &mut Grid, ids: Vec<AgentId>) {
        if ids.is_empty() { return; }
        for id in ids {
            if let Some(agent) = self.agents.get_mut(id as usize).and_then(|s| s.as_mut()) {
                if !agent.alive { continue; }
                agent.alive = false;
                grid.set(agent.loc, crate::grid::EMPTY);
            }
        }
        self.alive_ids.retain(|&id| {
            self.agents
                .get(id as usize)
                .and_then(|s| s.as_ref())
                .is_some_and(|a| a.alive)
        });
    }

    /// Apply all queued moves. Silently skips dead agents or occupied
    /// destinations. **Kill barriers**: if the destination is a kill
    /// barrier, the agent dies — its old cell is freed and the agent is
    /// removed from `alive_ids`. The kill barrier itself stays put.
    pub fn drain_move_queue(&mut self, grid: &mut Grid) {
        if self.move_queue.is_empty() { return; }
        let q = std::mem::take(&mut self.move_queue);
        self.apply_moves(grid, q);
    }

    /// Variant of `drain_move_queue` that consumes an externally-owned move
    /// list. Used by the parallel step pipeline so the merged rayon fold
    /// result goes straight to the grid without an extra extend through
    /// `self.move_queue`.
    pub fn drain_move_queue_from(&mut self, grid: &mut Grid, moves: Vec<(AgentId, Coord)>) {
        if moves.is_empty() && self.move_queue.is_empty() { return; }
        // Honour any pre-queued moves from `queue_for_move` (e.g. tooling),
        // then the externally-supplied list.
        let mut combined = std::mem::take(&mut self.move_queue);
        combined.extend(moves);
        self.apply_moves(grid, combined);
    }

    fn apply_moves(&mut self, grid: &mut Grid, moves: Vec<(AgentId, Coord)>) {
        let mut any_killed = false;
        for (id, new_loc) in moves {
            let agent = match self.agents.get_mut(id as usize).and_then(|s| s.as_mut()) {
                Some(a) if a.alive => a,
                _ => continue,
            };
            if grid.is_kill_barrier_at(new_loc) {
                let old_loc = agent.loc;
                agent.alive = false;
                grid.set(old_loc, crate::grid::EMPTY);
                any_killed = true;
                continue;
            }
            if !grid.is_empty_at(new_loc) { continue; }
            let old_loc = agent.loc;
            let new_dir = (new_loc - old_loc).as_dir();
            grid.set(old_loc, crate::grid::EMPTY);
            grid.set(new_loc, id);
            agent.loc = new_loc;
            agent.last_move_dir = new_dir;
            agent.heading = new_dir;
        }
        if any_killed {
            // Same single-pass retain as drain_death_queue: walk alive_ids
            // once, keep only entries whose Agent.alive is still true.
            self.alive_ids.retain(|&id| {
                self.agents
                    .get(id as usize)
                    .and_then(|s| s.as_ref())
                    .is_some_and(|a| a.alive)
            });
        }
    }
}
