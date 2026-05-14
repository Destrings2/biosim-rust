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

    /// Iterate alive agents mutably. O(N) — walks `alive_ids` directly and
    /// returns `&mut Agent` for each. Replaces a previous O(N²) implementation
    /// that did a linear `Vec::contains` per slot.
    pub fn iter_alive_mut(&mut self) -> impl Iterator<Item = &mut Agent> {
        // SAFETY: `alive_ids` contains unique AgentId values, so each call to
        // `next()` borrows a distinct slot of `agents`. The lifetime of every
        // yielded reference is bound to `&mut self` via the pointer's tied
        // lifetime, and the returned Iterator object holds the unique &mut
        // borrow of the population for its full lifetime.
        let ids: *const [AgentId] = self.alive_ids.as_slice();
        let agents: *mut Vec<Option<Agent>> = &mut self.agents;
        unsafe {
            (*ids).iter().filter_map(move |&id| {
                (*agents).as_mut_slice().get_mut(id as usize).and_then(|s| s.as_mut())
            })
        }
    }

    // ── Deferred queues ───────────────────────────────────────────────

    pub fn queue_for_death(&mut self, id: AgentId) {
        if !self.death_queue.contains(&id) {
            self.death_queue.push(id);
        }
    }

    pub fn queue_for_move(&mut self, id: AgentId, new_loc: Coord) {
        self.move_queue.push((id, new_loc));
    }

    /// Apply all queued deaths. Clears corresponding grid cells.
    pub fn drain_death_queue(&mut self, grid: &mut Grid) {
        for id in self.death_queue.drain(..) {
            if let Some(agent) = self.agents.get_mut(id as usize).and_then(|s| s.as_mut()) {
                agent.alive = false;
                grid.set(agent.loc, crate::grid::EMPTY);
            }
            self.alive_ids.retain(|&x| x != id);
        }
    }

    /// Apply all queued moves. Silently skips dead agents or occupied
    /// destinations. **Kill barriers**: if the destination is a kill
    /// barrier, the agent dies — its old cell is freed and the agent is
    /// removed from `alive_ids`. The kill barrier itself stays put.
    pub fn drain_move_queue(&mut self, grid: &mut Grid) {
        let mut killed = Vec::new();
        for (id, new_loc) in self.move_queue.drain(..) {
            let agent = match self.agents.get_mut(id as usize).and_then(|s| s.as_mut()) {
                Some(a) if a.alive => a,
                _ => continue,
            };
            // Touching a kill barrier kills the agent. The cell itself
            // stays as KILL_BARRIER so subsequent agents also die.
            if grid.is_kill_barrier_at(new_loc) {
                let old_loc = agent.loc;
                agent.alive = false;
                grid.set(old_loc, crate::grid::EMPTY);
                killed.push(id);
                continue;
            }
            if !grid.is_empty_at(new_loc) { continue; }
            let old_loc = agent.loc;
            let new_dir = (new_loc - old_loc).as_dir();
            grid.set(old_loc, crate::grid::EMPTY);
            grid.set(new_loc, id);
            agent.loc = new_loc;
            agent.last_move_dir = new_dir;
            agent.heading = new_dir; // persistent heading updated on move
        }
        if !killed.is_empty() {
            self.alive_ids.retain(|id| !killed.contains(id));
        }
    }
}
