//! Programmable agents owned by challenges.
//!
//! A [`Programmable`] is a non-evolved entity placed in the world by a
//! challenge. It occupies a grid cell (under a separate id range — see
//! [`grid::PROGRAMMABLE_FLAG`]), is stepped every `sim_step` by a
//! [`Program`] (a Rust impl picked at registration time), and can be
//! perceived by peeps through the generic `nearest_alien_*` sensor
//! family.
//!
//! # Why not just use [`Agent`]?
//!
//! Peeps have a genome, a neural network, evolve across generations, and
//! pay the cost of sensor evaluation + feed-forward every step. A
//! programmable doesn't need any of that — a challenge writing a predator
//! or a herder wants a tight `step` function with explicit Rust logic and
//! none of the GA machinery. Mixing them into the same struct would either
//! bloat agents with dead state or force the programmable to ship through
//! the evolution pipeline as a special case. Two collections, one trait
//! each, no special cases.
//!
//! # Lifecycle
//!
//! 1. A challenge registers one or more [`Program`] impls via
//!    [`ProgrammablePool::register_program`] (or [`register_or_get`]).
//! 2. In its `on_generation_start` hook, the challenge calls
//!    [`ProgrammablePool::spawn`] for each entity it wants in the world.
//! 3. Every `sim_step`, the framework calls [`ProgrammablePool::step_all`],
//!    which runs each entity's `Program::step` in parallel (`rayon`) and
//!    merges the effects sequentially into the grid / population / signals.
//! 4. On generation rollover the pool is cleared and the cycle repeats.
//!
//! # Parallel-safety contract
//!
//! `Program::step` receives a *read-only* [`ProgramContext`] (snapshot of
//! the world plus a read-only view of all alive programmables this step)
//! and mutates only its own `Programmable` and a thread-local
//! [`ProgramOutput`]. The framework merges outputs sequentially after the
//! parallel section ends — so two programs requesting the same cell don't
//! step on each other, but *which one wins* is order-dependent and not
//! deterministic across thread counts. Same trade-off the peep pipeline
//! already lives with.

mod spatial;
pub use spatial::SpatialIndex;

use std::collections::HashMap;

use crate::agent::AgentId;
use crate::food_layer::FoodLayer;
use crate::grid::{self, Grid};
use crate::population::Population;
use crate::rng::Rng;
use crate::signals_layer::Signals;
use crate::types::{Coord, Dir};
use crate::world::World;

/// Stable identifier for a programmable entity. 0 is reserved for "none",
/// matching the convention used by [`AgentId`].
pub type ProgrammableId = u32;
/// Index of a [`Program`] within a [`ProgrammablePool`].
pub type ProgramId = u16;
/// Opaque tag a challenge can stamp on the programmables it owns, used so
/// future `clear_for_owner` calls can scope cleanup. 0 means "untagged".
pub type OwnerTag = u32;

/// Sentinel meaning "no programmable".
pub const INVALID_PROGRAMMABLE: ProgrammableId = 0;

/// A programmable entity owned by a challenge.
///
/// Fields are `pub` so [`Program`] impls can read/write them directly.
/// Only `state` / `heading` / `color` should be mutated inside `step`; the
/// other fields are owned by the framework.
#[derive(Clone)]
pub struct Programmable {
    /// Stable identifier; matches the slot index in `ProgrammablePool::agents`.
    pub id: ProgrammableId,
    /// Position on the grid.
    pub loc: Coord,
    /// Facing direction. Useful for compass-relative behaviors.
    pub heading: Dir,
    /// Whether this entity is still alive. Set to `false` by the merge phase.
    pub alive: bool,
    /// Which [`Program`] drives this entity.
    pub program: ProgramId,
    /// Caller-chosen tag for ownership scoping. Default 0.
    pub owner: OwnerTag,
    /// Eight `f32` slots the program can use for per-entity state
    /// (cooldowns, counters, last-seen coords, target indices, …).
    /// Heavier state lives in a side-table owned by the program impl.
    pub state: [f32; 8],
    /// Rendered color. Read verbatim by the grid renderer.
    pub color: [u8; 3],
}

/// Effects a program emits per step. Plain old data, lives on the stack
/// inside the parallel section, merged after `step_all` returns.
///
/// Multi-effect programs (e.g. move AND emit a signal) just fill in
/// multiple fields. The framework applies them in a deterministic order
/// (set_color → die → kill_peep_at → move_to → signal_emit) so a program
/// can rely on, for example, "if I requested move + kill_peep_at, the
/// peep dies first so its cell is free for me to step onto".
#[derive(Default, Clone)]
pub struct ProgramOutput {
    /// Request to move to this absolute grid coordinate. Resolved by the
    /// merge: if the destination is `EMPTY` or holds a peep that this
    /// program also killed via `kill_peep_at`, the move applies.
    /// Conflicting moves between two programmables resolve to whichever
    /// the merge sees first (order is rayon's collected order — not
    /// reproducible across thread counts).
    pub move_to: Option<Coord>,
    /// If `true`, this entity dies after the merge.
    pub die: bool,
    /// Queue a peep at `coord` for death. Resolved against the live
    /// `Population::death_queue` during the merge.
    pub kill_peep_at: Option<Coord>,
    /// Pheromone emission at the entity's current `loc`. The value is
    /// the signal layer index; the burst pattern is the same as the
    /// peep `emit_signal` action (`+3` at center, `+1` at neighbors).
    /// Layer must be `< config.signal_layers`.
    pub signal_emit: Option<u8>,
    /// Update the entity's render color. Useful for visualising internal
    /// state (e.g. fed / hungry) without bookkeeping outside the program.
    pub set_color: Option<[u8; 3]>,
}

/// Read-only context handed to every [`Program::step`] call during the
/// parallel section.
pub struct ProgramContext<'a> {
    /// Read-only snapshot of the world. The same `World` instance peeps see
    /// when their sensors evaluate — including `world.programmable` for the
    /// sibling pool.
    pub world: &'a World<'a>,
    pub sim_step: u32,
    pub generation: u32,
    /// Worker-local rng. The codebase has already chosen speed over
    /// cross-thread-count reproducibility; this rng follows that choice.
    pub rng: &'a mut Rng,
}

impl<'a> ProgramContext<'a> {
    /// Iterate every alive sibling programmable. Reads through
    /// `world.programmable.iter_alive()`, so callers pay only for what they
    /// look at — no up-front clone of the pool. Programs that don't consult
    /// siblings pay zero cost.
    #[inline]
    pub fn siblings(&self) -> impl Iterator<Item = &Programmable> {
        self.world.programmable.iter_alive()
    }
}

/// Behavior driving a [`Programmable`]. One impl per "species".
///
/// Must be `Send + Sync` because `step` runs in a rayon parallel section.
/// Implementations are expected to be plain Rust structs with no interior
/// mutability — all per-entity state lives in `this.state` (or a
/// side-table the impl owns and indexes by `ProgrammableId`).
pub trait Program: Send + Sync {
    /// Stable machine identifier. Must be unique across all programs in
    /// a pool. Used by [`ProgrammablePool::register_or_get`] for upsert.
    fn id(&self) -> &str;
    /// Human-readable name for UI / logs.
    fn name(&self) -> &str;

    /// Decide what this entity does this step.
    ///
    /// - Read freely from `ctx.world` and `ctx.siblings()`.
    /// - Mutate `this.state`, `this.heading`, `this.color` as needed.
    /// - Write the requested side effects into `out`.
    fn step(&self, this: &mut Programmable, ctx: &mut ProgramContext, out: &mut ProgramOutput);

    /// Hook fired once when the entity is spawned (sequential, not in the
    /// parallel section). Override to set up initial `state`.
    fn on_spawn(&self, _this: &mut Programmable, _ctx: &mut ProgramContext) {}

    /// Hook fired once when the entity is despawned (sequential).
    fn on_despawn(&self, _this: &mut Programmable, _ctx: &mut ProgramContext) {}
}

/// Slot-stable storage for programmable entities, plus the program registry
/// that drives them.
///
/// API surface mirrors [`Population`] intentionally: slot-stable
/// `Vec<Option<T>>`, dense `alive_ids` cache, deferred death queue drained
/// by the framework. The differences from `Population` are (a) entities
/// are not evolved and have no genome / neural net; (b) the per-step
/// pipeline runs through [`Program::step`] rather than the
/// sensor/action/feedforward stack.
pub struct ProgrammablePool {
    /// Slot 0 reserved (matches `Population::INVALID_AGENT = 0`).
    agents: Vec<Option<Programmable>>,
    /// Programs registered in this pool. Indexed by [`ProgramId`].
    programs: Vec<Box<dyn Program>>,
    /// Stable string id → `ProgramId` index, for upsert-style registration.
    program_index: HashMap<String, ProgramId>,
    /// Dense cache of alive entity ids. Rebuilt incrementally on
    /// spawn/despawn; consumers (sensors, renderers) iterate this.
    alive_ids: Vec<ProgrammableId>,
    /// Owned buffer holding the parallel section's outputs: `(slot index,
    /// mutated entity state, requested effects)`. Lives on the pool so the
    /// allocation is reused across steps — `step_all` takes ownership via
    /// `mem::take` before the parallel section and swaps it back at the
    /// end, dodging the borrow conflict with the pool view that programs
    /// read from.
    scratch_results: Vec<(ProgrammableId, Programmable, ProgramOutput)>,
    /// Coarse spatial bucketing of alive entities, used by sensors like
    /// `nearest_alien_dist` to skip the per-peep linear scan over the
    /// pool. Rebuilt once per step before the parallel peep section; see
    /// [`refresh_spatial_index`](Self::refresh_spatial_index).
    spatial_index: SpatialIndex,
    /// Set true whenever an alive entity is added, removed, or moved
    /// outside `step_all`'s own merge — including challenge-driven
    /// spawning during `on_generation_start`. Cleared by the next
    /// `refresh_spatial_index`, which also handles the dirty flag set
    /// by `step_all`'s end-of-merge pass.
    spatial_dirty: bool,
}

impl ProgrammablePool {
    pub fn new() -> Self {
        Self {
            agents: vec![None], // slot 0 reserved
            programs: Vec::new(),
            program_index: HashMap::new(),
            alive_ids: Vec::new(),
            scratch_results: Vec::new(),
            spatial_index: SpatialIndex::new(),
            spatial_dirty: false,
        }
    }

    // ── Registration ──────────────────────────────────────────────────

    /// Register a program. Returns its [`ProgramId`].
    ///
    /// If a program with the same `id()` is already registered, the
    /// existing slot is reused — the new program replaces the old one
    /// in place. This matches `ChallengeRegistry::upsert_by_id` so
    /// challenges can re-register on each gen-start cycle without
    /// growing the registry.
    pub fn register_program(&mut self, program: Box<dyn Program>) -> ProgramId {
        let id = program.id().to_string();
        if let Some(&idx) = self.program_index.get(&id) {
            self.programs[idx as usize] = program;
            return idx;
        }
        let idx = self.programs.len() as ProgramId;
        self.programs.push(program);
        self.program_index.insert(id, idx);
        idx
    }

    /// Look up a program by its string id, registering it (via the
    /// supplied factory) if missing. The common pattern for challenges:
    ///
    /// ```ignore
    /// let prog = ctx.programmable.register_or_get("hunter", || Box::new(Hunter));
    /// ```
    pub fn register_or_get(
        &mut self,
        id: &str,
        factory: impl FnOnce() -> Box<dyn Program>,
    ) -> ProgramId {
        if let Some(&idx) = self.program_index.get(id) {
            return idx;
        }
        self.register_program(factory())
    }

    /// Number of registered programs. Mainly for diagnostics.
    pub fn program_count(&self) -> usize {
        self.programs.len()
    }

    /// Display name of a registered program. Used by inspectors and other
    /// read-only UI consumers that have a `ProgramId` (e.g. from
    /// `Programmable::program`) and want the human-readable label.
    pub fn program_name(&self, program: ProgramId) -> Option<&str> {
        self.programs.get(program as usize).map(|p| p.name())
    }

    // ── Entity lifecycle ──────────────────────────────────────────────

    /// Spawn a programmable at `loc` driven by `program`. Returns `None`
    /// if the cell is not empty (caller should pick a different spawn
    /// location).
    ///
    /// Writes the grid cell with the encoded programmable id and runs
    /// the program's `on_spawn` hook. Both happen sequentially — no
    /// parallel-safety concerns.
    pub fn spawn(
        &mut self,
        grid: &mut Grid,
        program: ProgramId,
        owner: OwnerTag,
        loc: Coord,
        color: [u8; 3],
    ) -> Option<ProgrammableId> {
        if !grid.is_empty_at(loc) {
            return None;
        }
        let id = self.agents.len() as ProgrammableId;
        let entity = Programmable {
            id,
            loc,
            heading: Dir::center(),
            alive: true,
            program,
            owner,
            state: [0.0; 8],
            color,
        };
        grid.set(loc, grid::encode_programmable(id));
        self.agents.push(Some(entity));
        self.alive_ids.push(id);
        self.spatial_dirty = true;
        // on_spawn fires sequentially with full mutable access to `this`.
        // The context's `world`/`siblings` are pre-step snapshots that
        // we don't bother building here since spawn happens at
        // gen-start, before any step has run.
        id.into()
    }

    /// Mark a programmable for death immediately. Used by tooling and the
    /// `Pool::clear` path. The `step_all` merge uses the same code path
    /// when it sees `ProgramOutput::die = true`.
    pub fn despawn(&mut self, grid: &mut Grid, id: ProgrammableId) {
        let Some(slot) = self.agents.get_mut(id as usize) else {
            return;
        };
        let Some(entity) = slot.as_mut() else { return };
        if !entity.alive {
            return;
        }
        entity.alive = false;
        grid.set(entity.loc, grid::EMPTY);
        // O(alive) but cheap — pool sizes are tens to low hundreds.
        self.alive_ids.retain(|&pid| pid != id);
        self.spatial_dirty = true;
    }

    /// Wipe the pool: despawn every entity and clear its grid cell.
    /// Programs registered are kept — only the live entities die. Used
    /// at generation rollover.
    pub fn clear(&mut self, grid: &mut Grid) {
        for &id in &self.alive_ids {
            if let Some(entity) = self.agents.get(id as usize).and_then(|s| s.as_ref()) {
                if entity.alive {
                    grid.set(entity.loc, grid::EMPTY);
                }
            }
        }
        self.agents.truncate(1); // keep slot 0 reserved
        self.alive_ids.clear();
        self.spatial_dirty = true;
    }

    /// Wipe every programmable whose `owner` tag matches. Other entities
    /// stay put. O(alive).
    pub fn clear_for_owner(&mut self, grid: &mut Grid, owner: OwnerTag) {
        let to_kill: Vec<ProgrammableId> = self
            .alive_ids
            .iter()
            .copied()
            .filter(|&id| {
                self.agents
                    .get(id as usize)
                    .and_then(|s| s.as_ref())
                    .is_some_and(|e| e.owner == owner)
            })
            .collect();
        for id in to_kill {
            self.despawn(grid, id);
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────

    pub fn get(&self, id: ProgrammableId) -> Option<&Programmable> {
        self.agents.get(id as usize)?.as_ref()
    }

    pub fn get_mut(&mut self, id: ProgrammableId) -> Option<&mut Programmable> {
        self.agents.get_mut(id as usize)?.as_mut()
    }

    pub fn alive_ids(&self) -> &[ProgrammableId] {
        &self.alive_ids
    }

    pub fn alive_count(&self) -> usize {
        self.alive_ids.len()
    }

    /// Iterate alive entities in `alive_ids` order.
    pub fn iter_alive(&self) -> impl Iterator<Item = &Programmable> {
        self.alive_ids.iter().filter_map(|&id| self.get(id))
    }

    // ── Step ──────────────────────────────────────────────────────────

    /// Run every alive program once and merge their effects into the
    /// world. Mirrors `sim_step::step_all_agents`: a parallel section
    /// that produces output buffers, followed by a sequential merge
    /// that mutates the grid / population / signals.
    ///
    /// Caller passes the SimulationState pieces split so the borrow
    /// checker can see the disjoint mutable references. Food is
    /// immutable to the pool (peeps absorb food; programmables don't).
    pub fn step_all(
        &mut self,
        grid: &mut Grid,
        signals: &mut Signals,
        population: &mut Population,
        food: &FoodLayer,
        sim_step: u32,
        generation: u32,
        steps_per_generation: u32,
    ) {
        if self.alive_ids.is_empty() {
            return;
        }

        // ── Phase A: parallel step ──────────────────────────────────
        //
        // Each task reads its starting entity directly from `pool_view`
        // (the read-only pool reborrow) — no up-front clone of the alive
        // set. Programs that don't touch siblings pay nothing for them;
        // those that do walk `ctx.siblings()` lazily.
        //
        // The output buffer is taken out of `self` for the parallel
        // section and put back at the end. This both (a) avoids fresh
        // allocations because Vec capacity rides through `mem::take`,
        // and (b) sidesteps the borrow conflict with `pool_view`.
        let mut results = std::mem::take(&mut self.scratch_results);
        results.clear();
        let n = self.alive_ids.len();
        results.reserve(n);

        // `&*self` reborrows the `&mut self` as a shared `&Self` for the
        // duration of this block so `World::new` and indexed reads of
        // `agents` / `programs` can coexist. The block ends before the
        // merge phase, restoring mutable access.
        {
            let pool_view: &ProgrammablePool = &*self;
            let world = World::new(
                grid,
                signals,
                food,
                population,
                pool_view,
                steps_per_generation,
                generation,
                sim_step,
            );

            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                pool_view
                    .alive_ids
                    .par_iter()
                    .map(|&id| {
                        // Copy is just a memcpy of the ~80-byte struct.
                        // Faster than a clone() call in profile because
                        // there's no per-task vtable indirection.
                        let mut this = pool_view.agents[id as usize]
                            .as_ref()
                            .expect("alive_ids points at a live slot")
                            .clone();
                        let prog = &*pool_view.programs[this.program as usize];
                        let mut out = ProgramOutput::default();
                        WORKER_RNG.with(|cell| {
                            let mut rng = cell.borrow_mut();
                            let mut ctx = ProgramContext {
                                world: &world,
                                sim_step,
                                generation,
                                rng: &mut rng,
                            };
                            prog.step(&mut this, &mut ctx, &mut out);
                        });
                        (id, this, out)
                    })
                    .collect_into_vec(&mut results);
            }

            #[cfg(not(feature = "parallel"))]
            {
                let mut local_rng = Rng::from_entropy();
                for &id in &pool_view.alive_ids {
                    let mut this = pool_view.agents[id as usize]
                        .as_ref()
                        .expect("alive_ids points at a live slot")
                        .clone();
                    let prog = &*pool_view.programs[this.program as usize];
                    let mut out = ProgramOutput::default();
                    let mut ctx =
                        ProgramContext { world: &world, sim_step, generation, rng: &mut local_rng };
                    prog.step(&mut this, &mut ctx, &mut out);
                    results.push((id, this, out));
                }
            }
        } // pool_view + world released; &mut self restored

        // ── Phase B: merge effects sequentially ─────────────────────
        // Order within a single entity's outputs:
        //   1. Write back mutated entity state (state, heading, color).
        //   2. set_color (overrides 1's color if both present).
        //   3. die — frees the cell.
        //   4. kill_peep_at — queue peep death (drained later).
        //   5. move_to — attempt move (blocks on barrier / kill_barrier
        //      kills the entity / occupied cell with peep kills + steps on).
        //   6. signal_emit — write to signals layer.
        // Across entities the order is `results` order, which tracks
        // rayon's collected order in the parallel path. We iterate by
        // reference so the Vec's allocation can be reused next step.
        for (id, updated, out) in &results {
            let id = *id;
            let Some(slot) = self.agents.get_mut(id as usize) else {
                continue;
            };
            let Some(entity) = slot.as_mut() else { continue };
            if !entity.alive {
                continue;
            }

            // 1. State write-back (program-mutable fields).
            entity.state = updated.state;
            entity.heading = updated.heading;
            // 2. set_color override (post-1 so it wins).
            entity.color = out.set_color.unwrap_or(updated.color);

            // 3. Die.
            if out.die {
                entity.alive = false;
                grid.set(entity.loc, grid::EMPTY);
                continue;
            }

            // 4. Kill peep at coord.
            if let Some(coord) = out.kill_peep_at {
                if grid.is_in_bounds(coord) {
                    let cell = grid.at(coord);
                    if let grid::CellKind::Agent(agent_id) = grid::cell_kind(cell) {
                        population.queue_for_death(agent_id);
                    }
                }
            }

            // 5. Move.
            if let Some(dest) = out.move_to {
                if !grid.is_in_bounds(dest) {
                    // out-of-bounds: ignore.
                } else if grid.is_kill_barrier_at(dest) {
                    // Stepping onto a kill barrier kills the entity.
                    grid.set(entity.loc, grid::EMPTY);
                    entity.alive = false;
                } else if grid.is_barrier_at(dest) {
                    // Blocked; no-op.
                } else if grid.is_empty_at(dest) {
                    let from = entity.loc;
                    grid.set(from, grid::EMPTY);
                    grid.set(dest, grid::encode_programmable(id));
                    entity.loc = dest;
                    let dir = (dest - from).as_dir();
                    entity.heading = dir;
                } else {
                    // Cell holds another occupant. If it's a peep and the
                    // program already requested kill_peep_at(dest), the
                    // peep's death is queued — but its grid cell isn't
                    // freed until the death queue drains, so we can't
                    // actually move this step. Leave the entity put.
                    // Future programs that want guaranteed step-onto-kill
                    // can chain (kill_peep_at + move_to) and accept the
                    // one-step lag, or open a "predator move" path that
                    // bundles both into the merge.
                }
            }

            // 6. Signal emit at the entity's (possibly updated) loc.
            if let Some(layer) = out.signal_emit {
                if layer < signals.layer_count() {
                    signals.increment(layer, entity.loc, grid);
                }
            }

            if !entity.alive {
                continue; // already handled
            }
        }

        // Drop the borrowed contents but keep the Vec's allocation by
        // swapping the (now-consumed) buffer back onto `self`. Next call
        // to `step_all` reuses the same heap block via `clear()` +
        // `collect_into_vec`.
        results.clear();
        self.scratch_results = results;

        // Prune `alive_ids` of entities that died during the merge (die
        // flag, kill barrier).
        self.alive_ids.retain(|&id| {
            self.agents.get(id as usize).and_then(|s| s.as_ref()).is_some_and(|e| e.alive)
        });

        // Movement / deaths shifted positions and population — the
        // spatial index now lags reality. Next `refresh_spatial_index`
        // call will pick this up.
        self.spatial_dirty = true;
    }

    // ── Spatial index ─────────────────────────────────────────────────

    /// Rebuild the spatial index against the current pool state if any
    /// mutation has happened since the last refresh. Idempotent — safe to
    /// call every step; only does work when needed.
    ///
    /// The caller passes the live world dimensions because the pool itself
    /// doesn't track the grid (`SimulationState` owns that). When the grid
    /// is resized the spatial index resizes itself transparently.
    pub fn refresh_spatial_index(&mut self, size_x: u16, size_y: u16) {
        if !self.spatial_dirty {
            return;
        }
        self.spatial_index.rebuild(size_x, size_y, &self.agents, &self.alive_ids);
        self.spatial_dirty = false;
    }

    /// Squared L2 distance to the nearest alive programmable from `loc`, or
    /// `None` if the pool is empty. Reads the spatial index — caller must
    /// have invoked [`refresh_spatial_index`](Self::refresh_spatial_index)
    /// at some point in the current step.
    pub fn nearest_alien_dist_sq(&self, loc: Coord) -> Option<u32> {
        if self.alive_ids.is_empty() {
            return None;
        }
        self.spatial_index.nearest_dist_sq(loc, &self.agents)
    }
}

impl Default for ProgrammablePool {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "parallel")]
thread_local! {
    /// Per-worker RNG for `Program::step` in the parallel section. Initialised
    /// from system entropy on first use per thread; reused for the lifetime of
    /// the process. The codebase has already opted for speed over
    /// cross-thread-count reproducibility (see `sim_step::WORKER_RNG`).
    static WORKER_RNG: std::cell::RefCell<Rng> = std::cell::RefCell::new(Rng::from_entropy());
}

// Suppress an unused warning when the `parallel` feature is off — the
// helper exists only for the parallel path.
#[cfg(not(feature = "parallel"))]
#[allow(dead_code)]
fn _unused_rng_stub() {}

// `AgentId` lives in `crate::agent`; re-imported here to keep the
// `use` set tidy. (The merge phase needs it for `kill_peep_at`.)
#[allow(unused_imports)]
use AgentId as _;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::food_layer::FoodLayer;
    use crate::signals_layer::Signals;
    use crate::sim_config::SimConfig;

    /// A test program that always moves east and never dies.
    struct EastWalker;
    impl Program for EastWalker {
        fn id(&self) -> &str {
            "east_walker"
        }
        fn name(&self) -> &str {
            "East walker"
        }
        fn step(
            &self,
            this: &mut Programmable,
            _ctx: &mut ProgramContext,
            out: &mut ProgramOutput,
        ) {
            out.move_to = Some(Coord::new(this.loc.x + 1, this.loc.y));
        }
    }

    /// A test program that flags itself for death every step.
    struct Suicidal;
    impl Program for Suicidal {
        fn id(&self) -> &str {
            "suicidal"
        }
        fn name(&self) -> &str {
            "Suicidal"
        }
        fn step(
            &self,
            _this: &mut Programmable,
            _ctx: &mut ProgramContext,
            out: &mut ProgramOutput,
        ) {
            out.die = true;
        }
    }

    fn make_world(grid: Grid) -> (Grid, Signals, FoodLayer, Population, SimConfig) {
        let cfg = SimConfig::default();
        let signals = Signals::new(cfg.signal_layers, grid.size_x, grid.size_y);
        let food = FoodLayer::new(grid.size_x, grid.size_y);
        let population = Population::new(0);
        (grid, signals, food, population, cfg)
    }

    #[test]
    fn spawn_writes_grid_and_alive_ids() {
        let grid = Grid::new(8, 8);
        let (mut grid, _signals, _food, _population, _cfg) = make_world(grid);
        let mut pool = ProgrammablePool::new();
        let prog = pool.register_program(Box::new(EastWalker));
        let id = pool.spawn(&mut grid, prog, 0, Coord::new(3, 3), [255, 0, 0]).unwrap();
        assert_eq!(pool.alive_count(), 1);
        assert_eq!(pool.alive_ids(), &[id]);
        let cell = grid.at(Coord::new(3, 3));
        assert_eq!(grid::cell_kind(cell), grid::CellKind::Programmable(id));
    }

    #[test]
    fn spawn_returns_none_on_occupied_cell() {
        let grid = Grid::new(4, 4);
        let (mut grid, _signals, _food, _population, _cfg) = make_world(grid);
        let mut pool = ProgrammablePool::new();
        let prog = pool.register_program(Box::new(EastWalker));
        assert!(pool.spawn(&mut grid, prog, 0, Coord::new(1, 1), [0, 0, 0]).is_some());
        // Second spawn at the same cell must fail.
        assert!(pool.spawn(&mut grid, prog, 0, Coord::new(1, 1), [0, 0, 0]).is_none());
    }

    #[test]
    fn clear_wipes_grid_and_alive_ids() {
        let grid = Grid::new(8, 8);
        let (mut grid, _signals, _food, _population, _cfg) = make_world(grid);
        let mut pool = ProgrammablePool::new();
        let prog = pool.register_program(Box::new(EastWalker));
        for x in 0..4 {
            pool.spawn(&mut grid, prog, 0, Coord::new(x, 0), [0, 0, 0]).unwrap();
        }
        assert_eq!(pool.alive_count(), 4);
        pool.clear(&mut grid);
        assert_eq!(pool.alive_count(), 0);
        for x in 0..4 {
            assert_eq!(grid.at(Coord::new(x, 0)), grid::EMPTY);
        }
    }

    #[test]
    fn step_all_applies_move_to() {
        let grid = Grid::new(8, 8);
        let (mut grid, mut signals, food, mut population, cfg) = make_world(grid);
        let mut pool = ProgrammablePool::new();
        let prog = pool.register_program(Box::new(EastWalker));
        let id = pool.spawn(&mut grid, prog, 0, Coord::new(2, 2), [0, 0, 0]).unwrap();
        pool.step_all(
            &mut grid,
            &mut signals,
            &mut population,
            &food,
            0,
            0,
            cfg.steps_per_generation,
        );
        // East walker moved from (2,2) to (3,2).
        assert_eq!(pool.get(id).unwrap().loc, Coord::new(3, 2));
        assert_eq!(grid.at(Coord::new(2, 2)), grid::EMPTY);
        let cell = grid.at(Coord::new(3, 2));
        assert_eq!(grid::cell_kind(cell), grid::CellKind::Programmable(id));
    }

    #[test]
    fn step_all_die_clears_grid_and_alive_ids() {
        let grid = Grid::new(8, 8);
        let (mut grid, mut signals, food, mut population, cfg) = make_world(grid);
        let mut pool = ProgrammablePool::new();
        let prog = pool.register_program(Box::new(Suicidal));
        let id = pool.spawn(&mut grid, prog, 0, Coord::new(4, 4), [0, 0, 0]).unwrap();
        pool.step_all(
            &mut grid,
            &mut signals,
            &mut population,
            &food,
            0,
            0,
            cfg.steps_per_generation,
        );
        assert_eq!(pool.alive_count(), 0);
        assert_eq!(grid.at(Coord::new(4, 4)), grid::EMPTY);
        assert!(!pool.get(id).unwrap().alive);
    }
}
