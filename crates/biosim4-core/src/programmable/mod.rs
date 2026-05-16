//! Programmable agents owned by challenges.
//!
//! A [`Programmable`] is a non-evolved entity placed in the world by a
//! challenge. It occupies a grid cell (under a separate id range — see
//! [`grid::PROGRAMMABLE_FLAG`]), is stepped every `sim_step` by a
//! [`Program`] (a Rust impl picked at registration time), and can be
//! perceived by peeps through the generic `longprobe_alien_fwd` sensor.
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

pub mod library;

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

    /// Called once when the entity is spawned. Override to initialise `this.state`.
    ///
    /// Runs sequentially at gen-start before any `World` snapshot exists,
    /// so only the entity and a local rng are available.
    fn on_spawn(&self, _this: &mut Programmable, _rng: &mut Rng) {}

    /// Called once when the entity is despawned (sequential).
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
}

impl ProgrammablePool {
    pub fn new() -> Self {
        Self {
            agents: vec![None], // slot 0 reserved
            programs: Vec::new(),
            program_index: HashMap::new(),
            alive_ids: Vec::new(),
            scratch_results: Vec::new(),
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

        // on_spawn runs sequentially — no parallel-safety concerns.
        // A scratch Rng is sufficient here; no World snapshot exists at gen-start.
        let prog_idx = program as usize;
        if let Some(entity) = self.agents[id as usize].as_mut() {
            let mut scratch_rng = Rng::from_entropy();
            let prog = &*self.programs[prog_idx];
            prog.on_spawn(entity, &mut scratch_rng);
        }

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
            //
            // When `kill_peep_at` and `move_to` target the same cell (the
            // predator-eats-and-steps case), apply the kill inline and
            // free the grid cell so step 5 below sees it as empty.
            // Otherwise the death stays queued for the end-of-step drain.
            //
            // The `queue_for_death` push runs unconditionally so
            // `alive_ids` gets pruned during the next `drain_death_queue`;
            // `apply_deaths` skips the already-dead inline kills, but
            // still walks the retain pass that updates the cache.
            if let Some(coord) = out.kill_peep_at {
                if grid.is_in_bounds(coord) {
                    if let grid::CellKind::Agent(agent_id) = grid::cell_kind(grid.at(coord)) {
                        if out.move_to == Some(coord) {
                            if let Some(peep) = population.get_mut(agent_id) {
                                if peep.alive {
                                    peep.alive = false;
                                    grid.set(coord, grid::EMPTY);
                                }
                            }
                        }
                        population.queue_for_death(agent_id);
                    }
                }
            }

            // 5. Move. Use `grid.wrap` so a target that crossed a
            // wrapping edge lands on the canonical cell; on the bounded
            // plane this is identical to the old `is_in_bounds` check.
            if let Some(raw_dest) = out.move_to {
                let Some(dest) = grid.wrap(raw_dest) else {
                    // OOB on a non-wrapping axis: ignore.
                    if let Some(layer) = out.signal_emit {
                        if layer < signals.layer_count() {
                            signals.increment(layer, entity.loc, grid);
                        }
                    }
                    continue;
                };
                if grid.is_kill_barrier_at(dest) {
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
                    // Derive heading from the wrap-aware delta so seam
                    // crossings don't flip the entity's heading 180°.
                    let (dx, dy) = grid.delta(from, dest);
                    entity.heading = crate::types::Coord::new(dx as i16, dy as i16).as_dir();
                } else {
                    // Occupied by something other than the just-killed peep;
                    // leave the entity in place.
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

    /// Test program that issues `kill_peep_at` + `move_to` on the same cell
    /// (the cell one east of itself), modelling a predator that eats and
    /// steps onto its prey in one tick.
    struct EatAndStepEast;
    impl Program for EatAndStepEast {
        fn id(&self) -> &str {
            "eat_and_step_east"
        }
        fn name(&self) -> &str {
            "Eat-and-step (test)"
        }
        fn step(
            &self,
            this: &mut Programmable,
            _ctx: &mut ProgramContext,
            out: &mut ProgramOutput,
        ) {
            let target = Coord::new(this.loc.x + 1, this.loc.y);
            out.kill_peep_at = Some(target);
            out.move_to = Some(target);
        }
    }

    #[test]
    fn step_all_predator_kills_and_steps_in_one_merge() {
        // Without the inline-kill path, this test would observe the
        // predator still at (3, 4) and the peep id still at (4, 4): the
        // merge would queue the peep's death but the move would see an
        // occupied cell and silently block. The fix applies the kill
        // before step 5's empty-check so the move resolves atomically.
        use crate::agent::Agent;
        use crate::genome::neural_net::{create_wiring, WiringConfig};
        use crate::genome::ops::make_random_genome;
        use crate::rng::Rng;

        let grid = Grid::new(8, 8);
        let (mut grid, mut signals, food, mut population, cfg) = make_world(grid);
        let mut pool = ProgrammablePool::new();

        // Real `Agent` in the population — the inline-kill path looks the
        // peep up via `population.get_mut` and only frees the grid cell
        // when it finds an alive entry, so a placeholder grid id isn't
        // enough. Wiring stays tiny (4 sensors / 4 actions) because the
        // agent never actually runs.
        let mut rng = Rng::seeded(7);
        let peep_loc = Coord::new(4, 4);
        let genome = make_random_genome(&cfg, &mut rng);
        let wiring = create_wiring(
            &genome,
            WiringConfig { sensor_count: 4, action_count: 4, max_neurons: 2 },
        );
        let peep_id = population.spawn(Agent::new(population.next_id(), peep_loc, genome, wiring));
        grid.set(peep_loc, peep_id);

        let prog = pool.register_program(Box::new(EatAndStepEast));
        let predator_id = pool.spawn(&mut grid, prog, 0, Coord::new(3, 4), [200, 0, 0]).unwrap();
        pool.step_all(
            &mut grid,
            &mut signals,
            &mut population,
            &food,
            0,
            0,
            cfg.steps_per_generation,
        );

        // Predator's old cell is now empty.
        assert_eq!(grid.at(Coord::new(3, 4)), grid::EMPTY);
        // The peep's cell now holds the predator.
        assert_eq!(
            grid::cell_kind(grid.at(peep_loc)),
            grid::CellKind::Programmable(predator_id),
            "predator should occupy the just-eaten peep's cell"
        );
        assert_eq!(pool.get(predator_id).unwrap().loc, peep_loc);
        // The peep died inline; the end-of-step `drain_death_queue` call
        // in sim_step prunes `alive_ids`, but the inline `peep.alive =
        // false` is already visible here.
        assert!(
            !population.get(peep_id).unwrap().alive,
            "peep should be marked dead by the inline kill"
        );
        assert!(
            population.death_queue.contains(&peep_id),
            "peep id must be queued so the end-of-step drain prunes alive_ids"
        );
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
