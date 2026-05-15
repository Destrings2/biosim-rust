//! Seedable RNG wrapper.
//!
//! Wraps `rand::rngs::SmallRng` (a fast, non-cryptographic PRNG). When
//! `SimConfig.rng_seed != 0`, the simulation uses `Rng::seeded(seed)` for
//! full reproducibility. When `rng_seed == 0`, `Rng::from_entropy()` pulls
//! from OS entropy.
//!
//! `fork(offset)` derives a child `Rng` by XORing the parent's next `u64`
//! with `offset`. In `step_one_agent`, each agent forks the simulation's
//! main `Rng` using its `AgentId` as the offset, giving every agent a
//! per-step independent stochastic stream without locks or shared state.

use rand::rngs::SmallRng;
use rand::{Rng as RandRng, SeedableRng};

/// Lightweight, seedable RNG wrapper used throughout the simulation.
pub struct Rng {
    inner: SmallRng,
}

impl Rng {
    /// Deterministic seed.
    pub fn seeded(seed: u64) -> Self {
        Self { inner: SmallRng::seed_from_u64(seed) }
    }

    /// Non-deterministic seed from system entropy.
    pub fn from_entropy() -> Self {
        Self { inner: SmallRng::from_entropy() }
    }

    /// Fork a child RNG from a deterministic offset of this one's next u64.
    pub fn fork(&mut self, offset: u64) -> Self {
        let base: u64 = self.inner.gen();
        Self { inner: SmallRng::seed_from_u64(base ^ offset) }
    }

    pub fn gen_u32(&mut self) -> u32 {
        self.inner.gen()
    }

    /// Uniform integer in `[lo, hi)` (exclusive upper bound — matches
    /// `gen_range_usize` and Rust's standard half-open range convention).
    pub fn gen_range_u32(&mut self, lo: u32, hi: u32) -> u32 {
        self.inner.gen_range(lo..hi)
    }

    pub fn gen_range_usize(&mut self, lo: usize, hi: usize) -> usize {
        self.inner.gen_range(lo..hi)
    }

    pub fn gen_f32(&mut self) -> f32 {
        self.inner.gen()
    }

    pub fn gen_bool(&mut self, probability: f32) -> bool {
        self.inner.gen::<f32>() < probability
    }
}

impl rand::RngCore for Rng {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest)
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.inner.try_fill_bytes(dest)
    }
}
