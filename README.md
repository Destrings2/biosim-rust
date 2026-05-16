# biosim4-rs

A Rust port of a genetic neural-net artificial-life simulator. Agents
evolve over generations on a 2D grid, guided by pluggable sensors,
actions, and survival challenges. Each agent's behavior is encoded in
a variable-length genome, compiled into a feed-forward neural net at
spawn time, and selected by whichever challenge is active.

Two frontends share one engine:

- **Bevy GUI** (`biosim4-bevy`) — live inspection, tool palette,
  parameter editing, fast-forward over multiple generations. Primary
  interface.
- **Headless CLI** (`biosim4-native`) — JSON config in, per-generation
  stats out. Used for batch runs and CI.

## Crates

| Crate | Role |
|---|---|
| [`biosim4-core`](crates/biosim4-core) | Platform-agnostic engine. Genome, neural net, world state, registries, stepping, reproduction. |
| [`biosim4-sensors`](crates/biosim4-sensors) | 40 built-in sensors. |
| [`biosim4-actions`](crates/biosim4-actions) | 23 built-in actions. |
| [`biosim4-challenges`](crates/biosim4-challenges) | 27 built-in survival challenges. |
| [`biosim4-breeds`](crates/biosim4-breeds) | Curated sensor/action/challenge presets. |
| [`biosim4-native`](crates/biosim4-native) | Headless CLI binary. |
| [`biosim4-bevy`](crates/biosim4-bevy) | Bevy + egui frontend. |

`biosim4-core` has no dependency on the catalogue crates. Adding a
sensor in `biosim4-sensors` does not trigger a core rebuild.

## Quick start

```sh
# GUI
cargo run -p biosim4-bevy --release

# Headless CLI (uses built-in defaults if no --config is passed)
cargo run -p biosim4-native --release -- --generations 200
```

Use `--release`. Without it the engine runs roughly 10× slower because
the workspace's own code is unoptimized in dev mode (dependencies are
optimized; see `[profile.dev.package."*"]` in `Cargo.toml`).

CLI flags:

```text
biosim4-native [--config FILE] [--generations N] [--threads N]
               [--seed N] [--quiet] [--verbose]
```

`--threads 0` uses every available core. `--seed 0` draws from system
entropy. CLI flags override values in `--config`.

## Toolchain

Rust 1.95, pinned in [`rust-toolchain.toml`](rust-toolchain.toml).
`rustup` installs the toolchain on first build. If both Homebrew's
`rustc` and `rustup`'s are on `PATH`, put rustup first:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
```

The Bevy frontend needs system libraries on Linux:

```sh
sudo apt-get install -y libasound2-dev libudev-dev libwayland-dev \
                        libxkbcommon-dev pkg-config
```

## Tests and lints

```sh
cargo test --workspace --all-features
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo deny check
```

The core engine has 17 integration test files covering sensors,
actions, challenges, registry behavior, parallel determinism,
programmable entities, and end-to-end simulation. The GUI and CLI
crates have no tests.

`cargo fmt --check` runs in CI but does not block. The codebase uses
single-line guard clauses (`if cond { return; }`) that stable rustfmt
cannot preserve.

## Determinism

Determinism is conditional on thread count:

- `num_threads = 1` (or the `parallel` feature off) — fully
  reproducible at a fixed `rng_seed`. Same seed produces the same
  evolution byte-for-byte.
- `num_threads > 1` with `parallel` on — intentionally
  non-deterministic. rayon merges chunk-local move/death queues in
  work-stealing order. Trades roughly 3× throughput at 8 threads for
  bit-exact reproducibility. Per-agent sensor randomness (Phase 1)
  stays reproducible regardless of thread count; only Phase 2 action
  draws diverge.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
contract and [`crates/biosim4-core/tests/parallel_determinism.rs`](crates/biosim4-core/tests/parallel_determinism.rs)
for the executable specification.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module DAG, generation
  lifecycle, cross-cutting patterns (registry commit, deferred queues,
  raw-pointer Phase 1/2 split, determinism contract).
- [`docs/SIMULATION_LOOP.md`](docs/SIMULATION_LOOP.md) — per-step
  execution path, `step_one_agent` two-phase design, `feed_forward`
  invariant, generation transition.
- [`docs/CONFIG.md`](docs/CONFIG.md) — `SimConfig` field reference
  with defaults and effective ranges.
- [`docs/EXTENDING.md`](docs/EXTENDING.md) — how to add a sensor,
  action, challenge, breed, or programmable entity.
- [`docs/BUILTINS.md`](docs/BUILTINS.md) — catalogue of every built-in
  sensor, action, challenge, and breed.
- [`crates/biosim4-bevy/README.md`](crates/biosim4-bevy/README.md) —
  GUI controls and panel layout.
- [`crates/biosim4-core/src/programmable/README.md`](crates/biosim4-core/src/programmable/README.md)
  — programmable (non-evolved, scripted) entity developer guide.

## License

Apache License 2.0. See [LICENSE](LICENSE).
