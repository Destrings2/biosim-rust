# biosim4-rs

A Rust port of a genetic neural-net artificial-life simulator. Agents
evolve over generations on a 2D grid, guided by pluggable sensors,
actions, and survival challenges.

The same simulation engine powers two frontends:

- **Native GUI** (`biosim4-bevy`) — Bevy + egui frontend with live
  inspection, parameter tweaking, and a GPU fast-forward path.
  This is the primary interface.
- **Native CLI** (`biosim4-native`) — JSON config in, stats out.

## Crate map

| Crate | Role |
|---|---|
| [`biosim4-core`](crates/biosim4-core) | Platform-agnostic engine. All genetics, neural nets, environment, stepping, and reproduction logic. |
| [`biosim4-native`](crates/biosim4-native) | CLI binary. Reads a JSON config, runs the simulation, prints stats. |
| [`biosim4-bevy`](crates/biosim4-bevy) | Bevy frontend. Parallel stepping via rayon, optional GPU fast-forward. |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the module DAG,
generation lifecycle, and cross-cutting patterns
(`registry/commit`, deferred queues, `step_one_agent`'s two-phase
design, determinism contract). [`docs/SIMULATION_LOOP.md`](docs/SIMULATION_LOOP.md)
documents the per-step execution path.

## Quick start

```sh
# Native GUI (recommended for exploration)
cargo run -p biosim4-bevy --release

# Native CLI
cargo run -p biosim4-native --release -- --config configs/default.json
```

Release mode is strongly recommended for both native binaries. Dev mode
leaves the crate code unoptimized for incremental rebuilds; the
simulation runs ~10× slower without `--release`.

## Toolchain

The workspace pins Rust 1.95 in [`rust-toolchain.toml`](rust-toolchain.toml).
On first build, `rustup` installs the toolchain automatically. If you
have a Homebrew rustc and a rustup rustc, make sure rustup's `cargo`
is first on `PATH`:

```sh
export PATH="$HOME/.cargo/bin:$PATH"
```

## Testing

```sh
cargo test --workspace --all-features
cargo clippy --workspace --all-targets -- -D warnings
```

The core engine has ~14 integration test files covering sensors,
actions, challenges, registry behavior, parallel determinism, and
end-to-end simulation. The GUI and CLI crates currently have no tests
(tracked in the tech-debt audit).

## Code style

`rustfmt.toml` documents the project style (max 100, max small
heuristics). The CI `fmt` job runs `cargo fmt --check` but does not
block — the codebase deliberately uses dense single-line guard
clauses (`if cond { return; }`) that stable rustfmt cannot preserve.

## Determinism

Determinism is conditional on thread count:

- `num_threads = 1` → fully reproducible at a fixed `rng_seed`.
- `num_threads > 1` → intentionally non-deterministic (rayon
  work-stealing). Trades ~3× throughput at 8 threads.

The GPU fast-forward path uses a different RNG and is **never**
bit-identical to the CPU path. See `docs/ARCHITECTURE.md` for the
full contract.

## Extending

Adding a sensor, action, or challenge is a single trait impl. See
"Extension Points" in `docs/ARCHITECTURE.md`.

## License

(Not yet specified.)
