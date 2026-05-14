# biosim4-bevy

Native [Bevy](https://bevy.org) frontend for biosim4-rs, with parallel
simulation stepping via the `parallel` feature of `biosim4-core` (rayon under
the hood).

## Run

```sh
cargo run -p biosim4-bevy --release
```

Release mode is recommended — dev mode runs the simulation slower because the
crate's own code isn't optimized (dependencies are; see the
`[profile.dev.package."*"]` override in the workspace `Cargo.toml`).

> If you have both Homebrew's `rustc` and `rustup`, make sure `rustup`'s
> `cargo` is first on PATH (`PATH=~/.cargo/bin:$PATH`). Bevy 0.18 requires
> rustc ≥ 1.89.

## Controls

| Action                  | Input                                       |
| ----------------------- | ------------------------------------------- |
| Play / pause            | `Space`                                     |
| Select tool             | `I` inspect · `B` barrier · `K` kill · `R` reproduce (or `1`–`4`) |
| Pan camera              | Middle-mouse drag, or right-mouse drag      |
| Zoom (cursor-anchored)  | Scroll wheel                                |
| Inspect agent           | Click an agent with the Inspect tool        |
| Paint barriers          | Hold left-mouse with Barrier tool (right-click to erase) |
| Close inspector         | `Esc` or click the `✕` on the window        |

## What's in the UI

- **Top bar**: brand, live stats (gen / step / alive / grid size / threads /
  FPS / speed), challenge chip, telemetry toggle.
- **Floating toolbar**: tool picker above the grid.
- **Floating playback bar**: play, pause, step, step-generation, run-epoch,
  reset, speed slider, pixel-scale slider.
- **Hover badge**: shows the cell under the cursor and what's in it.
- **Telemetry overlay**: sparklines for survival rate / diversity / population
  across the last 64 generations.
- **Right panel**: tabbed sub-views —
  - **Stats** — same data as the top bar, plus last-epoch summary.
  - **Challenge** — dropdown of built-in survival challenges; apply or clear.
  - **Registry** — toggle sensors and actions on/off. Takes effect next
    generation (genome wiring is fixed mid-generation).
  - **Config** — drag-edit world / evolution / energy / environment fields,
    then **APPLY · RESET** to rebuild the simulation.
- **Agent inspector** — click an agent with the Inspect tool. Renders the
  agent's neural network with sensors → neurons → actions, edge color encoding
  weight sign and thickness encoding magnitude.

## Parallelism

The crate enables `biosim4-core/parallel`, which uses rayon for Phase 1
(per-agent sensor evaluation + neural feed-forward) and Phase 2 chunking
(action execution). The rayon pool is sized once at startup from the
`num_threads` field of the default config (4 by default — edit in
`sim.rs::default_config` or the Config tab).

> Mid-run thread-count changes to **Phase 1** require an app restart because
> rayon's global thread pool is initialize-once. **Phase 2** chunking picks
> up the new `num_threads` on the next step.

## Layout

```
src/
├── main.rs            App entry, plugin registration
├── theme.rs           Color tokens + egui visual install
├── sim.rs             SimulationState resource, step system, command queue
├── grid_render.rs     RGBA texture upload, challenge gizmo overlays
├── camera.rs          2D camera pan/zoom, fit-to-grid
├── tool.rs            Tool input dispatcher + keyboard shortcuts
└── ui/
    ├── mod.rs         UI plugin, shared widgets
    ├── topbar.rs      Top stats bar
    ├── toolbar.rs     Floating tool picker + hover badge
    ├── playback.rs    Bottom playback bar
    ├── telemetry.rs   Sparkline overlay
    ├── right_panel.rs Tabbed right side panel
    └── inspector.rs   Neural-network agent inspector
```
