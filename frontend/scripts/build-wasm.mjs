#!/usr/bin/env node
// Build the biosim4-wasm crate via wasm-pack.
//
// Skips the build when nothing under `crates/` has changed since the last
// successful build (cargo + wasm-pack already do their own incremental work,
// but spawning wasm-pack itself takes ~1s — this short-circuits that.)
//
// Flags:
//   --dev      build with --dev profile (faster, no wasm-opt)
//   --watch    rebuild on .rs / Cargo.toml changes (uses fs.watch)
//   --force    skip the freshness check

import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import { existsSync, statSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const FRONTEND = path.resolve(__dirname, "..");
const ROOT = path.resolve(FRONTEND, "..");
const WASM_CRATE = path.resolve(ROOT, "crates/biosim4-wasm");
const CORE_CRATE = path.resolve(ROOT, "crates/biosim4-core");
const PKG_DIR = path.resolve(FRONTEND, "pkg");
const STAMP = path.resolve(PKG_DIR, ".build-stamp");

// Prefer rustup-managed toolchain over any system-wide rustc (e.g. Homebrew)
// — rustup's toolchain has the wasm32-unknown-unknown std lib, the others
// often don't.
const cargoBin = path.join(os.homedir(), ".cargo/bin");
if (existsSync(cargoBin)) {
  process.env.PATH = `${cargoBin}${path.delimiter}${process.env.PATH || ""}`;
}

const args = new Set(process.argv.slice(2));
const isDev = args.has("--dev");
const isWatch = args.has("--watch");
const isForce = args.has("--force");

// ── Freshness check ──────────────────────────────────────────────────────
async function newestSourceMtime() {
  let newest = 0;
  async function walk(dir) {
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const e of entries) {
      if (e.name === "target" || e.name === "node_modules" || e.name.startsWith(".")) continue;
      const full = path.join(dir, e.name);
      if (e.isDirectory()) {
        await walk(full);
      } else if (e.isFile() && (e.name.endsWith(".rs") || e.name === "Cargo.toml" || e.name === "Cargo.lock")) {
        const st = await fs.stat(full);
        if (st.mtimeMs > newest) newest = st.mtimeMs;
      }
    }
  }
  await walk(WASM_CRATE);
  await walk(CORE_CRATE);
  // Cargo.lock at workspace root affects builds too
  const lock = path.resolve(ROOT, "Cargo.lock");
  if (existsSync(lock)) {
    const st = statSync(lock);
    if (st.mtimeMs > newest) newest = st.mtimeMs;
  }
  return newest;
}

async function isFresh() {
  if (isForce) return false;
  if (!existsSync(STAMP)) return false;
  // Stamp file format: `<profile>\n<isoTimestamp>`. Mismatched profile
  // forces a rebuild even when sources are otherwise unchanged so a `--dev`
  // pkg/ doesn't accidentally ship in `npm run build`.
  const stampContent = await fs.readFile(STAMP, "utf8");
  const [stampedProfile] = stampContent.split("\n", 1);
  const wantProfile = isDev ? "dev" : "release";
  if (stampedProfile.trim() !== wantProfile) return false;
  const stampMtime = (await fs.stat(STAMP)).mtimeMs;
  const sourceMtime = await newestSourceMtime();
  return stampMtime >= sourceMtime;
}

// ── wasm-pack runner ─────────────────────────────────────────────────────
function runWasmPack() {
  return new Promise((resolve, reject) => {
    const profile = isDev ? "--dev" : "--release";
    const wasmPackArgs = [
      "build",
      WASM_CRATE,
      "--target", "web",
      "--out-dir", PKG_DIR,
      "--out-name", "biosim4_wasm",
      profile,
    ];
    if (isDev) {
      wasmPackArgs.push("--features", "debug-hooks");
    }
    console.log(`\n▶ wasm-pack ${wasmPackArgs.join(" ")}\n`);

    const child = spawn("wasm-pack", wasmPackArgs, {
      stdio: "inherit",
      shell: false,
    });
    child.on("error", (err) => {
      if (err.code === "ENOENT") {
        console.error(
          "\n✗ wasm-pack not found. Install with:\n  cargo install wasm-pack\n" +
          "  or: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh\n"
        );
      }
      reject(err);
    });
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`wasm-pack exited with code ${code}`));
    });
  });
}

async function build() {
  if (await isFresh()) {
    console.log("✓ wasm pkg/ is up-to-date with crate sources — skipping build.");
    return;
  }
  await runWasmPack();
  await fs.mkdir(PKG_DIR, { recursive: true });
  const profile = isDev ? "dev" : "release";
  await fs.writeFile(STAMP, `${profile}\n${new Date().toISOString()}\n`);
  console.log(`\n✓ wasm-pack ${profile} build finished.\n`);
}

// ── Watch mode ───────────────────────────────────────────────────────────
async function watch() {
  await build();
  console.log("👀 watching for changes…");
  let pending = false;
  let timer = null;
  const trigger = () => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(async () => {
      if (pending) return;
      pending = true;
      try {
        await build();
      } catch (e) {
        console.error("✗ build failed:", e.message);
      } finally {
        pending = false;
      }
    }, 200);
  };
  for (const dir of [WASM_CRATE, CORE_CRATE]) {
    fs.watch?.(dir, { recursive: true }, trigger);
  }
}

(async () => {
  try {
    if (isWatch) await watch();
    else await build();
  } catch (e) {
    console.error("\n✗", e.message);
    process.exit(1);
  }
})();
