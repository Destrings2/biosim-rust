import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  // Treat the wasm-pack output as a "library" we import directly. Vite handles
  // .wasm assets natively when imported by the wasm-pack glue.
  server: {
    fs: {
      // Allow serving files from one level up so we can reach pkg/ which
      // wasm-pack generates inside frontend/. (Already inside root, harmless.)
      allow: [path.resolve(__dirname)],
    },
    watch: {
      // Don't watch pkg/ — it's regenerated externally by wasm-pack.
      ignored: ["**/pkg/.build-stamp"],
    },
  },
  build: {
    target: "esnext", // top-level await for wasm init
    sourcemap: true,
  },
  optimizeDeps: {
    // Vite's pre-bundler chokes on the wasm-pack ES module unless we exclude it.
    exclude: ["biosim4_wasm"],
  },
});
