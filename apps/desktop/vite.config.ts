import { defineConfig } from "vite";

// https://v2.tauri.app/start/frontend/vite/
const host = process.env.TAURI_DEV_HOST;

export default defineConfig({
  // Keep Rust errors visible in the terminal.
  clearScreen: false,
  server: {
    // tauri.conf.json `build.devUrl` expects this port.
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host ? { protocol: "ws", host, port: 1421 } : undefined,
    watch: { ignored: ["**/src-tauri/**"] },
  },
  envPrefix: ["VITE_", "TAURI_ENV_*"],
  build: {
    // Tauri uses WebKit on macOS and Linux, WebView2 on Windows.
    target: process.env.TAURI_ENV_PLATFORM === "windows" ? "chrome105" : "safari13",
    sourcemap: !!process.env.TAURI_ENV_DEBUG,
  },
});
