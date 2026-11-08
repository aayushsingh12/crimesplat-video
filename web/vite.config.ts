import { defineConfig } from "vite";

export default defineConfig({
  server: { port: 5173 },
  // Spark ships WASM and workers; let Vite serve them as-is.
  optimizeDeps: { exclude: ["@sparkjsdev/spark"] },
});
