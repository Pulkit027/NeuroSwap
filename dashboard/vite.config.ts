import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// base "./" so the built dashboard works from any directory or static host.
export default defineConfig({
  base: "./",
  plugins: [react()],
});
