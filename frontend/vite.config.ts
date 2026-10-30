import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The Flask backend runs on :5000; /api is proxied so no CORS setup is needed in development.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5173, proxy: { "/api": process.env.API_URL || "http://localhost:5000" } },
});
