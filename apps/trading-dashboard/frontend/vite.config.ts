import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { nodePolyfills } from 'vite-plugin-node-polyfills';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:3001';
const BACKEND_WS  = BACKEND_URL.replace(/^http/, 'ws');
const FRONTEND_PORT = Number(process.env.FRONTEND_PORT ?? process.env.VITE_FRONTEND_PORT ?? 5174);

export default defineConfig({
  plugins: [react(), nodePolyfills()],
  server: {
    port: Number.isFinite(FRONTEND_PORT) ? FRONTEND_PORT : 5174,
    proxy: {
      '/api': {
        target: BACKEND_URL,
        changeOrigin: true,
      },
      '/ws': {
        target: BACKEND_WS,
        ws: true,
      },
    },
  },
});
