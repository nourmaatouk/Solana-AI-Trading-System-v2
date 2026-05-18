import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:3001';
const BACKEND_WS  = BACKEND_URL.replace(/^http/, 'ws');

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
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
