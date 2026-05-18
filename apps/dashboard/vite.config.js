import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const BACKEND_URL = process.env.BACKEND_URL ?? 'http://localhost:8000';

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // In production the dashboard is mounted at /app/ inside FastAPI.
  // In dev (npm run dev) it runs at the root of the Vite server.
  base: command === 'build' ? '/app/' : '/',
  server: {
    port: 5173,
    proxy: {
      // Proxy /health, /signal, etc. to the FastAPI backend during dev
      '/health':        { target: BACKEND_URL, changeOrigin: true },
      '/signal':        { target: BACKEND_URL, changeOrigin: true },
      '/status':        { target: BACKEND_URL, changeOrigin: true },
      '/paper-trading': { target: BACKEND_URL, changeOrigin: true },
      '/preferences':   { target: BACKEND_URL, changeOrigin: true },
      '/system':        { target: BACKEND_URL, changeOrigin: true },
    },
  },
}))
