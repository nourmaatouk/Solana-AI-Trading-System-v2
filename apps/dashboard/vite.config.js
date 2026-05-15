import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ command }) => ({
  plugins: [react()],
  // In production the dashboard is mounted at /app/ inside FastAPI.
  // In dev (npm run dev) it runs at the root of the Vite server.
  base: command === 'build' ? '/app/' : '/',
  server: {
    port: 5173,
    proxy: {
      // Proxy /health, /signal, etc. to the FastAPI backend during dev
      '/health':        { target: 'http://localhost:8000', changeOrigin: true },
      '/signal':        { target: 'http://localhost:8000', changeOrigin: true },
      '/status':        { target: 'http://localhost:8000', changeOrigin: true },
      '/paper-trading': { target: 'http://localhost:8000', changeOrigin: true },
      '/preferences':   { target: 'http://localhost:8000', changeOrigin: true },
      '/system':        { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
}))
