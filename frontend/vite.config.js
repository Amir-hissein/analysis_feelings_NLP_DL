import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In production the frontend is served by FastAPI, so it calls the API with a
// relative URL. In dev, proxy those calls to the local API on port 8000.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict': 'http://127.0.0.1:8000',
      '/health': 'http://127.0.0.1:8000',
    },
  },
})
