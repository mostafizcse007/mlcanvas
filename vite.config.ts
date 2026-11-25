import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
    cssCodeSplit: true,
    chunkSizeWarningLimit: 600,
    modulePreload: {
      resolveDependencies(filename, deps) {
        // Prevent preloading heavy non-initial stages on initial page entry
        return deps.filter(
          (dep) =>
            !dep.includes('stage02-solvers') &&
            !dep.includes('stage03-solvers') &&
            !dep.includes('stage04-solvers') &&
            !dep.includes('stage05-solvers') &&
            !dep.includes('stage06-solvers') &&
            !dep.includes('stage07-solvers') &&
            !dep.includes('Stage02Canvas') &&
            !dep.includes('Stage03Canvas') &&
            !dep.includes('Stage04Canvas') &&
            !dep.includes('Stage05Canvas') &&
            !dep.includes('Stage06Canvas') &&
            !dep.includes('Stage07Canvas') &&
            !dep.includes('CommandPalette') &&
            !dep.includes('vendor-katex')
        );
      },
    },
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/react/') || id.includes('node_modules/react-dom/')) {
            return 'vendor-react';
          }
          if (id.includes('node_modules/katex/')) {
            return 'vendor-katex';
          }
          if (id.includes('node_modules/lucide-react/')) {
            return 'vendor-icons';
          }
          if (id.includes('src/algorithms/stage01') || id.includes('src\\algorithms\\stage01')) {
            return 'stage01-solvers';
          }
          if (id.includes('src/algorithms/stage02') || id.includes('src\\algorithms\\stage02')) {
            return 'stage02-solvers';
          }
          if (id.includes('src/algorithms/stage03') || id.includes('src\\algorithms\\stage03')) {
            return 'stage03-solvers';
          }
          if (id.includes('src/algorithms/stage04') || id.includes('src\\algorithms\\stage04')) {
            return 'stage04-solvers';
          }
          if (id.includes('src/algorithms/stage05') || id.includes('src\\algorithms\\stage05')) {
            return 'stage05-solvers';
          }
          if (id.includes('src/algorithms/stage06') || id.includes('src\\algorithms\\stage06')) {
            return 'stage06-solvers';
          }
          if (id.includes('src/algorithms/stage07') || id.includes('src\\algorithms\\stage07')) {
            return 'stage07-solvers';
          }
        },
      },
    },
  },
  server: {
    port: 3000,
    open: true,
    allowedHosts: [
      '.trycloudflare.com',
    ],
  },
  preview: {
    port: 3000,
    allowedHosts: [
      '.trycloudflare.com',
    ],
  },
});