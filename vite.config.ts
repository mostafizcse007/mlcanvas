import { defineConfig, Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

/**
 * Inlines the entry CSS bundle into dist/index.html during production builds
 * to eliminate render-blocking stylesheet round-trips while keeping KaTeX CSS lazy.
 */
function inlineEntryCssPlugin(): Plugin {
  return {
    name: 'inline-entry-css',
    apply: 'build',
    enforce: 'post',
    transformIndexHtml(html, ctx) {
      if (!ctx.bundle) return html;
      return html.replace(
        /<link\s+rel="stylesheet"[^>]*href="([^"]*\/assets\/index-[^"]+\.css)"[^>]*>/g,
        (fullTag, href) => {
          const assetFileName = href.replace(/^\//, '');
          const chunk = ctx.bundle?.[assetFileName];
          if (chunk && chunk.type === 'asset' && typeof chunk.source === 'string') {
            delete ctx.bundle[assetFileName];
            return `<style>${chunk.source}</style>`;
          }
          return fullTag;
        }
      );
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    inlineEntryCssPlugin(),
  ],
  worker: {
    format: 'es',
  },
  build: {
    target: 'esnext',
    cssCodeSplit: true,
    chunkSizeWarningLimit: 600,
    modulePreload: {
      resolveDependencies(_filename, deps) {
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
