import { DirtyManager } from './dirty-manager';

export interface CanvasDimensions {
  width: number;
  height: number;
}

/**
 * Syncs canvas backing buffer resolution to logical dimensions * capped DPR
 * without mutating inline CSS styles or reading layout during render.
 */
export function syncCanvasBufferSize(
  canvas: HTMLCanvasElement,
  width: number,
  height: number,
  maxDpr: number = 2
): boolean {
  if (width <= 0 || height <= 0) return false;
  const dpr = Math.min(window.devicePixelRatio || 1, maxDpr);
  const targetW = Math.floor(width * dpr);
  const targetH = Math.floor(height * dpr);
  const resized = canvas.width !== targetW || canvas.height !== targetH;
  if (resized) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  const ctx = canvas.getContext('2d');
  if (ctx) {
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  return resized;
}

/**
 * Observes an element's content-box dimensions using ResizeObserver contentRect
 * to eliminate synchronous getBoundingClientRect() / clientWidth forced reflows.
 */
export function observeElementSize(
  element: HTMLElement | null,
  onSize: (dims: CanvasDimensions) => void
): () => void {
  if (!element || typeof ResizeObserver === 'undefined') {
    return () => {};
  }

  let hasObserved = false;
  const ro = new ResizeObserver((entries) => {
    for (const entry of entries) {
      hasObserved = true;
      onSize({
        width: entry.contentRect.width,
        height: entry.contentRect.height,
      });
    }
  });
  ro.observe(element);

  // Fallback only for mocked ResizeObservers (e.g., jsdom tests) that never fire callbacks.
  // Using setTimeout ensures real browsers deliver ResizeObserver after frame-1 layout first,
  // avoiding forced synchronous getBoundingClientRect() reflows before initial paint.
  const fallbackId = window.setTimeout(() => {
    if (!hasObserved) {
      const rect = element.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        onSize({ width: rect.width, height: rect.height });
      }
    }
  }, 120);

  return () => {
    window.clearTimeout(fallbackId);
    ro.disconnect();
  };
}

/**
 * Schedules a single requestAnimationFrame callback only when DirtyManager is dirty,
 * maintaining 0% idle CPU and zero idle rAF ticks.
 */
export function subscribeDirtyRender(
  dirtyManager: DirtyManager,
  renderFrame: () => void
): () => void {
  let animId: number | null = null;
  let disposed = false;

  const schedule = () => {
    if (disposed || animId !== null) return;
    animId = requestAnimationFrame(() => {
      animId = null;
      if (disposed) return;
      if (dirtyManager.checkAndClear()) {
        renderFrame();
      }
    });
  };

  const unsubscribe = dirtyManager.onDirty(schedule);
  if (dirtyManager.getDirty()) {
    schedule();
  }

  return () => {
    disposed = true;
    unsubscribe();
    if (animId !== null) {
      cancelAnimationFrame(animId);
      animId = null;
    }
  };
}
