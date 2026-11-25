import { mulberry32, gaussianRandom } from '../utils/prng';
import { SpatialVisionDataset, SpatialVisionSample } from '../types/algorithm';

export type VisionPresetKind = 'digit' | 'edges' | 'corner-cross' | 'checkerboard' | 'custom' | 'vertical' | 'horizontal';

export const CLASS_LABELS: [string, string, string, string] = [
  'Vertical Bars',
  'Horizontal Bars',
  'Diagonal Crosses',
  'Hollow Rings',
];

export function createBlankGrid(size: number = 16): Float32Array {
  return new Float32Array(size * size);
}

/**
 * Canonical 16x16 Presets (Values in [0, 1])
 */
export function getPresetGrid(kind: VisionPresetKind, variant: 'default' | 'digit0' | 'digit7' = 'default'): Float32Array {
  const grid = new Float32Array(256);

  if (kind === 'digit') {
    if (variant === 'digit0') {
      // Draw Digit 0
      for (let r = 2; r <= 13; r++) {
        for (let c = 3; c <= 12; c++) {
          const isBorder = (r === 2 || r === 13) && c >= 4 && c <= 11;
          const isSide = (c === 3 || c === 12) && r >= 3 && r <= 12;
          if (isBorder || isSide) {
            grid[r * 16 + c] = 1.0;
          }
        }
      }
    } else {
      // Draw Digit 7 (crisp top horizontal bar and diagonal stroke)
      for (let c = 3; c <= 12; c++) {
        grid[2 * 16 + c] = 1.0;
        grid[3 * 16 + c] = 0.9;
      }
      for (let r = 3; r <= 13; r++) {
        const c = Math.round(12 - (r - 3) * 0.7);
        if (c >= 0 && c < 16) {
          grid[r * 16 + c] = 1.0;
          if (c > 0) grid[r * 16 + c - 1] = Math.max(grid[r * 16 + c - 1], 0.7);
        }
      }
    }
  } else if (kind === 'edges') {
    // Left half: vertical edge (columns 0..7 high intensity, columns 8..15 low)
    // Bottom right quadrant: sharp horizontal edge
    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 16; c++) {
        if (c < 8) {
          grid[r * 16 + c] = 1.0;
        } else if (r >= 8) {
          grid[r * 16 + c] = 0.85;
        }
      }
    }
  } else if (kind === 'corner-cross') {
    // Cross centered at (8, 8)
    for (let i = 2; i <= 13; i++) {
      grid[8 * 16 + i] = 1.0; // horizontal line
      grid[i * 16 + 8] = 1.0; // vertical line
    }
    // L-corner in top-left
    for (let c = 2; c <= 5; c++) grid[2 * 16 + c] = 0.9;
    for (let r = 2; r <= 5; r++) grid[r * 16 + 2] = 0.9;
  } else if (kind === 'checkerboard') {
    // High-frequency 2x2 checkerboard pattern
    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 16; c++) {
        const check = (Math.floor(r / 2) + Math.floor(c / 2)) % 2;
        grid[r * 16 + c] = check === 0 ? 1.0 : 0.0;
      }
    }
  } else if (kind === 'vertical') {
    return generateSyntheticVisionClass(0, 0, 42, 0.0);
  } else if (kind === 'horizontal') {
    return generateSyntheticVisionClass(1, 0, 42, 0.0);
  }

  return grid;
}

/**
 * Brush drawing tool on 16x16 grid
 */
export function applyBrush(
  grid: Float32Array,
  col: number,
  row: number,
  value: number = 1.0,
  brushSize: 1 | 2 = 1
): Float32Array {
  const next = new Float32Array(grid);
  const radius = brushSize - 1;

  for (let dr = -radius; dr <= radius; dr++) {
    for (let dc = -radius; dc <= radius; dc++) {
      const r = row + dr;
      const c = col + dc;
      if (r >= 0 && r < 16 && c >= 0 && c < 16) {
        next[r * 16 + c] = Math.max(0, Math.min(1, value));
      }
    }
  }
  return next;
}

/**
 * Shifts grid horizontally to test translation invariance in pooling vs stride
 */
export function shiftGridRight(grid: Float32Array, shiftPx: number = 1): Float32Array {
  const next = new Float32Array(256);
  for (let r = 0; r < 16; r++) {
    for (let c = 0; c < 16; c++) {
      const srcCol = c - shiftPx;
      if (srcCol >= 0 && srcCol < 16) {
        next[r * 16 + c] = grid[r * 16 + srcCol];
      } else {
        next[r * 16 + c] = 0.0; // zero padding
      }
    }
  }
  return next;
}

/**
 * Applies contrast scaling, noise injection, and optional inversion
 */
export function applyNoiseAndContrast(
  grid: Float32Array,
  noise: number = 0.0,
  contrast: number = 1.0,
  inverted: boolean = false,
  seed: number = 42
): Float32Array {
  const next = new Float32Array(grid.length);
  const rand = mulberry32(seed);

  for (let i = 0; i < grid.length; i++) {
    let v = grid[i];

    if (inverted) {
      v = 1.0 - v;
    }

    // Contrast centering around 0.5
    if (contrast !== 1.0) {
      v = (v - 0.5) * contrast + 0.5;
    }

    // Gaussian noise injection
    if (noise > 0.001) {
      v += gaussianRandom(rand, 0, noise);
    }

    next[i] = Math.max(0, Math.min(1, v));
  }

  return next;
}

/**
 * Generates synthetic samples for 4-class micro-training
 * Class 0: Vertical Bars
 * Class 1: Horizontal Bars
 * Class 2: Diagonal Crosses
 * Class 3: Hollow Rings
 */
export function generateSyntheticVisionClass(
  classIndex: number,
  sampleIndex: number,
  seed: number,
  noiseLevel: number = 0.05
): Float32Array {
  const grid = new Float32Array(256);
  const rand = mulberry32(seed + classIndex * 1000 + sampleIndex * 17);

  const jitterX = Math.floor(rand() * 3) - 1; // -1, 0, 1
  const jitterY = Math.floor(rand() * 3) - 1;

  if (classIndex === 0) {
    // Vertical Bars: 3 vertical lines at columns 3, 8, 13 (+ jitter)
    const baseCols = [3, 8, 13];
    for (const bc of baseCols) {
      const c = Math.max(0, Math.min(15, bc + jitterX));
      for (let r = 1; r < 15; r++) {
        grid[r * 16 + c] = 0.95 + rand() * 0.05;
        if (c + 1 < 16 && rand() > 0.6) grid[r * 16 + c + 1] = 0.6;
      }
    }
  } else if (classIndex === 1) {
    // Horizontal Bars: 3 horizontal lines at rows 3, 8, 13 (+ jitter)
    const baseRows = [3, 8, 13];
    for (const br of baseRows) {
      const r = Math.max(0, Math.min(15, br + jitterY));
      for (let c = 1; c < 15; c++) {
        grid[r * 16 + c] = 0.95 + rand() * 0.05;
        if (r + 1 < 16 && rand() > 0.6) grid[(r + 1) * 16 + c] = 0.6;
      }
    }
  } else if (classIndex === 2) {
    // Diagonal Crosses: X shape centered around (8, 8)
    const cx = 8 + jitterX;
    const cy = 8 + jitterY;
    const radius = 5 + Math.floor(rand() * 2);
    for (let d = -radius; d <= radius; d++) {
      const r1 = cy + d;
      const c1 = cx + d;
      const c2 = cx - d;
      if (r1 >= 0 && r1 < 16) {
        if (c1 >= 0 && c1 < 16) grid[r1 * 16 + c1] = 0.95;
        if (c2 >= 0 && c2 < 16) grid[r1 * 16 + c2] = 0.95;
      }
    }
  } else if (classIndex === 3) {
    // Hollow Rings: Circular/elliptical loop centered around (8, 8)
    const cx = 8 + jitterX;
    const cy = 8 + jitterY;
    const radOuter = 5.2 + rand() * 0.8;
    const radInner = 2.8 + rand() * 0.5;

    for (let r = 0; r < 16; r++) {
      for (let c = 0; c < 16; c++) {
        const dist = Math.sqrt((r - cy) ** 2 + (c - cx) ** 2);
        if (dist >= radInner && dist <= radOuter) {
          grid[r * 16 + c] = 0.95;
        }
      }
    }
  }

  // Inject subtle background noise
  for (let i = 0; i < 256; i++) {
    if (noiseLevel > 0) {
      grid[i] = Math.max(0, Math.min(1, grid[i] + gaussianRandom(rand, 0, noiseLevel)));
    }
  }

  return grid;
}

/**
 * Builds full SpatialVisionDataset container
 */
export function generateSpatialVisionDataset(options: {
  preset?: VisionPresetKind;
  noise?: number;
  contrast?: number;
  inverted?: boolean;
  seed?: number;
  customGrid?: Float32Array;
}): SpatialVisionDataset {
  const preset = options.preset || 'digit';
  const noise = options.noise ?? 0.0;
  const contrast = options.contrast ?? 1.0;
  const inverted = options.inverted ?? false;
  const seed = options.seed ?? 42;

  const rawBase = options.customGrid ? new Float32Array(options.customGrid) : getPresetGrid(preset);
  const inputGrid = applyNoiseAndContrast(rawBase, noise, contrast, inverted, seed);

  const trainSamples: SpatialVisionSample[] = [];
  const testSamples: SpatialVisionSample[] = [];

  // Generate 8 train samples and 4 test samples per class
  let idCounter = 1;
  for (let cls = 0; cls < 4; cls++) {
    const labelName = CLASS_LABELS[cls];
    for (let i = 0; i < 8; i++) {
      trainSamples.push({
        id: idCounter++,
        pixels: generateSyntheticVisionClass(cls, i, seed, 0.04),
        label: cls,
        labelName,
      });
    }
    for (let i = 0; i < 4; i++) {
      testSamples.push({
        id: idCounter++,
        pixels: generateSyntheticVisionClass(cls, 8 + i, seed, 0.04),
        label: cls,
        labelName,
      });
    }
  }

  return {
    preset,
    inputGrid,
    noise,
    contrast,
    inverted,
    trainSamples,
    testSamples,
  };
}
