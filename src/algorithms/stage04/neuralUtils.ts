import { gaussianRandom } from '../../utils/prng';

export type ActivationType = 'relu' | 'leaky-relu' | 'tanh' | 'sigmoid' | 'linear';
export type WeightInitType = 'xavier' | 'he' | 'zeros-trap';
export type OptimizerType = 'sgd' | 'momentum' | 'adam';

export const BASIS_FEATURES = [
  'x1',
  'x2',
  'x1_sq',
  'x2_sq',
  'x1_x2',
  'sin_x1',
  'sin_x2',
] as const;

export type BasisFeature = typeof BASIS_FEATURES[number];

export function computeFeature(feat: BasisFeature, x1: number, x2: number): number {
  switch (feat) {
    case 'x1':
      return x1;
    case 'x2':
      return x2;
    case 'x1_sq':
      return x1 * x1;
    case 'x2_sq':
      return x2 * x2;
    case 'x1_x2':
      return x1 * x2;
    case 'sin_x1':
      return Math.sin(x1);
    case 'sin_x2':
      return Math.sin(x2);
    default:
      return 0;
  }
}

export function activate(z: number, type: ActivationType): number {
  switch (type) {
    case 'relu':
      return z > 0 ? z : 0;
    case 'leaky-relu':
      return z > 0 ? z : 0.01 * z;
    case 'tanh':
      return Math.tanh(z);
    case 'sigmoid':
      return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z))));
    case 'linear':
      return z;
    default:
      return z;
  }
}

export function activateDerivative(z: number, a: number, type: ActivationType): number {
  switch (type) {
    case 'relu':
      return z > 0 ? 1 : 0;
    case 'leaky-relu':
      return z > 0 ? 1 : 0.01;
    case 'tanh':
      return 1 - a * a;
    case 'sigmoid':
      return a * (1 - a);
    case 'linear':
      return 1;
    default:
      return 1;
  }
}

export function initializeWeights(
  fanIn: number,
  fanOut: number,
  initType: WeightInitType,
  rand: () => number
): Float32Array {
  const count = fanIn * fanOut;
  const weights = new Float32Array(count);

  if (initType === 'zeros-trap') {
    weights.fill(0);
    return weights;
  }

  if (initType === 'he') {
    const std = Math.sqrt(2 / Math.max(1, fanIn));
    for (let i = 0; i < count; i++) {
      weights[i] = gaussianRandom(rand, 0, std);
    }
  } else {
    // Xavier / Glorot
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    for (let i = 0; i < count; i++) {
      weights[i] = -limit + rand() * 2 * limit;
    }
  }

  return weights;
}

// Pre-computed 24x24 grid coordinates for zero-allocation neuron mini heatmaps
export const MINI_RES = 24;
export const MINI_COUNT = MINI_RES * MINI_RES;
export const MINI_COORDS_X1 = new Float32Array(MINI_COUNT);
export const MINI_COORDS_X2 = new Float32Array(MINI_COUNT);

(() => {
  const minX = -6.0;
  const maxX = 6.0;
  const minY = -6.0;
  const maxY = 6.0;
  const stepX = (maxX - minX) / (MINI_RES - 1);
  const stepY = (maxY - minY) / (MINI_RES - 1);

  let idx = 0;
  for (let r = 0; r < MINI_RES; r++) {
    const x2 = maxY - r * stepY;
    for (let c = 0; c < MINI_RES; c++) {
      const x1 = minX + c * stepX;
      MINI_COORDS_X1[idx] = x1;
      MINI_COORDS_X2[idx] = x2;
      idx++;
    }
  }
})();
