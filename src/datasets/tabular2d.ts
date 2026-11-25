import { mulberry32, gaussianRandom } from '../utils/prng';

export type DatasetKind =
  | 'linear-blobs'
  | 'concentric-circles'
  | 'xor-quadrants'
  | 'twin-spirals'
  | 'sine-wave'
  | 'polynomial-cubic'
  | 'linear-trend'
  | 'linear-surface';

export const CLASSIFICATION_DATASETS: DatasetKind[] = [
  'linear-blobs',
  'concentric-circles',
  'xor-quadrants',
  'twin-spirals',
];

export const REGRESSION_DATASETS: DatasetKind[] = [
  'sine-wave',
  'polynomial-cubic',
  'linear-trend',
  'linear-surface',
];

export function getDatasetTaskType(kind: DatasetKind): 'classification' | 'regression' {
  return REGRESSION_DATASETS.includes(kind) ? 'regression' : 'classification';
}

export type FeatureScalingMode = 'raw' | 'zscore' | 'minmax';

export type PointerToolMode = 'add-drag' | 'flip' | 'delete';

export interface DataPoint2D {
  id: number;
  x1: number;
  x2: number;
  y: number; // +1, -1, or +2
  isTest?: boolean;
}

export interface DatasetOptions {
  kind: DatasetKind;
  sampleCount: number;
  noise: number; // 0.0 to 0.5
  testSplit: number; // 0.1 to 0.4
  seed: number;
  numClasses: 2 | 3;
  scaling: FeatureScalingMode;
  stretchX2: boolean; // 10x anisotropic stretching
}

export function generateTabular2dDataset(options: DatasetOptions): {
  train: DataPoint2D[];
  test: DataPoint2D[];
} {
  const rand = mulberry32(options.seed);
  const n = options.sampleCount;
  const noise = options.noise;
  const points: DataPoint2D[] = [];

  let idCounter = 1;

  if (options.numClasses === 3 && getDatasetTaskType(options.kind) === 'classification') {
    // 3-Class Multiclass generation
    const perClass = Math.floor(n / 3);
    const centers = [
      { x1: -2.4, x2: -2.0, y: 1 },
      { x1: 2.4, x2: -2.0, y: -1 },
      { x1: 0.0, x2: 2.8, y: 2 },
    ];
    for (let c = 0; c < 3; c++) {
      const cnt = c === 2 ? n - 2 * perClass : perClass;
      const center = centers[c];
      for (let i = 0; i < cnt; i++) {
        points.push({
          id: idCounter++,
          x1: center.x1 + gaussianRandom(rand, 0, 0.8 + noise * 1.5),
          x2: center.x2 + gaussianRandom(rand, 0, 0.8 + noise * 1.5),
          y: center.y,
        });
      }
    }
  } else {
    // 2-Class Binary generation
    switch (options.kind) {
      case 'linear-blobs': {
        const half = Math.floor(n / 2);
        // Class +1: cluster around (-2.0, -1.8)
        for (let i = 0; i < half; i++) {
          points.push({
            id: idCounter++,
            x1: -2.0 + gaussianRandom(rand, 0, 0.75 + noise * 1.6),
            x2: -1.8 + gaussianRandom(rand, 0, 0.75 + noise * 1.6),
            y: 1,
          });
        }
        // Class -1: cluster around (2.0, 1.8)
        for (let i = 0; i < n - half; i++) {
          points.push({
            id: idCounter++,
            x1: 2.0 + gaussianRandom(rand, 0, 0.75 + noise * 1.6),
            x2: 1.8 + gaussianRandom(rand, 0, 0.75 + noise * 1.6),
            y: -1,
          });
        }
        break;
      }

      case 'concentric-circles': {
        const half = Math.floor(n / 2);
        // Inner circle (+1): radius ~ 1.5
        for (let i = 0; i < half; i++) {
          const r = rand() * 1.8 + gaussianRandom(rand, 0, noise * 0.7);
          const theta = rand() * 2 * Math.PI;
          points.push({
            id: idCounter++,
            x1: r * Math.cos(theta),
            x2: r * Math.sin(theta),
            y: 1,
          });
        }
        // Outer ring (-1): radius ~ 3.5 to 5.2
        for (let i = 0; i < n - half; i++) {
          const r = 3.4 + rand() * 1.8 + gaussianRandom(rand, 0, noise * 0.7);
          const theta = rand() * 2 * Math.PI;
          points.push({
            id: idCounter++,
            x1: r * Math.cos(theta),
            x2: r * Math.sin(theta),
            y: -1,
          });
        }
        break;
      }

      case 'xor-quadrants': {
        const quarter = Math.floor(n / 4);
        for (let q = 0; q < 4; q++) {
          const cnt = q === 3 ? n - 3 * quarter : quarter;
          const signX = q === 0 || q === 3 ? 1 : -1;
          const signY = q === 0 || q === 1 ? 1 : -1;
          const label = signX * signY > 0 ? 1 : -1;

          for (let i = 0; i < cnt; i++) {
            points.push({
              id: idCounter++,
              x1: signX * (0.6 + rand() * 3.4 + gaussianRandom(rand, 0, noise * 0.8)),
              x2: signY * (0.6 + rand() * 3.4 + gaussianRandom(rand, 0, noise * 0.8)),
              y: label,
            });
          }
        }
        break;
      }

      case 'twin-spirals': {
        const half = Math.floor(n / 2);
        for (let i = 0; i < half; i++) {
          const r = (i / half) * 4.8;
          const t = 1.75 * (i / half) * 2 * Math.PI;
          points.push({
            id: idCounter++,
            x1: r * Math.sin(t) + gaussianRandom(rand, 0, 0.1 + noise * 0.5),
            x2: r * Math.cos(t) + gaussianRandom(rand, 0, 0.1 + noise * 0.5),
            y: 1,
          });
        }
        for (let i = 0; i < n - half; i++) {
          const r = (i / (n - half)) * 4.8;
          const t = 1.75 * (i / (n - half)) * 2 * Math.PI + Math.PI;
          points.push({
            id: idCounter++,
            x1: r * Math.sin(t) + gaussianRandom(rand, 0, 0.1 + noise * 0.5),
            x2: r * Math.cos(t) + gaussianRandom(rand, 0, 0.1 + noise * 0.5),
            y: -1,
          });
        }
        break;
      }

      case 'sine-wave': {
        for (let i = 0; i < n; i++) {
          const x1 = -5.0 + (i / (n - 1 || 1)) * 10.0 + gaussianRandom(rand, 0, 0.1);
          const y = 2.4 * Math.sin(0.85 * x1) + gaussianRandom(rand, 0, 0.2 + noise * 1.5);
          points.push({
            id: idCounter++,
            x1,
            x2: y,
            y,
          });
        }
        break;
      }

      case 'polynomial-cubic': {
        for (let i = 0; i < n; i++) {
          const x1 = -4.2 + (i / (n - 1 || 1)) * 8.4 + gaussianRandom(rand, 0, 0.1);
          const y = 0.08 * Math.pow(x1, 3) - 0.45 * x1 + gaussianRandom(rand, 0, 0.25 + noise * 1.8);
          points.push({
            id: idCounter++,
            x1,
            x2: y,
            y,
          });
        }
        break;
      }

      case 'linear-trend': {
        for (let i = 0; i < n; i++) {
          const x1 = -5.0 + (i / (n - 1 || 1)) * 10.0 + gaussianRandom(rand, 0, 0.1);
          const y = 0.65 * x1 + 0.4 + gaussianRandom(rand, 0, 0.3 + noise * 2.0);
          points.push({
            id: idCounter++,
            x1,
            x2: y,
            y,
          });
        }
        break;
      }

      case 'linear-surface': {
        for (let i = 0; i < n; i++) {
          const x1 = (rand() - 0.5) * 8.5;
          const x2 = (rand() - 0.5) * 8.5;
          const y = 0.55 * x1 - 0.45 * x2 + 0.2 + gaussianRandom(rand, 0, 0.25 + noise * 1.6);
          points.push({
            id: idCounter++,
            x1,
            x2,
            y,
          });
        }
        break;
      }
    }
  }

  // Feature Scaling & Anisotropic Distortion
  applyScalingInPlace(points, options.scaling, options.stretchX2);

  // Fisher-Yates deterministic shuffle
  for (let i = points.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const tmp = points[i];
    points[i] = points[j];
    points[j] = tmp;
  }

  const testCount = Math.floor(n * options.testSplit);
  const train: DataPoint2D[] = [];
  const test: DataPoint2D[] = [];

  for (let i = 0; i < points.length; i++) {
    if (i < testCount) {
      points[i].isTest = true;
      test.push(points[i]);
    } else {
      points[i].isTest = false;
      train.push(points[i]);
    }
  }

  return { train, test };
}

function applyScalingInPlace(
  points: DataPoint2D[],
  mode: FeatureScalingMode,
  stretchX2: boolean
): void {
  const len = points.length;
  if (len === 0) return;

  // 10x stretching of X2 if requested (anisotropic ill-conditioned problem)
  if (stretchX2) {
    for (let i = 0; i < len; i++) {
      points[i].x2 *= 10;
    }
  }

  if (mode === 'raw') return;

  if (mode === 'zscore') {
    let sum1 = 0, sum2 = 0;
    for (let i = 0; i < len; i++) {
      sum1 += points[i].x1;
      sum2 += points[i].x2;
    }
    const mean1 = sum1 / len;
    const mean2 = sum2 / len;

    let var1 = 0, var2 = 0;
    for (let i = 0; i < len; i++) {
      const d1 = points[i].x1 - mean1;
      const d2 = points[i].x2 - mean2;
      var1 += d1 * d1;
      var2 += d2 * d2;
    }
    const std1 = Math.sqrt(var1 / len) || 1;
    const std2 = Math.sqrt(var2 / len) || 1;

    for (let i = 0; i < len; i++) {
      points[i].x1 = ((points[i].x1 - mean1) / std1) * 2.5;
      points[i].x2 = ((points[i].x2 - mean2) / std2) * 2.5;
    }
  } else if (mode === 'minmax') {
    let min1 = Infinity, max1 = -Infinity;
    let min2 = Infinity, max2 = -Infinity;

    for (let i = 0; i < len; i++) {
      if (points[i].x1 < min1) min1 = points[i].x1;
      if (points[i].x1 > max1) max1 = points[i].x1;
      if (points[i].x2 < min2) min2 = points[i].x2;
      if (points[i].x2 > max2) max2 = points[i].x2;
    }

    const range1 = max1 - min1 || 1;
    const range2 = max2 - min2 || 1;

    for (let i = 0; i < len; i++) {
      points[i].x1 = (((points[i].x1 - min1) / range1) * 2 - 1) * 3.5;
      points[i].x2 = (((points[i].x2 - min2) / range2) * 2 - 1) * 3.5;
    }
  }
}

