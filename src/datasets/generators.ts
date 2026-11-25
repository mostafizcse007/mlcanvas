import { DataPoint, DatasetConfig } from '../types/dataset';

// Simple fast deterministic pseudo-random generator
function pseudoRandom(seed: number): () => number {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

function gaussianRandom(rand: () => number): number {
  const u1 = Math.max(1e-7, rand());
  const u2 = rand();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

export function generateDataset(config: DatasetConfig): { train: DataPoint[]; test: DataPoint[] } {
  const points: DataPoint[] = [];
  const rand = pseudoRandom(42);
  const n = config.sampleCount;
  const noise = config.noise;

  let idCounter = 1;

  switch (config.type) {
    case 'separable-gaussians': {
      const half = Math.floor(n / 2);
      // Class A (+1): center at (-2, -1.5)
      for (let i = 0; i < half; i++) {
        const x1 = -2.2 + gaussianRandom(rand) * (0.8 + noise * 1.5);
        const x2 = -1.8 + gaussianRandom(rand) * (0.8 + noise * 1.5);
        points.push({ id: idCounter++, x1, x2, y: 1 });
      }
      // Class B (-1): center at (2, 1.8)
      for (let i = 0; i < n - half; i++) {
        const x1 = 2.2 + gaussianRandom(rand) * (0.8 + noise * 1.5);
        const x2 = 1.8 + gaussianRandom(rand) * (0.8 + noise * 1.5);
        points.push({ id: idCounter++, x1, x2, y: -1 });
      }
      break;
    }

    case 'circles': {
      const half = Math.floor(n / 2);
      // Inner circle: class +1, radius ~ 1.5
      for (let i = 0; i < half; i++) {
        const r = (rand() * 1.8) + gaussianRandom(rand) * noise * 0.8;
        const theta = rand() * 2 * Math.PI;
        points.push({
          id: idCounter++,
          x1: r * Math.cos(theta),
          x2: r * Math.sin(theta),
          y: 1,
        });
      }
      // Outer ring: class -1, radius ~ 3.5 to 5.0
      for (let i = 0; i < n - half; i++) {
        const r = 3.4 + (rand() * 1.8) + gaussianRandom(rand) * noise * 0.8;
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

    case 'moons': {
      const half = Math.floor(n / 2);
      // Top moon: center at (-1, 0), radius ~ 2.5
      for (let i = 0; i < half; i++) {
        const theta = rand() * Math.PI;
        const r = 2.4 + gaussianRandom(rand) * (0.2 + noise * 0.8);
        const x1 = r * Math.cos(theta) - 1.2;
        const x2 = r * Math.sin(theta) - 0.5;
        points.push({ id: idCounter++, x1, x2, y: 1 });
      }
      // Bottom moon: center at (1, 0), flipped
      for (let i = 0; i < n - half; i++) {
        const theta = rand() * Math.PI;
        const r = 2.4 + gaussianRandom(rand) * (0.2 + noise * 0.8);
        const x1 = 1.2 - r * Math.cos(theta);
        const x2 = 0.5 - r * Math.sin(theta);
        points.push({ id: idCounter++, x1, x2, y: -1 });
      }
      break;
    }

    case 'xor': {
      const quarter = Math.floor(n / 4);
      for (let q = 0; q < 4; q++) {
        const count = q === 3 ? n - 3 * quarter : quarter;
        const signX = q === 0 || q === 3 ? 1 : -1;
        const signY = q === 0 || q === 1 ? 1 : -1;
        // XOR logic: Quadrants 1 & 3 are +1, Quadrants 2 & 4 are -1
        const label = signX * signY > 0 ? 1 : -1;

        for (let i = 0; i < count; i++) {
          const x1 = signX * (0.6 + rand() * 3.5 + gaussianRandom(rand) * noise * 0.8);
          const x2 = signY * (0.6 + rand() * 3.5 + gaussianRandom(rand) * noise * 0.8);
          points.push({ id: idCounter++, x1, x2, y: label });
        }
      }
      break;
    }

    case 'spiral': {
      const half = Math.floor(n / 2);
      for (let i = 0; i < half; i++) {
        const r = (i / half) * 5.0;
        const t = 1.75 * (i / half) * 2 * Math.PI;
        const x1 = r * Math.sin(t) + gaussianRandom(rand) * (0.1 + noise * 0.6);
        const x2 = r * Math.cos(t) + gaussianRandom(rand) * (0.1 + noise * 0.6);
        points.push({ id: idCounter++, x1, x2, y: 1 });
      }
      for (let i = 0; i < n - half; i++) {
        const r = (i / (n - half)) * 5.0;
        const t = 1.75 * (i / (n - half)) * 2 * Math.PI + Math.PI;
        const x1 = r * Math.sin(t) + gaussianRandom(rand) * (0.1 + noise * 0.6);
        const x2 = r * Math.cos(t) + gaussianRandom(rand) * (0.1 + noise * 0.6);
        points.push({ id: idCounter++, x1, x2, y: -1 });
      }
      break;
    }

    case 'linear-regression': {
      for (let i = 0; i < n; i++) {
        const x1 = (rand() - 0.5) * 8.0;
        const x2 = (rand() - 0.5) * 8.0;
        // True equation: y = 1.8 * x1 - 0.9 * x2 + 0.5 + noise
        const target = 1.8 * x1 - 0.9 * x2 + 0.5 + gaussianRandom(rand) * (0.3 + noise * 2.0);
        points.push({ id: idCounter++, x1, x2, y: target });
      }
      break;
    }
  }

  // Shuffle dataset and partition into train & test
  for (let i = points.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    const temp = points[i];
    points[i] = points[j];
    points[j] = temp;
  }

  const testCount = Math.floor(n * config.testRatio);
  const train: DataPoint[] = [];
  const test: DataPoint[] = [];

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

