import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { dist2D } from './clusteringUtils';
import { mulberry32, gaussianRandom } from '../../utils/prng';

export interface UMAPState {
  nNeighbors: number;
  minDist: number;
  learningRate: number;
  fuzzyGraphEdges: { sourceId: number; targetId: number; weight: number }[];
  Y: [number, number][]; // 2D low-dimensional embedding coordinates
  epoch: number;
  converged: boolean;
  crossEntropyLoss: number;
  lossHistory: { iteration: number; loss: number; name: string }[];
  trainPoints?: [number, number][];
}

export interface UMAPParams {
  nNeighbors?: number;
  minDist?: number;
  learningRate?: number;
}

function constructFuzzySimplicialGraph(
  points: [number, number][],
  k: number
): {
  edges: { sourceId: number; targetId: number; weight: number }[];
  edgeMap: Map<string, number>;
} {
  const N = points.length;
  if (N <= 1) return { edges: [], edgeMap: new Map() };

  const effectiveK = Math.min(k, N - 1);
  const targetSum = Math.log2(effectiveK);

  // 1. Find k-nearest neighbors and distances
  const knnDist: { idx: number; d: number }[][] = [];
  const rho: number[] = new Array(N);

  for (let i = 0; i < N; i++) {
    const list: { idx: number; d: number }[] = [];
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      list.push({ idx: j, d: dist2D(points[i], points[j]) });
    }
    list.sort((a, b) => a.d - b.d);
    const topK = list.slice(0, effectiveK);
    knnDist.push(topK);
    rho[i] = topK.length > 0 ? topK[0].d : 0;
  }

  // 2. Binary search for sigma_i for each point
  const pCond = new Map<string, number>();

  for (let i = 0; i < N; i++) {
    let sigmaMin = 1e-4;
    let sigmaMax = 100.0;
    let sigma = 1.0;

    for (let iter = 0; iter < 30; iter++) {
      let sumExp = 0;
      for (const n of knnDist[i]) {
        const dDiff = Math.max(0, n.d - rho[i]);
        sumExp += Math.exp(-dDiff / sigma);
      }

      if (Math.abs(sumExp - targetSum) < 1e-4) break;

      if (sumExp > targetSum) {
        sigmaMax = sigma;
        sigma = (sigmaMin + sigma) / 2;
      } else {
        sigmaMin = sigma;
        sigma = (sigma + sigmaMax) / 2;
      }
    }

    for (const n of knnDist[i]) {
      const dDiff = Math.max(0, n.d - rho[i]);
      const pVal = Math.exp(-dDiff / Math.max(1e-4, sigma));
      pCond.set(`${i}_${n.idx}`, pVal);
    }
  }

  // 3. Fuzzy union: w_ij = p_i|j + p_j|i - p_i|j * p_j|i
  const edges: { sourceId: number; targetId: number; weight: number }[] = [];
  const edgeMap = new Map<string, number>();

  for (let i = 0; i < N; i++) {
    for (const n of knnDist[i]) {
      const j = n.idx;
      if (i < j) {
        const pij = pCond.get(`${i}_${j}`) ?? 0;
        const pji = pCond.get(`${j}_${i}`) ?? 0;
        const w = pij + pji - pij * pji;
        if (w > 0.02) {
          edges.push({ sourceId: i, targetId: j, weight: w });
          edgeMap.set(`${i}_${j}`, w);
          edgeMap.set(`${j}_${i}`, w);
        }
      }
    }
  }

  return { edges, edgeMap };
}

// Low-dimensional curve parameters matching minDist
function findCurveParameters(minDist: number): { a: number; b: number } {
  // Approximate standard UMAP fit: for minDist = 0.1, a ~ 1.57, b ~ 0.89
  const clamped = Math.max(0.01, Math.min(0.99, minDist));
  const b = 1.0 / (1.0 + clamped * 0.8);
  const a = 1.0 / Math.pow(clamped, 2 * b);
  return { a: Math.min(2.0, Math.max(0.5, a * 0.2)), b: Math.min(1.2, Math.max(0.6, b)) };
}

function computeCrossEntropyLoss(
  edges: { sourceId: number; targetId: number; weight: number }[],
  Y: [number, number][],
  a: number,
  b: number
): number {
  let totalCELoss = 0;
  for (const edge of edges) {
    const i = edge.sourceId;
    const j = edge.targetId;
    const w = edge.weight;
    const dy0 = Y[i][0] - Y[j][0];
    const dy1 = Y[i][1] - Y[j][1];
    const distSq = dy0 * dy0 + dy1 * dy1;
    const dist2b = Math.pow(Math.max(1e-6, distSq), b);
    const qVal = 1.0 / (1.0 + a * dist2b);
    if (w > 1e-4 && qVal > 1e-4) {
      totalCELoss += -w * Math.log(qVal);
    }
  }
  return totalCELoss;
}

export const umapPlugin: AlgorithmPlugin<UMAPState, UMAPParams> = {
  id: 'umap',
  name: 'UMAP (Uniform Manifold Approximation and Projection)',
  shortCode: 'UMAP',
  stage: 'unsupervised-2d',
  taskType: 'dimensionality-reduction',
  executionMode: 'epoch-loop',
  description:
    'Riemannian geometric manifold approximation constructing fuzzy simplicial simplicial sets and optimizing layout via fuzzy set cross-entropy.',

  primaryParams: [
    {
      id: 'nNeighbors',
      name: 'nNeighbors (k)',
      type: 'slider',
      min: 2,
      max: 40,
      step: 1,
      default: 15,
      description: 'Size of local fuzzy neighborhood graph in source space.',
    },
    {
      id: 'minDist',
      name: 'Min Dist (Tightly Packed)',
      type: 'slider',
      min: 0.01,
      max: 0.99,
      step: 0.05,
      default: 0.1,
      description: 'Clustering tightness preventing points from collapsing directly on top of each other.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.1,
      max: 2.0,
      step: 0.1,
      default: 1.0,
      description: 'Optimization step size for stochastic gradient descent.',
    },
  ],

  advancedParams: [],

  init(dataset: DatasetContainer, params: UMAPParams, seed: number = 42): UMAPState {
    const nNeighbors = Math.max(2, Math.min(40, Math.round(Number(params.nNeighbors ?? 15))));
    const minDist = Math.max(0.01, Math.min(0.99, Number(params.minDist ?? 0.1)));
    const learningRate = Math.max(0.1, Number(params.learningRate ?? 1.0));

    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const { edges } = constructFuzzySimplicialGraph(points, nNeighbors);

    const rand = mulberry32(seed);
    const N = points.length;
    const Y: [number, number][] = [];

    // Initialize with small random spread
    for (let i = 0; i < N; i++) {
      Y.push([gaussianRandom(rand, 0, 1.0), gaussianRandom(rand, 0, 1.0)]);
    }

    const { a, b } = findCurveParameters(minDist);
    const initialCELoss = computeCrossEntropyLoss(edges, Y, a, b);

    return {
      nNeighbors,
      minDist,
      learningRate,
      fuzzyGraphEdges: edges,
      Y,
      epoch: 0,
      converged: false,
      crossEntropyLoss: initialCELoss,
      lossHistory: [{ iteration: 0, loss: initialCELoss, name: 'Cross-Entropy' }],
      trainPoints: points,
    };
  },

  step(state: UMAPState, dataset: DatasetContainer, params: UMAPParams): StepResult<UMAPState> {
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const N = state.Y.length;
    if (N <= 1) {
      return { state, epoch: state.epoch, metrics: this.loss(state, dataset, params), converged: true };
    }

    const learningRate = Number(params.learningRate ?? state.learningRate);
    const minDist = Number(params.minDist ?? state.minDist);
    const { a, b } = findCurveParameters(minDist);

    const nextY = state.Y.map((y) => [...y] as [number, number]);
    let totalCELoss = 0;

    // 1. Attractive force along edges in fuzzy topological graph
    for (const edge of state.fuzzyGraphEdges) {
      const i = edge.sourceId;
      const j = edge.targetId;
      const w = edge.weight;

      const dy0 = nextY[i][0] - nextY[j][0];
      const dy1 = nextY[i][1] - nextY[j][1];
      const distSq = dy0 * dy0 + dy1 * dy1;

      // q_ij = 1 / (1 + a * distSq^b)
      const dist2b = Math.pow(Math.max(1e-6, distSq), b);
      const qVal = 1.0 / (1.0 + a * dist2b);

      if (w > 1e-4 && qVal > 1e-4) {
        totalCELoss += -w * Math.log(qVal);
      }

      // Attractive gradient: -2 * a * b * distSq^(b-1) / (1 + a * distSq^b)
      const gradAttr = (-2.0 * a * b * Math.pow(Math.max(1e-6, distSq), b - 1)) / (1.0 + a * dist2b);
      const clampGrad = Math.max(-4.0, Math.min(4.0, gradAttr * w * learningRate));

      nextY[i][0] += clampGrad * dy0;
      nextY[i][1] += clampGrad * dy1;
      nextY[j][0] -= clampGrad * dy0;
      nextY[j][1] -= clampGrad * dy1;
    }

    // 2. Repulsive force via negative sampling (sample random pairs)
    const nNeg = 3;
    for (let i = 0; i < N; i++) {
      for (let s = 0; s < nNeg; s++) {
        const j = Math.floor(Math.random() * N);
        if (i === j) continue;

        const dy0 = nextY[i][0] - nextY[j][0];
        const dy1 = nextY[i][1] - nextY[j][1];
        const distSq = dy0 * dy0 + dy1 * dy1;
        if (distSq < 1e-5) continue;

        const dist2b = Math.pow(distSq, b);
        const qVal = 1.0 / (1.0 + a * dist2b);

        // Repulsive gradient: 2 * b / ((0.001 + distSq) * (1 + a * distSq^b))
        const gradRep = (2.0 * b) / ((0.001 + distSq) * (1.0 + a * dist2b));
        const clampRep = Math.max(-3.0, Math.min(3.0, gradRep * (1.0 - qVal) * (learningRate * 0.4)));

        nextY[i][0] += clampRep * dy0;
        nextY[i][1] += clampRep * dy1;
      }
    }

    // Center embeddings
    let sx = 0;
    let sy = 0;
    for (let i = 0; i < N; i++) {
      sx += nextY[i][0];
      sy += nextY[i][1];
    }
    sx /= N;
    sy /= N;
    for (let i = 0; i < N; i++) {
      nextY[i][0] -= sx;
      nextY[i][1] -= sy;
    }

    const lossHistory = [
      ...state.lossHistory.slice(-50),
      { iteration: state.epoch + 1, loss: totalCELoss, name: 'Cross-Entropy' },
    ];

    const nextState: UMAPState = {
      ...state,
      Y: nextY,
      crossEntropyLoss: totalCELoss,
      epoch: state.epoch + 1,
      lossHistory,
      trainPoints: points,
      converged: state.epoch > 80,
    };

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: this.loss(nextState, dataset, params),
      converged: nextState.converged,
    };
  },

  predict(input: Float32Array, state: UMAPState, _params: UMAPParams): number {
    if (!state.Y || state.Y.length === 0 || !state.trainPoints || state.trainPoints.length === 0) {
      return state.Y.length > 0 ? state.Y[0][0] : 0;
    }
    const pt: [number, number] = [input[0], input[1]];
    let minDist = Infinity;
    let bestIdx = 0;
    for (let i = 0; i < state.trainPoints.length; i++) {
      const d = dist2D(pt, state.trainPoints[i]);
      if (d < minDist) {
        minDist = d;
        bestIdx = i;
      }
    }
    return state.Y[bestIdx] ? state.Y[bestIdx][0] : 0;
  },

  loss(state: UMAPState, _dataset: DatasetContainer, _params: UMAPParams): TelemetryMetrics {
    return {
      trainLoss: state.crossEntropyLoss,
      testLoss: state.crossEntropyLoss,
      trainAccuracy: Math.max(0, Math.min(100, Math.round(100 - state.crossEntropyLoss * 8))),
      custom: {
        crossEntropyLoss: Number(state.crossEntropyLoss.toFixed(3)),
        graphEdges: state.fuzzyGraphEdges.length,
        nNeighbors: state.nNeighbors,
        minDist: state.minDist,
      },
    };
  },

  getVisualArtifacts(state: UMAPState, dataset: DatasetContainer, _params: UMAPParams): VisualArtifact[] {
    const embeddedPoints = state.Y.map((y, i) => ({
      id: dataset.train[i]?.id ?? i,
      x: y[0],
      y: y[1],
      label: dataset.train[i]?.y ?? 1,
    }));

    return [
      {
        type: 'manifold-overlay',
        method: 'umap',
        embeddedPoints,
        fuzzyGraphEdges: state.fuzzyGraphEdges,
        lossTelemetry: state.lossHistory,
      },
    ];
  },

  serializeSnapshot(state: UMAPState): Float32Array {
    const N = state.Y.length;
    const buf = new Float32Array(3 + N * 2);
    buf[0] = state.epoch;
    buf[1] = state.crossEntropyLoss;
    buf[2] = N;
    for (let i = 0; i < N; i++) {
      buf[3 + i * 2] = state.Y[i][0];
      buf[3 + i * 2 + 1] = state.Y[i][1];
    }
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: UMAPState): UMAPState {
    const N = Math.round(buffer[2]);
    const Y: [number, number][] = [];
    for (let i = 0; i < N; i++) {
      Y.push([buffer[3 + i * 2], buffer[3 + i * 2 + 1]]);
    }
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      crossEntropyLoss: buffer[1],
      Y,
    };
  },

  explain(state: UMAPState, _params: UMAPParams, _lastStep?: StepResult<UMAPState>): ExplanationPayload {
    return {
      summary: `UMAP models manifold topology by constructing a fuzzy simplicial graph (${state.fuzzyGraphEdges.length} edges) and minimizing fuzzy set cross-entropy to preserve both local and global cluster hierarchy.`,
      coreAssumption:
        'Data lies on a connected Riemannian manifold with uniform local metric scaling (distances normalize locally by closest neighbor radius ρ_i).',
      whenItFails:
        'Sparse disjoint clusters where local connectivity assumption breaks down, or parameter extremes (min_dist = 0 collapses points completely; min_dist = 1 scatters points uniformly).',
      microLog: [
        `Construct fuzzy simplicial graph with k = ${state.nNeighbors} neighbors per point`,
        `Compute edge weights w_ij using local Riemannian scaling metric (ρ_i, σ_i)`,
        `Optimize 2D layout: attractive edges + negative sampling repulsion (Loss = ${state.crossEntropyLoss.toFixed(2)})`,
      ],
      mathLatex: {
        objective: `C = \\sum_{e \\in E} \\left( w_{ij} \\log \\frac{w_{ij}}{q_{ij}} + (1 - w_{ij}) \\log \\frac{1 - w_{ij}}{1 - q_{ij}} \\right)`,
        lossOrUpdate: `w_{i|j} = \\exp\\left(-\\frac{\\max(0, d(x_i, x_j) - \\rho_i)}{\\sigma_i}\\right), \\quad q_{ij} = \\frac{1}{1 + a \\|y_i - y_j\\|^{2b}}`,
      },
      codeSnippet: {
        sklearn: `import umap\nreducer = umap.UMAP(n_neighbors=${state.nNeighbors}, min_dist=${state.minDist})\nY = reducer.fit_transform(X)`,
        pytorch: `# PyTorch fuzzy simplicial set construction & SGD layout optimization`,
      },
    };
  },
};

