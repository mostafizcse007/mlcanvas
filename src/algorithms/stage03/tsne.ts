import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { mulberry32, gaussianRandom } from '../../utils/prng';
import { dist2D } from './clusteringUtils';

export interface TSNEState {
  perplexity: number;
  learningRate: number;
  exaggeration: number;
  P: Float32Array; // N x N high-D joint probabilities
  Y: [number, number][]; // N x 2 low-D embedding coordinates
  dY: [number, number][]; // Momentum velocity
  gains: [number, number][]; // Adaptive step gains
  klLoss: number;
  epoch: number;
  converged: boolean;
  lossHistory: { iteration: number; loss: number; name: string }[];
  trainPoints?: [number, number][];
}

export interface TSNEParams {
  perplexity?: number;
  learningRate?: number;
  exaggeration?: number;
}

function computeAffinityMatrixP(
  points: [number, number][],
  targetPerplexity: number
): Float32Array {
  const N = points.length;
  const P = new Float32Array(N * N);
  if (N <= 1) return P;

  const targetEntropy = Math.log2(Math.max(1, targetPerplexity));

  // Compute pairwise squared distances
  const distSq = new Float32Array(N * N);
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const dx = points[i][0] - points[j][0];
      const dy = points[i][1] - points[j][1];
      distSq[i * N + j] = dx * dx + dy * dy;
    }
  }

  // Binary search for each point's sigma_i
  const condP = new Float32Array(N * N);

  for (let i = 0; i < N; i++) {
    let betaMin = -Infinity;
    let betaMax = Infinity;
    let beta = 1.0; // 1 / (2 * sigma_i^2)

    for (let iter = 0; iter < 40; iter++) {
      let sumExp = 0;
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const pVal = Math.exp(-distSq[i * N + j] * beta);
        condP[i * N + j] = pVal;
        sumExp += pVal;
      }

      sumExp = Math.max(1e-12, sumExp);

      // Compute Shannon entropy
      let H = 0;
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const pj = condP[i * N + j] / sumExp;
        condP[i * N + j] = pj;
        if (pj > 1e-12) {
          H -= pj * Math.log2(pj);
        }
      }

      const Hdiff = H - targetEntropy;
      if (Math.abs(Hdiff) < 1e-4) break;

      if (Hdiff > 0) {
        betaMin = beta;
        beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
      } else {
        betaMax = beta;
        beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
      }
    }
  }

  // Symmetrize joint probabilities: p_ij = (p_j|i + p_i|j) / (2N)
  const normFactor = 2 * N;
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      if (i === j) continue;
      const pSym = (condP[i * N + j] + condP[j * N + i]) / normFactor;
      P[i * N + j] = Math.max(1e-12, pSym);
    }
  }

  return P;
}

export const tsnePlugin: AlgorithmPlugin<TSNEState, TSNEParams> = {
  id: 't-sne',
  name: 't-SNE (t-Distributed Stochastic Neighbor Embedding)',
  shortCode: 't-SNE',
  stage: 'unsupervised-2d',
  taskType: 'dimensionality-reduction',
  executionMode: 'epoch-loop',
  description:
    'Non-linear probabilistic manifold unfolding matching high-dimensional Gaussian neighborhoods to low-dimensional Student-t affinities via KL-divergence.',

  primaryParams: [
    {
      id: 'perplexity',
      name: 'Perplexity (Effective k)',
      type: 'slider',
      min: 2,
      max: 50,
      step: 1,
      default: 15,
      description: 'Effective number of local neighbors preserved in source space.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 10,
      max: 500,
      step: 10,
      default: 100,
      description: 'Gradient descent step size in the low-dimensional embedding space.',
    },
    {
      id: 'exaggeration',
      name: 'Early Exaggeration',
      type: 'slider',
      min: 1.0,
      max: 4.0,
      step: 0.5,
      default: 2.0,
      description: 'Multiplicative factor on early high-D affinities to form tight, isolated clusters.',
    },
  ],

  advancedParams: [],

  init(dataset: DatasetContainer, params: TSNEParams, seed: number = 42): TSNEState {
    const perplexity = Math.max(2, Math.min(50, Math.round(Number(params.perplexity ?? 15))));
    const learningRate = Math.max(10, Number(params.learningRate ?? 100));
    const exaggeration = Math.max(1.0, Number(params.exaggeration ?? 2.0));

    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const N = points.length;
    const P = computeAffinityMatrixP(points, perplexity);

    const rand = mulberry32(seed);
    const Y: [number, number][] = [];
    const dY: [number, number][] = [];
    const gains: [number, number][] = [];

    // Initialize Y with small Gaussian noise
    for (let i = 0; i < N; i++) {
      Y.push([gaussianRandom(rand, 0, 0.01), gaussianRandom(rand, 0, 0.01)]);
      dY.push([0, 0]);
      gains.push([1, 1]);
    }

    return {
      perplexity,
      learningRate,
      exaggeration,
      P,
      Y,
      dY,
      gains,
      klLoss: 2.5,
      epoch: 0,
      converged: false,
      lossHistory: [{ iteration: 0, loss: 2.5, name: 'KL Loss' }],
      trainPoints: points,
    };
  },

  step(state: TSNEState, dataset: DatasetContainer, params: TSNEParams): StepResult<TSNEState> {
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const N = state.Y.length;
    if (N <= 1) {
      return { state, epoch: state.epoch, metrics: this.loss(state, dataset, params), converged: true };
    }

    const learningRate = Number(params.learningRate ?? state.learningRate);
    const exaggeration = state.epoch < 30 ? Number(params.exaggeration ?? state.exaggeration) : 1.0;
    const momentum = state.epoch < 20 ? 0.5 : 0.8;

    // 1. Compute low-dimensional Student-t probabilities Q
    // q_ij = (1 + ||y_i - y_j||^2)^(-1) / sum_{k!=l} (1 + ||y_k - y_l||^2)^(-1)
    const numQ = new Float32Array(N * N);
    let sumNumQ = 0;

    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const dy0 = state.Y[i][0] - state.Y[j][0];
        const dy1 = state.Y[i][1] - state.Y[j][1];
        const qVal = 1.0 / (1.0 + dy0 * dy0 + dy1 * dy1);
        numQ[i * N + j] = qVal;
        sumNumQ += qVal;
      }
    }

    sumNumQ = Math.max(1e-12, sumNumQ);

    // 2. Compute KL-Divergence and Gradients
    // dC / dy_i = 4 * sum_j (p_ij * exag - q_ij) * qVal * (y_i - y_j)
    let totalKL = 0;
    const gradX = new Float32Array(N);
    const gradY = new Float32Array(N);

    for (let i = 0; i < N; i++) {
      let gx = 0;
      let gy = 0;

      for (let j = 0; j < N; j++) {
        if (i === j) continue;
        const p_ij = state.P[i * N + j] * exaggeration;
        const q_ij = Math.max(1e-12, numQ[i * N + j] / sumNumQ);

        if (state.P[i * N + j] > 1e-12) {
          totalKL += state.P[i * N + j] * Math.log(state.P[i * N + j] / q_ij);
        }

        const mult = 4.0 * (p_ij - q_ij) * numQ[i * N + j];
        gx += mult * (state.Y[i][0] - state.Y[j][0]);
        gy += mult * (state.Y[i][1] - state.Y[j][1]);
      }

      gradX[i] = gx;
      gradY[i] = gy;
    }

    // 3. Update positions with adaptive gains & momentum
    const nextY: [number, number][] = [];
    const nextDY: [number, number][] = [];
    const nextGains: [number, number][] = [];

    let meanY0 = 0;
    let meanY1 = 0;

    for (let i = 0; i < N; i++) {
      // Adaptive gains (Jacobs delta-bar-delta heuristic)
      let gain0 = (gradX[i] > 0) !== (state.dY[i][0] > 0) ? state.gains[i][0] + 0.2 : state.gains[i][0] * 0.8;
      let gain1 = (gradY[i] > 0) !== (state.dY[i][1] > 0) ? state.gains[i][1] + 0.2 : state.gains[i][1] * 0.8;
      gain0 = Math.max(0.01, gain0);
      gain1 = Math.max(0.01, gain1);

      const dy0 = momentum * state.dY[i][0] - learningRate * gain0 * gradX[i];
      const dy1 = momentum * state.dY[i][1] - learningRate * gain1 * gradY[i];

      const y0 = state.Y[i][0] + dy0;
      const y1 = state.Y[i][1] + dy1;

      meanY0 += y0;
      meanY1 += y1;

      nextY.push([y0, y1]);
      nextDY.push([dy0, dy1]);
      nextGains.push([gain0, gain1]);
    }

    // Zero-center embeddings
    meanY0 /= N;
    meanY1 /= N;
    for (let i = 0; i < N; i++) {
      nextY[i][0] -= meanY0;
      nextY[i][1] -= meanY1;
    }

    const lossHistory = [
      ...state.lossHistory.slice(-50),
      { iteration: state.epoch + 1, loss: totalKL, name: 'KL Loss' },
    ];

    const nextState: TSNEState = {
      ...state,
      Y: nextY,
      dY: nextDY,
      gains: nextGains,
      klLoss: totalKL,
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

  predict(input: Float32Array, state: TSNEState, _params: TSNEParams): number {
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

  loss(state: TSNEState, _dataset: DatasetContainer, _params: TSNEParams): TelemetryMetrics {
    return {
      trainLoss: state.klLoss,
      testLoss: state.klLoss,
      trainAccuracy: Math.max(0, Math.min(100, Math.round(100 - state.klLoss * 15))),
      custom: {
        klDivergence: Number(state.klLoss.toFixed(4)),
        perplexity: state.perplexity,
        learningRate: state.learningRate,
      },
    };
  },

  getVisualArtifacts(state: TSNEState, dataset: DatasetContainer, _params: TSNEParams): VisualArtifact[] {
    const embeddedPoints = state.Y.map((y, i) => ({
      id: dataset.train[i]?.id ?? i,
      x: y[0],
      y: y[1],
      label: dataset.train[i]?.y ?? 1,
    }));

    return [
      {
        type: 'manifold-overlay',
        method: 'tsne',
        embeddedPoints,
        lossTelemetry: state.lossHistory,
      },
    ];
  },

  serializeSnapshot(state: TSNEState): Float32Array {
    const N = state.Y.length;
    const buf = new Float32Array(3 + N * 2);
    buf[0] = state.epoch;
    buf[1] = state.klLoss;
    buf[2] = N;
    for (let i = 0; i < N; i++) {
      buf[3 + i * 2] = state.Y[i][0];
      buf[3 + i * 2 + 1] = state.Y[i][1];
    }
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: TSNEState): TSNEState {
    const N = Math.round(buffer[2]);
    const Y: [number, number][] = [];
    for (let i = 0; i < N; i++) {
      Y.push([buffer[3 + i * 2], buffer[3 + i * 2 + 1]]);
    }
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      klLoss: buffer[1],
      Y,
    };
  },

  explain(state: TSNEState, _params: TSNEParams, _lastStep?: StepResult<TSNEState>): ExplanationPayload {
    return {
      summary: `t-SNE optimizes a 2D non-linear manifold representation by minimizing the KL-divergence between Gaussian high-dimensional affinities and heavy-tailed Student-t embedding affinities (KL Loss = ${state.klLoss.toFixed(3)}).`,
      coreAssumption:
        'Local neighborhood proximity is informative while long-range Euclidean distance is unreliable; heavy-tailed Student-t distribution alleviates crowding problem.',
      whenItFails:
        'Perplexity mis-tuning (perplexity = 2 breaks data into meaningless micro-clumps; perplexity = N collapses points into a uniform sphere) and inter-cluster distances cannot be interpreted quantitatively.',
      microLog: [
        `Calibrate high-D Gaussian perplexity σ_i for Perplexity = ${state.perplexity}`,
        `Compute low-D Student-t joint affinities q_ij = (1 + ||y_i - y_j||^2)^(-1) / Z`,
        `Gradient descent step with momentum: untangling 2D embedding points (KL = ${state.klLoss.toFixed(3)})`,
      ],
      mathLatex: {
        objective: `KL(P \\parallel Q) = \\sum_{i \\neq j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}}, \\quad q_{ij} = \\frac{(1 + \\|y_i - y_j\\|^2)^{-1}}{\\sum_{k \\neq l}(1 + \\|y_k - y_l\\|^2)^{-1}}`,
        lossOrUpdate: `\\frac{\\partial KL}{\\partial y_i} = 4 \\sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + \\|y_i - y_j\\|^2)^{-1}`,
      },
      codeSnippet: {
        sklearn: `from sklearn.manifold import TSNE\ntsne = TSNE(n_components=2, perplexity=${state.perplexity}, learning_rate=${state.learningRate})\nY = tsne.fit_transform(X)`,
        pytorch: `# Custom PyTorch t-SNE with autograd and Adam optimizer`,
      },
    };
  },
};
