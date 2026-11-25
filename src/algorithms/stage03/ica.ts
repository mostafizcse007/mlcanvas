import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { eigen2x2Symmetric } from './clusteringUtils';
import { mulberry32 } from '../../utils/prng';

export interface ICAState {
  contrast: 'logcosh' | 'exp' | 'cube';
  mean: [number, number];
  whiteningMatrix: [[number, number], [number, number]];
  deWhiteningMatrix: [[number, number], [number, number]];
  unmixingVectors: [[number, number], [number, number]]; // In whitened space
  independentAxes: [[number, number], [number, number]]; // In original source space (can be non-orthogonal!)
  kurtosis: [number, number];
  projected1D: { pointId: number; value: number; label: number }[];
  dropLines: { from: [number, number]; to: [number, number]; pointId: number; label: number }[];
  epoch: number;
  converged: boolean;
}

export interface ICAParams {
  contrast?: 'logcosh' | 'exp' | 'cube';
  maxIter?: number;
}

function computeWhitening(
  points: { x1: number; x2: number }[]
): {
  mean: [number, number];
  whiteningMatrix: [[number, number], [number, number]];
  deWhiteningMatrix: [[number, number], [number, number]];
  whitenedPoints: [number, number][];
} {
  const N = points.length;
  if (N === 0) {
    return {
      mean: [0, 0],
      whiteningMatrix: [
        [1, 0],
        [0, 1],
      ],
      deWhiteningMatrix: [
        [1, 0],
        [0, 1],
      ],
      whitenedPoints: [],
    };
  }

  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += p.x1;
    sy += p.x2;
  }
  const mean: [number, number] = [sx / N, sy / N];

  // Covariance
  let c00 = 0;
  let c01 = 0;
  let c11 = 0;
  for (const p of points) {
    const dx = p.x1 - mean[0];
    const dy = p.x2 - mean[1];
    c00 += dx * dx;
    c01 += dx * dy;
    c11 += dy * dy;
  }
  c00 /= N;
  c01 /= N;
  c11 /= N;

  // Eigendecomposition: Sigma = E * D * E^T
  const eigen = eigen2x2Symmetric(c00, c01, c11);
  const l1 = Math.max(1e-6, eigen.values[0]);
  const l2 = Math.max(1e-6, eigen.values[1]);
  const [e1, e2] = eigen.vectors;

  // Whitening matrix: W_white = D^(-1/2) * E^T
  const invSqrt1 = 1.0 / Math.sqrt(l1);
  const invSqrt2 = 1.0 / Math.sqrt(l2);

  const w00 = invSqrt1 * e1[0];
  const w01 = invSqrt1 * e1[1];
  const w10 = invSqrt2 * e2[0];
  const w11 = invSqrt2 * e2[1];

  // De-whitening matrix: E * D^(1/2)
  const sqrt1 = Math.sqrt(l1);
  const sqrt2 = Math.sqrt(l2);
  const dw00 = e1[0] * sqrt1;
  const dw01 = e2[0] * sqrt2;
  const dw10 = e1[1] * sqrt1;
  const dw11 = e2[1] * sqrt2;

  // Transform points: z = W_white * (x - mean)
  const whitenedPoints: [number, number][] = points.map((p) => {
    const dx = p.x1 - mean[0];
    const dy = p.x2 - mean[1];
    return [w00 * dx + w01 * dy, w10 * dx + w11 * dy];
  });

  return {
    mean,
    whiteningMatrix: [
      [w00, w01],
      [w10, w11],
    ],
    deWhiteningMatrix: [
      [dw00, dw01],
      [dw10, dw11],
    ],
    whitenedPoints,
  };
}

function evaluateContrast(u: number, contrast: 'logcosh' | 'exp' | 'cube'): { g: number; gp: number } {
  if (contrast === 'logcosh') {
    const t = Math.tanh(u);
    return { g: t, gp: 1 - t * t };
  } else if (contrast === 'exp') {
    const expVal = Math.exp(-0.5 * u * u);
    return { g: u * expVal, gp: (1 - u * u) * expVal };
  } else {
    // cube
    return { g: u * u * u, gp: 3 * u * u };
  }
}

export const icaPlugin: AlgorithmPlugin<ICAState, ICAParams> = {
  id: 'ica',
  name: 'Independent Component Analysis (FastICA)',
  shortCode: 'FastICA',
  stage: 'unsupervised-2d',
  taskType: 'dimensionality-reduction',
  executionMode: 'epoch-loop',
  description:
    'Fixed-point non-Gaussianity maximization extracting statistically independent, non-orthogonal underlying signal sources.',

  primaryParams: [
    {
      id: 'contrast',
      name: 'Contrast Function',
      type: 'select',
      options: [
        { label: 'logcosh (Hyperbolic Tangent)', value: 'logcosh' },
        { label: 'Exponential Gaussian', value: 'exp' },
        { label: 'Kurtosis (Cubic Polynomial)', value: 'cube' },
      ],
      default: 'logcosh',
      description: 'Non-linear negentropy contrast proxy function g(u).',
    },
  ],

  advancedParams: [
    {
      id: 'maxIter',
      name: 'Max Iterations',
      type: 'slider',
      min: 5,
      max: 100,
      step: 5,
      default: 30,
      description: 'Maximum FastICA fixed-point iteration updates.',
    },
  ],

  init(dataset: DatasetContainer, params: ICAParams, seed: number = 42): ICAState {
    const contrast = params.contrast || 'logcosh';
    const { mean, whiteningMatrix, deWhiteningMatrix } = computeWhitening(dataset.train);
    const rand = mulberry32(seed);

    // Initial random orthogonal vectors in whitened space
    const theta = rand() * Math.PI * 2;
    const w1: [number, number] = [Math.cos(theta), Math.sin(theta)];
    const w2: [number, number] = [-Math.sin(theta), Math.cos(theta)];

    // Map whitened axes back to source space: axis = deWhitening * w
    const ax1X = deWhiteningMatrix[0][0] * w1[0] + deWhiteningMatrix[0][1] * w1[1];
    const ax1Y = deWhiteningMatrix[1][0] * w1[0] + deWhiteningMatrix[1][1] * w1[1];
    const n1 = Math.hypot(ax1X, ax1Y) || 1;

    const ax2X = deWhiteningMatrix[0][0] * w2[0] + deWhiteningMatrix[0][1] * w2[1];
    const ax2Y = deWhiteningMatrix[1][0] * w2[0] + deWhiteningMatrix[1][1] * w2[1];
    const n2 = Math.hypot(ax2X, ax2Y) || 1;

    // Projected 1D & Drop-Lines
    const projected1D: ICAState['projected1D'] = [];
    const dropLines: ICAState['dropLines'] = [];
    for (const p of dataset.train) {
      const dx = p.x1 - mean[0];
      const dy = p.x2 - mean[1];
      const z = (dx * ax1X + dy * ax1Y) / n1;
      projected1D.push({ pointId: p.id, value: z, label: p.y });
      dropLines.push({
        from: [p.x1, p.x2],
        to: [mean[0] + (z * ax1X) / n1, mean[1] + (z * ax1Y) / n1],
        pointId: p.id,
        label: p.y,
      });
    }

    return {
      contrast,
      mean,
      whiteningMatrix,
      deWhiteningMatrix,
      unmixingVectors: [w1, w2],
      independentAxes: [
        [ax1X / n1, ax1Y / n1],
        [ax2X / n2, ax2Y / n2],
      ],
      kurtosis: [0, 0],
      projected1D,
      dropLines,
      epoch: 0,
      converged: false,
    };
  },

  step(state: ICAState, dataset: DatasetContainer, params: ICAParams): StepResult<ICAState> {
    const contrast = params.contrast || state.contrast;
    const { mean, whiteningMatrix, deWhiteningMatrix, whitenedPoints } = computeWhitening(dataset.train);
    const N = whitenedPoints.length;

    if (N === 0) {
      return { state, epoch: state.epoch, metrics: this.loss(state, dataset, params), converged: true };
    }

    let [w1, w2] = state.unmixingVectors;

    // FastICA Fixed-point update for w1:
    // w1+ = E[z * g(w1^T z)] - E[g'(w1^T z)] * w1
    let sum1X = 0;
    let sum1Y = 0;
    let sumGp1 = 0;

    for (let i = 0; i < N; i++) {
      const z = whitenedPoints[i];
      const u = w1[0] * z[0] + w1[1] * z[1];
      const { g, gp } = evaluateContrast(u, contrast);
      sum1X += z[0] * g;
      sum1Y += z[1] * g;
      sumGp1 += gp;
    }

    const expGp1 = sumGp1 / N;
    const w1NewX = sum1X / N - expGp1 * w1[0];
    const w1NewY = sum1Y / N - expGp1 * w1[1];
    const norm1 = Math.hypot(w1NewX, w1NewY);
    if (norm1 > 1e-9 && Number.isFinite(norm1)) {
      w1 = [w1NewX / norm1, w1NewY / norm1];
    }

    // Fixed-point update for w2 with Gram-Schmidt orthogonalization in whitened space
    let sum2X = 0;
    let sum2Y = 0;
    let sumGp2 = 0;
    for (let i = 0; i < N; i++) {
      const z = whitenedPoints[i];
      const u = w2[0] * z[0] + w2[1] * z[1];
      const { g, gp } = evaluateContrast(u, contrast);
      sum2X += z[0] * g;
      sum2Y += z[1] * g;
      sumGp2 += gp;
    }
    const expGp2 = sumGp2 / N;
    let w2NewX = sum2X / N - expGp2 * w2[0];
    let w2NewY = sum2Y / N - expGp2 * w2[1];

    // Orthogonalize w2 against w1
    const proj = w2NewX * w1[0] + w2NewY * w1[1];
    w2NewX -= proj * w1[0];
    w2NewY -= proj * w1[1];
    const norm2 = Math.hypot(w2NewX, w2NewY);
    if (norm2 > 1e-9 && Number.isFinite(norm2)) {
      w2 = [w2NewX / norm2, w2NewY / norm2];
    } else {
      w2 = [-w1[1], w1[0]];
    }

    // Transform unmixing vectors back to original unwhitened space
    // Notice: in source space, these two vectors are free to be NON-ORTHOGONAL!
    const ax1X = deWhiteningMatrix[0][0] * w1[0] + deWhiteningMatrix[0][1] * w1[1];
    const ax1Y = deWhiteningMatrix[1][0] * w1[0] + deWhiteningMatrix[1][1] * w1[1];
    const n1 = Math.hypot(ax1X, ax1Y) || 1;

    const ax2X = deWhiteningMatrix[0][0] * w2[0] + deWhiteningMatrix[0][1] * w2[1];
    const ax2Y = deWhiteningMatrix[1][0] * w2[0] + deWhiteningMatrix[1][1] * w2[1];
    const n2 = Math.hypot(ax2X, ax2Y) || 1;

    // Compute empirical kurtosis along both axes
    let k1 = 0;
    let k2 = 0;
    for (let i = 0; i < N; i++) {
      const z = whitenedPoints[i];
      const u1 = w1[0] * z[0] + w1[1] * z[1];
      const u2 = w2[0] * z[0] + w2[1] * z[1];
      k1 += Math.pow(u1, 4);
      k2 += Math.pow(u2, 4);
    }
    k1 = Math.abs(k1 / N - 3.0); // Excess kurtosis vs Gaussian
    k2 = Math.abs(k2 / N - 3.0);

    // Projected 1D & Drop-Lines
    const projected1D: ICAState['projected1D'] = [];
    const dropLines: ICAState['dropLines'] = [];
    for (const p of dataset.train) {
      const dx = p.x1 - mean[0];
      const dy = p.x2 - mean[1];
      const z = (dx * ax1X + dy * ax1Y) / n1;
      projected1D.push({ pointId: p.id, value: z, label: p.y });
      dropLines.push({
        from: [p.x1, p.x2],
        to: [mean[0] + (z * ax1X) / n1, mean[1] + (z * ax1Y) / n1],
        pointId: p.id,
        label: p.y,
      });
    }

    const nextState: ICAState = {
      contrast,
      mean,
      whiteningMatrix,
      deWhiteningMatrix,
      unmixingVectors: [w1, w2],
      independentAxes: [
        [ax1X / n1, ax1Y / n1],
        [ax2X / n2, ax2Y / n2],
      ],
      kurtosis: [k1, k2],
      projected1D,
      dropLines,
      epoch: state.epoch + 1,
      converged: state.epoch > 20,
    };

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: this.loss(nextState, dataset, params),
      converged: nextState.converged,
    };
  },

  predict(input: Float32Array, state: ICAState, _params: ICAParams): number {
    const dx = input[0] - state.mean[0];
    const dy = input[1] - state.mean[1];
    const ax1 = state.independentAxes[0];
    return dx * ax1[0] + dy * ax1[1];
  },

  loss(state: ICAState, _dataset: DatasetContainer, _params: ICAParams): TelemetryMetrics {
    const avgKurtosis = (state.kurtosis[0] + state.kurtosis[1]) / 2;
    return {
      trainLoss: 1.0 / Math.max(0.01, avgKurtosis),
      testLoss: 1.0 / Math.max(0.01, avgKurtosis),
      trainAccuracy: Math.min(100, Math.round(avgKurtosis * 40)),
      custom: {
        kurtosis1: Number(state.kurtosis[0].toFixed(3)),
        kurtosis2: Number(state.kurtosis[1].toFixed(3)),
        contrast: state.contrast,
      },
    };
  },

  getVisualArtifacts(state: ICAState, _dataset: DatasetContainer, _params: ICAParams): VisualArtifact[] {
    const [ax1, ax2] = state.independentAxes;

    // Dot product between independent axes to reveal non-orthogonality
    const dotProduct = ax1[0] * ax2[0] + ax1[1] * ax2[1];
    const angleDeg = (Math.acos(Math.max(-1, Math.min(1, dotProduct))) * 180) / Math.PI;

    const axes = [
      {
        id: 'ica-ic1',
        direction: ax1,
        length: 4.0,
        label: `IC1 (Kurt = ${state.kurtosis[0].toFixed(2)})`,
        color: '#D97706', // Amber
      },
      {
        id: 'ica-ic2',
        direction: ax2,
        length: 3.8,
        label: `IC2 (Non-Orthogonal: ${angleDeg.toFixed(0)}°)`,
        color: '#059669', // Emerald
      },
    ];

    return [
      {
        type: 'projection-overlay',
        method: 'ica',
        center: state.mean,
        axes,
        dropLines: state.dropLines,
        projected1D: state.projected1D,
        icaKurtosis: [state.kurtosis[0], state.kurtosis[1]],
      },
    ];
  },

  serializeSnapshot(state: ICAState): Float32Array {
    const buf = new Float32Array(8);
    buf[0] = state.epoch;
    buf[1] = state.independentAxes[0][0];
    buf[2] = state.independentAxes[0][1];
    buf[3] = state.independentAxes[1][0];
    buf[4] = state.independentAxes[1][1];
    buf[5] = state.kurtosis[0];
    buf[6] = state.kurtosis[1];
    buf[7] = state.converged ? 1 : 0;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: ICAState): ICAState {
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      independentAxes: [
        [buffer[1], buffer[2]],
        [buffer[3], buffer[4]],
      ],
      kurtosis: [buffer[5], buffer[6]],
      converged: buffer[7] === 1,
    };
  },

  explain(state: ICAState, _params: ICAParams, _lastStep?: StepResult<ICAState>): ExplanationPayload {
    return {
      summary: `FastICA isolates non-Gaussian independent sources by rotating whitened projection axes to maximize negentropy (Kurtosis = [${state.kurtosis[0].toFixed(2)}, ${state.kurtosis[1].toFixed(2)}]).`,
      coreAssumption:
        'Observed features are linear mixtures of statistically independent, non-Gaussian source distributions.',
      whenItFails:
        'Pure Gaussian sources (rotational symmetry makes independent components unidentifiable; orthogonal rotations yield identical likelihoods).',
      microLog: [
        `Pre-whiten data to identity covariance Z = X Σ^(-1/2)`,
        `Evaluate FastICA fixed-point iteration using ${state.contrast} contrast function`,
        `Map unmixing vectors back to source space: independent axes are free to be non-orthogonal`,
      ],
      mathLatex: {
        objective: `J(y) \\approx \\frac{1}{12} E[y^3]^2 + \\frac{1}{48} \\text{kurt}(y)^2 \\quad (\\text{Negentropy Approximation})`,
        lossOrUpdate: `w^+ = E[z g(w^T z)] - E[g'(w^T z)] w, \\quad w = \\frac{w^+}{\\|w^+\\|}`,
      },
      codeSnippet: {
        sklearn: `from sklearn.decomposition import FastICA\nica = FastICA(n_components=2, fun='${state.contrast}', random_state=42)\nS_ = ica.fit_transform(X)`,
        pytorch: `# FastICA PyTorch batch expectation and Gram-Schmidt orthogonalization`,
      },
    };
  },
};
