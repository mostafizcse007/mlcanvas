import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { eigen2x2Symmetric } from './clusteringUtils';

export interface PCAState {
  nComponents: 1 | 2;
  whiten: boolean;
  centering: boolean;
  mean: [number, number];
  eigenvalues: [number, number]; // lambda1, lambda2
  eigenvectors: [[number, number], [number, number]]; // [v1, v2]
  varianceRatios: [number, number]; // [ratio1, ratio2]
  projected1D: { pointId: number; value: number; label: number }[];
  dropLines: { from: [number, number]; to: [number, number]; pointId: number; label: number }[];
  reconstructionMSE: number;
  epoch: number;
  converged: boolean;
}

export interface PCAParams {
  nComponents?: 1 | 2;
  whiten?: 'off' | 'on' | boolean;
  centering?: 'mean-centered' | 'uncentered' | boolean;
}

function computePCA(
  points: { id: number; x1: number; x2: number; y: number }[],
  centering: boolean,
  whiten: boolean
): {
  mean: [number, number];
  eigenvalues: [number, number];
  eigenvectors: [[number, number], [number, number]];
  varianceRatios: [number, number];
  projected1D: PCAState['projected1D'];
  dropLines: PCAState['dropLines'];
  reconstructionMSE: number;
} {
  const N = points.length;
  if (N === 0) {
    return {
      mean: [0, 0],
      eigenvalues: [1, 1],
      eigenvectors: [
        [1, 0],
        [0, 1],
      ],
      varianceRatios: [0.5, 0.5],
      projected1D: [],
      dropLines: [],
      reconstructionMSE: 0,
    };
  }

  // 1. Mean
  let sumX = 0;
  let sumY = 0;
  for (let i = 0; i < N; i++) {
    sumX += points[i].x1;
    sumY += points[i].x2;
  }
  const mean: [number, number] = centering ? [sumX / N, sumY / N] : [0, 0];

  // 2. Covariance matrix Sigma = (1 / N) * X_c^T * X_c
  let s00 = 0;
  let s01 = 0;
  let s11 = 0;

  for (let i = 0; i < N; i++) {
    const dx = points[i].x1 - mean[0];
    const dy = points[i].x2 - mean[1];
    s00 += dx * dx;
    s01 += dx * dy;
    s11 += dy * dy;
  }

  const covA = s00 / Math.max(1, N);
  const covB = s01 / Math.max(1, N);
  const covD = s11 / Math.max(1, N);

  // 3. Eigendecomposition
  const eigen = eigen2x2Symmetric(covA, covB, covD);
  let [l1, l2] = eigen.values;
  l1 = Math.max(1e-9, l1);
  l2 = Math.max(1e-9, l2);

  const [v1, v2] = eigen.vectors;
  const totalVar = l1 + l2;
  const varianceRatios: [number, number] = [l1 / totalVar, l2 / totalVar];

  // 4. Projections & Drop-Lines
  const projected1D: PCAState['projected1D'] = [];
  const dropLines: PCAState['dropLines'] = [];
  let totalSqError = 0;

  for (let i = 0; i < N; i++) {
    const pt = points[i];
    const dx = pt.x1 - mean[0];
    const dy = pt.x2 - mean[1];

    // Dot product with primary eigenvector v1
    let z = dx * v1[0] + dy * v1[1];
    if (whiten) {
      z /= Math.sqrt(l1);
    }

    projected1D.push({
      pointId: pt.id,
      value: z,
      label: pt.y,
    });

    // Reconstructed point in 2D space along v1
    const reconX = mean[0] + (whiten ? z * Math.sqrt(l1) : z) * v1[0];
    const reconY = mean[1] + (whiten ? z * Math.sqrt(l1) : z) * v1[1];

    dropLines.push({
      from: [pt.x1, pt.x2],
      to: [reconX, reconY],
      pointId: pt.id,
      label: pt.y,
    });

    const residualDist = Math.hypot(pt.x1 - reconX, pt.x2 - reconY);
    totalSqError += residualDist * residualDist;
  }

  const reconstructionMSE = totalSqError / Math.max(1, N);

  return {
    mean,
    eigenvalues: [l1, l2],
    eigenvectors: [v1, v2],
    varianceRatios,
    projected1D,
    dropLines,
    reconstructionMSE,
  };
}

export const pcaPlugin: AlgorithmPlugin<PCAState, PCAParams> = {
  id: 'pca',
  name: 'Principal Component Analysis (PCA)',
  shortCode: 'PCA',
  stage: 'unsupervised-2d',
  taskType: 'dimensionality-reduction',
  executionMode: 'closed-form',
  description:
    'Orthogonal linear transformation maximizing variance along orthogonal eigenvectors and minimizing residual projection error.',

  primaryParams: [
    {
      id: 'nComponents',
      name: 'Components',
      type: 'select',
      options: [
        { label: '1 Component (1D Axis)', value: 1 },
        { label: '2 Components (Full Basis)', value: 2 },
      ],
      default: 1,
      description: 'Number of principal eigenvectors retained for projection.',
    },
    {
      id: 'whiten',
      name: 'Whiten (Scale / √λ)',
      type: 'select',
      options: [
        { label: 'Off (Raw Variance)', value: 'off' },
        { label: 'On (Unit Variance)', value: 'on' },
      ],
      default: 'off',
      description: 'Scales projected components by the inverse square root of eigenvalues.',
    },
    {
      id: 'centering',
      name: 'Mean Centering',
      type: 'select',
      options: [
        { label: 'Mean-Centered (Standard)', value: 'mean-centered' },
        { label: 'Uncentered (Origin-Fixed)', value: 'uncentered' },
      ],
      default: 'mean-centered',
      description: 'Subtracts empirical mean before computing covariance matrix.',
    },
  ],

  advancedParams: [],

  init(dataset: DatasetContainer, params: PCAParams, _seed: number = 42): PCAState {
    const nComponents = (Number(params.nComponents ?? 1) as 1 | 2) || 1;
    const whiten = params.whiten === 'on' || params.whiten === true;
    const centering = params.centering !== 'uncentered';

    const result = computePCA(dataset.train, centering, whiten);

    return {
      nComponents,
      whiten,
      centering,
      mean: result.mean,
      eigenvalues: result.eigenvalues,
      eigenvectors: result.eigenvectors,
      varianceRatios: result.varianceRatios,
      projected1D: result.projected1D,
      dropLines: result.dropLines,
      reconstructionMSE: result.reconstructionMSE,
      epoch: 0,
      converged: true,
    };
  },

  step(state: PCAState, dataset: DatasetContainer, params: PCAParams): StepResult<PCAState> {
    const whiten = params.whiten === 'on' || params.whiten === true;
    const centering = params.centering !== 'uncentered';
    const result = computePCA(dataset.train, centering, whiten);

    const nextState: PCAState = {
      ...state,
      mean: result.mean,
      eigenvalues: result.eigenvalues,
      eigenvectors: result.eigenvectors,
      varianceRatios: result.varianceRatios,
      projected1D: result.projected1D,
      dropLines: result.dropLines,
      reconstructionMSE: result.reconstructionMSE,
      epoch: state.epoch + 1,
      converged: true,
    };

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: this.loss(nextState, dataset, params),
      converged: true,
    };
  },

  predict(input: Float32Array, state: PCAState, _params: PCAParams): number {
    const dx = input[0] - state.mean[0];
    const dy = input[1] - state.mean[1];
    const v1 = state.eigenvectors[0];
    return dx * v1[0] + dy * v1[1];
  },

  loss(state: PCAState, _dataset: DatasetContainer, _params: PCAParams): TelemetryMetrics {
    const explainedVarPct = Math.round(state.varianceRatios[0] * 1000) / 10;
    return {
      trainLoss: state.reconstructionMSE,
      testLoss: state.reconstructionMSE,
      trainAccuracy: explainedVarPct,
      custom: {
        explainedVariancePct: `${explainedVarPct}%`,
        eigenvalue1: Number(state.eigenvalues[0].toFixed(3)),
        eigenvalue2: Number(state.eigenvalues[1].toFixed(3)),
        reconstructionMSE: Number(state.reconstructionMSE.toFixed(4)),
      },
    };
  },

  getVisualArtifacts(state: PCAState, _dataset: DatasetContainer, _params: PCAParams): VisualArtifact[] {
    const [v1, v2] = state.eigenvectors;
    const [l1, l2] = state.eigenvalues;
    const [r1, r2] = state.varianceRatios;

    const scale1 = Math.sqrt(l1) * 2.2;
    const scale2 = Math.sqrt(l2) * 2.2;

    const axes = [
      {
        id: 'pc1',
        direction: v1,
        length: scale1,
        varianceRatio: r1,
        label: `PC1 (${(r1 * 100).toFixed(1)}%)`,
        color: '#2563EB',
      },
    ];

    if (state.nComponents === 2) {
      axes.push({
        id: 'pc2',
        direction: v2,
        length: scale2,
        varianceRatio: r2,
        label: `PC2 (${(r2 * 100).toFixed(1)}%)`,
        color: '#7C3AED',
      });
    }

    const screeData = [
      {
        component: 1,
        eigenvalue: l1,
        varianceRatio: r1,
        cumulativeVariance: r1,
      },
      {
        component: 2,
        eigenvalue: l2,
        varianceRatio: r2,
        cumulativeVariance: 1.0,
      },
    ];

    return [
      {
        type: 'projection-overlay',
        method: 'pca',
        center: state.mean,
        axes,
        dropLines: state.dropLines,
        screeData,
        projected1D: state.projected1D,
      },
    ];
  },

  serializeSnapshot(state: PCAState): Float32Array {
    const buf = new Float32Array(9);
    buf[0] = state.epoch;
    buf[1] = state.mean[0];
    buf[2] = state.mean[1];
    buf[3] = state.eigenvalues[0];
    buf[4] = state.eigenvalues[1];
    buf[5] = state.eigenvectors[0][0];
    buf[6] = state.eigenvectors[0][1];
    buf[7] = state.varianceRatios[0];
    buf[8] = state.reconstructionMSE;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: PCAState): PCAState {
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      mean: [buffer[1], buffer[2]],
      eigenvalues: [buffer[3], buffer[4]],
      eigenvectors: [
        [buffer[5], buffer[6]],
        [-buffer[6], buffer[5]],
      ],
      varianceRatios: [buffer[7], 1 - buffer[7]],
      reconstructionMSE: buffer[8],
    };
  },

  explain(state: PCAState, _params: PCAParams, _lastStep?: StepResult<PCAState>): ExplanationPayload {
    return {
      summary: `Principal Component Analysis aligns orthogonal projection axes along the directions of maximal variance (PC1 accounts for ${(state.varianceRatios[0] * 100).toFixed(1)}% of total variance).`,
      coreAssumption:
        'High variance directions represent the most informative signals; variables are linearly correlated with orthogonal underlying factors.',
      whenItFails:
        'Supervised separation (max-variance axis may align along class overlap rather than class distinction) and nonlinear manifolds (curved spirals collapse into overlapping noise).',
      microLog: [
        `Mean-center dataset at μ = [${state.mean[0].toFixed(2)}, ${state.mean[1].toFixed(2)}]`,
        `Solve covariance matrix eigendecomposition: λ₁ = ${state.eigenvalues[0].toFixed(2)}, λ₂ = ${state.eigenvalues[1].toFixed(2)}`,
        `Drop-lines show perpendicular orthogonal projection onto PC1 (Reconstruction MSE = ${state.reconstructionMSE.toFixed(3)})`,
      ],
      mathLatex: {
        objective: `\\max_{\\|v_1\\|=1} v_1^T \\Sigma v_1 = \\max_{\\|v_1\\|=1} \\frac{1}{N} \\sum_{i=1}^N (v_1^T(x_i - \\mu))^2`,
        lossOrUpdate: `\\Sigma v_1 = \\lambda_1 v_1, \\quad z_i = v_1^T (x_i - \\mu), \\quad \\text{MSE} = \\frac{1}{N}\\sum_{i=1}^N \\|x_i - (\\mu + z_i v_1)\\|^2`,
      },
      codeSnippet: {
        sklearn: `from sklearn.decomposition import PCA\npca = PCA(n_components=${state.nComponents}, whiten=${state.whiten})\nX_pca = pca.fit_transform(X)\nprint("Explained variance:", pca.explained_variance_ratio_)`,
        pytorch: `# PyTorch SVD\nU, S, V = torch.pca_lowrank(X, q=${state.nComponents}, center=True)`,
      },
    };
  },
};

