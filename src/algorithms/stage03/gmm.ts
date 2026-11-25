import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import {
  eigen2x2Symmetric,
  computeSilhouetteScore,
  CLUSTER_PALETTE,
} from './clusteringUtils';
import { mulberry32 } from '../../utils/prng';

export interface GMMComponent {
  id: number;
  weight: number; // pi_k
  mean: [number, number]; // mu_k
  cov: [[number, number], [number, number]]; // Sigma_k
  eigenvalues: [number, number];
  eigenvectors: [[number, number], [number, number]];
  angle: number;
}

export interface GMMState {
  k: number;
  covarianceType: 'full' | 'diag' | 'spherical';
  regCovar: number;
  components: GMMComponent[];
  responsibilities: number[][]; // N x K
  hardAssignments: number[]; // N
  subStep: 'e-step' | 'm-step';
  epoch: number;
  converged: boolean;
  logLikelihood: number;
  bic: number;
  silhouette: number;
}

export interface GMMParams {
  k?: number;
  covarianceType?: 'full' | 'diag' | 'spherical';
  regCovar?: number;
}

function gaussian2dPdf(
  x: [number, number],
  mean: [number, number],
  cov: [[number, number], [number, number]]
): number {
  const dx = x[0] - mean[0];
  const dy = x[1] - mean[1];

  const c00 = cov[0][0];
  const c01 = cov[0][1];
  const c10 = cov[1][0];
  const c11 = cov[1][1];

  const det = Math.max(1e-12, c00 * c11 - c01 * c10);
  // Inverse 2x2
  const inv00 = c11 / det;
  const inv01 = -c01 / det;
  const inv10 = -c10 / det;
  const inv11 = c00 / det;

  // Mahalanobis distance squared: dx^T * inv * dx
  const quad = dx * (inv00 * dx + inv01 * dy) + dy * (inv10 * dx + inv11 * dy);
  const normConst = 1.0 / (2 * Math.PI * Math.sqrt(det));

  return normConst * Math.exp(-0.5 * Math.max(0, quad));
}

function computeResponsibilities(
  points: [number, number][],
  components: GMMComponent[]
): { responsibilities: number[][]; hardAssignments: number[]; logLikelihood: number } {
  const N = points.length;
  const K = components.length;
  const responsibilities: number[][] = Array.from({ length: N }, () => new Array(K).fill(0));
  const hardAssignments: number[] = new Array(N).fill(0);
  let totalLogLikelihood = 0;

  for (let i = 0; i < N; i++) {
    const pt = points[i];
    let sumWeightProb = 0;

    for (let k = 0; k < K; k++) {
      const comp = components[k];
      const prob = gaussian2dPdf(pt, comp.mean, comp.cov);
      const wp = Math.max(1e-15, comp.weight * prob);
      responsibilities[i][k] = wp;
      sumWeightProb += wp;
    }

    totalLogLikelihood += Math.log(Math.max(1e-15, sumWeightProb));

    // Normalize
    let maxResp = -1;
    let bestK = 0;
    for (let k = 0; k < K; k++) {
      responsibilities[i][k] /= Math.max(1e-15, sumWeightProb);
      if (responsibilities[i][k] > maxResp) {
        maxResp = responsibilities[i][k];
        bestK = k;
      }
    }
    hardAssignments[i] = bestK;
  }

  return { responsibilities, hardAssignments, logLikelihood: totalLogLikelihood };
}

function updateComponentsFromResponsibilities(
  points: [number, number][],
  responsibilities: number[][],
  covType: 'full' | 'diag' | 'spherical',
  regCovar: number,
  oldComponents: GMMComponent[]
): GMMComponent[] {
  const N = points.length;
  const K = oldComponents.length;
  const newComponents: GMMComponent[] = [];

  for (let k = 0; k < K; k++) {
    let Nk = 0;
    for (let i = 0; i < N; i++) {
      Nk += responsibilities[i][k];
    }
    Nk = Math.max(1e-5, Nk);

    // Update weight pi_k
    const weight = Nk / Math.max(1, N);

    // Update mean mu_k
    let meanX = 0;
    let meanY = 0;
    for (let i = 0; i < N; i++) {
      meanX += responsibilities[i][k] * points[i][0];
      meanY += responsibilities[i][k] * points[i][1];
    }
    const mean: [number, number] = [meanX / Nk, meanY / Nk];

    // Update covariance Sigma_k
    let cov00 = 0;
    let cov01 = 0;
    let cov11 = 0;

    for (let i = 0; i < N; i++) {
      const dx = points[i][0] - mean[0];
      const dy = points[i][1] - mean[1];
      const r = responsibilities[i][k];
      cov00 += r * dx * dx;
      cov01 += r * dx * dy;
      cov11 += r * dy * dy;
    }

    cov00 = cov00 / Nk + regCovar;
    cov11 = cov11 / Nk + regCovar;
    cov01 = cov01 / Nk;

    if (covType === 'diag') {
      cov01 = 0;
    } else if (covType === 'spherical') {
      const avgVar = (cov00 + cov11) / 2;
      cov00 = avgVar;
      cov11 = avgVar;
      cov01 = 0;
    }

    const cov: [[number, number], [number, number]] = [
      [cov00, cov01],
      [cov01, cov11],
    ];

    const eigen = eigen2x2Symmetric(cov00, cov01, cov11);

    newComponents.push({
      id: k,
      weight,
      mean,
      cov,
      eigenvalues: eigen.values,
      eigenvectors: eigen.vectors,
      angle: eigen.angle,
    });
  }

  return newComponents;
}

export const gmmPlugin: AlgorithmPlugin<GMMState, GMMParams> = {
  id: 'gmm',
  name: 'Gaussian Mixture Model (EM Algorithm)',
  shortCode: 'GMM (EM)',
  stage: 'unsupervised-2d',
  taskType: 'density-estimation',
  executionMode: 'phase-walkthrough',
  description:
    'Soft probabilistic clustering via Expectation-Maximization with full covariance ellipsoids and Bayesian Information Criterion.',

  primaryParams: [
    {
      id: 'k',
      name: 'Components (K)',
      type: 'slider',
      min: 1,
      max: 6,
      step: 1,
      default: 3,
      description: 'Number of Gaussian mixture components to fit.',
    },
    {
      id: 'covarianceType',
      name: 'Covariance Type',
      type: 'select',
      options: [
        { label: 'Full (Tilted Ellipses)', value: 'full' },
        { label: 'Tied / Diagonal (Axis-Aligned)', value: 'diag' },
        { label: 'Spherical (Equal Radii)', value: 'spherical' },
      ],
      default: 'full',
      description: 'Geometric flexibility of component covariance matrices.',
    },
    {
      id: 'regCovar',
      name: 'Covariance Reg (ε)',
      type: 'slider',
      min: 0.0001,
      max: 0.1,
      step: 0.001,
      default: 0.005,
      description: 'Diagonal regularization added to covariance matrix to prevent singular collapse.',
    },
  ],

  advancedParams: [],

  init(dataset: DatasetContainer, params: GMMParams, seed: number = 42): GMMState {
    const k = Math.max(1, Math.min(6, Math.round(Number(params.k ?? 3))));
    const covarianceType = params.covarianceType || 'full';
    const regCovar = Math.max(1e-4, Number(params.regCovar ?? 0.005));

    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const rand = mulberry32(seed);

    // Initial random component centers
    const components: GMMComponent[] = [];
    for (let c = 0; c < k; c++) {
      const idx = points.length > 0 ? Math.floor(rand() * points.length) : 0;
      const initialMean: [number, number] = points.length > 0 ? [...points[idx]] : [(rand() - 0.5) * 4, (rand() - 0.5) * 4];

      const initialCov: [[number, number], [number, number]] = [
        [1.0 + regCovar, 0],
        [0, 1.0 + regCovar],
      ];
      const eigen = eigen2x2Symmetric(initialCov[0][0], 0, initialCov[1][1]);

      components.push({
        id: c,
        weight: 1.0 / k,
        mean: initialMean,
        cov: initialCov,
        eigenvalues: eigen.values,
        eigenvectors: eigen.vectors,
        angle: eigen.angle,
      });
    }

    const { responsibilities, hardAssignments, logLikelihood } = computeResponsibilities(points, components);
    const silhouette = k > 1 ? computeSilhouetteScore(points, hardAssignments, k) : 0;

    // BIC = k_params * ln(N) - 2 * ln(L)
    const nParams = k * (2 + (covarianceType === 'full' ? 3 : covarianceType === 'diag' ? 2 : 1) + 1);
    const bic = nParams * Math.log(Math.max(1, points.length)) - 2 * logLikelihood;

    return {
      k,
      covarianceType,
      regCovar,
      components,
      responsibilities,
      hardAssignments,
      subStep: 'e-step',
      epoch: 0,
      converged: false,
      logLikelihood,
      bic,
      silhouette,
    };
  },

  step(state: GMMState, dataset: DatasetContainer, params: GMMParams): StepResult<GMMState> {
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    if (points.length === 0) {
      return { state, epoch: state.epoch, metrics: this.loss(state, dataset, params), converged: true };
    }

    let nextComponents = [...state.components];
    let nextSubStep: 'e-step' | 'm-step' = state.subStep;
    let nextResp = state.responsibilities;
    let nextHard = state.hardAssignments;
    let logLikelihood = state.logLikelihood;

    if (state.subStep === 'e-step') {
      // E-Step: Compute soft responsibilities
      const res = computeResponsibilities(points, state.components);
      nextResp = res.responsibilities;
      nextHard = res.hardAssignments;
      logLikelihood = res.logLikelihood;
      nextSubStep = 'm-step';
    } else {
      // M-Step: Update means, covariances, and mixing weights
      nextComponents = updateComponentsFromResponsibilities(
        points,
        state.responsibilities,
        state.covarianceType,
        state.regCovar,
        state.components
      );
      nextSubStep = 'e-step';
    }

    const silhouette = state.k > 1 ? computeSilhouetteScore(points, nextHard, state.k) : 0;
    const nParams = state.k * 5;
    const bic = nParams * Math.log(Math.max(1, points.length)) - 2 * logLikelihood;

    const nextState: GMMState = {
      ...state,
      components: nextComponents,
      responsibilities: nextResp,
      hardAssignments: nextHard,
      subStep: nextSubStep,
      epoch: state.epoch + 1,
      logLikelihood,
      bic,
      silhouette,
    };

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: this.loss(nextState, dataset, params),
      converged: false,
    };
  },

  predict(input: Float32Array, state: GMMState, _params: GMMParams): number {
    const pt: [number, number] = [input[0], input[1]];
    if (!state.components || state.components.length === 0) return 0;
    if (state.components.length === 2) {
      const p0 = state.components[0].weight * gaussian2dPdf(pt, state.components[0].mean, state.components[0].cov);
      const p1 = state.components[1].weight * gaussian2dPdf(pt, state.components[1].mean, state.components[1].cov);
      return Math.log(Math.max(1e-12, p0)) - Math.log(Math.max(1e-12, p1));
    }
    let pSum = 0;
    let maxP = -1;
    let bestK = 0;
    for (let k = 0; k < state.components.length; k++) {
      const comp = state.components[k];
      const p = comp.weight * gaussian2dPdf(pt, comp.mean, comp.cov);
      pSum += p;
      if (p > maxP) {
        maxP = p;
        bestK = k;
      }
    }
    const resp = maxP / Math.max(1e-12, pSum);
    return bestK % 2 === 0 ? (resp - 0.5) * 4 : -(resp - 0.5) * 4;
  },

  loss(state: GMMState, _dataset: DatasetContainer, _params: GMMParams): TelemetryMetrics {
    const negLogLikelihood = -state.logLikelihood;
    return {
      trainLoss: negLogLikelihood,
      testLoss: negLogLikelihood,
      trainAccuracy: Math.round(state.silhouette * 1000) / 10,
      custom: {
        logLikelihood: Number(state.logLikelihood.toFixed(2)),
        bic: Number(state.bic.toFixed(2)),
        silhouette: Number(state.silhouette.toFixed(3)),
        activeComponents: state.components.length,
        subStep: state.subStep === 'e-step' ? 'E-Step (Responsibilities)' : 'M-Step (Mean/Cov)',
      },
    };
  },

  getVisualArtifacts(state: GMMState, _dataset: DatasetContainer, _params: GMMParams): VisualArtifact[] {
    const centroids = state.components.map((c) => ({
      id: c.id,
      current: c.mean,
      color: CLUSTER_PALETTE[c.id % CLUSTER_PALETTE.length],
    }));

    const gmmEllipses = state.components.map((c) => ({
      componentId: c.id,
      mean: c.mean,
      cov: c.cov,
      eigenvalues: c.eigenvalues,
      eigenvectors: c.eigenvectors,
      angle: c.angle,
      weight: c.weight,
      color: CLUSTER_PALETTE[c.id % CLUSTER_PALETTE.length],
    }));

    return [
      {
        type: 'clustering-overlay',
        centroids,
        assignments: state.hardAssignments,
        softResponsibilities: state.responsibilities,
        gmmEllipses,
      },
    ];
  },

  serializeSnapshot(state: GMMState): Float32Array {
    const k = state.components.length;
    // [epoch, k, subStep, logLikelihood, silhouette, ...means(2*k), ...cov(4*k)]
    const buf = new Float32Array(5 + k * 6);
    buf[0] = state.epoch;
    buf[1] = k;
    buf[2] = state.subStep === 'e-step' ? 0 : 1;
    buf[3] = state.logLikelihood;
    buf[4] = state.silhouette;
    let offset = 5;
    for (let c = 0; c < k; c++) {
      buf[offset++] = state.components[c].mean[0];
      buf[offset++] = state.components[c].mean[1];
      buf[offset++] = state.components[c].cov[0][0];
      buf[offset++] = state.components[c].cov[0][1];
      buf[offset++] = state.components[c].cov[1][0];
      buf[offset++] = state.components[c].cov[1][1];
    }
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: GMMState): GMMState {
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      k: Math.round(buffer[1]),
      subStep: buffer[2] === 0 ? 'e-step' : 'm-step',
      logLikelihood: buffer[3],
      silhouette: buffer[4],
    };
  },

  explain(state: GMMState, _params: GMMParams, _lastStep?: StepResult<GMMState>): ExplanationPayload {
    return {
      summary: `Gaussian Mixture Models represent data as a superposition of ${state.k} multivariate normal components with soft posterior responsibilities and full covariance ellipsoids.`,
      coreAssumption:
        'Data is generated from a mixture of K Gaussian probability distributions with distinct means and covariance tensors.',
      whenItFails:
        'Singularity collapse when a component contracts onto a single point (variance σ² → 0, likelihood → ∞), resolved via covariance regularization ε_cov.',
      microLog: [
        `Evaluate E-Step: posterior responsibilities γ_ik = P(z_k=1 | x_i)`,
        `Evaluate M-Step: re-estimate means μ_k and covariance matrices Σ_k`,
        `Current Log-Likelihood = ${state.logLikelihood.toFixed(2)} (BIC = ${state.bic.toFixed(2)})`,
      ],
      mathLatex: {
        objective: `\\ln p(X | \\pi, \\mu, \\Sigma) = \\sum_{i=1}^N \\ln \\left( \\sum_{k=1}^K \\pi_k \\mathcal{N}(x_i \\mid \\mu_k, \\Sigma_k) \\right)`,
        lossOrUpdate: `\\gamma_{ik} = \\frac{\\pi_k \\mathcal{N}(x_i \\mid \\mu_k, \\Sigma_k)}{\\sum_j \\pi_j \\mathcal{N}(x_i \\mid \\mu_j, \\Sigma_j)}, \\quad \\Sigma_k = \\frac{1}{N_k}\\sum_{i=1}^N \\gamma_{ik}(x_i - \\mu_k)(x_i - \\mu_k)^T + \\varepsilon I`,
      },
      codeSnippet: {
        sklearn: `from sklearn.mixture import GaussianMixture\ngmm = GaussianMixture(n_components=${state.k}, covariance_type='${state.covarianceType}', reg_covar=${state.regCovar})\ngmm.fit(X)\nprobs = gmm.predict_proba(X)`,
        pytorch: `# PyTorch MultivariateNormal Mixture with torch.logsumexp`,
      },
    };
  },
};
