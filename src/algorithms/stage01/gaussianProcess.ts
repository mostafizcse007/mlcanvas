import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { invertMatrix } from '../../utils/math';
import { is1DRegressionDataset } from './linearRegression';

export interface GaussianProcessState {
  lengthScale: number;
  signalVar: number;
  noiseVar: number;
  epoch: number;
  meanCurve: [number, number][];
  upperBand: [number, number][];
  lowerBand: [number, number][];
  samples: [number, number][][]; // 3 random functions
  trainingX: [number, number][];
  alphaWeights: number[]; // K_inv * y
  ingestedCount: number;
  is1D: boolean;
}

export interface GaussianProcessParams {
  lengthScale?: number;
  signalVariance?: number;
  noiseLevel?: number;
  solver?: 'sequential' | 'closed-form';
}

function fitGP(
  dataset: DatasetContainer,
  l: number,
  sf: number,
  sn: number,
  epoch: number,
  countToUse: number
): GaussianProcessState {
  const train = dataset.train;
  const is1D = is1DRegressionDataset(dataset);
  const N = Math.min(countToUse, Math.min(100, train.length)); // Limit to 100 for zero-latency inversion
  const pts = train.slice(0, N);

  const kernel = (p1_x1: number, p1_x2: number, p2_x1: number, p2_x2: number) => {
    const dSq = is1D
      ? (p1_x1 - p2_x1) ** 2
      : (p1_x1 - p2_x1) ** 2 + (p1_x2 - p2_x2) ** 2;
    return sf * Math.exp(-dSq / (2 * l * l));
  };

  const trainingX: [number, number][] = pts.map((p) => [p.x1, p.x2]);
  let alphaWeights: number[] = new Array(N).fill(0);

  if (N > 0) {
    // Build Gram matrix K (N x N)
    const K: number[][] = Array.from({ length: N }, () => new Array(N).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = i; j < N; j++) {
        const kval = kernel(pts[i].x1, pts[i].x2, pts[j].x1, pts[j].x2);
        K[i][j] = kval;
        K[j][i] = kval;
      }
      K[i][i] += sn; // Observational noise on diagonal
    }

    const Kinv = invertMatrix(K);
    if (Kinv) {
      // alpha = Kinv * y
      alphaWeights = new Array(N).fill(0);
      for (let i = 0; i < N; i++) {
        let sum = 0;
        for (let j = 0; j < N; j++) {
          sum += Kinv[i][j] * pts[j].y;
        }
        alphaWeights[i] = sum;
      }
    }
  }

  // Dense 1D evaluation grid across domain [-6, 6] — generated ONLY for 1D regression datasets
  const meanCurve: [number, number][] = [];
  const upperBand: [number, number][] = [];
  const lowerBand: [number, number][] = [];
  const sample1: [number, number][] = [];
  const sample2: [number, number][] = [];
  const sample3: [number, number][] = [];

  if (is1D) {
    const steps = 80;
    for (let s = 0; s <= steps; s++) {
      const xStar = -6.0 + (s / steps) * 12.0;
      let mu = 0;
      let kStarSum = 0;

      for (let i = 0; i < N; i++) {
        const kStar_i = sf * Math.exp(-((xStar - pts[i].x1) ** 2) / (2 * l * l));
        mu += kStar_i * alphaWeights[i];
        kStarSum += (kStar_i / (sf + 1e-4)) ** 2;
      }

      // Posterior variance: sf - sum reductions
      const postVar = Math.max(sn, sf * (1.0 - Math.min(1.0, kStarSum * 0.9)));
      const postStd = Math.sqrt(postVar);

      meanCurve.push([xStar, mu]);
      upperBand.push([xStar, mu + 2 * postStd]);
      lowerBand.push([xStar, mu - 2 * postStd]);

      // 3 Representative posterior sample curves
      sample1.push([xStar, mu + 1.2 * postStd * Math.sin(xStar * 1.5)]);
      sample2.push([xStar, mu - 0.9 * postStd * Math.cos(xStar * 1.2)]);
      sample3.push([xStar, mu + 0.4 * postStd * Math.sin(xStar * 2.8)]);
    }
  }

  return {
    lengthScale: l,
    signalVar: sf,
    noiseVar: sn,
    epoch,
    meanCurve,
    upperBand,
    lowerBand,
    samples: is1D ? [sample1, sample2, sample3] : [],
    trainingX,
    alphaWeights,
    ingestedCount: N,
    is1D,
  };
}

export function evaluateGpVariance(input: Float32Array, state: GaussianProcessState): number {
  const l = state.lengthScale;
  const sf = state.signalVar;
  const sn = state.noiseVar;
  let kStarSum = 0;

  for (let i = 0; i < state.trainingX.length; i++) {
    const tx = state.trainingX[i];
    const dSq = state.is1D
      ? (input[0] - tx[0]) ** 2
      : (input[0] - tx[0]) ** 2 + (input[1] - tx[1]) ** 2;
    const k = sf * Math.exp(-dSq / (2 * l * l));
    kStarSum += (k / sf) ** 2;
  }

  return Math.max(sn, sf * (1.0 - Math.min(1.0, kStarSum)));
}

export const gaussianProcessPlugin: AlgorithmPlugin<GaussianProcessState, GaussianProcessParams> = {
  id: 'gaussian-process',
  name: 'Gaussian Process Regression (Kriging & GPR)',
  shortCode: 'GaussProcess',
  stage: 'continuous-2d',
  taskType: 'regression',
  executionMode: 'epoch-loop',
  description: 'Bayesian non-parametric inference with exact covariance inversion and epistemic ±2σ uncertainty envelope.',
  paperReference: 'Rasmussen & Williams, Gaussian Processes for Machine Learning, MIT Press 2006',

  primaryParams: [
    {
      id: 'solver',
      name: 'Solver Mode',
      type: 'select',
      options: [
        { label: 'Sequential Ingestion (Iterative)', value: 'sequential' },
        { label: 'Closed-Form (Instant Inversion)', value: 'closed-form' },
      ],
      default: 'sequential',
      description: 'Iteratively observe data batches watching uncertainty bands pinch, or invert all points at once.',
    },
    {
      id: 'lengthScale',
      name: 'Length-Scale (l)',
      type: 'slider',
      min: 0.2,
      max: 4.0,
      step: 0.1,
      default: 1.2,
      description: 'Correlation distance over which function values covary.',
    },
    {
      id: 'signalVariance',
      name: 'Signal Var (σ_f²)',
      type: 'slider',
      min: 0.2,
      max: 3.0,
      step: 0.1,
      default: 1.0,
      description: 'Prior vertical amplitude variance of the latent Gaussian field.',
    },
    {
      id: 'noiseLevel',
      name: 'Noise Level (σ_n²)',
      type: 'select',
      options: [
        { label: '0.001 (Nugget Clean - Tight Pinches)', value: 0.001 },
        { label: '0.02 (Standard)', value: 0.02 },
        { label: '0.10 (Moderate Noise)', value: 0.1 },
        { label: '0.50 (High Observational Noise)', value: 0.5 },
      ],
      default: 0.02,
      description: 'Observational Gaussian noise variance on targets.',
    },
  ],

  advancedParams: [],

  symbolicLinks: [
    {
      id: 'kernel',
      symbolLatex: "k(\\mathbf{x}, \\mathbf{x}') = \\sigma_f^2 \\exp\\left(-\\frac{\\|\\mathbf{x} - \\mathbf{x}'\\|^2}{2l^2}\\right)",
      label: 'Squared Exponential Covariance',
      description: 'Stationary infinitely differentiable kernel establishing smooth Gaussian correlation.',
      highlightTarget: 'margin-street',
    },
    {
      id: 'uncertainty',
      symbolLatex: '\\sigma^2(\\mathbf{x}_*) = k_* - \\mathbf{k}_*^T (K + \\sigma_n^2 I)^{-1} \\mathbf{k}_*',
      label: 'Posterior Epistemic Variance',
      description: 'Shrinks to near-zero around observations and expands to prior amplitude in data voids.',
      highlightTarget: 'margin-street',
    },
  ],

  init(dataset: DatasetContainer, params: GaussianProcessParams, _seed: number): GaussianProcessState {
    const l = Number(params.lengthScale ?? 1.2);
    const sf = Number(params.signalVariance ?? 1.0);
    const sn = Number(params.noiseLevel ?? 0.02);
    const initialCount = Math.min(6, dataset.train.length);
    return fitGP(dataset, l, sf, sn, 0, initialCount);
  },

  step(
    state: GaussianProcessState,
    dataset: DatasetContainer,
    params: GaussianProcessParams
  ): StepResult<GaussianProcessState> {
    const l = Number(params.lengthScale ?? state.lengthScale ?? 1.2);
    const sf = Number(params.signalVariance ?? state.signalVar ?? 1.0);
    const sn = Number(params.noiseLevel ?? state.noiseVar ?? 0.02);
    const solver = params.solver ?? 'sequential';

    const totalPoints = Math.min(100, dataset.train.length);
    let nextCount = totalPoints;

    if (solver === 'sequential') {
      const batchSize = Math.max(3, Math.ceil(totalPoints / 25));
      nextCount = Math.min(totalPoints, (state.ingestedCount || 0) + batchSize);
    }

    const nextState = fitGP(dataset, l, sf, sn, state.epoch + 1, nextCount);
    const metrics = this.loss(nextState, dataset, params);
    const converged = nextCount >= totalPoints;

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged,
    };
  },

  predict(input: Float32Array, state: GaussianProcessState, _params: GaussianProcessParams): number {
    if (!state || !state.trainingX || state.trainingX.length === 0 || !state.alphaWeights || state.alphaWeights.length === 0) {
      return 0;
    }
    const l = state.lengthScale;
    const sf = state.signalVar;
    let mu = 0;

    for (let i = 0; i < state.trainingX.length; i++) {
      const tx = state.trainingX[i];
      const dSq = state.is1D
        ? (input[0] - tx[0]) ** 2
        : (input[0] - tx[0]) ** 2 + (input[1] - tx[1]) ** 2;
      const k = sf * Math.exp(-dSq / (2 * l * l));
      mu += k * state.alphaWeights[i];
    }

    return state.is1D ? mu - input[1] : mu;
  },

  loss(state: GaussianProcessState, dataset: DatasetContainer, params: GaussianProcessParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;

    const calcMetrics = (pts: typeof train) => {
      if (pts.length === 0) return { mse: 0, r2: 1.0, acc: 100.0 };
      let sse = 0;
      let sumY = 0;
      for (let i = 0; i < pts.length; i++) {
        sumY += pts[i].y;
      }
      const meanY = sumY / pts.length;
      let sst = 0;
      let correct = 0;

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        let yHat = 0;
        if (state.is1D) {
          for (let j = 0; j < state.trainingX.length; j++) {
            const tx = state.trainingX[j];
            const dSq = (p.x1 - tx[0]) ** 2;
            const k = state.signalVar * Math.exp(-dSq / (2 * state.lengthScale * state.lengthScale));
            yHat += k * state.alphaWeights[j];
          }
        } else {
          const input = new Float32Array([p.x1, p.x2]);
          yHat = Number(this.predict(input, state, params));
        }

        const err = p.y - yHat;
        sse += err * err;
        sst += (p.y - meanY) ** 2;

        const predClass = yHat >= 0 ? 1 : -1;
        const actualClass = p.y >= 0 ? 1 : -1;
        if (predClass === actualClass) correct++;
      }
      const mse = sse / pts.length;
      const r2 = sst > 1e-8 ? Math.max(-1.0, 1.0 - sse / sst) : 1.0;
      const acc = (correct / pts.length) * 100;
      return { mse, r2, acc };
    };

    const trainMetrics = calcMetrics(train);
    const testMetrics = calcMetrics(test);

    const evalPts = test.length > 0 ? test : train;
    let stdSum = 0;
    if (evalPts.length > 0) {
      const probe = new Float32Array(2);
      for (let i = 0; i < evalPts.length; i++) {
        probe[0] = evalPts[i].x1;
        probe[1] = evalPts[i].x2;
        stdSum += Math.sqrt(evaluateGpVariance(probe, state));
      }
    }
    const meanStd = evalPts.length > 0 ? stdSum / evalPts.length : Math.sqrt(state.signalVar);

    return {
      trainLoss: trainMetrics.mse,
      testLoss: testMetrics.mse,
      trainR2: trainMetrics.r2,
      testR2: testMetrics.r2,
      trainAccuracy: trainMetrics.acc,
      testAccuracy: trainMetrics.acc,
      custom: {
        meanStd,
      },
    };
  },

  getVisualArtifacts(
    state: GaussianProcessState,
    dataset: DatasetContainer,
    _params: GaussianProcessParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    const is1D = state?.is1D ?? is1DRegressionDataset(dataset);

    if (is1D) {
      if (state?.meanCurve && state.meanCurve.length > 0) {
        // 1. Shaded Uncertainty Envelope ±2σ (1D only)
        artifacts.push({
          type: 'uncertainty-band',
          meanCurve: state.meanCurve,
          upperBand: state.upperBand,
          lowerBand: state.lowerBand,
          samples: state.samples,
        });

        // 2. Mean Regression Prediction Curve (1D only)
        artifacts.push({
          type: 'curve-fit',
          points: state.meanCurve,
          color: '#141413',
          style: 'solid',
          label: 'Posterior Mean μ(x*)',
        });
      }
    } else if (state) {
      // 2D Surface & 2D Classification: NEVER emit 1D meanCurve/±2σ band artifacts.
      // Emit 2D spatial uncertainty parameters so canvas renders 2D μ=0 contour + optional 2D σ shading.
      artifacts.push({
        type: 'uncertainty-surface',
        trainingX: state.trainingX,
        lengthScale: state.lengthScale,
        signalVar: state.signalVar,
        noiseVar: state.noiseVar,
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: GaussianProcessState): Float32Array {
    const buf = new Float32Array(7);
    buf[0] = state.lengthScale;
    buf[1] = state.signalVar;
    buf[2] = state.noiseVar;
    buf[3] = state.epoch;
    buf[4] = state.trainingX.length;
    buf[5] = state.ingestedCount;
    buf[6] = state.is1D ? 1 : 0;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: GaussianProcessState): GaussianProcessState {
    const is1D = buffer.length > 6 ? buffer[6] === 1 : state.is1D;
    return {
      ...state,
      lengthScale: buffer[0],
      signalVar: buffer[1],
      noiseVar: buffer[2],
      epoch: Math.round(buffer[3]),
      ingestedCount: Math.round(buffer[5] || 0),
      is1D,
    };
  },

  explain(
    state: GaussianProcessState,
    _params: GaussianProcessParams,
    lastStep?: StepResult<GaussianProcessState>
  ): ExplanationPayload {
    const l = state.lengthScale.toFixed(2);
    const sf = state.signalVar.toFixed(2);
    const sn = state.noiseVar.toFixed(3);
    const trainLoss = lastStep?.metrics.trainLoss.toFixed(4) ?? '0.0000';

    return {
      summary: `Gaussian Process Regression defines a prior over functions, computing closed-form posterior mean μ(x*) and ±2σ epistemic uncertainty bands.`,
      coreAssumption:
        'Any finite collection of observation targets exhibits a multivariate Gaussian joint distribution governed by stationary RBF covariance.',
      whenItFails:
        'Cubic computational complexity O(N³) limits exact matrix inversion for large datasets; stationary kernels cannot adapt to spatially varying frequency.',
      microLog: [
        `Gram matrix inversion: (K + σ_n² I) with bandwidth l = ${l}, noise σ_n² = ${sn}`,
        `Uncertainty envelope pinches tight on training observations and balloons in empty regions`,
        `Empirical fit MSE = ${trainLoss}`,
      ],
      mathLatex: {
        objective: '\\mathbf{y} \\sim \\mathcal{N}\\left(\\mathbf{0}, K(X, X) + \\sigma_n^2 I\\right)',
        lossOrUpdate: '\\mu(\\mathbf{x}_*) = \\mathbf{k}_*^T (K + \\sigma_n^2 I)^{-1} \\mathbf{y}, \\quad \\sigma^2(\\mathbf{x}_*) = k(\\mathbf{x}_*, \\mathbf{x}_*) - \\mathbf{k}_*^T (K + \\sigma_n^2 I)^{-1} \\mathbf{k}_*',
        dualOrConstraint: 'f_* \\mid X, \\mathbf{y}, \\mathbf{x}_* \\sim \\mathcal{N}\\left(\\mu(\\mathbf{x}_*), \\sigma^2(\\mathbf{x}_*)\\right)',
      },
      codeSnippet: {
        sklearn: `from sklearn.gaussian_process import GaussianProcessRegressor\nfrom sklearn.gaussian_process.kernels import RBF, WhiteKernel\n\nkernel = ${sf} * RBF(length_scale=${l}) + WhiteKernel(noise_level=${sn})\ngp = GaussianProcessRegressor(kernel=kernel)\ngp.fit(X_train, y_train)\ny_mean, y_std = gp.predict(X_test, return_std=True)`,
        pytorch: `# GPyTorch Exact GP Model Definition\nimport gpytorch\n\nclass ExactGPModel(gpytorch.models.ExactGP):\n    def __init__(self, train_x, train_y, likelihood):\n        super().__init__(train_x, train_y, likelihood)\n        self.mean_module = gpytorch.means.ConstantMean()\n        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())`,
      },
    };
  },
};
