import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { invertMatrix } from '../../utils/math';
import { mulberry32 } from '../../utils/prng';
import { is1DRegressionDataset } from './linearRegression';

export interface PolynomialRegressionState {
  coefficients: Float32Array; // 1D: [w0..wd] (K=d+1), 2D: [w0..w_{K-1}] (K=(d+1)(d+2)/2)
  degree: number;
  epoch: number;
  curvePoints: [number, number][]; // 1D fitted curve points
  residuals: { id: number; x1: number; x2: number; y: number; yHat: number; error: number }[];
  rungeWarning: boolean; // True if boundary derivatives/magnitudes oscillate wildly
  conditioningWarning: boolean; // True if Vandermonde Gram matrix condition ratio is severe
  is1D: boolean;
}

export interface PolynomialRegressionParams {
  degree?: number;
  regularization?: number;
  solver?: 'sgd' | 'closed-form';
  learningRate?: number;
  batchPasses?: number;
  cyclesPerStep?: number;
  subEpochs?: number;
}

export interface PolyTerm2D {
  p1: number;
  p2: number;
  degree: number;
}

/**
 * Returns all bivariate monomial terms (p1, p2) with total degree p1 + p2 <= degree,
 * ordered by total degree ascending, then descending power of x1:
 * Degree 0: (0,0) -> 1
 * Degree 1: (1,0) -> x1, (0,1) -> x2
 * Degree 2: (2,0) -> x1^2, (1,1) -> x1*x2, (0,2) -> x2^2
 * Degree 3: (3,0) -> x1^3, (2,1) -> x1^2*x2, (1,2) -> x1*x2^2, (0,3) -> x2^3
 */
export function get2DPolynomialTerms(degree: number): PolyTerm2D[] {
  const terms: PolyTerm2D[] = [];
  for (let d = 0; d <= degree; d++) {
    for (let p2 = 0; p2 <= d; p2++) {
      const p1 = d - p2;
      terms.push({ p1, p2, degree: d });
    }
  }
  return terms;
}

export function getFeatureCount(degree: number, is1D: boolean): number {
  return is1D ? degree + 1 : ((degree + 1) * (degree + 2)) / 2;
}

/**
 * Computes monomial feature expansion with 0.25 coordinate normalization.
 * In 1D: [1, (0.25 x1), (0.25 x1)^2, ..., (0.25 x1)^d]
 * In 2D: all (0.25 x1)^p1 * (0.25 x2)^p2 for p1 + p2 <= degree
 */
export function computePolyFeatures(
  x1: number,
  x2: number,
  degree: number,
  is1D: boolean
): Float32Array {
  const scale = 0.25;
  const sx1 = x1 * scale;
  if (is1D) {
    const feats = new Float32Array(degree + 1);
    feats[0] = 1.0;
    let t = 1.0;
    for (let p = 1; p <= degree; p++) {
      t *= sx1;
      feats[p] = t;
    }
    return feats;
  } else {
    const sx2 = x2 * scale;
    const K = ((degree + 1) * (degree + 2)) / 2;
    const feats = new Float32Array(K);
    const pow1 = new Float32Array(degree + 1);
    const pow2 = new Float32Array(degree + 1);
    pow1[0] = 1.0;
    pow2[0] = 1.0;
    for (let p = 1; p <= degree; p++) {
      pow1[p] = pow1[p - 1] * sx1;
      pow2[p] = pow2[p - 1] * sx2;
    }
    let idx = 0;
    for (let d = 0; d <= degree; d++) {
      for (let p2 = 0; p2 <= d; p2++) {
        const p1 = d - p2;
        feats[idx++] = pow1[p1] * pow2[p2];
      }
    }
    return feats;
  }
}

/**
 * Distinguishes numerical conditioning of the design matrix from physical edge oscillation (Runge's phenomenon).
 */
export function checkStabilityDiagnostics(
  coeffs: Float32Array,
  degree: number,
  lambda: number,
  is1D: boolean,
  gramDiagRatio = 1.0
): { rungeWarning: boolean; conditioningWarning: boolean } {
  // 1. Numerical Conditioning:
  // Vandermonde matrices have exponential eigenvalue decay; severe without sufficient ridge regularization
  const conditioningWarning = (degree >= 7 && lambda < 0.005) || gramDiagRatio > 1e7;

  // 2. Runge Boundary Oscillation:
  // Detects whether polynomial derivative/magnitudes near domain edges (|x| >= 4.5)
  // oscillate substantially faster than in the interior (|x| <= 3.0)
  let rungeWarning = false;
  if (is1D) {
    const step = 0.25;
    let maxInteriorSlope = 0;
    let prevY: number | null = null;
    for (let x = -3.0; x <= 3.0; x += step) {
      const f = computePolyFeatures(x, 0, degree, true);
      let y = 0;
      for (let i = 0; i < f.length && i < coeffs.length; i++) y += coeffs[i] * f[i];
      if (prevY !== null) {
        maxInteriorSlope = Math.max(maxInteriorSlope, Math.abs(y - prevY) / step);
      }
      prevY = y;
    }

    let maxEdgeSlope = 0;
    prevY = null;
    for (let x = -6.0; x <= -4.5; x += step) {
      const f = computePolyFeatures(x, 0, degree, true);
      let y = 0;
      for (let i = 0; i < f.length && i < coeffs.length; i++) y += coeffs[i] * f[i];
      if (prevY !== null) {
        maxEdgeSlope = Math.max(maxEdgeSlope, Math.abs(y - prevY) / step);
      }
      prevY = y;
    }
    prevY = null;
    for (let x = 4.5; x <= 6.0; x += step) {
      const f = computePolyFeatures(x, 0, degree, true);
      let y = 0;
      for (let i = 0; i < f.length && i < coeffs.length; i++) y += coeffs[i] * f[i];
      if (prevY !== null) {
        maxEdgeSlope = Math.max(maxEdgeSlope, Math.abs(y - prevY) / step);
      }
      prevY = y;
    }

    // Runge oscillation manifests at high degrees without adequate ridge damping (lambda < 0.02)
    if (degree >= 5 && lambda < 0.02 && maxEdgeSlope > 2.5 * Math.max(0.5, maxInteriorSlope)) {
      rungeWarning = true;
    }
  } else {
    // 2D boundary oscillation detection
    if (degree >= 6 && lambda < 0.01) {
      rungeWarning = true;
    }
  }

  return { rungeWarning, conditioningWarning };
}

/**
 * Closed-form least squares with Tikhonov/Ridge regularization.
 * Intercept (w0) is left unpenalized (except for 1e-6 numerical baseline) while
 * all non-trivial monomials (k >= 1) receive the selected lambda stabilizer.
 */
function solvePolynomialClosedForm(
  train: { x1: number; x2: number; y: number }[],
  degree: number,
  lambda: number,
  is1D: boolean
): { coeffs: Float32Array; condRatio: number } {
  const N = train.length;
  const K = getFeatureCount(degree, is1D);
  const coeffs = new Float32Array(K);

  if (N === 0) return { coeffs, condRatio: 1.0 };

  let sumY = 0;
  for (let i = 0; i < N; i++) sumY += train[i].y;
  const meanY = sumY / N;

  const XtX: number[][] = Array.from({ length: K }, () => new Array(K).fill(0));
  const Xty: number[] = new Array(K).fill(0);

  for (let i = 0; i < N; i++) {
    const pt = train[i];
    const phi = computePolyFeatures(pt.x1, pt.x2, degree, is1D);
    const yi = pt.y;
    for (let j = 0; j < K; j++) {
      Xty[j] += phi[j] * yi;
      for (let k = j; k < K; k++) {
        XtX[j][k] += phi[j] * phi[k];
      }
    }
  }

  for (let j = 0; j < K; j++) {
    for (let k = j + 1; k < K; k++) {
      XtX[k][j] = XtX[j][k];
    }
  }

  let diagMax = 1e-6;
  let diagMin = Infinity;
  for (let j = 0; j < K; j++) {
    diagMax = Math.max(diagMax, XtX[j][j]);
    diagMin = Math.min(diagMin, Math.max(1e-12, XtX[j][j]));
  }
  const condRatio = diagMax / diagMin;

  const solveWithReg = (regL2: number): Float32Array | null => {
    const M: number[][] = XtX.map((row) => [...row]);
    M[0][0] += 1e-6; // Unpenalized intercept baseline
    for (let j = 1; j < K; j++) {
      M[j][j] += Math.max(1e-6, regL2);
    }
    const inv = invertMatrix(M);
    if (!inv) return null;

    const sol = new Float32Array(K);
    for (let j = 0; j < K; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += inv[j][k] * Xty[k];
      }
      if (!isFinite(sum) || isNaN(sum)) return null;
      sol[j] = sum;
    }
    return sol;
  };

  let sol = solveWithReg(lambda);
  if (!sol && lambda < 1e-4) {
    sol = solveWithReg(1e-4);
  }
  if (!sol && lambda < 1e-2) {
    sol = solveWithReg(1e-2);
  }

  if (sol) {
    for (let i = 0; i < K; i++) coeffs[i] = sol[i];
  } else {
    // Singular fallback: horizontal plane/line at target mean
    coeffs[0] = meanY;
  }

  return { coeffs, condRatio };
}

/**
 * Stochastic gradient descent with L2 monomial weight decay (Tikhonov stabilizer).
 * Intercept w0 receives no L2 decay to avoid shrinking the vertical baseline.
 */
function stepPolynomialSGD(
  coeffs: Float32Array,
  train: { x1: number; x2: number; y: number }[],
  degree: number,
  lambda: number,
  lr: number,
  passes: number,
  is1D: boolean
): Float32Array {
  const K = getFeatureCount(degree, is1D);
  const updated = new Float32Array(coeffs);
  const N = Math.max(1, train.length);

  for (let pass = 0; pass < passes; pass++) {
    for (let i = 0; i < N; i++) {
      const pt = train[i];
      const phi = computePolyFeatures(pt.x1, pt.x2, degree, is1D);
      let yHat = 0;
      for (let k = 0; k < K; k++) {
        yHat += updated[k] * phi[k];
      }
      const err = pt.y - yHat;

      // Intercept update (no weight decay)
      const grad0 = Math.max(-10, Math.min(10, err * phi[0]));
      updated[0] += (lr * grad0) / N;

      // Monomial updates (k >= 1) with L2 weight decay
      const decay = Math.max(0, 1.0 - lr * lambda);
      for (let k = 1; k < K; k++) {
        const grad = Math.max(-10, Math.min(10, err * phi[k]));
        updated[k] = decay * updated[k] + (lr * grad) / N;
      }
    }
  }

  return updated;
}

export const polynomialRegressionPlugin: AlgorithmPlugin<PolynomialRegressionState, PolynomialRegressionParams> = {
  id: 'polynomial-regression',
  name: 'Polynomial Feature Regression',
  shortCode: 'PolyReg',
  stage: 'continuous-2d',
  taskType: 'regression',
  executionMode: 'epoch-loop',
  description: 'Vandermonde basis expansion with Runge oscillation detection and Tikhonov stabilizer.',
  paperReference: 'Runge (1901), Über empirische Funktionen und die Interpolation zwischen äquidistanten Ordinaten',

  primaryParams: [
    {
      id: 'solver',
      name: 'Solver Mode',
      type: 'select',
      options: [
        { label: 'Iterative Monomial SGD', value: 'sgd' },
        { label: 'Closed-Form (Vandermonde OLS)', value: 'closed-form' },
      ],
      default: 'sgd',
      description: 'Epoch-by-epoch monomial gradient descent or instantaneous normal equation inversion.',
    },
    {
      id: 'degree',
      name: 'Degree (d)',
      type: 'slider',
      min: 1,
      max: 12,
      step: 1,
      default: 3,
      description: 'Maximum degree of polynomial basis expansion.',
    },
    {
      id: 'regularization',
      name: 'Ridge Stabilizer (λ)',
      type: 'select',
      options: [
        { label: '0.000 (Pure OLS - High Oscillation)', value: 0.0 },
        { label: '0.001 (Light)', value: 0.001 },
        { label: '0.05 (Balanced)', value: 0.05 },
        { label: '0.50 (Heavy Damping)', value: 0.5 },
      ],
      default: 0.001,
      description: 'L2 penalty on higher-order monomials dampening boundary swings.',
    },
  ],

  advancedParams: [
    {
      id: 'learningRate',
      name: 'Sub-Epoch Step',
      type: 'slider',
      min: 0.001,
      max: 0.1,
      step: 0.005,
      default: 0.01,
      description: 'Refinement step size for parameter updates.',
    },
  ],

  symbolicLinks: [
    {
      id: 'phi',
      symbolLatex: '\\phi(\\mathbf{x}) = [x_1^{p_1} x_2^{p_2}]_{p_1+p_2 \\le d}',
      label: 'Polynomial Basis Expansion',
      description: 'Non-linear feature map projecting input coordinates into d-degree polynomial space.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: PolynomialRegressionParams, seed: number): PolynomialRegressionState {
    const rand = mulberry32(seed);
    const d = Math.max(1, Math.min(12, Number(params.degree ?? 3)));
    const l2 = Number(params.regularization ?? 0.001);
    const solver = params.solver ?? 'sgd';
    const train = dataset.train;
    const is1D = is1DRegressionDataset(dataset);
    const K = getFeatureCount(d, is1D);

    let coeffs = new Float32Array(K);

    if (solver === 'closed-form' && train.length >= 2) {
      const closed = solvePolynomialClosedForm(train, d, l2, is1D);
      coeffs.set(closed.coeffs);
    } else {
      coeffs[0] = (rand() - 0.5) * 0.5;
      if (is1D) {
        coeffs[1] = 0.8 + (rand() - 0.5) * 0.4;
        for (let i = 2; i < K; i++) {
          coeffs[i] = ((rand() - 0.5) * 0.2) / i;
        }
      } else {
        if (K > 1) coeffs[1] = 0.6 + (rand() - 0.5) * 0.3;
        if (K > 2) coeffs[2] = 0.4 + (rand() - 0.5) * 0.3;
        for (let i = 3; i < K; i++) {
          coeffs[i] = ((rand() - 0.5) * 0.1) / Math.sqrt(i);
        }
      }
    }

    const curvePoints: [number, number][] = [];
    if (is1D) {
      for (let x = -6.0; x <= 6.0; x += 0.1) {
        const f = computePolyFeatures(x, 0, d, true);
        let yHat = 0;
        for (let j = 0; j < K; j++) yHat += coeffs[j] * f[j];
        curvePoints.push([x, Math.max(-20, Math.min(20, yHat))]);
      }
    }

    const residuals = train.map((p) => {
      const f = computePolyFeatures(p.x1, p.x2, d, is1D);
      let yHat = 0;
      for (let j = 0; j < K; j++) yHat += coeffs[j] * f[j];
      return { id: p.id, x1: p.x1, x2: p.x2, y: p.y, yHat, error: p.y - yHat };
    });

    const { rungeWarning, conditioningWarning } = checkStabilityDiagnostics(coeffs, d, l2, is1D);

    return {
      coefficients: coeffs,
      degree: d,
      epoch: solver === 'closed-form' ? 1 : 0,
      curvePoints,
      residuals,
      rungeWarning,
      conditioningWarning,
      is1D,
    };
  },

  step(
    state: PolynomialRegressionState,
    dataset: DatasetContainer,
    params: PolynomialRegressionParams
  ): StepResult<PolynomialRegressionState> {
    const solver = params.solver ?? 'sgd';
    const lr = Number(params.learningRate ?? 0.02);
    const d = Math.max(1, Math.min(12, Number(params.degree ?? 3)));
    const l2 = Number(params.regularization ?? 0.001);
    const passes = Math.max(1, Math.round(Number(params.batchPasses ?? params.cyclesPerStep ?? params.subEpochs ?? 5)));
    const train = dataset.train;
    const is1D = is1DRegressionDataset(dataset);
    const K = getFeatureCount(d, is1D);

    let coeffs = new Float32Array(K);
    for (let i = 0; i < K && i < state.coefficients.length; i++) {
      coeffs[i] = state.coefficients[i];
    }

    let condRatio = 1.0;

    if (solver === 'closed-form') {
      if (train.length >= 2) {
        const closed = solvePolynomialClosedForm(train, d, l2, is1D);
        coeffs.set(closed.coeffs);
        condRatio = closed.condRatio;
      }
    } else {
      coeffs.set(stepPolynomialSGD(coeffs, train, d, l2, lr, passes, is1D));
    }

    const curvePoints: [number, number][] = [];
    if (is1D) {
      for (let x = -6.0; x <= 6.0; x += 0.1) {
        const f = computePolyFeatures(x, 0, d, true);
        let yHat = 0;
        for (let j = 0; j < K; j++) yHat += coeffs[j] * f[j];
        curvePoints.push([x, Math.max(-20, Math.min(20, yHat))]);
      }
    }

    const residuals = train.map((p) => {
      const f = computePolyFeatures(p.x1, p.x2, d, is1D);
      let yHat = 0;
      for (let j = 0; j < K; j++) yHat += coeffs[j] * f[j];
      return { id: p.id, x1: p.x1, x2: p.x2, y: p.y, yHat, error: p.y - yHat };
    });

    const { rungeWarning, conditioningWarning } = checkStabilityDiagnostics(coeffs, d, l2, is1D, condRatio);

    const nextState: PolynomialRegressionState = {
      coefficients: coeffs,
      degree: d,
      epoch: solver === 'closed-form' ? 1 : state.epoch + 1,
      curvePoints,
      residuals,
      rungeWarning,
      conditioningWarning,
      is1D,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged: solver === 'closed-form',
    };
  },

  predict(input: Float32Array, state: PolynomialRegressionState, _params: PolynomialRegressionParams): number {
    if (!state || !state.coefficients) return 0;
    const is1D = state.is1D ?? true;
    const d = state.degree ?? 3;
    const feats = computePolyFeatures(input[0], input[1], d, is1D);
    let yHat = 0;
    for (let i = 0; i < feats.length && i < state.coefficients.length; i++) {
      yHat += state.coefficients[i] * feats[i];
    }
    return yHat;
  },

  loss(state: PolynomialRegressionState, dataset: DatasetContainer, _params: PolynomialRegressionParams): TelemetryMetrics {
    const is1D = state.is1D ?? (dataset.train && dataset.train.length >= 2 ? is1DRegressionDataset(dataset) : true);
    const d = state.degree ?? 3;
    const K = getFeatureCount(d, is1D);

    const calcMetrics = (pts: typeof dataset.train) => {
      if (!pts || pts.length === 0) return { mse: 0, rmse: 0, r2: 1.0, acc: 100.0 };
      let sse = 0;
      let sumY = 0;
      for (let i = 0; i < pts.length; i++) sumY += pts[i].y;
      const meanY = sumY / pts.length;
      let sst = 0;
      let correct = 0;

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const feats = computePolyFeatures(p.x1, p.x2, d, is1D);
        let yHat = 0;
        for (let j = 0; j < K && j < state.coefficients.length; j++) {
          yHat += state.coefficients[j] * feats[j];
        }
        const err = p.y - yHat;
        sse += err * err;
        sst += (p.y - meanY) ** 2;

        const predClass = yHat >= 0 ? 1 : -1;
        const actualClass = p.y >= 0 ? 1 : -1;
        if (predClass === actualClass) correct++;
      }

      const mse = sse / pts.length;
      const rmse = Math.sqrt(mse);
      const r2 = sst > 1e-8 ? Math.max(-1.0, 1.0 - sse / sst) : (sse < 1e-8 ? 1.0 : 0.0);
      const acc = (correct / pts.length) * 100;
      return { mse, rmse, r2, acc };
    };

    const train = calcMetrics(dataset.train);
    const test = calcMetrics(dataset.test);

    return {
      trainLoss: train.mse,
      testLoss: test.mse,
      trainR2: train.r2,
      testR2: test.r2,
      trainAccuracy: train.acc,
      testAccuracy: test.acc,
      sparsity: { zeroCount: 0, totalCount: K },
    };
  },

  getVisualArtifacts(
    state: PolynomialRegressionState,
    _dataset: DatasetContainer,
    _params: PolynomialRegressionParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    const is1D = state.is1D ?? true;
    const d = state.degree;
    const K = getFeatureCount(d, is1D);

    if (is1D) {
      if (state.curvePoints && state.curvePoints.length > 1) {
        artifacts.push({
          type: 'curve-fit',
          points: state.curvePoints,
          color: state.rungeWarning ? '#D94E34' : '#141413',
          style: 'solid',
          label: `Fit: Degree ${d} Polynomial (K=${d + 1})`,
          rungeWarning: state.rungeWarning,
        });
      }

      if (state.residuals && state.residuals.length > 0) {
        artifacts.push({
          type: 'residual-bars',
          residuals: state.residuals,
        });
      }
    } else {
      artifacts.push({
        type: 'curve-fit',
        points: [],
        color: state.rungeWarning ? '#D94E34' : '#141413',
        style: 'solid',
        label: `Fit Surface: 2D Degree ${d} Polynomial (K=${K} terms)`,
        rungeWarning: state.rungeWarning,
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: PolynomialRegressionState): Float32Array {
    const buf = new Float32Array(state.coefficients.length + 5);
    buf[0] = state.degree;
    buf[1] = state.epoch;
    buf[2] = state.rungeWarning ? 1 : 0;
    buf[3] = state.conditioningWarning ? 1 : 0;
    buf[4] = state.is1D ? 1 : 0;
    for (let i = 0; i < state.coefficients.length; i++) {
      buf[5 + i] = state.coefficients[i];
    }
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: PolynomialRegressionState): PolynomialRegressionState {
    const degree = Math.round(buffer[0]);
    const epoch = Math.round(buffer[1]);
    const rungeWarning = buffer[2] === 1;

    let conditioningWarning = false;
    let is1D = state.is1D ?? true;
    let coeffOffset = 3;

    if (buffer.length >= 5) {
      conditioningWarning = buffer[3] === 1;
      is1D = buffer[4] === 1;
      coeffOffset = 5;
    }

    const K = getFeatureCount(degree, is1D);
    const coeffs = new Float32Array(K);
    for (let i = 0; i < K && (coeffOffset + i) < buffer.length; i++) {
      coeffs[i] = buffer[coeffOffset + i];
    }

    return {
      ...state,
      degree,
      epoch,
      rungeWarning,
      conditioningWarning,
      is1D,
      coefficients: coeffs,
    };
  },

  explain(
    state: PolynomialRegressionState,
    params: PolynomialRegressionParams,
    lastStep?: StepResult<PolynomialRegressionState>
  ): ExplanationPayload {
    const d = state.degree;
    const is1D = state.is1D ?? true;
    const K = getFeatureCount(d, is1D);
    const l2 = Number(params.regularization ?? 0.001);
    const solver = params.solver ?? 'sgd';
    const passes = Math.max(1, Math.round(Number(params.batchPasses ?? params.cyclesPerStep ?? params.subEpochs ?? 5)));

    const trainLoss = lastStep?.metrics.trainLoss.toFixed(4) ?? '0.0000';
    const testLoss = lastStep?.metrics.testLoss.toFixed(4) ?? '0.0000';
    const trainR2 = lastStep?.metrics.trainR2 !== undefined ? lastStep.metrics.trainR2.toFixed(3) : '1.000';
    const trainRmse = Math.sqrt(Math.max(0, Number(trainLoss))).toFixed(4);
    const testRmse = Math.sqrt(Math.max(0, Number(testLoss))).toFixed(4);

    let formulaDesc = '';
    if (is1D) {
      const c0 = (state.coefficients[0] ?? 0).toFixed(2);
      const c1 = (state.coefficients[1] ?? 0).toFixed(2);
      if (d === 1) {
        formulaDesc = `y_hat = (${c0}) + (${c1})(0.25 x1)`;
      } else if (d === 2) {
        const c2 = (state.coefficients[2] ?? 0).toFixed(2);
        formulaDesc = `y_hat = (${c0}) + (${c1})(0.25 x1) + (${c2})(0.25 x1)²`;
      } else {
        const cd = (state.coefficients[d] ?? 0).toFixed(2);
        formulaDesc = `y_hat = (${c0}) + (${c1})(0.25 x1) + ... + (${cd})(0.25 x1)^${d} [${d + 1} monomial terms]`;
      }
    } else {
      const c0 = (state.coefficients[0] ?? 0).toFixed(2);
      const c1 = (state.coefficients[1] ?? 0).toFixed(2);
      const c2 = (state.coefficients[2] ?? 0).toFixed(2);
      formulaDesc = `y_hat = (${c0}) + (${c1})(0.25 x1) + (${c2})(0.25 x2) + ... [${K} bivariate cross terms i+j <= ${d}]`;
    }

    let stabilityDesc = `STABLE: Well-conditioned Vandermonde basis; ridge stabilizer (λ=${l2.toFixed(3)}) bounds condition number and prevents boundary swing.`;
    if (state.rungeWarning && state.conditioningWarning) {
      stabilityDesc = `CRITICAL: Runge boundary swing detected (edge oscillation > 2.5x interior) and ill-conditioned Vandermonde matrix. Increase Ridge Stabilizer (λ ≥ 0.05).`;
    } else if (state.rungeWarning) {
      stabilityDesc = `WARNING: Runge edge oscillation detected: higher-order monomials swing at domain boundaries. Increase Ridge Stabilizer (λ) to damp edge curvature.`;
    } else if (state.conditioningWarning) {
      stabilityDesc = `CAUTION: Ill-conditioned design matrix with near-singular eigenvalues. Regularized with ridge λ=${l2.toFixed(3)} to prevent numerical divergence.`;
    }

    const solverDesc = solver === 'closed-form'
      ? `analytical normal equations (Ridge λ=${l2})`
      : `monomial SGD (${passes} pass${passes > 1 ? 'es' : ''}/step, λ=${l2})`;

    return {
      summary: is1D
        ? `Polynomial Regression expands coordinate x₁ into a ${d}-degree Vandermonde basis (${d + 1} features), fitting non-linear curves via linear least squares.`
        : `Polynomial Regression expands coordinates (x₁, x₂) into a bivariate polynomial basis with cross terms of total degree ≤ ${d} (${K} features).`,
      coreAssumption:
        'Continuous smooth non-linear target surface well-approximated by Taylor/monomial expansions.',
      whenItFails:
        "High degrees induce Runge's phenomenon with catastrophic edge oscillations unless damped with Tikhonov L2 ridge regularization.",
      microLog: [
        `${is1D ? `Vandermonde expansion: dim(Φ) = ${d + 1}` : `Bivariate polynomial expansion: dim(Φ) = ${K} (cross-terms i+j <= ${d})`} | ${formulaDesc}`,
        `Fit evaluation: Train MSE = ${trainLoss} (R² = ${trainR2}, RMSE = ${trainRmse}), Test MSE = ${testLoss} (RMSE = ${testRmse})`,
        `Parameter vector updated via ${solverDesc} | ${stabilityDesc}`,
      ],
      mathLatex: {
        objective: is1D
          ? '\\min_{\\mathbf{w}} \\frac{1}{2N} \\sum_{i=1}^N (y_i - \\mathbf{w}^T \\phi(x_{1,i}))^2 + \\frac{\\lambda}{2} \\sum_{j=1}^d w_j^2'
          : '\\min_{\\mathbf{w}} \\frac{1}{2N} \\sum_{i=1}^N (y_i - \\mathbf{w}^T \\phi(x_{1,i}, x_{2,i}))^2 + \\frac{\\lambda}{2} \\sum_{j=1}^{K-1} w_j^2',
        lossOrUpdate: '(\\Phi^T \\Phi + \\Lambda) \\mathbf{w} = \\Phi^T \\mathbf{y} \\quad (\\Lambda_{00}=0, \\Lambda_{jj}=\\lambda)',
        dualOrConstraint: is1D
          ? '\\phi(x_1) = [1, \\tilde{x}_1, \\tilde{x}_1^2, \\dots, \\tilde{x}_1^d]^T, \\quad \\tilde{x}_1 = 0.25 x_1'
          : '\\phi(x_1, x_2) = [\\tilde{x}_1^i \\tilde{x}_2^j]_{i+j \\le d}, \\quad K = \\frac{(d+1)(d+2)}{2}',
      },
      codeSnippet: {
        sklearn: is1D
          ? `from sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import Ridge\nfrom sklearn.pipeline import make_pipeline\n\n# 1D Vandermonde expansion: phi(x1) = [1, x1, ..., x1^d]\nmodel = make_pipeline(PolynomialFeatures(degree=${d}, include_bias=True), Ridge(alpha=${l2}))\nmodel.fit(X_train[:, :1], y_train)`
          : `from sklearn.preprocessing import PolynomialFeatures\nfrom sklearn.linear_model import Ridge\nfrom sklearn.pipeline import make_pipeline\n\n# 2D Bivariate expansion with cross terms: i + j <= ${d} (${K} features)\nmodel = make_pipeline(PolynomialFeatures(degree=${d}, include_bias=True), Ridge(alpha=${l2}))\nmodel.fit(X_train[:, :2], y_train)`,
        pytorch: is1D
          ? `import torch\nimport torch.nn as nn\n\nclass Poly1D(nn.Module):\n    def __init__(self, degree=${d}):\n        super().__init__()\n        self.linear = nn.Linear(degree + 1, 1, bias=False)\n    def forward(self, x1):\n        powers = torch.cat([(0.25 * x1) ** p for p in range(degree + 1)], dim=-1)\n        return self.linear(powers)`
          : `import torch\nimport torch.nn as nn\n\nclass Poly2D(nn.Module):\n    def __init__(self, degree=${d}):\n        super().__init__()\n        self.linear = nn.Linear(${K}, 1, bias=False)\n    def forward(self, x): # x: [N, 2]\n        terms = []\n        for d_sum in range(degree + 1):\n            for p2 in range(d_sum + 1):\n                p1 = d_sum - p2\n                terms.append(((0.25 * x[:, :1]) ** p1) * ((0.25 * x[:, 1:2]) ** p2))\n        return self.linear(torch.cat(terms, dim=-1))`,
      },
    };
  },
};
