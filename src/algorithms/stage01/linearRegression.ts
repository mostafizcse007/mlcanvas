import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { solveNormalEquations, huberLoss, huberDerivative } from '../../utils/math';
import { mulberry32 } from '../../utils/prng';

export interface LinearRegressionState {
  weights: Float32Array; // [w1, w2]
  bias: number;
  epoch: number;
  residuals: { id: number; x1: number; x2: number; y: number; yHat: number; error: number }[];
  gradientNorm: number;
  is1D?: boolean;
}

export interface LinearRegressionParams {
  solver?: 'closed-form' | 'sgd';
  learningRate?: number;
  criterion?: 'mse' | 'huber';
  huberDelta?: number;
  batchPasses?: number;
  cyclesPerStep?: number;
  subEpochs?: number;
}

/**
 * Robustly detects whether a regression dataset is 1D (x1 -> y) or 2D (x1, x2 -> y),
 * even when 10x anisotropic stretching or z-score/minmax feature scaling is applied.
 */
export function is1DRegressionDataset(dataset: DatasetContainer): boolean {
  const train = dataset.train;
  const test = dataset.test;
  const points = train && train.length >= 2 ? train : test;
  if (!points || points.length < 2) return false;

  const sample = points.slice(0, Math.min(points.length, 25));
  // Raw 1D: x2 == y
  if (sample.every((p) => Math.abs(p.x2 - p.y) < 1e-3)) return true;
  // 10x-stretched 1D: x2 == 10 * y
  if (sample.every((p) => Math.abs(p.x2 - 10 * p.y) < 1e-3)) return true;

  // Linear correlation check for scaled modes (zscore, minmax):
  // In 1D regression datasets, x2 is generated directly from y before scaling, so x2 is a strict linear transform of y.
  // In 2D surface regression, x2 and y are independent uniform coordinates (|r| < 0.85).
  let meanX2 = 0, meanY = 0;
  for (let i = 0; i < sample.length; i++) {
    meanX2 += sample[i].x2;
    meanY += sample[i].y;
  }
  meanX2 /= sample.length;
  meanY /= sample.length;
  let varX2 = 0, varY = 0, cov = 0;
  for (let i = 0; i < sample.length; i++) {
    const dx = sample[i].x2 - meanX2;
    const dy = sample[i].y - meanY;
    varX2 += dx * dx;
    varY += dy * dy;
    cov += dx * dy;
  }
  if (varX2 > 1e-8 && varY > 1e-8) {
    const r2 = (cov * cov) / (varX2 * varY);
    if (r2 > 0.999) return true;
  }
  return false;
}

/**
 * Iteratively Reweighted Least Squares (IRLS) for Huber M-Estimation.
 * Solves (X^T W X) w = X^T W y iteratively, robustly downweighting residual outliers past delta.
 */
function solveHuberIRLS(
  X: number[][],
  y: number[],
  delta = 1.0,
  maxIter = 10,
  tol = 1e-4
): number[] | null {
  const N = X.length;
  if (N === 0) return null;
  const D = X[0].length;

  let w = solveNormalEquations(X, y, 1e-4) || solveNormalEquations(X, y, 1e-2);
  if (!w) return null;

  for (let iter = 0; iter < maxIter; iter++) {
    const Xw: number[][] = new Array(N);
    const yw: number[] = new Array(N);

    for (let i = 0; i < N; i++) {
      let pred = 0;
      for (let j = 0; j < D; j++) pred += X[i][j] * w![j];
      const res = Math.abs(y[i] - pred);
      const weight = res <= delta ? 1.0 : delta / Math.max(res, 1e-6);
      const s = Math.sqrt(weight);

      Xw[i] = new Array(D);
      for (let j = 0; j < D; j++) {
        Xw[i][j] = X[i][j] * s;
      }
      yw[i] = y[i] * s;
    }

    const nextW = solveNormalEquations(Xw, yw, 1e-4) || solveNormalEquations(Xw, yw, 1e-2);
    if (!nextW) break;

    let maxChange = 0;
    for (let j = 0; j < D; j++) {
      maxChange = Math.max(maxChange, Math.abs(nextW[j] - w![j]));
    }
    w = nextW;
    if (maxChange < tol) break;
  }

  return w;
}

/**
 * Stable OLS Normal Equations & Huber IRLS solver with ill-conditioned matrix fallbacks.
 * Prevents false perfect fits (loss=0, R2=1) under extreme noise, collinearity, or singular design.
 */
function solveOLS(
  train: { x1: number; x2: number; y: number }[],
  is1D: boolean,
  criterion: 'mse' | 'huber' = 'mse',
  delta = 1.0
): { w0: number; w1: number; bias: number } {
  const N = train.length;
  if (N === 0) {
    return { w0: 0, w1: is1D ? -1 : 0, bias: 0 };
  }

  let sumY = 0;
  for (let i = 0; i < N; i++) sumY += train[i].y;
  const meanY = sumY / N;

  const solveFn = (X: number[][], y: number[]) => {
    if (criterion === 'huber') {
      const hSol = solveHuberIRLS(X, y, delta);
      if (hSol) return hSol;
    }
    return solveNormalEquations(X, y, 1e-4) || solveNormalEquations(X, y, 1e-2);
  };

  if (is1D) {
    // 1D regression: y = m * x1 + c. Matrix X is N x 2 with columns [x1, 1.0].
    // Target y is never leaked through x2.
    const X = train.map((p) => [p.x1, 1.0]);
    const y = train.map((p) => p.y);
    const sol = solveFn(X, y);
    if (sol && !isNaN(sol[0]) && !isNaN(sol[1]) && isFinite(sol[0]) && isFinite(sol[1])) {
      return { w0: sol[0], w1: -1, bias: sol[1] };
    }
    // Fallback: horizontal least-squares line at target mean (R2 = 0.0, no false 1.0)
    return { w0: 0, w1: -1, bias: meanY };
  } else {
    // 2D regression: y = w0 * x1 + w1 * x2 + b. Matrix X is N x 3 with columns [x1, x2, 1.0].
    const X = train.map((p) => [p.x1, p.x2, 1.0]);
    const y = train.map((p) => p.y);
    const sol = solveFn(X, y);
    if (sol && !isNaN(sol[0]) && !isNaN(sol[1]) && !isNaN(sol[2]) &&
        isFinite(sol[0]) && isFinite(sol[1]) && isFinite(sol[2])) {
      return { w0: sol[0], w1: sol[1], bias: sol[2] };
    }
    return { w0: 0, w1: 0, bias: meanY };
  }
}

export const linearRegressionPlugin: AlgorithmPlugin<LinearRegressionState, LinearRegressionParams> = {
  id: 'linear-regression',
  name: 'Linear Regression (OLS / SGD)',
  shortCode: 'LinReg (OLS)',
  stage: 'continuous-2d',
  taskType: 'regression',
  executionMode: 'epoch-loop',
  description: 'Ordinary Least Squares & Stochastic Gradient Descent with residual error bars and Huber loss.',
  paperReference: 'Gauss (1809), Legendre (1805); Robbins & Monro (1951)',

  primaryParams: [
    {
      id: 'solver',
      name: 'Solver Mode',
      type: 'select',
      options: [
        { label: 'Iterative (SGD)', value: 'sgd' },
        { label: 'Closed-Form (Normal Eq)', value: 'closed-form' },
      ],
      default: 'sgd',
      description: 'Choose stochastic iterative gradient descent or instant analytical inversion.',
    },
    {
      id: 'criterion',
      name: 'Loss Criterion',
      type: 'select',
      options: [
        { label: 'MSE (L2 Squared Error)', value: 'mse' },
        { label: 'Huber Loss (Robust Outlier)', value: 'huber' },
      ],
      default: 'mse',
      description: 'Mean Squared Error or robust Huber loss switching to linear penalty past delta.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'select',
      options: [
        { label: '0.005 (Gentle)', value: 0.005 },
        { label: '0.02 (Standard)', value: 0.02 },
        { label: '0.08 (Fast)', value: 0.08 },
      ],
      default: 0.02,
      description: 'SGD step size per data pass.',
    },
  ],

  advancedParams: [
    {
      id: 'huberDelta',
      name: 'Huber Threshold (δ)',
      type: 'slider',
      min: 0.2,
      max: 3.0,
      step: 0.1,
      default: 1.0,
      description: 'Threshold at which Huber loss transitions from quadratic to linear.',
    },
  ],

  symbolicLinks: [
    {
      id: 'w',
      symbolLatex: '\\mathbf{w} = (X^T X)^{-1} X^T y',
      label: 'Normal Equation Weights',
      description: 'Analytical least-squares solution projecting targets onto feature column space.',
      highlightTarget: 'normal-vector',
    },
    {
      id: 'residuals',
      symbolLatex: 'e_i = y_i - \\hat{y}_i',
      label: 'Vertical Residuals',
      description: 'Geometric distance between each empirical observation and the fitted hyperplane.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: LinearRegressionParams, seed: number): LinearRegressionState {
    const rand = mulberry32(seed);
    const train = dataset.train;
    const is1D = is1DRegressionDataset(dataset);
    const solver = params?.solver ?? 'sgd';
    const criterion = params?.criterion ?? 'mse';
    const delta = Number(params?.huberDelta ?? 1.0);

    let w0: number;
    let w1: number;
    let bias: number;

    if (solver === 'closed-form' && train.length >= 2) {
      const ols = solveOLS(train, is1D, criterion, delta);
      w0 = ols.w0;
      w1 = ols.w1;
      bias = ols.bias;
    } else {
      w0 = 0.5 + (rand() - 0.5) * 0.4;
      w1 = is1D ? -1 : (0.4 + (rand() - 0.5) * 0.4);
      bias = (rand() - 0.5) * 0.5;
    }

    const weights = new Float32Array([w0, w1]);

    const residuals = train.map((p) => {
      const yHat = is1D ? w0 * p.x1 + bias : w0 * p.x1 + w1 * p.x2 + bias;
      return {
        id: p.id,
        x1: p.x1,
        x2: p.x2,
        y: p.y,
        yHat,
        error: p.y - yHat,
      };
    });

    return {
      weights,
      bias,
      epoch: solver === 'closed-form' ? 1 : 0,
      residuals,
      gradientNorm: 0.1,
      is1D,
    };
  },

  step(
    state: LinearRegressionState,
    dataset: DatasetContainer,
    params: LinearRegressionParams
  ): StepResult<LinearRegressionState> {
    const solver = params.solver ?? 'sgd';
    const criterion = params.criterion ?? 'mse';
    const delta = Number(params.huberDelta ?? 1.0);
    const lr = Number(params.learningRate ?? 0.02);
    const passes = Math.max(1, Math.round(Number(params.batchPasses ?? params.cyclesPerStep ?? params.subEpochs ?? 5)));

    const train = dataset.train;
    const N = Math.max(1, train.length);
    const is1D = is1DRegressionDataset(dataset);

    const w = new Float32Array(state.weights);
    let b = state.bias;

    if (solver === 'closed-form') {
      const ols = solveOLS(train, is1D, criterion, delta);
      w[0] = ols.w0;
      w[1] = ols.w1;
      b = ols.bias;
    } else {
      if (is1D) {
        let m = w[0];
        let c = b;

        for (let pass = 0; pass < passes; pass++) {
          for (let i = 0; i < N; i++) {
            const pt = train[i];
            const yHat = m * pt.x1 + c;
            const residual = pt.y - yHat;

            const gradFactor = criterion === 'huber'
              ? huberDerivative(residual, delta)
              : residual;

            m += (lr * gradFactor * pt.x1) / N;
            c += (lr * gradFactor) / N;
          }
        }

        w[0] = m;
        w[1] = -1;
        b = c;
      } else {
        for (let pass = 0; pass < passes; pass++) {
          for (let i = 0; i < N; i++) {
            const pt = train[i];
            const yHat = w[0] * pt.x1 + w[1] * pt.x2 + b;
            const residual = pt.y - yHat;

            const gradFactor = criterion === 'huber'
              ? huberDerivative(residual, delta)
              : residual;

            w[0] += (lr * gradFactor * pt.x1) / N;
            w[1] += (lr * gradFactor * pt.x2) / N;
            b += (lr * gradFactor) / N;
          }
        }
      }
    }

    const residuals = train.map((p) => {
      const yHat = is1D ? w[0] * p.x1 + b : w[0] * p.x1 + w[1] * p.x2 + b;
      return { id: p.id, x1: p.x1, x2: p.x2, y: p.y, yHat, error: p.y - yHat };
    });

    const nextState: LinearRegressionState = {
      weights: w,
      bias: b,
      epoch: solver === 'closed-form' ? 1 : state.epoch + 1,
      residuals,
      gradientNorm: Math.hypot(w[0], is1D ? 0 : w[1]),
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

  predict(input: Float32Array, state: LinearRegressionState, _params: LinearRegressionParams): number {
    if (!state || !state.weights) return 0;
    const is1D = state.is1D ?? (Math.abs(state.weights[1] + 1.0) < 1e-4);
    if (is1D) {
      return state.weights[0] * input[0] + (state.bias ?? 0);
    }
    return state.weights[0] * input[0] + state.weights[1] * input[1] + (state.bias ?? 0);
  },

  loss(state: LinearRegressionState, dataset: DatasetContainer, params: LinearRegressionParams): TelemetryMetrics {
    const criterion = params.criterion ?? 'mse';
    const delta = Number(params.huberDelta ?? 1.0);
    const is1D = state.is1D ?? is1DRegressionDataset(dataset);

    const calcMetrics = (pts: typeof dataset.train) => {
      if (!pts || pts.length === 0) return { loss: 0, r2: 1.0, acc: 100.0 };
      let sum = 0;
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
        const yHat = is1D ? state.weights[0] * p.x1 + state.bias : state.weights[0] * p.x1 + state.weights[1] * p.x2 + state.bias;
        const err = p.y - yHat;
        sum += criterion === 'huber' ? huberLoss(err, delta) : err * err;
        sse += err * err;
        sst += (p.y - meanY) ** 2;

        const predClass = yHat >= 0 ? 1 : -1;
        const actualClass = p.y >= 0 ? 1 : -1;
        if (predClass === actualClass) correct++;
      }

      const loss = sum / pts.length;
      const r2 = sst > 1e-8 ? Math.max(-1.0, 1.0 - sse / sst) : (sse < 1e-8 ? 1.0 : 0.0);
      const acc = (correct / pts.length) * 100;
      return { loss, r2, acc };
    };

    const trainMetrics = calcMetrics(dataset.train);
    const testMetrics = calcMetrics(dataset.test);

    return {
      trainLoss: trainMetrics.loss,
      testLoss: testMetrics.loss,
      trainR2: trainMetrics.r2,
      testR2: testMetrics.r2,
      trainAccuracy: trainMetrics.acc,
      testAccuracy: testMetrics.acc,
      gradientNorm: state.gradientNorm,
      sparsity: { zeroCount: 0, totalCount: is1D ? 1 : 2 },
    };
  },

  getVisualArtifacts(
    state: LinearRegressionState,
    dataset: DatasetContainer,
    _params: LinearRegressionParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    const is1D = state.is1D ?? is1DRegressionDataset(dataset);

    // 1. Fitted Line / Surface Contour
    if (Math.hypot(state.weights[0], state.weights[1]) > 1e-5) {
      artifacts.push({
        type: 'hyperplane',
        normal: [state.weights[0], state.weights[1]],
        bias: state.bias,
        color: '#141413',
        style: 'solid',
        label: is1D ? undefined : 'ŷ = 0 Contour',
      });
    }

    // 2. Residual Error Bars (rendered on 1D regression where vertical axis is target y)
    if (is1D && state.residuals && state.residuals.length > 0) {
      artifacts.push({
        type: 'residual-bars',
        residuals: state.residuals,
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: LinearRegressionState): Float32Array {
    const buf = new Float32Array(6);
    buf[0] = state.weights[0];
    buf[1] = state.weights[1];
    buf[2] = state.bias;
    buf[3] = state.epoch;
    buf[4] = state.gradientNorm;
    buf[5] = state.is1D ? 1 : 0;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: LinearRegressionState): LinearRegressionState {
    const w = new Float32Array([buffer[0], buffer[1]]);
    const is1D = buffer.length > 5 ? buffer[5] === 1 : (Math.abs(w[1] + 1.0) < 1e-4);
    return {
      ...state,
      weights: w,
      bias: buffer[2],
      epoch: Math.round(buffer[3]),
      gradientNorm: buffer[4],
      is1D,
    };
  },

  explain(
    state: LinearRegressionState,
    params: LinearRegressionParams,
    lastStep?: StepResult<LinearRegressionState>
  ): ExplanationPayload {
    const is1D = state?.is1D ?? (state?.weights ? Math.abs(state.weights[1] + 1.0) < 1e-4 : true);
    const w0 = (state?.weights?.[0] ?? 0).toFixed(3);
    const w1 = (state?.weights?.[1] ?? 0).toFixed(3);
    const b = (state?.bias ?? 0).toFixed(3);
    const trainLoss = lastStep?.metrics.trainLoss !== undefined ? lastStep.metrics.trainLoss.toFixed(4) : '0.0000';
    const passes = Math.max(1, Math.round(Number(params.batchPasses ?? params.cyclesPerStep ?? params.subEpochs ?? 5)));
    const solver = params.solver ?? 'sgd';
    const criterion = params.criterion ?? 'mse';
    const delta = Number(params.huberDelta ?? 1.0);

    const affineFormula = is1D
      ? `Affine projection: y_hat = (${w0}) x1 + (${b}) [units: x1 -> y]`
      : `Affine projection: y_hat = (${w0}) x1 + (${w1}) x2 + (${b}) [units: (x1,x2) -> y]`;

    const lossLabel = criterion === 'huber' ? `Huber Loss (δ=${delta.toFixed(1)})` : 'MSE';
    const solverLabel = solver === 'sgd'
      ? `SGD (${passes} pass${passes > 1 ? 'es' : ''}/step)`
      : criterion === 'huber'
      ? `closed-form Huber M-estimator (IRLS, δ=${delta.toFixed(1)})`
      : 'exact normal equations (OLS)';

    return {
      summary:
        'Linear Regression models the conditional expectation E[Y|X] as an affine hyper-plane minimizing residual sum of squares.',
      coreAssumption:
        'Homoscedastic Gaussian error residuals e_i ~ N(0, σ²) independent of inputs, with linear parameter dependence.',
      whenItFails:
        'Susceptible to catastrophic leverage from single rogue outliers under squared L2 loss, and singular matrix inversion under multicollinearity.',
      microLog: [
        affineFormula,
        `Residual error calculation across observations with ${lossLabel} = ${trainLoss}`,
        `Parameter vector updated via ${solverLabel}`,
      ],
      mathLatex: {
        objective: criterion === 'huber'
          ? `\\min_{\\mathbf{w}, b} \\frac{1}{N} \\sum_{i=1}^N L_\\delta(y_i - \\mathbf{w}^T \\mathbf{x}_i - b)`
          : '\\min_{\\mathbf{w}, b} \\frac{1}{2N} \\sum_{i=1}^N (y_i - \\mathbf{w}^T \\mathbf{x}_i - b)^2',
        lossOrUpdate:
          solver === 'sgd'
            ? criterion === 'huber'
              ? '\\mathbf{w} \\leftarrow \\mathbf{w} + \\frac{\\eta}{N} \\sum_{i=1}^N \\psi_\\delta(y_i - \\hat{y}_i) \\mathbf{x}_i'
              : '\\mathbf{w} \\leftarrow \\mathbf{w} + \\frac{\\eta}{N} \\sum_{i=1}^N (y_i - \\hat{y}_i) \\mathbf{x}_i'
            : criterion === 'huber'
              ? '(X^T W X) \\mathbf{w}^* = X^T W \\mathbf{y} \\quad (\\text{IRLS})'
              : '\\mathbf{w}^* = (X^T X)^{-1} X^T \\mathbf{y}',
        dualOrConstraint: criterion === 'huber'
          ? 'L_\\delta(r) = \\begin{cases} \\frac{1}{2}r^2 & |r| \\le \\delta \\\\ \\delta(|r| - \\frac{1}{2}\\delta) & |r| > \\delta \\end{cases}'
          : 'e_i = y_i - \\hat{y}_i \\quad (\\text{Orthogonal residual condition: } X^T \\mathbf{e} = \\mathbf{0})',
      },
      codeSnippet: {
        sklearn: criterion === 'huber'
          ? `from sklearn.linear_model import HuberRegressor\n\nmodel = HuberRegressor(epsilon=${delta.toFixed(2)})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f"Weights: {model.coef_}, Intercept: {model.intercept_}")`
          : `from sklearn.linear_model import LinearRegression\n\nmodel = LinearRegression(fit_intercept=True)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)\nprint(f"Weights: {model.coef_}, Intercept: {model.intercept_}")`,
        pytorch: `import torch\nimport torch.nn as nn\n\nmodel = nn.Linear(in_features=${is1D ? 1 : 2}, out_features=1)\noptimizer = torch.optim.SGD(model.parameters(), lr=${Number(params.learningRate ?? 0.02)})\ncriterion = ${criterion === 'huber' ? `nn.HuberLoss(delta=${delta.toFixed(2)})` : 'nn.MSELoss()'}\n\nfor epoch in range(100):\n    optimizer.zero_grad()\n    y_hat = model(X)\n    loss = criterion(y_hat, y)\n    loss.backward()\n    optimizer.step()`,
      },
    };
  },
};

