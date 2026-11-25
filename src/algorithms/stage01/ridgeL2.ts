import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { solveNormalEquations, l2Norm } from '../../utils/math';
import { mulberry32 } from '../../utils/prng';

export interface RidgeL2State {
  weights: Float32Array; // [w1, w2]
  bias: number;
  epoch: number;
  effectiveRadius: number;
  gradientNorm: number;
}

export interface RidgeL2Params {
  lambdaPenalty?: number;
  solver?: 'closed-form' | 'sgd';
  learningRate?: number;
}

export const ridgeL2Plugin: AlgorithmPlugin<RidgeL2State, RidgeL2Params> = {
  id: 'ridge-l2',
  name: 'Ridge L2 Regularized Regression',
  shortCode: 'Ridge L2',
  stage: 'continuous-2d',
  taskType: 'regression',
  executionMode: 'epoch-loop',
  description: 'Tikhonov L2 quadratic penalty with analytical Gram stabilization and circular weight-space inset.',
  paperReference: 'Hoerl & Kennard, Ridge Regression: Biased Estimation for Nonorthogonal Problems, Technometrics 1970',

  primaryParams: [
    {
      id: 'lambdaPenalty',
      name: 'Ridge Penalty (λ)',
      type: 'select',
      options: [
        { label: '0.001 (Minimal Shrinkage)', value: 0.001 },
        { label: '0.05', value: 0.05 },
        { label: '0.20 (Balanced)', value: 0.2 },
        { label: '1.00 (Heavy Shrinkage)', value: 1.0 },
        { label: '5.00 (Extreme Damping)', value: 5.0 },
      ],
      default: 0.2,
      description: 'Quadratic penalty coefficient shrinking weight norms proportionally toward the origin.',
    },
    {
      id: 'solver',
      name: 'Solver Mode',
      type: 'select',
      options: [
        { label: 'Weight Decay SGD', value: 'sgd' },
        { label: 'Closed-Form (Tikhonov Inversion)', value: 'closed-form' },
      ],
      default: 'sgd',
      description: 'Choose gradient descent with L2 weight decay or instant analytical normal equation.',
    },
  ],

  advancedParams: [
    {
      id: 'learningRate',
      name: 'SGD Step Size (η)',
      type: 'slider',
      min: 0.001,
      max: 0.1,
      step: 0.005,
      default: 0.02,
      description: 'Learning rate when SGD solver is active.',
    },
  ],

  symbolicLinks: [
    {
      id: 'l2',
      symbolLatex: '\\|\\mathbf{w}\\|_2^2 \\le r^2',
      label: 'L2 Ball Constraint',
      description: 'Spherical convex constraint region where tangential intersection smoothly contracts all weights.',
      highlightTarget: 'weights-inset',
    },
  ],

  init(dataset: DatasetContainer, params: RidgeL2Params, seed: number): RidgeL2State {
    const rand = mulberry32(seed);
    const train = dataset.train;
    const is1D = train.length >= 2 && train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);
    const lambda = Number(params?.lambdaPenalty ?? 0.2);
    const w0 = (0.6 + rand() * 0.4) / (1 + lambda);
    const w1 = is1D ? -1 : ((rand() - 0.5) * 0.5) / (1 + lambda);
    const bias = (rand() - 0.5) * 0.4;
    const weights = new Float32Array([w0, w1]);
    const norm = Math.hypot(w0, is1D ? 0 : w1);
    const effectiveRadius = Math.max(0.2, norm);

    return {
      weights,
      bias,
      epoch: 0,
      effectiveRadius,
      gradientNorm: 0.1,
    };
  },

  step(
    state: RidgeL2State,
    dataset: DatasetContainer,
    params: RidgeL2Params
  ): StepResult<RidgeL2State> {
    const lambda = Number(params.lambdaPenalty ?? 0.2);
    const solver = params.solver ?? 'sgd';
    const lr = Number(params.learningRate ?? 0.02);

    const w = new Float32Array(state.weights);
    let b = state.bias;
    const train = dataset.train;
    const N = Math.max(1, train.length);
    const is1D = train.length >= 2 && train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);

    if (is1D) {
      let m = w[0];
      let c = b;

      if (solver === 'closed-form') {
        const X = train.map((p) => [p.x1, 1.0]);
        const y = train.map((p) => p.y);
        const sol = solveNormalEquations(X, y, lambda);
        if (sol) {
          m = sol[0];
          c = sol[1];
        }
      } else {
        // SGD with L2 weight decay: accumulate gradient over pass, apply weight decay once per pass
        for (let pass = 0; pass < 5; pass++) {
          let gm = 0;
          let gc = 0;
          for (let i = 0; i < N; i++) {
            const pt = train[i];
            const yHat = m * pt.x1 + c;
            const err = pt.y - yHat;

            gm += err * pt.x1;
            gc += err;
          }
          m = (1.0 - lr * lambda) * m + (lr * gm) / N;
          c += (lr * gc) / N;
        }
      }

      w[0] = m;
      w[1] = -1;
      b = c;
    } else {
      if (solver === 'closed-form') {
        const X = train.map((p) => [p.x1, p.x2, 1.0]);
        const y = train.map((p) => p.y);
        const sol = solveNormalEquations(X, y, lambda);
        if (sol) {
          w[0] = sol[0];
          w[1] = sol[1];
          b = sol[2];
        }
      } else {
        // SGD with L2 weight decay: accumulate gradient over pass, apply weight decay once per pass
        for (let pass = 0; pass < 5; pass++) {
          let g0 = 0;
          let g1 = 0;
          let gb = 0;
          for (let i = 0; i < N; i++) {
            const pt = train[i];
            const yHat = w[0] * pt.x1 + w[1] * pt.x2 + b;
            const err = pt.y - yHat;

            g0 += err * pt.x1;
            g1 += err * pt.x2;
            gb += err;
          }
          w[0] = (1.0 - lr * lambda) * w[0] + (lr * g0) / N;
          w[1] = (1.0 - lr * lambda) * w[1] + (lr * g1) / N;
          b += (lr * gb) / N;
        }
      }
    }

    const norm = is1D ? Math.abs(w[0]) : l2Norm(w);

    const nextState: RidgeL2State = {
      weights: w,
      bias: b,
      epoch: state.epoch + 1,
      effectiveRadius: Math.max(0.2, norm),
      gradientNorm: norm,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged: solver === 'closed-form',
    };
  },

  predict(input: Float32Array, state: RidgeL2State, _params: RidgeL2Params): number {
    if (!state || !state.weights) return 0;
    return state.weights[0] * input[0] + state.weights[1] * input[1] + (state.bias ?? 0);
  },

  loss(state: RidgeL2State, dataset: DatasetContainer, params: RidgeL2Params): TelemetryMetrics {
    const lambda = Number(params.lambdaPenalty ?? 0.2);
    const train = dataset.train;
    const is1D = train.length >= 2 && train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);

    const calcMetrics = (pts: typeof dataset.train) => {
      if (pts.length === 0) return { loss: 0, r2: 1.0, acc: 100.0 };
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
        sse += err * err;
        sst += (p.y - meanY) ** 2;

        const predClass = yHat >= 0 ? 1 : -1;
        const actualClass = p.y >= 0 ? 1 : -1;
        if (predClass === actualClass) correct++;
      }
      const mse = sse / pts.length;
      const penalty = 0.5 * lambda * (is1D ? state.weights[0] ** 2 : state.weights[0] ** 2 + state.weights[1] ** 2);
      const r2 = sst > 1e-8 ? Math.max(-1.0, 1.0 - sse / sst) : 1.0;
      const acc = (correct / pts.length) * 100;
      return { loss: mse + penalty, r2, acc };
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
      sparsity: { zeroCount: 0, totalCount: 2 },
    };
  },

  getVisualArtifacts(
    state: RidgeL2State,
    _dataset: DatasetContainer,
    _params: RidgeL2Params
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];

    // 1. Separating / Regressing Hyperplane
    if (Math.hypot(state.weights[0], state.weights[1]) > 1e-5) {
      artifacts.push({
        type: 'hyperplane',
        normal: [state.weights[0], state.weights[1]],
        bias: state.bias,
        color: '#141413',
        style: 'solid',
      });
    }

    // 2. Weight Space (w1, w2) Inset with L2 Circle Ball
    artifacts.push({
      type: 'weight-constraint',
      constraintType: 'l2-circle',
      radius: state.effectiveRadius,
      currentWeight: [state.weights[0], state.weights[1]],
    });

    return artifacts;
  },

  serializeSnapshot(state: RidgeL2State): Float32Array {
    const buf = new Float32Array(5);
    buf[0] = state.weights[0];
    buf[1] = state.weights[1];
    buf[2] = state.bias;
    buf[3] = state.epoch;
    buf[4] = state.effectiveRadius;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: RidgeL2State): RidgeL2State {
    const w = new Float32Array([buffer[0], buffer[1]]);
    return {
      ...state,
      weights: w,
      bias: buffer[2],
      epoch: Math.round(buffer[3]),
      effectiveRadius: buffer[4],
      gradientNorm: l2Norm(w),
    };
  },

  explain(
    state: RidgeL2State,
    params: RidgeL2Params,
    lastStep?: StepResult<RidgeL2State>
  ): ExplanationPayload {
    const w0 = (state?.weights?.[0] ?? 0).toFixed(3);
    const w1 = (state?.weights?.[1] ?? 0).toFixed(3);
    const lambda = Number(params?.lambdaPenalty ?? 0.2).toFixed(3);
    const trainLoss = lastStep?.metrics.trainLoss !== undefined ? lastStep.metrics.trainLoss.toFixed(4) : '0.0000';

    return {
      summary:
        'Ridge Regression adds an L2 quadratic norm penalty ||w||_2^2, conditioning ill-posed inverse problems and shrinking weights proportionally.',
      coreAssumption:
        'Parameters are distributed under a Gaussian prior w ~ N(0, σ_w² I); smooth shrinkage without exact coordinate sparsity.',
      whenItFails:
        'Does not perform automatic feature selection; irrelevant features receive small non-zero weights rather than strict zeros.',
      microLog: [
        `Regularized Gram matrix condition: (X^T X + ${lambda} I)`,
        `Estimated parameters: w₁ = ${w0}, w₂ = ${w1} (L2 Norm = ${state.gradientNorm.toFixed(3)})`,
        `Empirical penalized loss = ${trainLoss}`,
      ],
      mathLatex: {
        objective: '\\min_{\\mathbf{w}} \\frac{1}{2N} \\|\\mathbf{y} - X \\mathbf{w}\\|_2^2 + \\frac{\\lambda}{2} \\|\\mathbf{w}\\|_2^2',
        lossOrUpdate: '\\mathbf{w}^* = (X^T X + \\lambda I)^{-1} X^T \\mathbf{y}',
        dualOrConstraint: '\\|\\mathbf{w}\\|_2^2 \\le r^2 \\quad (\\text{Smooth circular L2 constraint ball})',
      },
      codeSnippet: {
        sklearn: `from sklearn.linear_model import Ridge\n\nmodel = Ridge(alpha=${lambda}, fit_intercept=True)\nmodel.fit(X_train, y_train)\nprint(f"Weights: {model.coef_}, Intercept: {model.intercept_}")`,
        pytorch: `import torch\nimport torch.nn as nn\n\nmodel = nn.Linear(2, 1)\n# weight_decay in PyTorch Adam/SGD directly applies L2 Tikhonov penalty\noptimizer = torch.optim.SGD(model.parameters(), lr=0.02, weight_decay=${lambda})`,
      },
    };
  },
};

