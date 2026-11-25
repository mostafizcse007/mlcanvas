import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { softThreshold, l2Norm } from '../../utils/math';
import { mulberry32 } from '../../utils/prng';

export interface ElasticNetState {
  weights: Float32Array; // [w1, w2]
  bias: number;
  epoch: number;
  zeroWeightIndices: number[];
  maxDelta: number;
}

export interface ElasticNetParams {
  lambdaPenalty?: number;
  l1Ratio?: number; // α: 0 = pure Ridge, 1 = pure Lasso
  cyclesPerStep?: number;
}

export const elasticNetPlugin: AlgorithmPlugin<ElasticNetState, ElasticNetParams> = {
  id: 'elastic-net',
  name: 'Elastic Net Regularization',
  shortCode: 'ElasticNet',
  stage: 'continuous-2d',
  taskType: 'regression',
  executionMode: 'epoch-loop',
  description: 'Hybrid L1/L2 penalty combining Lasso coordinate sparsity with Ridge grouping for collinear features.',
  paperReference: 'Zou & Hastie, Regularization and variable selection via the elastic net, JRSS-B 2005',

  primaryParams: [
    {
      id: 'lambdaPenalty',
      name: 'Penalty Strength (λ)',
      type: 'select',
      options: [
        { label: '0.01 (Light)', value: 0.01 },
        { label: '0.05', value: 0.05 },
        { label: '0.20 (Balanced)', value: 0.2 },
        { label: '0.80 (Aggressive)', value: 0.8 },
      ],
      default: 0.2,
      description: 'Overall shrinkage scale multiplying the combined regularization term.',
    },
    {
      id: 'l1Ratio',
      name: 'L1 Ratio (α)',
      type: 'slider',
      min: 0.0,
      max: 1.0,
      step: 0.05,
      default: 0.5,
      description: '0.0 = Pure Ridge (Circle), 0.5 = Balanced Hybrid, 1.0 = Pure Lasso (Diamond).',
    },
  ],

  advancedParams: [
    {
      id: 'cyclesPerStep',
      name: 'Cycles per Epoch',
      type: 'slider',
      min: 1,
      max: 20,
      step: 1,
      default: 5,
      description: 'Number of full coordinate passes per frame.',
    },
  ],

  symbolicLinks: [
    {
      id: 'elastic',
      symbolLatex: '\\lambda \\left( \\alpha \\|\\mathbf{w}\\|_1 + \\frac{1-\\alpha}{2} \\|\\mathbf{w}\\|_2^2 \\right)',
      label: 'Hybrid Penalty Contour',
      description: 'Interpolates between strict diamond corners and smooth spherical curvature.',
      highlightTarget: 'weights-inset',
    },
  ],

  init(dataset: DatasetContainer, _params: ElasticNetParams, seed: number): ElasticNetState {
    const rand = mulberry32(seed);
    const train = dataset.train;
    const is1D = train.length >= 2 && train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);
    const w0 = 0.6 + rand() * 0.3;
    const w1 = is1D ? -1 : 0.4 + rand() * 0.2;
    let bias = 0;

    if (train.length > 0) {
      let sumY = 0;
      for (const p of train) sumY += p.y;
      bias = sumY / train.length;
    } else {
      bias = (rand() - 0.5) * 0.4;
    }

    return {
      weights: new Float32Array([w0, w1]),
      bias,
      epoch: 0,
      zeroWeightIndices: [],
      maxDelta: 0,
    };
  },

  step(
    state: ElasticNetState,
    dataset: DatasetContainer,
    params: ElasticNetParams
  ): StepResult<ElasticNetState> {
    const lambda = Number(params.lambdaPenalty ?? 0.2);
    const alpha = Number(params.l1Ratio ?? 0.5);
    const cycles = Number(params.cyclesPerStep ?? 5);

    const train = dataset.train;
    const N = Math.max(1, train.length);
    const is1D = train.length >= 2 && train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);

    const w = new Float32Array(state.weights);
    let b = state.bias;
    let maxDelta = 0;

    const l1Pen = lambda * alpha;
    const l2Pen = lambda * (1.0 - alpha);

    if (is1D) {
      let m = w[0];
      let z0 = 0;
      for (let i = 0; i < N; i++) {
        z0 += train[i].x1 * train[i].x1;
      }
      z0 = Math.max(1e-6, z0 / N);

      for (let c = 0; c < cycles; c++) {
        // Intercept update
        let sumErr = 0;
        for (let i = 0; i < N; i++) {
          sumErr += train[i].y - m * train[i].x1;
        }
        b = sumErr / N;

        // Coordinate update on slope m
        let rho0 = 0;
        for (let i = 0; i < N; i++) {
          rho0 += train[i].x1 * (train[i].y - b);
        }
        rho0 /= N;

        const nextM = softThreshold(rho0, l1Pen) / (z0 + l2Pen);
        maxDelta = Math.max(maxDelta, Math.abs(nextM - m));
        m = Math.abs(nextM) < 1e-4 ? 0.0 : nextM;
      }

      w[0] = m;
      w[1] = -1;
    } else {
      // Feature normalization norms: z_j = sum_i x_ij^2 / N
      const z = [0, 0];
      for (let i = 0; i < N; i++) {
        z[0] += train[i].x1 * train[i].x1;
        z[1] += train[i].x2 * train[i].x2;
      }
      z[0] = Math.max(1e-6, z[0] / N);
      z[1] = Math.max(1e-6, z[1] / N);

      for (let c = 0; c < cycles; c++) {
        // 1. Update w0 (X1)
        {
          let rho0 = 0;
          for (let i = 0; i < N; i++) {
            const pt = train[i];
            const partialYHat = w[1] * pt.x2 + b;
            rho0 += pt.x1 * (pt.y - partialYHat);
          }
          rho0 /= N;

          const nextW0 = softThreshold(rho0, l1Pen) / (z[0] + l2Pen);
          maxDelta = Math.max(maxDelta, Math.abs(nextW0 - w[0]));
          w[0] = Math.abs(nextW0) < 1e-4 ? 0.0 : nextW0;
        }

        // 2. Update w1 (X2)
        {
          let rho1 = 0;
          for (let i = 0; i < N; i++) {
            const pt = train[i];
            const partialYHat = w[0] * pt.x1 + b;
            rho1 += pt.x2 * (pt.y - partialYHat);
          }
          rho1 /= N;

          const nextW1 = softThreshold(rho1, l1Pen) / (z[1] + l2Pen);
          maxDelta = Math.max(maxDelta, Math.abs(nextW1 - w[1]));
          w[1] = Math.abs(nextW1) < 1e-4 ? 0.0 : nextW1;
        }

        // 3. Update intercept b
        {
          let sumErr = 0;
          for (let i = 0; i < N; i++) {
            sumErr += train[i].y - (w[0] * train[i].x1 + w[1] * train[i].x2);
          }
          b = sumErr / N;
        }
      }
    }

    const zeros: number[] = [];
    if (w[0] === 0.0) zeros.push(0);
    if (!is1D && w[1] === 0.0) zeros.push(1);

    const nextState: ElasticNetState = {
      weights: w,
      bias: b,
      epoch: state.epoch + 1,
      zeroWeightIndices: zeros,
      maxDelta,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged: maxDelta < 1e-4,
    };
  },

  predict(input: Float32Array, state: ElasticNetState, _params: ElasticNetParams): number {
    if (!state || !state.weights) return 0;
    if (state.weights[1] === -1 || input.length === 1) {
      return state.weights[0] * input[0] + state.bias;
    }
    return state.weights[0] * input[0] + state.weights[1] * input[1] + state.bias;
  },

  loss(state: ElasticNetState, dataset: DatasetContainer, params: ElasticNetParams): TelemetryMetrics {
    const lambda = Number(params.lambdaPenalty ?? 0.2);
    const alpha = Number(params.l1Ratio ?? 0.5);
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
      const l1Part = alpha * (is1D ? Math.abs(state.weights[0]) : Math.abs(state.weights[0]) + Math.abs(state.weights[1]));
      const l2Part = 0.5 * (1.0 - alpha) * (is1D ? state.weights[0] ** 2 : state.weights[0] ** 2 + state.weights[1] ** 2);
      const r2 = sst > 1e-8 ? Math.max(-1.0, 1.0 - sse / sst) : 1.0;
      const acc = (correct / pts.length) * 100;
      return { loss: mse + lambda * (l1Part + l2Part), r2, acc };
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
      gradientNorm: state.maxDelta,
      sparsity: {
        zeroCount: state.zeroWeightIndices.length,
        totalCount: is1D ? 1 : 2,
      },
    };
  },

  getVisualArtifacts(
    state: ElasticNetState,
    _dataset: DatasetContainer,
    params: ElasticNetParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];

    const alpha = Number(params.l1Ratio ?? 0.5);

    // 1. Separating / Regressing Hyperplane
    artifacts.push({
      type: 'hyperplane',
      normal: [state.weights[0], state.weights[1]],
      bias: state.bias,
      color: '#141413',
      style: 'solid',
    });

    // 2. Weight Space (w1, w2) Inset with Morphing Elastic Net Contour
    artifacts.push({
      type: 'weight-constraint',
      constraintType: 'elastic-net',
      radius: Math.max(0.2, l2Norm(state.weights)),
      currentWeight: [state.weights[0], state.weights[1] === -1 ? 0 : state.weights[1]],
      l1Ratio: alpha,
    });

    return artifacts;
  },

  serializeSnapshot(state: ElasticNetState): Float32Array {
    const buf = new Float32Array(5);
    buf[0] = state.weights[0];
    buf[1] = state.weights[1];
    buf[2] = state.bias;
    buf[3] = state.epoch;
    buf[4] = state.maxDelta;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: ElasticNetState): ElasticNetState {
    const w = new Float32Array([buffer[0], buffer[1]]);
    const zeros: number[] = [];
    if (w[0] === 0.0) zeros.push(0);
    if (w[1] === 0.0) zeros.push(1);
    return {
      ...state,
      weights: w,
      bias: buffer[2],
      epoch: Math.round(buffer[3]),
      maxDelta: buffer[4],
      zeroWeightIndices: zeros,
    };
  },

  explain(
    state: ElasticNetState,
    params: ElasticNetParams,
    lastStep?: StepResult<ElasticNetState>
  ): ExplanationPayload {
    const w0 = state?.weights ? state.weights[0].toFixed(3) : '0.000';
    const w1 = state?.weights ? state.weights[1].toFixed(3) : '0.000';
    const alpha = Number(params.l1Ratio ?? 0.5).toFixed(2);
    const lambda = Number(params.lambdaPenalty ?? 0.2).toFixed(3);
    const trainLoss = lastStep?.metrics.trainLoss.toFixed(4) ?? '0.0000';
    const zeroCount = state?.zeroWeightIndices ? state.zeroWeightIndices.length : 0;

    return {
      summary: `Elastic Net balances L1 sparsity (α = ${alpha}) and L2 grouping (1 - α = ${(1 - Number(alpha)).toFixed(2)}), selecting grouped collinear features.`,
      coreAssumption:
        'Features exhibit strong correlations where pure Lasso arbitrarily breaks ties; strictly convex hybrid penalty stabilizes feature selection.',
      whenItFails:
        'Introduces double shrinkage bias requiring post-hoc unshrink scaling or cross-validated tuning over dual hyperparameter grid (λ, α).',
      microLog: [
        `Coordinate update with soft-threshold S_${(Number(lambda) * Number(alpha)).toFixed(3)} and denominator (z_j + ${(Number(lambda) * (1 - Number(alpha))).toFixed(3)})`,
        `Fitted coefficients: w₁ = ${w0}, w₂ = ${w1} (${zeroCount} clamped to exact 0.000)`,
        `Objective value: MSE + Hybrid Penalty = ${trainLoss}`,
      ],
      mathLatex: {
        objective: '\\min_{\\mathbf{w}} \\frac{1}{2N} \\|\\mathbf{y} - X \\mathbf{w}\\|_2^2 + \\lambda \\left( \\alpha \\|\\mathbf{w}\\|_1 + \\frac{1-\\alpha}{2} \\|\\mathbf{w}\\|_2^2 \\right)',
        lossOrUpdate: 'w_j \\leftarrow \\frac{\\mathcal{S}_{\\lambda \\alpha}(\\rho_j)}{z_j + \\lambda (1 - \\alpha)}',
        dualOrConstraint: '\\alpha \\|\\mathbf{w}\\|_1 + (1-\\alpha) \\|\\mathbf{w}\\|_2^2 \\le C \\quad (\\text{Morphing Convex Shape})',
      },
      codeSnippet: {
        sklearn: `from sklearn.linear_model import ElasticNet\n\nmodel = ElasticNet(alpha=${lambda}, l1_ratio=${alpha})\nmodel.fit(X_train, y_train)\nprint(f"Weights: {model.coef_}, Intercept: {model.intercept_}")`,
        pytorch: `import torch\nimport torch.nn as nn\n\n# PyTorch Custom Elastic Net Loss\ndef elastic_net_loss(y_hat, y, weights, l1_ratio=${alpha}, lambda_reg=${lambda}):\n    mse = nn.functional.mse_loss(y_hat, y)\n    l1 = torch.sum(torch.abs(weights))\n    l2 = torch.sum(weights ** 2)\n    return mse + lambda_reg * (l1_ratio * l1 + 0.5 * (1 - l1_ratio) * l2)`,
      },
    };
  },
};

