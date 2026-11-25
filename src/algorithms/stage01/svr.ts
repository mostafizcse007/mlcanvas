import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { mulberry32 } from '../../utils/prng';
import { is1DRegressionDataset } from './linearRegression';

export interface SVRState {
  weights: Float32Array; // Linear weights [w1, w2]
  bias: number;
  epoch: number;
  epsilon: number;
  supportVectorIds: number[];
  curvePoints: [number, number][];
  upperTubePoints: [number, number][];
  lowerTubePoints: [number, number][];
  residuals: { id: number; x1: number; x2: number; y: number; yHat: number; error: number }[];
  is1D?: boolean;
}

export interface SVRParams {
  epsilon?: number;
  cPenalty?: number;
  kernel?: 'linear' | 'rbf';
  learningRate?: number;
}

export const svrPlugin: AlgorithmPlugin<SVRState, SVRParams> = {
  id: 'svr',
  name: 'Support Vector Regression (ε-Insensitive)',
  shortCode: 'SVR',
  stage: 'continuous-2d',
  taskType: 'regression',
  executionMode: 'epoch-loop',
  description: 'Vapnik ε-insensitive loss corridor with zero penalty inside the tube and sparse support vectors.',
  paperReference: 'Drucker, Burges, Kaufman, Smola & Vapnik, Support Vector Regression Machines, NIPS 1996',

  primaryParams: [
    {
      id: 'epsilon',
      name: 'Insensitive Tube (ε)',
      type: 'slider',
      min: 0.05,
      max: 2.0,
      step: 0.05,
      default: 0.4,
      description: 'Radius of the insensitive corridor; points inside this band contribute zero loss.',
    },
    {
      id: 'cPenalty',
      name: 'Slack Penalty (C)',
      type: 'select',
      options: [
        { label: '0.1 (Soft)', value: 0.1 },
        { label: '1.0 (Standard)', value: 1.0 },
        { label: '10.0 (Strict)', value: 10.0 },
      ],
      default: 1.0,
      description: 'Trade-off between flatness of function and tolerance for deviations exceeding ε.',
    },
    {
      id: 'kernel',
      name: 'Kernel',
      type: 'select',
      options: [
        { label: 'Linear', value: 'linear' },
        { label: 'RBF (Non-Linear)', value: 'rbf' },
      ],
      default: 'linear',
      description: 'Linear affine corridor or non-linear RBF tube.',
    },
  ],

  advancedParams: [
    {
      id: 'learningRate',
      name: 'Step Size (η)',
      type: 'slider',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      default: 0.02,
      description: 'Subgradient step size for primal SVR solver.',
    },
  ],

  symbolicLinks: [
    {
      id: 'eps',
      symbolLatex: '|y - f(\\mathbf{x})|_\\epsilon = \\max(0, |y - f(\\mathbf{x})| - \\epsilon)',
      label: 'ε-Insensitive Loss Function',
      description: 'Corridor ignoring errors smaller than ε, giving zero gradient inside the band.',
      highlightTarget: 'margin-street',
    },
  ],

  init(dataset: DatasetContainer, params: SVRParams, seed: number): SVRState {
    const rand = mulberry32(seed);
    const eps = Number(params.epsilon ?? 0.4);
    const train = dataset.train;
    const is1D = is1DRegressionDataset(dataset);
    const w0 = 0.6 + rand() * 0.3;
    const w1 = is1D ? -1 : 0.4 + rand() * 0.2;
    const weights = new Float32Array([w0, w1]);
    const bias = (rand() - 0.5) * 0.3;

    const curvePoints: [number, number][] = [];
    const upperTubePoints: [number, number][] = [];
    const lowerTubePoints: [number, number][] = [];
    if (is1D) {
      const steps = 60;
      for (let i = 0; i <= steps; i++) {
        const x = -6.0 + (i / steps) * 12.0;
        const yHat = w0 * x + bias;
        curvePoints.push([x, yHat]);
        upperTubePoints.push([x, yHat + eps]);
        lowerTubePoints.push([x, yHat - eps]);
      }
    }

    const svIds: number[] = [];
    const residuals: SVRState['residuals'] = [];
    for (let i = 0; i < train.length; i++) {
      const pt = train[i];
      const yHat = is1D ? w0 * pt.x1 + bias : w0 * pt.x1 + w1 * pt.x2 + bias;
      const diff = pt.y - yHat;
      residuals.push({ id: pt.id, x1: pt.x1, x2: pt.x2, y: pt.y, yHat, error: diff });
      if (Math.abs(diff) >= eps * 0.95) {
        svIds.push(pt.id);
      }
    }

    return {
      weights,
      bias,
      epoch: 0,
      epsilon: eps,
      supportVectorIds: svIds,
      curvePoints,
      upperTubePoints,
      lowerTubePoints,
      residuals,
      is1D,
    };
  },

  step(
    state: SVRState,
    dataset: DatasetContainer,
    params: SVRParams
  ): StepResult<SVRState> {
    const eps = Number(params.epsilon ?? 0.4);
    const C = Number(params.cPenalty ?? 1.0);
    const lr = Number(params.learningRate ?? 0.02);

    const train = dataset.train;
    const N = Math.max(1, train.length);
    const is1D = is1DRegressionDataset(dataset);
    const lambda = 1.0 / (C * N);

    const w = new Float32Array(state.weights);
    let b = state.bias;

    if (is1D) {
      let m = w[0];
      let c = b;

      for (let pass = 0; pass < 5; pass++) {
        let subgM = 0;
        let subgC = 0;
        for (let i = 0; i < N; i++) {
          const pt = train[i];
          const yHat = m * pt.x1 + c;
          const diff = pt.y - yHat;
          const absDiff = Math.abs(diff);

          if (absDiff > eps) {
            const signErr = diff > 0 ? 1 : -1;
            subgM += signErr * pt.x1;
            subgC += signErr * 0.5;
          }
        }
        m = (1.0 - lr * lambda) * m + (lr * subgM) / N;
        c += (lr * subgC) / N;
      }

      w[0] = m;
      w[1] = -1;
      b = c;
    } else {
      // Primal subgradient passes on ε-insensitive loss with L2 weight decay
      for (let pass = 0; pass < 5; pass++) {
        let subg0 = 0;
        let subg1 = 0;
        let subgB = 0;
        for (let i = 0; i < N; i++) {
          const pt = train[i];
          const yHat = w[0] * pt.x1 + w[1] * pt.x2 + b;
          const diff = pt.y - yHat;
          const absDiff = Math.abs(diff);

          if (absDiff > eps) {
            const signErr = diff > 0 ? 1 : -1;
            subg0 += signErr * pt.x1;
            subg1 += signErr * pt.x2;
            subgB += signErr * 0.5;
          }
        }
        w[0] = (1.0 - lr * lambda) * w[0] + (lr * subg0) / N;
        w[1] = (1.0 - lr * lambda) * w[1] + (lr * subg1) / N;
        b += (lr * subgB) / N;
      }
    }

    // Support vectors are points strictly outside the tube (|diff| >= eps)
    const svIds: number[] = [];
    const residuals: typeof state.residuals = [];

    for (let i = 0; i < N; i++) {
      const pt = train[i];
      const yHat = is1D ? w[0] * pt.x1 + b : w[0] * pt.x1 + w[1] * pt.x2 + b;
      const diff = pt.y - yHat;
      residuals.push({ id: pt.id, x1: pt.x1, x2: pt.x2, y: pt.y, yHat, error: diff });

      if (Math.abs(diff) >= eps * 0.95) {
        svIds.push(pt.id);
      }
    }

    // Build 1D curve and tube corridor points ONLY on 1D regression datasets
    const curvePoints: [number, number][] = [];
    const upperTubePoints: [number, number][] = [];
    const lowerTubePoints: [number, number][] = [];
    if (is1D) {
      const steps = 60;
      for (let i = 0; i <= steps; i++) {
        const x = -6.0 + (i / steps) * 12.0;
        const yHat = w[0] * x + b;
        curvePoints.push([x, yHat]);
        upperTubePoints.push([x, yHat + eps]);
        lowerTubePoints.push([x, yHat - eps]);
      }
    }

    const nextState: SVRState = {
      weights: w,
      bias: b,
      epoch: state.epoch + 1,
      epsilon: eps,
      supportVectorIds: svIds,
      curvePoints,
      upperTubePoints,
      lowerTubePoints,
      residuals,
      is1D,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
    };
  },

  predict(input: Float32Array, state: SVRState, _params: SVRParams): number {
    if (!state || !state.weights) return 0;
    const is1D = state.is1D ?? (state.weights[1] === -1 || input.length === 1);
    if (is1D) {
      return state.weights[0] * input[0] + state.bias;
    }
    return state.weights[0] * input[0] + state.weights[1] * input[1] + state.bias;
  },

  loss(state: SVRState, dataset: DatasetContainer, _params: SVRParams): TelemetryMetrics {
    const eps = state.epsilon;
    const is1D = state.is1D ?? is1DRegressionDataset(dataset);

    const calcMetrics = (pts: typeof dataset.train) => {
      if (pts.length === 0) return { loss: 0, r2: 1.0, acc: 100.0 };
      let sumEps = 0;
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
        const diff = Math.abs(p.y - yHat);
        if (diff > eps) sumEps += diff - eps;
        sse += (p.y - yHat) ** 2;
        sst += (p.y - meanY) ** 2;

        const predClass = yHat >= 0 ? 1 : -1;
        const actualClass = p.y >= 0 ? 1 : -1;
        if (predClass === actualClass) correct++;
      }
      const loss = sumEps / pts.length;
      const r2 = sst > 1e-8 ? Math.max(-1.0, 1.0 - sse / sst) : 1.0;
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
      marginWidth: eps * 2,
      sparsity: {
        zeroCount: dataset.train.length - state.supportVectorIds.length,
        totalCount: dataset.train.length,
      },
    };
  },

  getVisualArtifacts(
    state: SVRState,
    dataset: DatasetContainer,
    _params: SVRParams
  ): VisualArtifact[] {
    const is1D = state.is1D ?? is1DRegressionDataset(dataset);
    const artifacts: VisualArtifact[] = [];

    if (is1D) {
      // 1. Shaded Epsilon Tube (1D only)
      if (state.upperTubePoints.length > 1 && state.lowerTubePoints.length > 1) {
        artifacts.push({
          type: 'epsilon-tube',
          pointsUpper: state.upperTubePoints,
          pointsLower: state.lowerTubePoints,
          epsilon: state.epsilon,
        });
      }

      // 2. Center Regression Fit (1D only)
      if (state.curvePoints.length > 1) {
        artifacts.push({
          type: 'curve-fit',
          points: state.curvePoints,
          color: '#141413',
          style: 'solid',
        });
      }

      // 3. Support Vector Rings for points outside the tube
      artifacts.push({
        type: 'support-vector-rings',
        pointIds: state.supportVectorIds,
      });

      // 4. Slack residual bars only for violations (1D only)
      const slackResiduals = state.residuals.filter((r) => Math.abs(r.error) >= state.epsilon * 0.95);
      artifacts.push({
        type: 'residual-bars',
        residuals: slackResiduals,
      });
    } else {
      // 2D Surface & 2D Classification: render 2D hyperplane contour (ŷ = 0 / boundary) & SV rings only
      if (Math.hypot(state.weights[0], state.weights[1]) > 1e-5) {
        artifacts.push({
          type: 'hyperplane',
          normal: [state.weights[0], state.weights[1]],
          bias: state.bias,
          color: '#141413',
          style: 'solid',
        });
      }

      artifacts.push({
        type: 'support-vector-rings',
        pointIds: state.supportVectorIds,
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: SVRState): Float32Array {
    const buf = new Float32Array(6);
    buf[0] = state.weights[0];
    buf[1] = state.weights[1];
    buf[2] = state.bias;
    buf[3] = state.epoch;
    buf[4] = state.epsilon;
    buf[5] = state.is1D ? 1 : 0;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: SVRState): SVRState {
    const w = new Float32Array([buffer[0], buffer[1]]);
    const is1D = buffer.length > 5 ? buffer[5] === 1 : (state.is1D ?? Math.abs(w[1] + 1.0) < 1e-4);
    return {
      ...state,
      weights: w,
      bias: buffer[2],
      epoch: Math.round(buffer[3]),
      epsilon: buffer[4],
      is1D,
    };
  },

  explain(
    state: SVRState,
    _params: SVRParams,
    lastStep?: StepResult<SVRState>
  ): ExplanationPayload {
    const svCount = state.supportVectorIds.length;
    const eps = state.epsilon.toFixed(2);
    const loss = lastStep?.metrics.trainLoss.toFixed(4) ?? '0.0000';

    return {
      summary: `Support Vector Regression forms an ε-insensitive corridor (±${eps}) where errors within the tube are ignored, resulting in ${svCount} sparse support vectors.`,
      coreAssumption:
        'Small fluctuations |y - f(x)| ≤ ε are uninformative measurement noise; true structural losses occur only outside the tube.',
      whenItFails:
        'If ε is set too large relative to the signal variance, the tube engulfs all data points and fits a constant mean with zero support vectors.',
      microLog: [
        `Corridor width 2ε = ${(state.epsilon * 2).toFixed(2)}; points inside tube contribute zero slack penalty`,
        `${svCount} observations exceed the tube and act as boundary-defining support vectors`,
        `Average ε-insensitive slack loss = ${loss}`,
      ],
      mathLatex: {
        objective: '\\min_{\\mathbf{w}, b} \\frac{1}{2} \\|\\mathbf{w}\\|_2^2 + C \\sum_{i=1}^N \\max(0, |y_i - \\mathbf{w}^T \\mathbf{x}_i - b| - \\epsilon)',
        lossOrUpdate: 'L_\\epsilon(y, \\hat{y}) = \\begin{cases} 0 & \\text{if } |y - \\hat{y}| \\le \\epsilon \\\\ |y - \\hat{y}| - \\epsilon & \\text{otherwise} \\end{cases}',
        dualOrConstraint: '\\xi_i, \\xi_i^* \\ge 0 \\quad (\\text{Slack variables exceeding upper and lower tube})',
      },
      codeSnippet: {
        sklearn: `from sklearn.svm import SVR\n\nmodel = SVR(epsilon=${eps}, C=1.0, kernel='linear')\nmodel.fit(X_train, y_train)\nprint(f"Number of Support Vectors: {len(model.support_)}")`,
        pytorch: `# Custom PyTorch Epsilon-Insensitive Loss\ndef epsilon_loss(y_hat, y, eps=${eps}):\n    diff = torch.abs(y_hat - y)\n    return torch.mean(torch.clamp(diff - eps, min=0.0))`,
      },
    };
  },
};
