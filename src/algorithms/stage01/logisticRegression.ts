import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { sigmoid, softmax } from '../../utils/math';
import { mulberry32 } from '../../utils/prng';
import { binarizeTarget, binarizeTarget01, isMultiClassClassificationDataset } from './crossTask';

export interface LogisticRegressionState {
  // Binary classification weights [w1, w2] and bias
  weights: Float32Array;
  bias: number;
  // Multi-class weights [3][2] and biases [3] when 3 classes are active
  isMultiClass: boolean;
  multiWeights: Float32Array[]; // 3 weight vectors
  multiBiases: Float32Array;
  epoch: number;
  threshold: number;
  gradientNorm: number;
}

export interface LogisticRegressionParams {
  learningRate?: number;
  cPenalty?: number;
  threshold?: number;
}

export const LOGREG_MIN_THRESHOLD = 0.01;
export const LOGREG_MAX_THRESHOLD = 0.99;

/**
 * Clamps the Logistic Regression probability cutoff τ to [0.01, 0.99]
 * so logit ln(τ / (1 - τ)) remains finite and never produces ±Infinity or NaN.
 */
export function clampThreshold(raw: unknown): number {
  const val = Number(raw);
  if (!Number.isFinite(val)) return 0.5;
  return Math.min(LOGREG_MAX_THRESHOLD, Math.max(LOGREG_MIN_THRESHOLD, val));
}

export const logisticRegressionPlugin: AlgorithmPlugin<LogisticRegressionState, LogisticRegressionParams> = {
  id: 'logistic-regression',
  name: 'Logistic Regression (Sigmoid & Softmax)',
  shortCode: 'LogReg',
  stage: 'continuous-2d',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Bernoulli likelihood maximization with calibrated probabilities and live iso-probability contour shifts.',
  paperReference: 'Cox, The regression analysis of binary sequences, JRSS-B 1958; Hastie et al., ESL 2009',

  primaryParams: [
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'select',
      options: [
        { label: '0.01 (Slow)', value: 0.01 },
        { label: '0.05 (Balanced)', value: 0.05 },
        { label: '0.15 (Fast)', value: 0.15 },
        { label: '0.50 (Aggressive)', value: 0.5 },
      ],
      default: 0.05,
      description: 'Gradient ascent step size on log-likelihood surface.',
    },
    {
      id: 'cPenalty',
      name: 'Regularization (C)',
      type: 'select',
      options: [
        { label: '0.01 (Strong L2)', value: 0.01 },
        { label: '0.10', value: 0.1 },
        { label: '1.00 (Standard)', value: 1.0 },
        { label: '10.0 (Minimal Reg)', value: 10.0 },
      ],
      default: 1.0,
      description: 'Inverse regularization strength C = 1/λ. Smaller values specify stronger regularization.',
    },
    {
      id: 'threshold',
      name: 'Decision Threshold (τ)',
      type: 'slider',
      min: 0.01,
      max: 0.99,
      step: 0.01,
      default: 0.5,
      description: 'Probability threshold τ for classifying into positive class y = +1.',
    },
  ],

  advancedParams: [],

  symbolicLinks: [
    {
      id: 'sigmoid',
      symbolLatex: 'P(y=1|\\mathbf{x}) = \\sigma(\\mathbf{w}^T \\mathbf{x} + b)',
      label: 'Sigmoid Probability Function',
      description: 'Monotonic S-shaped curve mapping affine real scores into calibrated probabilities [0, 1].',
      highlightTarget: 'decision-hyperplane',
    },
    {
      id: 'threshold',
      symbolLatex: '\\tau \\in (0, 1)',
      label: 'Decision Cutoff Threshold',
      description: 'Operational risk threshold classifying positive when P(y=1|x) ≥ τ.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: LogisticRegressionParams, seed: number): LogisticRegressionState {
    const rand = mulberry32(seed);
    const hasClass2 = isMultiClassClassificationDataset(dataset);
    const threshold = clampThreshold(params?.threshold ?? 0.5);

    const w0 = 0.5 + (rand() - 0.5) * 0.4;
    const w1 = -0.6 + (rand() - 0.5) * 0.4;
    const weights = new Float32Array([w0, w1]);
    const bias = (rand() - 0.5) * 0.2;

    const multiWeights = [
      new Float32Array([0.5 + (rand() - 0.5) * 0.2, -0.4 + (rand() - 0.5) * 0.2]),
      new Float32Array([-0.4 + (rand() - 0.5) * 0.2, 0.5 + (rand() - 0.5) * 0.2]),
      new Float32Array([-0.2 + (rand() - 0.5) * 0.2, -0.3 + (rand() - 0.5) * 0.2]),
    ];
    const multiBiases = new Float32Array([0.1, -0.1, 0.0]);

    return {
      weights,
      bias,
      isMultiClass: hasClass2,
      multiWeights,
      multiBiases,
      epoch: 0,
      threshold,
      gradientNorm: Math.hypot(w0, w1),
    };
  },

  step(
    state: LogisticRegressionState,
    dataset: DatasetContainer,
    params: LogisticRegressionParams
  ): StepResult<LogisticRegressionState> {
    const lr = Number(params.learningRate ?? 0.05);
    const C = Number(params.cPenalty ?? 1.0);
    const threshold = clampThreshold(params?.threshold ?? state.threshold ?? 0.5);

    const train = dataset.train;
    const N = Math.max(1, train.length);
    const hasClass2 = isMultiClassClassificationDataset(dataset);
    const lambda = 1.0 / (C * N);

    if (!hasClass2) {
      // Binary Logistic Regression with Sigmoid
      const w = new Float32Array(state.weights);
      let b = state.bias;
      let gradNorm = 0;

      for (let pass = 0; pass < 5; pass++) {
        let g0 = 0, g1 = 0, gb = 0;
        for (let i = 0; i < N; i++) {
          const pt = train[i];
          const yTarget = binarizeTarget01(pt.y);
          const z = w[0] * pt.x1 + w[1] * pt.x2 + b;
          const p = sigmoid(z);
          const err = p - yTarget;

          g0 += err * pt.x1;
          g1 += err * pt.x2;
          gb += err;
        }

        g0 = g0 / N + lambda * w[0];
        g1 = g1 / N + lambda * w[1];
        gb = gb / N;

        w[0] -= lr * g0;
        w[1] -= lr * g1;
        b -= lr * gb;

        gradNorm = Math.hypot(g0, g1);
      }

      const nextState: LogisticRegressionState = {
        weights: w,
        bias: b,
        isMultiClass: false,
        multiWeights: state.multiWeights,
        multiBiases: state.multiBiases,
        epoch: state.epoch + 1,
        threshold,
        gradientNorm: gradNorm,
      };

      const metrics = this.loss(nextState, dataset, params);
      return {
        state: nextState,
        epoch: nextState.epoch,
        metrics,
        converged: gradNorm < 1e-4,
      };
    } else {
      // 3-Class Multinomial Softmax
      const mw = state.multiWeights.map((w) => new Float32Array(w));
      const mb = new Float32Array(state.multiBiases);

      for (let pass = 0; pass < 5; pass++) {
        const gw = [new Float32Array(2), new Float32Array(2), new Float32Array(2)];
        const gb = new Float32Array(3);

        for (let i = 0; i < N; i++) {
          const pt = train[i];
          const classIdx = pt.y === 1 ? 0 : pt.y === -1 ? 1 : 2;

          const logits = [
            mw[0][0] * pt.x1 + mw[0][1] * pt.x2 + mb[0],
            mw[1][0] * pt.x1 + mw[1][1] * pt.x2 + mb[1],
            mw[2][0] * pt.x1 + mw[2][1] * pt.x2 + mb[2],
          ];
          const probs = softmax(logits);

          for (let k = 0; k < 3; k++) {
            const indicator = k === classIdx ? 1.0 : 0.0;
            const diff = probs[k] - indicator;
            gw[k][0] += diff * pt.x1;
            gw[k][1] += diff * pt.x2;
            gb[k] += diff;
          }
        }

        for (let k = 0; k < 3; k++) {
          mw[k][0] -= lr * (gw[k][0] / N + lambda * mw[k][0]);
          mw[k][1] -= lr * (gw[k][1] / N + lambda * mw[k][1]);
          mb[k] -= lr * (gb[k] / N);
        }
      }

      const nextState: LogisticRegressionState = {
        weights: mw[0],
        bias: mb[0],
        isMultiClass: true,
        multiWeights: mw,
        multiBiases: mb,
        epoch: state.epoch + 1,
        threshold,
        gradientNorm: Math.hypot(mw[0][0], mw[0][1]),
      };

      const metrics = this.loss(nextState, dataset, params);
      return {
        state: nextState,
        epoch: nextState.epoch,
        metrics,
      };
    }
  },

  predict(input: Float32Array, state: LogisticRegressionState, params: LogisticRegressionParams): number {
    if (!state || (!state.weights && !state.multiWeights)) return 0;
    const tau = clampThreshold(params?.threshold ?? state.threshold);

    if (!state.isMultiClass) {
      if (!state.weights) return 0;
      const z = state.weights[0] * input[0] + state.weights[1] * input[1] + state.bias;
      const prob = sigmoid(z);
      // Return smooth continuous probability centered at 0: (prob - tau) * 2
      // So when prob == tau, predict == 0 (decision boundary)
      return (prob - tau) * 2.0;
    } else {
      if (!state.multiWeights || state.multiWeights.length < 3) return 0;
      const logits = [
        state.multiWeights[0][0] * input[0] + state.multiWeights[0][1] * input[1] + state.multiBiases[0],
        state.multiWeights[1][0] * input[0] + state.multiWeights[1][1] * input[1] + state.multiBiases[1],
        state.multiWeights[2][0] * input[0] + state.multiWeights[2][1] * input[1] + state.multiBiases[2],
      ];
      const probs = softmax(logits);
      // Return the class index mapped to [-1, 1, 2]
      if (probs[0] >= probs[1] && probs[0] >= probs[2]) return 1.0;
      if (probs[1] >= probs[0] && probs[1] >= probs[2]) return -1.0;
      return 2.0;
    }
  },

  loss(state: LogisticRegressionState, dataset: DatasetContainer, params: LogisticRegressionParams): TelemetryMetrics {
    const tau = clampThreshold(params?.threshold ?? state.threshold);

    const calcCrossEntropy = (pts: typeof dataset.train) => {
      if (pts.length === 0) return { loss: 0, acc: 0 };
      let totalLoss = 0;
      let correct = 0;

      for (let i = 0; i < pts.length; i++) {
        const pt = pts[i];
        if (!state.isMultiClass) {
          const z = state.weights[0] * pt.x1 + state.weights[1] * pt.x2 + state.bias;
          const p = Math.max(1e-7, Math.min(1.0 - 1e-7, sigmoid(z)));
          const yTarget = binarizeTarget01(pt.y);
          totalLoss -= yTarget * Math.log(p) + (1 - yTarget) * Math.log(1 - p);

          const pred = p >= tau ? 1 : -1;
          if (pred === binarizeTarget(pt.y)) correct++;
        } else {
          const logits = [
            state.multiWeights[0][0] * pt.x1 + state.multiWeights[0][1] * pt.x2 + state.multiBiases[0],
            state.multiWeights[1][0] * pt.x1 + state.multiWeights[1][1] * pt.x2 + state.multiBiases[1],
            state.multiWeights[2][0] * pt.x1 + state.multiWeights[2][1] * pt.x2 + state.multiBiases[2],
          ];
          const probs = softmax(logits);
          const classIdx = pt.y === 1 ? 0 : pt.y === -1 ? 1 : 2;
          totalLoss -= Math.log(Math.max(1e-7, probs[classIdx]));

          let maxK = 0;
          for (let k = 1; k < 3; k++) {
            if (probs[k] > probs[maxK]) maxK = k;
          }
          const predLabel = maxK === 0 ? 1 : maxK === 1 ? -1 : 2;
          if (predLabel === pt.y) correct++;
        }
      }

      return {
        loss: totalLoss / pts.length,
        acc: (correct / pts.length) * 100,
      };
    };

    const trainRes = calcCrossEntropy(dataset.train);
    const testRes = calcCrossEntropy(dataset.test);

    return {
      trainLoss: trainRes.loss,
      testLoss: testRes.loss,
      trainAccuracy: trainRes.acc,
      testAccuracy: testRes.acc,
      gradientNorm: state.gradientNorm,
      sparsity: { zeroCount: 0, totalCount: 2 },
    };
  },

  getVisualArtifacts(
    state: LogisticRegressionState,
    _dataset: DatasetContainer,
    params: LogisticRegressionParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    const tau = clampThreshold(params?.threshold ?? state.threshold);

    if (!state.isMultiClass) {
      // 1. Primary Decision Hyperplane at P = tau
      // Logit z_tau = ln(tau / (1 - tau))
      // w0 * x1 + w1 * x2 + b = z_tau => w0 * x1 + w1 * x2 + (b - z_tau) = 0
      const logitTau = Math.log(tau / (1.0 - tau));
      artifacts.push({
        type: 'hyperplane',
        normal: [state.weights[0], state.weights[1]],
        bias: state.bias - logitTau,
        color: '#141413',
        style: 'solid',
        label: `P = ${tau.toFixed(2)}`,
      });

      // 2. Iso-probability contour lines at P = 0.25 and P = 0.75
      const logit25 = Math.log(0.25 / 0.75);
      const logit75 = Math.log(0.75 / 0.25);

      artifacts.push({
        type: 'margin-line',
        normal: [state.weights[0], state.weights[1]],
        bias: state.bias,
        offset: logit75,
        label: 'P = 0.75',
      });
      artifacts.push({
        type: 'margin-line',
        normal: [state.weights[0], state.weights[1]],
        bias: state.bias,
        offset: logit25,
        label: 'P = 0.25',
      });
    } else {
      // Multi-class decision boundaries between pairs
      // Class 0 vs Class 1
      artifacts.push({
        type: 'hyperplane',
        normal: [
          state.multiWeights[0][0] - state.multiWeights[1][0],
          state.multiWeights[0][1] - state.multiWeights[1][1],
        ],
        bias: state.multiBiases[0] - state.multiBiases[1],
        color: '#D94E34',
        style: 'solid',
      });
      // Class 1 vs Class 2
      artifacts.push({
        type: 'hyperplane',
        normal: [
          state.multiWeights[1][0] - state.multiWeights[2][0],
          state.multiWeights[1][1] - state.multiWeights[2][1],
        ],
        bias: state.multiBiases[1] - state.multiBiases[2],
        color: '#1E40AF',
        style: 'dashed',
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: LogisticRegressionState): Float32Array {
    const buf = new Float32Array(6);
    buf[0] = state.weights[0];
    buf[1] = state.weights[1];
    buf[2] = state.bias;
    buf[3] = state.epoch;
    buf[4] = state.threshold;
    buf[5] = state.isMultiClass ? 1 : 0;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: LogisticRegressionState): LogisticRegressionState {
    const w = new Float32Array([buffer[0], buffer[1]]);
    return {
      ...state,
      weights: w,
      bias: buffer[2],
      epoch: Math.round(buffer[3]),
      threshold: clampThreshold(buffer[4]),
      isMultiClass: buffer[5] === 1,
    };
  },

  explain(
    state: LogisticRegressionState,
    params: LogisticRegressionParams,
    lastStep?: StepResult<LogisticRegressionState>
  ): ExplanationPayload {
    const tau = clampThreshold(params?.threshold ?? state.threshold).toFixed(2);
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const loss = lastStep?.metrics.trainLoss.toFixed(4) ?? '0.0000';

    return {
      summary: `Logistic Regression models class posteriors via sigmoid/softmax transformations, estimating calibrated conditional odds with ${acc}% accuracy.`,
      coreAssumption:
        'Log-odds (logit) of class probabilities are linear in feature coordinates; class conditional densities belong to exponential family with shared covariance.',
      whenItFails:
        'Fails on non-linearly separable manifolds (XOR, concentric circles) and suffers complete separation where weights diverge to infinity without L2 regularization.',
      microLog: [
        `Logit score evaluation with calibrated threshold τ = ${tau}`,
        `Batch cross-entropy loss = ${loss}, Train Accuracy = ${acc}%`,
        `Gradient descent step applied with L2 penalty factor C = ${params.cPenalty ?? 1.0}`,
      ],
      mathLatex: {
        objective: '\\min_{\\mathbf{w}, b} -\\frac{1}{N} \\sum_{i=1}^N \\left[ y_i \\ln \\sigma(z_i) + (1 - y_i) \\ln(1 - \\sigma(z_i)) \\right] + \\frac{1}{2C} \\|\\mathbf{w}\\|_2^2',
        lossOrUpdate: '\\mathbf{w} \\leftarrow \\mathbf{w} - \\eta \\left( \\frac{1}{N} X^T (\\mathbf{p} - \\mathbf{y}) + \\frac{1}{C} \\mathbf{w} \\right)',
        dualOrConstraint: 'z_\\tau = \\ln\\left(\\frac{\\tau}{1 - \\tau}\\right) \\quad \\Longrightarrow \\quad \\mathbf{w}^T \\mathbf{x} + b = z_\\tau',
      },
      codeSnippet: {
        sklearn: `from sklearn.linear_model import LogisticRegression\n\nmodel = LogisticRegression(C=${params.cPenalty ?? 1.0}, solver='lbfgs')\nmodel.fit(X_train, y_train)\nprobs = model.predict_proba(X_test)\n# Classify with customized operational threshold tau\ny_pred = (probs[:, 1] >= ${tau}).astype(int)`,
        pytorch: `import torch\nimport torch.nn as nn\n\nmodel = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())\ncriterion = nn.BCELoss()\noptimizer = torch.optim.SGD(model.parameters(), lr=0.05, weight_decay=${1.0 / (Number(params.cPenalty ?? 1.0) * 100)})`,
      },
    };
  },
};
