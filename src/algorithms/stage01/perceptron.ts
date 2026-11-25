import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { l2Norm } from '../../utils/math';
import { mulberry32 } from '../../utils/prng';
import { binarizeTarget } from './crossTask';

export interface PerceptronState {
  weights: Float32Array; // [w1, w2, (w_x1x2)]
  bias: number;
  epoch: number;
  mistakesCount: number;
  lastMistakePointId: number | null;
  lastMistakeDelta: [number, number];
  isQuadratic: boolean;
}

export interface PerceptronParams {
  learningRate?: number;
  featureMapping?: 'linear' | 'quadratic';
}

function computePerceptronScore(
  x1: number,
  x2: number,
  weights: Float32Array,
  bias: number,
  isQuadratic: boolean
): number {
  if (isQuadratic && weights.length >= 3) {
    return weights[0] * x1 + weights[1] * x2 + weights[2] * (x1 * x2) + bias;
  }
  return weights[0] * x1 + weights[1] * x2 + bias;
}

/**
 * Evaluates a dataset split using mean 0-1 misclassification rate in [0, 1]
 * so Train Loss and Test Loss strictly share identical scale and units.
 */
function evaluatePerceptronSplit(
  pts: DatasetContainer['train'],
  weights: Float32Array,
  bias: number,
  isQuadratic: boolean
): { mistakes: number; loss: number; accuracy: number } {
  const N = pts.length;
  if (N === 0) {
    return { mistakes: 0, loss: 0, accuracy: 0 };
  }
  let mistakes = 0;
  for (let i = 0; i < N; i++) {
    const p = pts[i];
    const y = binarizeTarget(p.y);
    const score = computePerceptronScore(p.x1, p.x2, weights, bias, isQuadratic);
    if (y * score <= 0) {
      mistakes++;
    }
  }
  return {
    mistakes,
    loss: mistakes / N,
    accuracy: ((N - mistakes) / N) * 100,
  };
}

export const perceptronPlugin: AlgorithmPlugin<PerceptronState, PerceptronParams> = {
  id: 'perceptron',
  name: 'Rosenblatt Perceptron',
  shortCode: 'Perceptron',
  stage: 'continuous-2d',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Historic 1958 mistake-driven linear threshold unit with geometric normal vector addition.',
  paperReference: 'Rosenblatt, The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain, Psych. Rev. 1958',

  primaryParams: [
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'select',
      options: [
        { label: '0.01', value: 0.01 },
        { label: '0.10 (Standard)', value: 0.1 },
        { label: '0.50 (Fast)', value: 0.5 },
        { label: '1.00 (Unit Steps)', value: 1.0 },
      ],
      default: 0.1,
      description: 'Step size multiplier for weight updates on mistake observations.',
    },
    {
      id: 'featureMapping',
      name: 'Basis Feature Map',
      type: 'select',
      options: [
        { label: 'Standard Linear [X₁, X₂]', value: 'linear' },
        { label: 'Quadratic Interaction [X₁·X₂]', value: 'quadratic' },
      ],
      default: 'linear',
      description: 'Expands feature vector with quadratic cross product solving the XOR non-separability deficit.',
    },
  ],

  advancedParams: [],

  symbolicLinks: [
    {
      id: 'update',
      symbolLatex: '\\mathbf{w} \\leftarrow \\mathbf{w} + \\eta y_i \\mathbf{x}_i',
      label: 'Rosenblatt Hebbian Update',
      description: 'Direct vector addition rotating the normal vector toward the misclassified sample.',
      highlightTarget: 'normal-vector',
    },
  ],

  init(dataset: DatasetContainer, params: PerceptronParams, seed: number): PerceptronState {
    const isQuad = params.featureMapping === 'quadratic';
    const rand = mulberry32(seed);
    const weights = new Float32Array(isQuad ? 3 : 2);
    weights[0] = 0.5 + (rand() - 0.5) * 0.4;
    weights[1] = -0.6 + (rand() - 0.5) * 0.4;
    if (isQuad) weights[2] = (rand() - 0.5) * 0.2;
    const bias = (rand() - 0.5) * 0.2;
    const initEval = evaluatePerceptronSplit(dataset.train, weights, bias, isQuad);

    return {
      weights,
      bias,
      epoch: 0,
      mistakesCount: initEval.mistakes,
      lastMistakePointId: null,
      lastMistakeDelta: [0, 0],
      isQuadratic: isQuad,
    };
  },

  step(
    state: PerceptronState,
    dataset: DatasetContainer,
    params: PerceptronParams
  ): StepResult<PerceptronState> {
    const lr = Number(params.learningRate ?? 0.1);
    const isQuad = params.featureMapping === 'quadratic';
    const train = dataset.train;
    const N = train.length;

    const w = new Float32Array(isQuad ? 3 : 2);
    w[0] = state.weights[0];
    w[1] = state.weights[1];
    if (isQuad) w[2] = state.weights[2] ?? 0.0;
    let b = state.bias;

    let mistakesThisEpoch = 0;
    let lastId: number | null = null;
    let lastDelta: [number, number] = [0, 0];

    // Single pass through data: update ONLY on misclassified points
    for (let i = 0; i < N; i++) {
      const pt = train[i];
      const y = binarizeTarget(pt.y);
      const score = computePerceptronScore(pt.x1, pt.x2, w, b, isQuad);

      if (y * score <= 0) {
        // Mistake! Apply update: w <- w + lr * y * x
        w[0] += lr * y * pt.x1;
        w[1] += lr * y * pt.x2;
        if (isQuad) w[2] += lr * y * (pt.x1 * pt.x2);
        b += lr * y * 0.5;

        mistakesThisEpoch++;
        lastId = pt.id;
        lastDelta = [lr * y * pt.x1, lr * y * pt.x2];
      }
    }

    const nextState: PerceptronState = {
      weights: w,
      bias: b,
      epoch: state.epoch + 1,
      mistakesCount: mistakesThisEpoch,
      lastMistakePointId: lastId,
      lastMistakeDelta: lastDelta,
      isQuadratic: isQuad,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged: mistakesThisEpoch === 0,
    };
  },

  predict(input: Float32Array, state: PerceptronState, _params: PerceptronParams): number {
    if (!state || !state.weights) return 0;
    return computePerceptronScore(input[0], input[1], state.weights, state.bias, state.isQuadratic);
  },

  loss(state: PerceptronState, dataset: DatasetContainer, _params: PerceptronParams): TelemetryMetrics {
    const trainStats = evaluatePerceptronSplit(dataset.train, state.weights, state.bias, state.isQuadratic);
    const testStats = evaluatePerceptronSplit(dataset.test, state.weights, state.bias, state.isQuadratic);

    return {
      trainLoss: trainStats.loss,
      testLoss: testStats.loss,
      trainAccuracy: trainStats.accuracy,
      testAccuracy: testStats.accuracy,
      gradientNorm: l2Norm(state.weights),
    };
  },

  getVisualArtifacts(
    state: PerceptronState,
    _dataset: DatasetContainer,
    _params: PerceptronParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    if (!state?.weights || (Math.abs(state.weights[0]) < 1e-6 && Math.abs(state.weights[1]) < 1e-6)) {
      return artifacts;
    }

    // 1. Separating Hyperplane
    artifacts.push({
      type: 'hyperplane',
      normal: [state.weights[0], state.weights[1]],
      bias: state.bias,
      color: '#141413',
      style: 'solid',
    });

    // 2. Vector Addition Artifact: normal vector and mistake update arrow
    artifacts.push({
      type: 'vector-addition',
      origin: [0, 0],
      wVector: [state.weights[0], state.weights[1]],
      deltaVector: state.lastMistakeDelta,
      currentPointId: state.lastMistakePointId ?? undefined,
    });

    return artifacts;
  },

  serializeSnapshot(state: PerceptronState): Float32Array {
    const buf = new Float32Array(7);
    buf[0] = state.weights[0];
    buf[1] = state.weights[1];
    buf[2] = state.weights[2] ?? 0;
    buf[3] = state.bias;
    buf[4] = state.epoch;
    buf[5] = state.mistakesCount;
    buf[6] = state.isQuadratic ? 1 : 0;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: PerceptronState): PerceptronState {
    const isQuad = buffer[6] === 1;
    const w = new Float32Array(isQuad ? 3 : 2);
    w[0] = buffer[0];
    w[1] = buffer[1];
    if (isQuad) w[2] = buffer[2];

    return {
      ...state,
      weights: w,
      bias: buffer[3],
      epoch: Math.round(buffer[4]),
      mistakesCount: Math.round(buffer[5]),
      isQuadratic: isQuad,
    };
  },

  explain(
    state: PerceptronState,
    params: PerceptronParams,
    lastStep?: StepResult<PerceptronState>
  ): ExplanationPayload {
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const mistakes = state.mistakesCount;
    const hasMistakes =
      mistakes > 0 || (lastStep?.metrics.trainLoss !== undefined && lastStep.metrics.trainLoss > 0);
    const lossRatePct = lastStep?.metrics.trainLoss !== undefined
      ? (lastStep.metrics.trainLoss * 100).toFixed(1)
      : '0.0';

    const firstLog =
      state.epoch === 0
        ? `Evaluated initial weights at Epoch 0: ${mistakes} misclassifications (${acc}% accuracy)`
        : `Evaluated dataset: ${mistakes} misclassifications recorded this pass`;

    const secondLog =
      state.lastMistakePointId !== null
        ? `Mistake on Point #${state.lastMistakePointId}: Rotated normal vector w by Δw = (${state.lastMistakeDelta[0].toFixed(2)}, ${state.lastMistakeDelta[1].toFixed(2)})`
        : !hasMistakes && state.epoch > 0
        ? 'Zero mistakes! Separating hyperplane has achieved 100% convergence.'
        : !hasMistakes
        ? 'Initial hyperplane already separates 100% of training samples (0 mistakes at Epoch 0).'
        : `${mistakes} training samples misclassified (${lossRatePct}% 0-1 error); press Step [→] or Play [Space] to train.`;

    return {
      summary: `Rosenblatt's Perceptron iteratively corrects misclassifications through direct Hebbian vector addition, currently at ${acc}% accuracy.`,
      coreAssumption:
        'Linearly separable classes in the chosen feature representation; Novikoff Theorem guarantees convergence in ≤ (R/γ)² steps.',
      whenItFails:
        'Minsky & Papert (1969) proved perceptrons cannot solve XOR or parity problems without non-linear feature maps like X₁·X₂.',
      microLog: [
        firstLog,
        secondLog,
        `Current weight vector: [${state?.weights ? state.weights[0].toFixed(2) : '0.00'}, ${state?.weights ? state.weights[1].toFixed(2) : '0.00'}]`,
      ],
      mathLatex: {
        objective: '\\min_{\\mathbf{w}, b} -\\sum_{i \\in \\mathcal{M}} y_i (\\mathbf{w}^T \\mathbf{x}_i + b) \\quad (\\mathcal{M} = \\text{Misclassified set})',
        lossOrUpdate: '\\text{If } y_i (\\mathbf{w}^T \\mathbf{x}_i + b) \\le 0 \\implies \\mathbf{w} \\leftarrow \\mathbf{w} + \\eta y_i \\mathbf{x}_i, \\quad b \\leftarrow b + \\eta y_i',
        dualOrConstraint: 'k \\le \\left(\\frac{R}{\\gamma}\\right)^2 \\quad (\\text{Novikoff Convergence Bound for Margin } \\gamma)',
      },
      codeSnippet: {
        sklearn: `from sklearn.linear_model import Perceptron\n\nmodel = Perceptron(eta0=${params.learningRate ?? 0.1}, max_iter=1000, tol=1e-3)\nmodel.fit(X_train, y_train)\nprint(f"Weights: {model.coef_}, Intercept: {model.intercept_}")`,
        pytorch: `import torch\n\nw = torch.zeros(2, requires_grad=False)\nb = 0.0\nlr = ${params.learningRate ?? 0.1}\n\nfor x_i, y_i in zip(X, y):\n    if y_i * (torch.dot(w, x_i) + b) <= 0:\n        w += lr * y_i * x_i\n        b += lr * y_i`,
      },
    };
  },
};

