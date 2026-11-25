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

export interface LinearSVMState {
  weights: Float32Array; // [w1, w2]
  bias: number;
  epoch: number;
  supportVectorIndices: number[]; // Point IDs
  marginWidth: number;
  gradientNorm: number;
}

export interface LinearSVMParams {
  learningRate?: number;
  cPenalty?: number;
  batchPasses?: number;
}

export const linearSvmPlugin: AlgorithmPlugin<LinearSVMState, LinearSVMParams> = {
  id: 'linear-svm',
  name: 'Linear Support Vector Machine',
  shortCode: 'Linear SVM',
  stage: 'continuous-2d',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Primal subgradient Pegasos solver with exact margin width computation and support vector detection.',
  paperReference: 'Shalev-Shwartz et al., Pegasos: Primal Estimated sub-GrAdient SOlver for SVM, ICML 2007',

  primaryParams: [
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'select',
      options: [
        { label: '0.001 (Slow)', value: 0.001 },
        { label: '0.01', value: 0.01 },
        { label: '0.03 (Balanced)', value: 0.03 },
        { label: '0.1', value: 0.1 },
        { label: '0.3 (Aggressive)', value: 0.3 },
      ],
      default: 0.03,
      description: 'Step size for stochastic subgradient descent updates.',
    },
    {
      id: 'cPenalty',
      name: 'Regularization (C)',
      type: 'select',
      options: [
        { label: '0.01 (Wide Margin)', value: 0.01 },
        { label: '0.1', value: 0.1 },
        { label: '1.0 (Standard)', value: 1.0 },
        { label: '10.0 (Hard Margin)', value: 10.0 },
        { label: '100.0', value: 100.0 },
      ],
      default: 1.0,
      description: 'Trade-off between maximum margin street width 2/||w|| and hinge slack penalties.',
    },
  ],

  advancedParams: [
    {
      id: 'batchPasses',
      name: 'Sub-epochs per step',
      type: 'select',
      options: [
        { label: '1 pass', value: 1 },
        { label: '5 passes', value: 5 },
        { label: '10 passes', value: 10 },
      ],
      default: 5,
      description: 'Number of stochastic passes over training data per animation frame.',
    },
  ],

  symbolicLinks: [
    {
      id: 'w',
      symbolLatex: '\\mathbf{w}',
      label: 'Weight Vector',
      description: 'Normal vector orthogonal to separating hyperplane, governing boundary orientation.',
      highlightTarget: 'normal-vector',
    },
    {
      id: 'b',
      symbolLatex: 'b',
      label: 'Bias Intercept',
      description: 'Scalar offset translating the decision plane along the normal vector from origin.',
      highlightTarget: 'decision-hyperplane',
    },
    {
      id: 'gamma',
      symbolLatex: '\\gamma = \\frac{2}{\\|\\mathbf{w}\\|}',
      label: 'Margin Street',
      description: 'Geometric distance between positive margin plane w^Tx + b = 1 and negative plane w^Tx + b = -1.',
      highlightTarget: 'margin-street',
    },
    {
      id: 'hinge',
      symbolLatex: '\\max(0, 1 - y_i f(\\mathbf{x}_i))',
      label: 'Hinge Loss',
      description: 'Penalizes margin violations and misclassified instances linearly.',
      highlightTarget: 'support-vectors',
    },
  ],

  init(dataset: DatasetContainer, _params: LinearSVMParams, seed: number): LinearSVMState {
    const rand = mulberry32(seed);
    const w0 = 0.5 + (rand() - 0.5) * 0.4;
    const w1 = -0.6 + (rand() - 0.5) * 0.4;
    const weights = new Float32Array([w0, w1]);
    const bias = (rand() - 0.5) * 0.2;
    const norm = Math.hypot(w0, w1);
    const marginWidth = norm > 1e-6 ? 2.0 / norm : 0;

    const svIndices: number[] = [];
    if (dataset.train.length > 0) {
      for (let i = 0; i < dataset.train.length; i++) {
        const pt = dataset.train[i];
        const y = binarizeTarget(pt.y);
        const score = w0 * pt.x1 + w1 * pt.x2 + bias;
        if (y * score <= 1.05) {
          svIndices.push(pt.id);
        }
      }
    }

    return {
      weights,
      bias,
      epoch: 0,
      supportVectorIndices: svIndices,
      marginWidth,
      gradientNorm: norm,
    };
  },

  step(
    state: LinearSVMState,
    dataset: DatasetContainer,
    params: LinearSVMParams
  ): StepResult<LinearSVMState> {
    const lr = Number(params.learningRate ?? 0.03);
    const C = Number(params.cPenalty ?? 1.0);
    const passes = Number(params.batchPasses ?? 5);

    const train = dataset.train;
    const N = Math.max(1, train.length);
    const lambda = 1.0 / (C * N);

    const w = new Float32Array(state.weights);
    let b = state.bias;

    let gradNormSum = 0;

    for (let p = 0; p < passes; p++) {
      let subg0 = 0;
      let subg1 = 0;
      let subgb = 0;

      for (let i = 0; i < N; i++) {
        const pt = train[i];
        const y = binarizeTarget(pt.y); // Binary projection

        const score = w[0] * pt.x1 + w[1] * pt.x2 + b;
        const marginScore = y * score;

        if (marginScore < 1.0) {
          // Margin violation: subgradient includes -y * x
          subg0 -= y * pt.x1;
          subg1 -= y * pt.x2;
          subgb += y * 0.1;
        }
      }

      // Regularization gradient + accumulated subgradient over N samples
      const g0 = lambda * w[0] + subg0 / N;
      const g1 = lambda * w[1] + subg1 / N;
      w[0] -= lr * g0;
      w[1] -= lr * g1;
      b += (lr * subgb) / N;
      gradNormSum += g0 * g0 + g1 * g1;
    }

    const norm = l2Norm(w);
    if (isNaN(norm) || norm > 1e5) {
      return {
        state,
        epoch: state.epoch,
        metrics: this.loss(state, dataset, params),
        circuitBreakerTripped: true,
      };
    }

    // Determine active support vectors (points on or inside margin)
    const svIndices: number[] = [];
    for (let i = 0; i < train.length; i++) {
      const pt = train[i];
      const y = binarizeTarget(pt.y);
      const score = w[0] * pt.x1 + w[1] * pt.x2 + b;
      if (y * score <= 1.05) {
        svIndices.push(pt.id);
      }
    }

    const marginWidth = norm > 1e-6 ? 2.0 / norm : 0;
    const gradientNorm = Math.sqrt(gradNormSum);

    const nextState: LinearSVMState = {
      weights: w,
      bias: b,
      epoch: state.epoch + 1,
      supportVectorIndices: svIndices,
      marginWidth,
      gradientNorm,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged: gradientNorm < 1e-4,
    };
  },

  predict(input: Float32Array, state: LinearSVMState, _params: LinearSVMParams): number {
    if (!state || !state.weights) return 0;
    return state.weights[0] * input[0] + state.weights[1] * input[1] + state.bias;
  },

  loss(
    state: LinearSVMState,
    dataset: DatasetContainer,
    params: LinearSVMParams
  ): TelemetryMetrics {
    const C = Number(params.cPenalty ?? 1.0);
    const N = Math.max(1, dataset.train.length);
    const lambda = 1.0 / (C * N);

    let trainHinge = 0;
    let trainCorrect = 0;
    const w = state.weights;
    const b = state.bias;

    for (let i = 0; i < dataset.train.length; i++) {
      const pt = dataset.train[i];
      const y = binarizeTarget(pt.y);
      const score = w[0] * pt.x1 + w[1] * pt.x2 + b;
      const marginVal = y * score;
      trainHinge += Math.max(0, 1.0 - marginVal);
      if ((score >= 0 ? 1 : -1) === y) trainCorrect++;
    }

    const regPenalty = 0.5 * (w[0] * w[0] + w[1] * w[1]);
    const trainLoss = lambda * regPenalty + trainHinge / N;
    const trainAccuracy = (trainCorrect / N) * 100;

    let testLoss = trainLoss;
    let testAccuracy = trainAccuracy;

    if (dataset.test.length > 0) {
      let testHinge = 0;
      let testCorrect = 0;
      for (let i = 0; i < dataset.test.length; i++) {
        const pt = dataset.test[i];
        const y = binarizeTarget(pt.y);
        const score = w[0] * pt.x1 + w[1] * pt.x2 + b;
        testHinge += Math.max(0, 1.0 - y * score);
        if ((score >= 0 ? 1 : -1) === y) testCorrect++;
      }
      testLoss = testHinge / dataset.test.length;
      testAccuracy = (testCorrect / dataset.test.length) * 100;
    }

    return {
      trainLoss,
      testLoss,
      trainAccuracy,
      testAccuracy,
      marginWidth: state.marginWidth,
      gradientNorm: state.gradientNorm,
      custom: {
        supportVectorCount: state.supportVectorIndices.length,
      },
    };
  },

  getVisualArtifacts(
    state: LinearSVMState,
    _dataset: DatasetContainer,
    _params: LinearSVMParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    if (!state?.weights) return artifacts;
    const w0 = state.weights[0];
    const w1 = state.weights[1];

    if (Math.abs(w0) < 1e-6 && Math.abs(w1) < 1e-6) {
      return artifacts;
    }

    // 1. Decision Hyperplane: w^T x + b = 0
    artifacts.push({
      type: 'hyperplane',
      normal: [w0, w1],
      bias: state.bias,
      style: 'solid',
    });

    // 2. Margin Plane +1: w^T x + b = 1
    artifacts.push({
      type: 'margin-line',
      normal: [w0, w1],
      bias: state.bias,
      offset: 1,
      label: '+1 Margin',
    });

    // 3. Margin Plane -1: w^T x + b = -1
    artifacts.push({
      type: 'margin-line',
      normal: [w0, w1],
      bias: state.bias,
      offset: -1,
      label: '-1 Margin',
    });

    // 4. Support Vector Target Rings
    if (state.supportVectorIndices.length > 0) {
      artifacts.push({
        type: 'support-vector-rings',
        pointIds: state.supportVectorIndices,
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: LinearSVMState): Float32Array {
    const svCount = state.supportVectorIndices.length;
    const buffer = new Float32Array(7 + svCount);
    buffer[0] = state.epoch;
    buffer[1] = state.bias;
    buffer[2] = state.marginWidth;
    buffer[3] = state.gradientNorm;
    buffer[4] = state.weights[0];
    buffer[5] = state.weights[1];
    buffer[6] = svCount;
    for (let i = 0; i < svCount; i++) {
      buffer[7 + i] = state.supportVectorIndices[i];
    }
    return buffer;
  },

  restoreSnapshot(buffer: Float32Array, _prevState: LinearSVMState): LinearSVMState {
    const epoch = buffer[0];
    const bias = buffer[1];
    const marginWidth = buffer[2];
    const gradientNorm = buffer[3];
    const weights = new Float32Array([buffer[4], buffer[5]]);
    const svCount = buffer[6];
    const supportVectorIndices: number[] = new Array(svCount);
    for (let i = 0; i < svCount; i++) {
      supportVectorIndices[i] = buffer[7 + i];
    }

    return {
      weights,
      bias,
      epoch,
      supportVectorIndices,
      marginWidth,
      gradientNorm,
    };
  },

  explain(state: LinearSVMState, params: LinearSVMParams): ExplanationPayload {
    const w0 = state?.weights ? state.weights[0].toFixed(3) : '0.000';
    const w1 = state?.weights ? state.weights[1].toFixed(3) : '0.000';
    const b = state?.bias !== undefined ? state.bias.toFixed(3) : '0.000';
    const cVal = params.cPenalty ?? 1.0;

    return {
      summary: 'Finds the unique oriented hyperplane that maximizes the geometric margin 2/||w|| separating two classes.',
      coreAssumption: 'Data is linearly separable or soft-separable with bounded margin violations in the feature space.',
      whenItFails: 'Points outside the margin street contribute zero gradient. Moving an outlier that is already outside has no effect on the boundary.',
      microLog: [
        'Forward: Project input point onto weight vector: w^T x + b.',
        'Measure: Calculate signed margin distance y_i (w^T x_i + b); check against 1.0 threshold.',
        'Update: If margin is violated (< 1), add η y_i x_i to w and adjust bias; otherwise apply L2 shrinkage only.',
      ],
      mathLatex: {
        objective: `\\min_{\\mathbf{w}, b} \\frac{1}{2}\\|\\mathbf{w}\\|^2 + ${cVal} \\sum_{i=1}^N \\max(0, 1 - y_i(\\mathbf{w}^T \\mathbf{x}_i + b))`,
        lossOrUpdate: `\\mathbf{w} \\leftarrow (1 - \\eta \\lambda)\\mathbf{w} + \\eta y_i \\mathbf{x}_i \\implies ${w0} x_1 + ${w1} x_2 + ${b} = 0`,
        dualOrConstraint: `\\gamma = \\frac{2}{\\|\\mathbf{w}\\|_2} = ${(state.marginWidth || 0).toFixed(3)}, \\quad \\text{Active SVs: } ${state.supportVectorIndices.length}`,
      },
      codeSnippet: {
        sklearn: `from sklearn.svm import LinearSVC
import numpy as np

# Linear SVM with exact C regularization
clf = LinearSVC(C=${cVal}, loss='hinge', max_iter=2000, random_state=42)
clf.fit(X_train, y_train)

w = clf.coef_[0]
b = clf.intercept_[0]
margin = 2.0 / np.linalg.norm(w)
print(f"Weights: {w}, Bias: {b:.4f}, Margin: {margin:.4f}")`,
        pytorch: `import torch
import torch.nn as nn

class LinearSVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

def pegasos_step(model, x, y, lr=0.03, c_param=1.0):
    with torch.no_grad():
        score = model(x)
        if y * score < 1.0:
            for p in model.parameters():
                p.grad = ...`,
      },
    };
  },
};
