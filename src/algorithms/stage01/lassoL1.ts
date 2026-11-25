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

export interface LassoL1State {
  weights: Float32Array; // [w1, w2]
  bias: number;
  epoch: number;
  zeroWeightIndices: number[]; // Dimensions strictly clamped to exact 0.000
  maxDelta: number;
}

export interface LassoL1Params {
  lambdaPenalty?: number;
  penaltyType?: 'l1' | 'l2';
  cyclesPerStep?: number;
}

export const lassoL1Plugin: AlgorithmPlugin<LassoL1State, LassoL1Params> = {
  id: 'lasso-l1',
  name: 'Lasso L1 Sparse Regression',
  shortCode: 'Lasso L1',
  stage: 'continuous-2d',
  taskType: 'regression',
  executionMode: 'epoch-loop',
  description: 'L1-regularized sparse regression via cyclical coordinate descent with exact soft-thresholding and weight-space inset.',
  paperReference: 'Tibshirani, Regression Shrinkage and Selection via the Lasso, JRSS-B 1996; Friedman et al., 2010',

  primaryParams: [
    {
      id: 'lambdaPenalty',
      name: 'Regularization (λ)',
      type: 'select',
      options: [
        { label: '0.001 (Dense)', value: 0.001 },
        { label: '0.05', value: 0.05 },
        { label: '0.1 (Selective)', value: 0.1 },
        { label: '0.3', value: 0.3 },
        { label: '1.0 (Sparse)', value: 1.0 },
      ],
      default: 0.1,
      description: 'L1 penalty strength. Higher values force non-informative feature weights strictly to exact 0.000.',
    },
    {
      id: 'penaltyType',
      name: 'Constraint Geometry',
      type: 'select',
      options: [
        { label: 'L1 Diamond (Lasso)', value: 'l1' },
        { label: 'L2 Circle (Ridge)', value: 'l2' },
      ],
      default: 'l1',
      description: 'Selects between non-differentiable diamond budget (corner sparsity) vs smooth circular budget.',
    },
  ],

  advancedParams: [
    {
      id: 'cyclesPerStep',
      name: 'Sweeps per step',
      type: 'select',
      options: [
        { label: '1 cycle', value: 1 },
        { label: '5 cycles', value: 5 },
        { label: '10 cycles', value: 10 },
      ],
      default: 5,
      description: 'Full cyclical passes over all active feature dimensions per step.',
    },
  ],

  symbolicLinks: [
    {
      id: 'soft-thresh',
      symbolLatex: '\\mathcal{S}_{N\\lambda}(z)',
      label: 'Soft-Thresholding Operator',
      description: 'Shrinks coefficients toward zero by Nλ; clamps exactly to 0 if correlation |ρ_j| ≤ Nλ.',
      highlightTarget: 'weights-inset',
    },
    {
      id: 'l1-norm',
      symbolLatex: '\\lambda \\|\\mathbf{w}\\|_1',
      label: 'L1 Diamond Constraint',
      description: 'Total absolute coefficient sum, producing diamond isocontours with sharp axis corners.',
      highlightTarget: 'weights-inset',
    },
    {
      id: 'residual',
      symbolLatex: 'y_i - \\hat{y}_i^{(-j)}',
      label: 'Partial Residual',
      description: 'Prediction error isolating the contribution of all features except j.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, _params: LassoL1Params, seed: number): LassoL1State {
    const rand = mulberry32(seed);
    const train = dataset.train;
    const is1D = train.length >= 2 && train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);
    const w0 = 0.7 + rand() * 0.3;
    const w1 = is1D ? -1 : 0.0;
    const bias = (rand() - 0.5) * 0.4;
    const zeroIndices = w1 === 0 ? [1] : [];
    return {
      weights: new Float32Array([w0, w1]),
      bias,
      epoch: 0,
      zeroWeightIndices: zeroIndices,
      maxDelta: 0,
    };
  },

  step(
    state: LassoL1State,
    dataset: DatasetContainer,
    params: LassoL1Params
  ): StepResult<LassoL1State> {
    const lambda = Number(params.lambdaPenalty ?? 0.1);
    const penaltyType = params.penaltyType ?? 'l1';
    const cycles = Number(params.cyclesPerStep ?? 5);

    const train = dataset.train;
    const N = Math.max(1, train.length);
    const is1D = train.length >= 2 && train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);

    const w = new Float32Array(state.weights);
    let b = state.bias;
    let maxParamDelta = 0;

    if (is1D) {
      let m = w[0];
      let z0 = 0;
      for (let i = 0; i < N; i++) {
        z0 += train[i].x1 * train[i].x1;
      }
      if (z0 < 1e-8) z0 = 1e-8;

      for (let c = 0; c < cycles; c++) {
        // Intercept update
        let resSum = 0;
        for (let i = 0; i < N; i++) {
          resSum += train[i].y - m * train[i].x1;
        }
        b = resSum / N;

        // Coordinate update for slope m
        let rho = 0;
        for (let i = 0; i < N; i++) {
          rho += train[i].x1 * (train[i].y - b);
        }

        const oldM = m;
        let newM = 0;
        if (penaltyType === 'l1') {
          newM = softThreshold(rho, N * lambda) / z0;
          if (Math.abs(newM) < 1e-5) newM = 0;
        } else {
          newM = rho / (z0 + N * lambda);
        }

        m = newM;
        maxParamDelta = Math.max(maxParamDelta, Math.abs(m - oldM));
      }

      w[0] = m;
      w[1] = -1;
    } else {
      const d = 2;
      const z = new Float32Array(d);
      for (let i = 0; i < N; i++) {
        z[0] += train[i].x1 * train[i].x1;
        z[1] += train[i].x2 * train[i].x2;
      }
      if (z[0] < 1e-8) z[0] = 1e-8;
      if (z[1] < 1e-8) z[1] = 1e-8;

      for (let c = 0; c < cycles; c++) {
        let residualSum = 0;
        for (let i = 0; i < N; i++) {
          residualSum += train[i].y - (w[0] * train[i].x1 + w[1] * train[i].x2);
        }
        b = residualSum / N;

        for (let j = 0; j < d; j++) {
          let rho_j = 0;
          for (let i = 0; i < N; i++) {
            const pt = train[i];
            const x_ij = j === 0 ? pt.x1 : pt.x2;
            const otherW = j === 0 ? w[1] : w[0];
            const otherX = j === 0 ? pt.x2 : pt.x1;
            const yHatMinusJ = otherW * otherX + b;
            rho_j += x_ij * (pt.y - yHatMinusJ);
          }

          const oldWj = w[j];
          let newWj = 0;

          if (penaltyType === 'l1') {
            newWj = softThreshold(rho_j, N * lambda) / z[j];
            if (Math.abs(newWj) < 1e-5) newWj = 0;
          } else {
            newWj = rho_j / (z[j] + N * lambda);
          }

          w[j] = newWj;
          maxParamDelta = Math.max(maxParamDelta, Math.abs(w[j] - oldWj));
        }
      }
    }

    const norm = is1D ? Math.abs(w[0]) : l2Norm(w);
    if (isNaN(norm) || norm > 1e5) {
      return {
        state,
        epoch: state.epoch,
        metrics: this.loss(state, dataset, params),
        circuitBreakerTripped: true,
      };
    }

    const zeroIndices: number[] = [];
    if (Math.abs(w[0]) === 0) zeroIndices.push(0);
    if (!is1D && Math.abs(w[1]) === 0) zeroIndices.push(1);

    const nextState: LassoL1State = {
      weights: w,
      bias: b,
      epoch: state.epoch + 1,
      zeroWeightIndices: zeroIndices,
      maxDelta: maxParamDelta,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged: maxParamDelta < 1e-5,
    };
  },

  predict(input: Float32Array, state: LassoL1State, _params: LassoL1Params): number {
    if (!state || !state.weights) return 0;
    if (state.weights[1] === -1 || input.length === 1) {
      return state.weights[0] * input[0] + state.bias;
    }
    return state.weights[0] * input[0] + state.weights[1] * input[1] + state.bias;
  },

  loss(
    state: LassoL1State,
    dataset: DatasetContainer,
    params: LassoL1Params
  ): TelemetryMetrics {
    const lambda = Number(params.lambdaPenalty ?? 0.1);
    const train = dataset.train;
    const is1D = train.length >= 2 && train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);
    const N = Math.max(1, train.length);
    const w = state.weights;
    const b = state.bias;

    let trainSse = 0;
    let trainSst = 0;
    let trainSumY = 0;
    for (let i = 0; i < train.length; i++) trainSumY += train[i].y;
    const trainMeanY = trainSumY / N;

    let trainCorrect = 0;
    for (let i = 0; i < train.length; i++) {
      const pt = train[i];
      const pred = is1D ? w[0] * pt.x1 + b : w[0] * pt.x1 + w[1] * pt.x2 + b;
      const diff = pt.y - pred;
      trainSse += diff * diff;
      trainSst += (pt.y - trainMeanY) ** 2;
      if ((pred >= 0 ? 1 : -1) === (pt.y >= 0 ? 1 : -1)) trainCorrect++;
    }

    const l1Sum = is1D ? Math.abs(w[0]) : Math.abs(w[0]) + Math.abs(w[1]);
    const trainLoss = (0.5 * trainSse) / N + lambda * l1Sum;
    const trainAccuracy = (trainCorrect / N) * 100;
    const trainR2 = trainSst > 1e-8 ? Math.max(-1.0, 1.0 - trainSse / trainSst) : 1.0;

    let testLoss = trainLoss;
    let testAccuracy = trainAccuracy;
    let testR2 = trainR2;

    if (dataset.test.length > 0) {
      let testSse = 0;
      let testSst = 0;
      let testSumY = 0;
      for (let i = 0; i < dataset.test.length; i++) testSumY += dataset.test[i].y;
      const testMeanY = testSumY / dataset.test.length;

      let testCorrect = 0;
      for (let i = 0; i < dataset.test.length; i++) {
        const pt = dataset.test[i];
        const pred = is1D ? w[0] * pt.x1 + b : w[0] * pt.x1 + w[1] * pt.x2 + b;
        const diff = pt.y - pred;
        testSse += diff * diff;
        testSst += (pt.y - testMeanY) ** 2;
        if ((pred >= 0 ? 1 : -1) === (pt.y >= 0 ? 1 : -1)) testCorrect++;
      }
      testLoss = (0.5 * testSse) / dataset.test.length + lambda * l1Sum;
      testAccuracy = (testCorrect / dataset.test.length) * 100;
      testR2 = testSst > 1e-8 ? Math.max(-1.0, 1.0 - testSse / testSst) : 1.0;
    }

    return {
      trainLoss,
      testLoss,
      trainAccuracy,
      testAccuracy,
      trainR2,
      testR2,
      sparsity: {
        zeroCount: state.zeroWeightIndices.length,
        totalCount: is1D ? 1 : 2,
      },
      gradientNorm: state.maxDelta,
    };
  },

  getVisualArtifacts(
    state: LassoL1State,
    _dataset: DatasetContainer,
    _params: LassoL1Params
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];

    const w0 = state.weights[0];
    const w1 = state.weights[1];

    // 1. Decision Hyperplane if not all zero
    if (Math.abs(w0) > 1e-6 || Math.abs(w1) > 1e-6) {
      artifacts.push({
        type: 'hyperplane',
        normal: [w0, w1],
        bias: state.bias,
        style: 'solid',
      });
    }

    // 2. Weight Space Constraint Inset
    artifacts.push({
      type: 'weight-constraint',
      constraintType: 'l1-diamond',
      radius: 1.0,
      currentWeight: [w0, w1 === -1 ? 0 : w1],
    });

    return artifacts;
  },

  serializeSnapshot(state: LassoL1State): Float32Array {
    const zeroCount = state.zeroWeightIndices.length;
    const buffer = new Float32Array(6 + zeroCount);
    buffer[0] = state.epoch;
    buffer[1] = state.bias;
    buffer[2] = state.maxDelta;
    buffer[3] = state.weights[0];
    buffer[4] = state.weights[1];
    buffer[5] = zeroCount;
    for (let i = 0; i < zeroCount; i++) {
      buffer[6 + i] = state.zeroWeightIndices[i];
    }
    return buffer;
  },

  restoreSnapshot(buffer: Float32Array, _prevState: LassoL1State): LassoL1State {
    const epoch = buffer[0];
    const bias = buffer[1];
    const maxDelta = buffer[2];
    const weights = new Float32Array([buffer[3], buffer[4]]);
    const zeroCount = buffer[5];
    const zeroWeightIndices: number[] = new Array(zeroCount);
    for (let i = 0; i < zeroCount; i++) {
      zeroWeightIndices[i] = buffer[6 + i];
    }

    return {
      weights,
      bias,
      epoch,
      zeroWeightIndices,
      maxDelta,
    };
  },

  explain(state: LassoL1State, params: LassoL1Params): ExplanationPayload {
    const w0 = state?.weights ? state.weights[0].toFixed(3) : '0.000';
    const w1 = state?.weights ? state.weights[1].toFixed(3) : '0.000';
    const b = state?.bias !== undefined ? state.bias.toFixed(3) : '0.000';
    const lam = params.lambdaPenalty ?? 0.1;
    const zeroCount = state?.zeroWeightIndices ? state.zeroWeightIndices.length : 0;

    return {
      summary: 'Drives redundant feature weights identically to zero via a non-differentiable diamond budget in weight space.',
      coreAssumption: 'A sparse linear combination of features underlies the true data generating process.',
      whenItFails: 'Arbitrarily selects one feature among highly correlated collinear groups rather than distributing weights evenly.',
      microLog: [
        'Forward: Predict target using active coordinates: y_hat = w1 x1 + w2 x2 + b.',
        'Measure: Calculate partial residual r_j isolating the contribution of dimension j.',
        'Update: Pass correlation through soft-threshold operator S_{N lambda}; snap to 0.000 if within dead zone.',
      ],
      mathLatex: {
        objective: `\\min_{\\mathbf{w}, b} \\frac{1}{2N}\\|\\mathbf{y} - (\\mathbf{X}\\mathbf{w} + b)\\|^2_2 + ${lam} \\|\\mathbf{w}\\|_1`,
        lossOrUpdate: `w_j \\leftarrow \\frac{\\mathcal{S}_{N\\lambda}(\\rho_j)}{\\sum x_{ij}^2} \\implies w = [${w0}, ${w1}], \\; b = ${b}`,
        dualOrConstraint: `\\|\\mathbf{w}\\|_1 \\le t \\iff |w_1| + |w_2| \\le t \\quad (\\text{Exact zeros: } ${zeroCount}/2)`,
      },
      codeSnippet: {
        sklearn: `from sklearn.linear_model import Lasso
import numpy as np

# Lasso L1 regression with exact coordinate descent
model = Lasso(alpha=${lam}, max_iter=2000, random_state=42)
model.fit(X_train, y_train)

weights = model.coef_
zero_count = np.sum(np.abs(weights) < 1e-5)
print(f"Weights: {weights}, Zero weights: {zero_count}/{len(weights)}")`,
        pytorch: `import torch
import torch.nn as nn

class LassoRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

def soft_threshold(w, thresh):
    return torch.sign(w) * torch.clamp(torch.abs(w) - thresh, min=0.0)`,
      },
    };
  },
};
