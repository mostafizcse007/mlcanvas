import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import {
  TreeNode,
  splitNode,
  predictTree,
  flattenTree,
  extractSplits,
  extractLeaves,
  serializeTreeToFloat32,
  deserializeTreeFromFloat32,
  SplitCandidate,
  isRegressionDataset,
  getAllowedFeatures,
} from './treeUtils';

export interface AdaBoostParams {
  nEstimators?: number;
  learningRate?: number;
  stumpDepth?: 1 | 2;
  selectedTree?: number;
}

export interface AdaBoostState {
  stumps: TreeNode[];
  alphas: number[]; // estimator weights alpha_m
  weights: Float32Array; // sample weights w_i of length N
  epoch: number;
  selectedTreeIndex: number;
  lastMisclassifiedIds: number[];
  isRegression?: boolean;
}

function findBestWeightedRegressionStump(
  train: { x1: number; x2: number; y: number }[],
  weights: Float32Array,
  maxDepth: 1 | 2,
  allowedFeatures: (0 | 1)[] = [0, 1]
): TreeNode {
  const n = train.length;
  let idCounter = 1;

  let sumWY = 0, sumW = 0;
  for (let i = 0; i < n; i++) {
    sumWY += weights[i] * train[i].y;
    sumW += weights[i];
  }
  const rootMean = sumW > 0 ? sumWY / sumW : 0;

  let rootVar = 0;
  for (let i = 0; i < n; i++) {
    const diff = train[i].y - rootMean;
    rootVar += weights[i] * diff * diff;
  }

  const root: TreeNode = {
    id: idCounter++,
    depth: 0,
    isLeaf: true,
    samples: n,
    impurity: rootVar,
    value: rootMean,
    region: [-6, 6, -6, 6],
  };

  function findWeightedRegressionSplit(indices: number[]): SplitCandidate | null {
    if (indices.length < 2) return null;
    let nodeW = 0, nodeWY = 0;
    for (const idx of indices) {
      nodeW += weights[idx];
      nodeWY += weights[idx] * train[idx].y;
    }
    if (nodeW <= 1e-7) return null;

    let bestGain = -Infinity;
    let bestSplit: SplitCandidate | null = null;

    for (const fIdx of allowedFeatures) {
      const pairs: { val: number; idx: number }[] = [];
      for (const idx of indices) {
        pairs.push({ val: fIdx === 0 ? train[idx].x1 : train[idx].x2, idx });
      }
      pairs.sort((a, b) => a.val - b.val);

      let leftW = 0, leftWY = 0;
      for (let i = 0; i < pairs.length - 1; i++) {
        const idx = pairs[i].idx;
        leftW += weights[idx];
        leftWY += weights[idx] * train[idx].y;

        if (pairs[i].val === pairs[i + 1].val) continue;

        const rightW = nodeW - leftW;
        const rightWY = nodeWY - leftWY;

        if (leftW < 1e-7 || rightW < 1e-7) continue;

        const leftMean = leftWY / leftW;
        const rightMean = rightWY / rightW;

        const gain = (leftWY * leftWY) / leftW + (rightWY * rightWY) / rightW - (nodeWY * nodeWY) / nodeW;

        if (gain > bestGain) {
          bestGain = gain;
          const thresh = (pairs[i].val + pairs[i + 1].val) / 2;
          const leftIdxs: number[] = [];
          const rightIdxs: number[] = [];
          for (const p of pairs) {
            if (p.val <= thresh) leftIdxs.push(p.idx);
            else rightIdxs.push(p.idx);
          }

          bestSplit = {
            featureIndex: fIdx,
            threshold: thresh,
            gain,
            leftIndices: leftIdxs,
            rightIndices: rightIdxs,
            leftImpurity: 0,
            rightImpurity: 0,
            leftValue: leftMean,
            rightValue: rightMean,
          };
        }
      }
    }
    return bestSplit;
  }

  const split1 = findWeightedRegressionSplit(train.map((_, i) => i));
  if (split1) {
    const { left, right } = splitNode(root, split1, () => idCounter++);
    if (maxDepth === 2) {
      const splitL = findWeightedRegressionSplit(split1.leftIndices);
      if (splitL && splitL.gain > 1e-5) splitNode(left, splitL, () => idCounter++);
      const splitR = findWeightedRegressionSplit(split1.rightIndices);
      if (splitR && splitR.gain > 1e-5) splitNode(right, splitR, () => idCounter++);
    }
  }

  return root;
}

function findBestWeightedStump(
  train: { x1: number; x2: number; y: number }[],
  weights: Float32Array,
  maxDepth: 1 | 2,
  allowedFeatures: (0 | 1)[] = [0, 1]
): TreeNode {
  const n = train.length;
  let idCounter = 1;

  // Weighted root value
  let sumPos = 0, sumNeg = 0;
  for (let i = 0; i < n; i++) {
    if (train[i].y === 1) sumPos += weights[i];
    else sumNeg += weights[i];
  }
  const rootVal = sumPos >= sumNeg ? 1 : -1;

  const root: TreeNode = {
    id: idCounter++,
    depth: 0,
    isLeaf: true,
    samples: n,
    impurity: 1 - (sumPos * sumPos + sumNeg * sumNeg),
    value: rootVal,
    region: [-6, 6, -6, 6],
  };

  function findWeightedSplit(indices: number[]): SplitCandidate | null {
    if (indices.length < 2) return null;
    let totalW = 0, totalWPos = 0, totalWNeg = 0;
    for (const idx of indices) {
      const w = weights[idx];
      totalW += w;
      if (train[idx].y === 1) totalWPos += w;
      else totalWNeg += w;
    }
    if (totalWPos <= 1e-7 || totalWNeg <= 1e-7) return null;

    let bestErr = Infinity;
    let bestSplit: SplitCandidate | null = null;

    for (const fIdx of allowedFeatures) {
      const pairs: { val: number; idx: number }[] = [];
      for (const idx of indices) {
        pairs.push({ val: fIdx === 0 ? train[idx].x1 : train[idx].x2, idx });
      }
      pairs.sort((a, b) => a.val - b.val);

      for (let i = 0; i < pairs.length - 1; i++) {
        if (pairs[i].val === pairs[i + 1].val) continue;
        const thresh = (pairs[i].val + pairs[i + 1].val) / 2;

        let leftWPos = 0, leftWNeg = 0;
        let rightWPos = 0, rightWNeg = 0;
        const leftIdxs: number[] = [];
        const rightIdxs: number[] = [];

        for (const p of pairs) {
          const idx = p.idx;
          const w = weights[idx];
          if (p.val <= thresh) {
            leftIdxs.push(idx);
            if (train[idx].y === 1) leftWPos += w;
            else leftWNeg += w;
          } else {
            rightIdxs.push(idx);
            if (train[idx].y === 1) rightWPos += w;
            else rightWNeg += w;
          }
        }

        if (leftIdxs.length === 0 || rightIdxs.length === 0) continue;

        const leftVal = leftWPos >= leftWNeg ? 1 : -1;
        const rightVal = rightWPos >= rightWNeg ? 1 : -1;

        // Weighted misclassification error
        const leftMis = leftVal === 1 ? leftWNeg : leftWPos;
        const rightMis = rightVal === 1 ? rightWNeg : rightWPos;
        const err = leftMis + rightMis;

        if (err < bestErr) {
          bestErr = err;
          bestSplit = {
            featureIndex: fIdx,
            threshold: thresh,
            gain: 1 - err,
            leftIndices: leftIdxs,
            rightIndices: rightIdxs,
            leftImpurity: leftMis,
            rightImpurity: rightMis,
            leftValue: leftVal,
            rightValue: rightVal,
          };
        }
      }
    }
    return bestSplit;
  }

  const split1 = findWeightedSplit(train.map((_, i) => i));
  if (split1) {
    const { left, right } = splitNode(root, split1, () => idCounter++);
    if (maxDepth === 2) {
      const splitL = findWeightedSplit(split1.leftIndices);
      if (splitL) splitNode(left, splitL, () => idCounter++);
      const splitR = findWeightedSplit(split1.rightIndices);
      if (splitR) splitNode(right, splitR, () => idCounter++);
    }
  }

  return root;
}

export const adaboostPlugin: AlgorithmPlugin<AdaBoostState, AdaBoostParams> = {
  id: 'adaboost',
  name: 'AdaBoost (Adaptive Boosting — SAMME)',
  shortCode: 'AdaBoost',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Sequential boosting reweighting sample distributions to focus exponentially on hard misclassified instances with dynamic point radius scaling.',
  paperReference: 'Freund & Schapire, A Decision-Theoretic Generalization of On-Line Learning, 1997; Zhu et al., SAMME, 2009',

  primaryParams: [
    {
      id: 'nEstimators',
      name: 'Rounds (M)',
      type: 'slider',
      min: 1,
      max: 60,
      step: 1,
      default: 20,
      description: 'Total number of sequential boosting rounds / decision stumps.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.05,
      max: 1.0,
      step: 0.05,
      default: 0.5,
      description: 'Shrinkage parameter scaling the contribution of each weak stump alpha.',
    },
    {
      id: 'stumpDepth',
      name: 'Stump Depth',
      type: 'select',
      options: [
        { label: 'Depth 1 (Single Split Stump)', value: 1 },
        { label: 'Depth 2 (Two-Level Tree)', value: 2 },
      ],
      default: 1,
      description: 'Maximum depth of each base weak classifier.',
    },
  ],

  advancedParams: [
    {
      id: 'selectedTree',
      name: 'Inspect Stump #',
      type: 'slider',
      min: -1,
      max: 60,
      step: 1,
      default: -1,
      description: '-1 for combined ensemble, or 0..M-1 for individual weak stump boundary.',
    },
  ],

  symbolicLinks: [
    {
      id: 'adaboost-alpha',
      symbolLatex: '\\alpha_m = \\frac{1}{2} \\ln \\left(\\frac{1 - \\epsilon_m}{\\epsilon_m}\\right)',
      label: 'Stump Importance Weight',
      description: 'Trust weight assigned to estimator m based on its weighted classification error.',
      highlightTarget: 'weights-inset',
    },
    {
      id: 'adaboost-reweight',
      symbolLatex: 'w_i^{(m+1)} = \\frac{w_i^{(m)} \\exp\\left(-\\alpha_m y_i h_m(\\mathbf{x}_i)\\right)}{Z_m}',
      label: 'Sample Weight Multiplier',
      description: 'Misclassified instances receive an exponential weight boost proportional to alpha.',
      highlightTarget: 'normal-vector',
    },
  ],

  init(dataset: DatasetContainer, params: AdaBoostParams, _seed: number): AdaBoostState {
    const n = dataset.train.length;
    const weights = new Float32Array(n);
    const initialWeight = n > 0 ? 1.0 / n : 1.0;
    for (let i = 0; i < n; i++) weights[i] = initialWeight;
    const isReg = isRegressionDataset(dataset, params);

    return {
      stumps: [],
      alphas: [],
      weights,
      epoch: 0,
      selectedTreeIndex: -1,
      lastMisclassifiedIds: [],
      isRegression: isReg,
    };
  },

  step(state: AdaBoostState, dataset: DatasetContainer, params: AdaBoostParams): StepResult<AdaBoostState> {
    const targetEstimators = Number(params.nEstimators ?? 20);
    const lr = Number(params.learningRate ?? 0.5);
    const stumpDepth = (Number(params.stumpDepth) === 2 ? 2 : 1) as 1 | 2;
    const selectedTree = Number(params.selectedTree ?? -1);
    const isReg = state.isRegression ?? isRegressionDataset(dataset, params);

    let updatedStumps = state.stumps;
    let updatedAlphas = state.alphas;
    let nextWeights = state.weights;
    let misIds = state.lastMisclassifiedIds;

    const n = dataset.train.length;

    if (state.stumps.length < targetEstimators) {
      const allowedFeatures = getAllowedFeatures(dataset, isReg);
      if (isReg) {
        // AdaBoost.R2 for Regression
        const currentStump = state.stumps.length === 0
          ? findBestWeightedRegressionStump(dataset.train, state.weights, stumpDepth, allowedFeatures)
          : null;

        if (state.stumps.length === 0 && currentStump) {
          const preds = dataset.train.map(p => predictTree(currentStump, p.x1, p.x2));
          let maxDiff = 0;
          for (let i = 0; i < n; i++) {
            const d = Math.abs(dataset.train[i].y - preds[i]);
            if (d > maxDiff) maxDiff = d;
          }
          maxDiff = Math.max(1e-4, maxDiff);

          let avgL = 0;
          const losses = new Float32Array(n);
          for (let i = 0; i < n; i++) {
            const relDiff = Math.abs(dataset.train[i].y - preds[i]) / maxDiff;
            losses[i] = relDiff * relDiff;
            avgL += state.weights[i] * losses[i];
          }
          avgL = Math.min(Math.max(avgL, 1e-6), 1 - 1e-6);
          const beta = avgL / (1 - avgL);
          const alpha = lr * Math.log(Math.max(1e-6, 1 / beta));

          nextWeights = new Float32Array(n);
          let zNorm = 0;
          for (let i = 0; i < n; i++) {
            nextWeights[i] = state.weights[i] * Math.pow(Math.max(1e-6, beta), 1 - losses[i]);
            zNorm += nextWeights[i];
          }
          if (zNorm > 1e-8) for (let i = 0; i < n; i++) nextWeights[i] /= zNorm;

          updatedStumps = [currentStump];
          updatedAlphas = [Math.max(0.01, alpha)];
        } else {
          const lastStump = state.stumps[state.stumps.length - 1];
          const preds = dataset.train.map(p => predictTree(lastStump, p.x1, p.x2));
          let maxDiff = 0;
          for (let i = 0; i < n; i++) {
            const d = Math.abs(dataset.train[i].y - preds[i]);
            if (d > maxDiff) maxDiff = d;
          }
          maxDiff = Math.max(1e-4, maxDiff);

          let avgL = 0;
          const losses = new Float32Array(n);
          for (let i = 0; i < n; i++) {
            const relDiff = Math.abs(dataset.train[i].y - preds[i]) / maxDiff;
            losses[i] = relDiff * relDiff;
            avgL += state.weights[i] * losses[i];
          }
          avgL = Math.min(Math.max(avgL, 1e-6), 1 - 1e-6);
          const beta = avgL / (1 - avgL);
          const alpha = lr * Math.log(Math.max(1e-6, 1 / beta));

          nextWeights = new Float32Array(n);
          let zNorm = 0;
          for (let i = 0; i < n; i++) {
            nextWeights[i] = state.weights[i] * Math.pow(Math.max(1e-6, beta), 1 - losses[i]);
            zNorm += nextWeights[i];
          }
          if (zNorm > 1e-8) for (let i = 0; i < n; i++) nextWeights[i] /= zNorm;

          const newStump = findBestWeightedRegressionStump(dataset.train, nextWeights, stumpDepth, allowedFeatures);
          updatedStumps = [...state.stumps, newStump];
          updatedAlphas = [...state.alphas, Math.max(0.01, alpha)];
        }
      } else {
        // Classification SAMME
        if (state.stumps.length === 0) {
          const stump1 = findBestWeightedStump(dataset.train, state.weights, stumpDepth);
          let err1 = 0;
          const currentMisIds: number[] = [];
          for (let i = 0; i < n; i++) {
            const pred = predictTree(stump1, dataset.train[i].x1, dataset.train[i].x2);
            if (pred !== dataset.train[i].y) {
              err1 += state.weights[i];
              currentMisIds.push(dataset.train[i].id);
            }
          }
          err1 = Math.min(Math.max(err1, 1e-6), 1 - 1e-6);
          const alpha1 = lr * 0.5 * Math.log((1 - err1) / err1);

          updatedStumps = [stump1];
          updatedAlphas = [alpha1];
          misIds = currentMisIds;
        } else {
          const lastStump = state.stumps[state.stumps.length - 1];
          let err = 0;
          const misclassified: boolean[] = new Array(n).fill(false);
          const currentMisIds: number[] = [];

          for (let i = 0; i < n; i++) {
            const p = dataset.train[i];
            const pred = predictTree(lastStump, p.x1, p.x2);
            if (pred !== p.y) {
              err += state.weights[i];
              misclassified[i] = true;
              currentMisIds.push(p.id);
            }
          }

          err = Math.min(Math.max(err, 1e-6), 1 - 1e-6);
          const alpha = lr * 0.5 * Math.log((1 - err) / err);

          nextWeights = new Float32Array(n);
          let zNorm = 0;
          for (let i = 0; i < n; i++) {
            const mult = misclassified[i] ? Math.exp(alpha) : Math.exp(-alpha);
            nextWeights[i] = state.weights[i] * mult;
            zNorm += nextWeights[i];
          }
          if (zNorm > 1e-8) {
            for (let i = 0; i < n; i++) nextWeights[i] /= zNorm;
          }

          const newStump = findBestWeightedStump(dataset.train, nextWeights, stumpDepth);
          updatedStumps = [...state.stumps, newStump];
          updatedAlphas = [...state.alphas, alpha];
          misIds = currentMisIds;
        }
      }
    } else if (state.stumps.length > targetEstimators) {
      updatedStumps = state.stumps.slice(0, targetEstimators);
      updatedAlphas = state.alphas.slice(0, targetEstimators);
    }

    const nextEpoch = state.epoch + 1;
    const nextState: AdaBoostState = {
      stumps: updatedStumps,
      alphas: updatedAlphas,
      weights: nextWeights,
      epoch: nextEpoch,
      selectedTreeIndex: selectedTree,
      lastMisclassifiedIds: misIds,
      isRegression: isReg,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
      converged: updatedStumps.length >= targetEstimators,
    };
  },

  predict(input: Float32Array, state: AdaBoostState, params: AdaBoostParams): number {
    if (!state.stumps || state.stumps.length === 0 || state.epoch === 0) return 0;

    const sel = params.selectedTree !== undefined ? Number(params.selectedTree) : state.selectedTreeIndex;
    if (sel >= 0 && sel < state.stumps.length) {
      return predictTree(state.stumps[sel], input[0], input[1]);
    }

    if (state.isRegression) {
      let sumAlpha = 0;
      let score = 0;
      for (let m = 0; m < state.stumps.length; m++) {
        const w = Math.max(0.001, state.alphas[m] ?? 1.0);
        score += w * predictTree(state.stumps[m], input[0], input[1]);
        sumAlpha += w;
      }
      return sumAlpha > 0 ? score / sumAlpha : 0;
    }

    let score = 0;
    for (let m = 0; m < state.stumps.length; m++) {
      const pred = predictTree(state.stumps[m], input[0], input[1]);
      score += state.alphas[m] * pred;
    }
    return score;
  },

  loss(state: AdaBoostState, dataset: DatasetContainer, _params: AdaBoostParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;
    if (train.length === 0) return { trainLoss: 0, testLoss: 0 };

    const isReg = state.isRegression ?? false;

    if (!state.stumps || state.stumps.length === 0) {
      return {
        trainLoss: isReg ? 1.0 : 0.693,
        testLoss: isReg ? 1.0 : 0.693,
        trainAccuracy: isReg ? 0.0 : 50.0,
        testAccuracy: isReg ? 0.0 : 50.0,
        trainR2: isReg ? 0.0 : undefined,
        testR2: isReg ? 0.0 : undefined,
        custom: {
          'Active Stumps': 0,
        },
      };
    }

    if (isReg) {
      let sumAlpha = 0;
      for (let m = 0; m < state.stumps.length; m++) sumAlpha += Math.max(0.001, state.alphas[m] ?? 1.0);

      const computeRegPred = (x1: number, x2: number) => {
        let score = 0;
        for (let m = 0; m < state.stumps.length; m++) {
          const w = Math.max(0.001, state.alphas[m] ?? 1.0);
          score += w * predictTree(state.stumps[m], x1, x2);
        }
        return sumAlpha > 0 ? score / sumAlpha : 0;
      };

      let trMse = 0, trMeanY = 0;
      for (let i = 0; i < train.length; i++) trMeanY += train[i].y;
      trMeanY /= Math.max(1, train.length);
      let trTss = 0;
      for (let i = 0; i < train.length; i++) {
        const yHat = computeRegPred(train[i].x1, train[i].x2);
        const err = train[i].y - yHat;
        trMse += err * err;
        trTss += (train[i].y - trMeanY) * (train[i].y - trMeanY);
      }
      trMse /= Math.max(1, train.length);
      const trR2 = trTss > 1e-7 ? Math.max(-1.0, 1.0 - (trMse * train.length) / trTss) : 0;

      let teMse = 0, teR2 = 0;
      if (test.length > 0) {
        let teMeanY = 0;
        for (let i = 0; i < test.length; i++) teMeanY += test[i].y;
        teMeanY /= test.length;
        let teTss = 0;
        for (let i = 0; i < test.length; i++) {
          const yHat = computeRegPred(test[i].x1, test[i].x2);
          const err = test[i].y - yHat;
          teMse += err * err;
          teTss += (test[i].y - teMeanY) * (test[i].y - teMeanY);
        }
        teMse /= test.length;
        teR2 = teTss > 1e-7 ? Math.max(-1.0, 1.0 - (teMse * test.length) / teTss) : 0;
      }

      return {
        trainLoss: trMse,
        testLoss: teMse,
        trainAccuracy: Math.max(0, Math.min(100, trR2 * 100)),
        testAccuracy: Math.max(0, Math.min(100, teR2 * 100)),
        trainR2: trR2,
        testR2: teR2,
        custom: {
          'Train MSE': Number(trMse.toFixed(4)),
          'Test MSE': Number(teMse.toFixed(4)),
          'Train R²': Number(trR2.toFixed(3)),
          'Test R²': Number(teR2.toFixed(3)),
          'Latest Alpha': Number(state.alphas[state.alphas.length - 1]?.toFixed(4) ?? 0),
          'Active Stumps': state.stumps.length,
        },
      };
    }

    let trCorrect = 0;
    let trExpLoss = 0;
    for (let i = 0; i < train.length; i++) {
      let score = 0;
      for (let m = 0; m < state.stumps.length; m++) {
        score += state.alphas[m] * predictTree(state.stumps[m], train[i].x1, train[i].x2);
      }
      if ((score >= 0 ? 1 : -1) === train[i].y) trCorrect++;
      trExpLoss += Math.exp(-train[i].y * score);
    }
    const trAcc = (trCorrect / train.length) * 100;
    trExpLoss /= train.length;

    let teCorrect = 0, teExpLoss = 0;
    if (test.length > 0) {
      for (let i = 0; i < test.length; i++) {
        let score = 0;
        for (let m = 0; m < state.stumps.length; m++) {
          score += state.alphas[m] * predictTree(state.stumps[m], test[i].x1, test[i].x2);
        }
        if ((score >= 0 ? 1 : -1) === test[i].y) teCorrect++;
        teExpLoss += Math.exp(-test[i].y * score);
      }
      teExpLoss /= test.length;
    }
    const teAcc = test.length > 0 ? (teCorrect / test.length) * 100 : 0;

    return {
      trainLoss: trExpLoss,
      testLoss: teExpLoss,
      trainAccuracy: trAcc,
      testAccuracy: teAcc,
      custom: {
        'Latest Alpha': Number(state.alphas[state.alphas.length - 1]?.toFixed(4) ?? 0),
        'Misclassified Samples': state.lastMisclassifiedIds.length,
        'Active Stumps': state.stumps.length,
      },
    };
  },

  getVisualArtifacts(
    state: AdaBoostState,
    dataset: DatasetContainer,
    params: AdaBoostParams
  ): VisualArtifact[] {
    const n = dataset.train.length;
    const avgW = n > 0 ? 1.0 / n : 1.0;
    const misSet = new Set(state.lastMisclassifiedIds);

    const weightsArtifact = dataset.train.map((p, idx) => {
      const w = state.weights[idx] ?? avgW;
      const normalizedWeight = w / avgW; // 1.0 = average
      return {
        id: p.id,
        weight: w,
        normalizedWeight,
        isMisclassified: misSet.has(p.id),
      };
    });

    if (!state.stumps || state.stumps.length === 0) {
      return [
        {
          type: 'tree-partition',
          root: undefined,
          allNodes: [],
          splits: [],
          leaves: [],
          totalTreesCount: 0,
          selectedTreeIndex: -1,
        },
        {
          type: 'adaboost-weights',
          weights: weightsArtifact,
        },
      ];
    }

    const sel = params.selectedTree !== undefined ? Number(params.selectedTree) : state.selectedTreeIndex;
    const activeStump = (sel >= 0 && sel < state.stumps.length)
      ? state.stumps[sel]
      : state.stumps[state.stumps.length - 1];

    const splits = extractSplits(activeStump);
    const leaves = extractLeaves(activeStump);
    const allNodes = flattenTree(activeStump);

    return [
      {
        type: 'tree-partition',
        root: allNodes[0],
        allNodes,
        splits,
        leaves,
        totalTreesCount: state.stumps.length,
        selectedTreeIndex: sel,
        treesRoots: state.stumps.slice(0, 12),
      },
      {
        type: 'adaboost-weights',
        weights: weightsArtifact,
      },
    ];
  },

  serializeSnapshot(state: AdaBoostState): Float32Array {
    if (state.stumps.length === 0) return new Float32Array(0);
    return serializeTreeToFloat32(state.stumps[state.stumps.length - 1]);
  },

  restoreSnapshot(buffer: Float32Array, state: AdaBoostState): AdaBoostState {
    if (!buffer || buffer.length === 0) {
      return { ...state, stumps: [], alphas: [], epoch: 0 };
    }
    const restored = deserializeTreeFromFloat32(buffer);
    return {
      ...state,
      stumps: [restored],
      alphas: [1.0],
      epoch: 1,
    };
  },

  explain(
    state: AdaBoostState,
    params: AdaBoostParams,
    lastStep?: StepResult<AdaBoostState>
  ): ExplanationPayload {
    const count = state.stumps.length;
    const alpha = state.alphas[state.alphas.length - 1]?.toFixed(3) ?? '0.000';
    const isReg = state.isRegression ?? false;
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const mse = lastStep?.metrics.trainLoss?.toFixed(4) ?? '0.0000';

    return {
      summary: isReg
        ? `AdaBoost.R2 combined M = ${count} regression stumps with current trust weight α = ${alpha}, achieving Train MSE = ${mse}.`
        : `AdaBoost (SAMME) combined M = ${count} decision stumps with current estimator weight α = ${alpha}, reaching ${acc}% classification accuracy.`,
      coreAssumption:
        'Additive loss minimization: sequentially adjusting instance sample weights converts weak base estimators into a highly accurate ensemble.',
      whenItFails:
        'Severe vulnerability to noisy outliers: instances with flipped labels or extreme residuals receive compounding weights, monopolizing subsequent stumps.',
      microLog: isReg
        ? [
            `Round [${count}]: Evaluated weighted regression stump on ${state.weights.length} samples`,
            `Computed relative loss L_i = (e_i / D)^2 and trust weight α = ${alpha}`,
            `Reweighted training distribution w_i ∝ w_i * β^(1 - L_i)`,
          ]
        : [
            `Round [${count}]: Evaluated weighted stump on ${state.weights.length} samples`,
            `Fitted stump with α = ${alpha} (lr = ${params.learningRate ?? 0.5})`,
            `${state.lastMisclassifiedIds.length} misclassified points swollen up to 2.5x radius`,
          ],
      mathLatex: {
        objective: isReg
          ? 'L_i = \\frac{(y_i - \\hat{y}_i)^2}{D^2}, \\quad D = \\max_i |y_i - \\hat{y}_i|'
          : 'L(y, f(\\mathbf{x})) = \\exp(-y f(\\mathbf{x})), \\quad f(\\mathbf{x}) = \\sum_{m=1}^M \\alpha_m h_m(\\mathbf{x})',
        lossOrUpdate: isReg
          ? 'w_i^{(m+1)} = \\frac{w_i^{(m)} \\beta_m^{1 - L_i}}{Z_m}, \\quad \\beta_m = \\frac{\\bar{L}_m}{1 - \\bar{L}_m}'
          : 'w_i^{(m+1)} = \\frac{w_i^{(m)} \\exp\\left(-\\alpha_m y_i h_m(\\mathbf{x}_i)\\right)}{\\sum_k w_k^{(m+1)}}',
        dualOrConstraint: isReg
          ? '\\hat{y}(\\mathbf{x}) = \\frac{\\sum_{m=1}^M \\alpha_m h_m(\\mathbf{x})}{\\sum_{m=1}^M \\alpha_m}, \\quad \\alpha_m = \\ln(1 / \\beta_m)'
          : '\\alpha_m = \\frac{1}{2} \\ln \\left(\\frac{1 - \\epsilon_m}{\\epsilon_m}\\right), \\quad \\epsilon_m = \\sum_{i: y_i \\neq h_m(x_i)} w_i^{(m)}',
      },
      codeSnippet: {
        sklearn: isReg
          ? `from sklearn.ensemble import AdaBoostRegressor\nfrom sklearn.tree import DecisionTreeRegressor\n\nmodel = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=${params.stumpDepth ?? 1}), n_estimators=${params.nEstimators ?? 20}, learning_rate=${params.learningRate ?? 0.5}, loss='square')\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`
          : `from sklearn.ensemble import AdaBoostClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n\nmodel = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=${params.stumpDepth ?? 1}), n_estimators=${params.nEstimators ?? 20}, learning_rate=${params.learningRate ?? 0.5}, algorithm='SAMME')\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`,
        pytorch: `# AdaBoost Weighted Voting in PyTorch\nimport torch\n\ndef adaboost_predict(x, stumps, alphas):\n    scores = torch.zeros(x.shape[0])\n    for stump, alpha in zip(stumps, alphas):\n        scores += alpha * stump(x)\n    return torch.sign(scores)`,
      },
    };
  },
};
