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

export interface XGBoostParams {
  nEstimators?: number;
  learningRate?: number;
  regLambda?: number;
  gammaGain?: number;
  regAlpha?: number;
  maxDepth?: number;
  selectedTree?: number;
}

export interface XGBoostState {
  trees: TreeNode[];
  learningRate: number;
  regLambda: number;
  gammaGain: number;
  epoch: number;
  selectedTreeIndex: number;
  baseScore?: number;
  isRegression?: boolean;
}

function sigmoid(z: number): number {
  if (z > 15) return 1.0;
  if (z < -15) return 0.0;
  return 1.0 / (1.0 + Math.exp(-z));
}

function computeSimilarity(gSum: number, hSum: number, lambda: number): number {
  return (gSum * gSum) / (hSum + lambda);
}

function computeOptimalLeafWeight(gSum: number, hSum: number, lambda: number, alpha: number): number {
  // L1 soft-thresholding and L2 ridge
  if (alpha > 0) {
    if (gSum > alpha) gSum -= alpha;
    else if (gSum < -alpha) gSum += alpha;
    else return 0;
  }
  return -gSum / (hSum + lambda);
}

function buildLevelWiseTree(
  train: { x1: number; x2: number }[],
  g: Float32Array,
  h: Float32Array,
  maxDepth: number,
  lambda: number,
  gamma: number,
  alpha: number,
  idGen: () => number,
  allowedFeatures: (0 | 1)[] = [0, 1]
): TreeNode {
  const n = train.length;
  let gTotal = 0, hTotal = 0;
  for (let i = 0; i < n; i++) {
    gTotal += g[i];
    hTotal += h[i];
  }

  const rootSim = computeSimilarity(gTotal, hTotal, lambda);
  const rootWeight = computeOptimalLeafWeight(gTotal, hTotal, lambda, alpha);

  const root: TreeNode = {
    id: idGen(),
    depth: 0,
    isLeaf: true,
    samples: n,
    impurity: rootSim,
    value: rootWeight,
    similarity: rootSim,
    weight: rootWeight,
    region: [-6, 6, -6, 6],
  };

  // Grow Level-Wise (row-by-row breadth-first)
  let currentLevel: { node: TreeNode; indices: number[] }[] = [{ node: root, indices: train.map((_, i) => i) }];

  for (let d = 0; d < maxDepth; d++) {
    const nextLevel: { node: TreeNode; indices: number[] }[] = [];

    for (const { node, indices } of currentLevel) {
      if (indices.length < 2) continue;

      let bestGain = -Infinity;
      let bestSplit: SplitCandidate | null = null;

      let gNode = 0, hNode = 0;
      for (const idx of indices) {
        gNode += g[idx];
        hNode += h[idx];
      }
      const simNode = computeSimilarity(gNode, hNode, lambda);

      for (const fIdx of allowedFeatures) {
        const pairs: { val: number; idx: number }[] = [];
        for (const idx of indices) {
          pairs.push({ val: fIdx === 0 ? train[idx].x1 : train[idx].x2, idx });
        }
        pairs.sort((a, b) => a.val - b.val);

        let gLeft = 0, hLeft = 0;
        for (let i = 0; i < pairs.length - 1; i++) {
          const idx = pairs[i].idx;
          gLeft += g[idx];
          hLeft += h[idx];

          if (pairs[i].val === pairs[i + 1].val) continue;

          const gRight = gNode - gLeft;
          const hRight = hNode - hLeft;

          const simL = computeSimilarity(gLeft, hLeft, lambda);
          const simR = computeSimilarity(gRight, hRight, lambda);

          // XGBoost Gain Formula
          const gain = 0.5 * (simL + simR - simNode) - gamma;

          if (gain > bestGain && gain > 0) {
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
              leftImpurity: simL,
              rightImpurity: simR,
              leftValue: computeOptimalLeafWeight(gLeft, hLeft, lambda, alpha),
              rightValue: computeOptimalLeafWeight(gRight, hRight, lambda, alpha),
            };
          }
        }
      }

      if (bestSplit) {
        const { left, right } = splitNode(node, bestSplit, idGen);
        left.similarity = bestSplit.leftImpurity;
        left.weight = bestSplit.leftValue;
        left.gain = bestSplit.gain;

        right.similarity = bestSplit.rightImpurity;
        right.weight = bestSplit.rightValue;
        right.gain = bestSplit.gain;

        nextLevel.push({ node: left, indices: bestSplit.leftIndices });
        nextLevel.push({ node: right, indices: bestSplit.rightIndices });
      }
    }

    currentLevel = nextLevel;
    if (currentLevel.length === 0) break;
  }

  return root;
}

export const xgboostPlugin: AlgorithmPlugin<XGBoostState, XGBoostParams> = {
  id: 'xgboost',
  name: 'XGBoost (Extreme Gradient Boosting)',
  shortCode: 'XGBoost',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Second-order Taylor expansion boosting with Newton-Raphson similarity scores, L1/L2 leaf regularization, and level-wise balanced tree growth.',
  paperReference: 'Chen & Guestrin, XGBoost: A Scalable Tree Boosting System, KDD, 2016',

  primaryParams: [
    {
      id: 'nEstimators',
      name: 'Rounds (M)',
      type: 'slider',
      min: 1,
      max: 60,
      step: 1,
      default: 20,
      description: 'Number of sequential boosting trees built via Newton-Raphson optimization.',
    },
    {
      id: 'maxDepth',
      name: 'Max Depth',
      type: 'slider',
      min: 1,
      max: 6,
      step: 1,
      default: 3,
      description: 'Maximum depth limit for level-wise tree growth.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.02,
      max: 1.0,
      step: 0.02,
      default: 0.1,
      description: 'Shrinkage step size scaling leaf weights.',
    },
    {
      id: 'regLambda',
      name: 'L2 Reg (λ)',
      type: 'slider',
      min: 0.0,
      max: 10.0,
      step: 0.5,
      default: 1.0,
      description: 'L2 regularization penalty on leaf weights, preventing extreme similarity spikes.',
    },
  ],

  advancedParams: [
    {
      id: 'gammaGain',
      name: 'Min Split Loss (γ)',
      type: 'slider',
      min: 0.0,
      max: 5.0,
      step: 0.1,
      default: 0.1,
      description: 'Minimum gain threshold required to justify creating a new split partition.',
    },
    {
      id: 'regAlpha',
      name: 'L1 Reg (α)',
      type: 'slider',
      min: 0.0,
      max: 5.0,
      step: 0.2,
      default: 0.0,
      description: 'L1 sparsity penalty on leaf weights.',
    },
    {
      id: 'selectedTree',
      name: 'Inspect Tree #',
      type: 'slider',
      min: -1,
      max: 60,
      step: 1,
      default: -1,
      description: '-1 for combined ensemble, or 0..M-1 for individual tree inspection.',
    },
  ],

  symbolicLinks: [
    {
      id: 'xgb-sim',
      symbolLatex: '\\text{Similarity} = \\frac{(\\sum g_i)^2}{\\sum h_i + \\lambda}',
      label: 'Newton-Raphson Similarity',
      description: 'Exact optimal loss reduction quality metric incorporating 1st order gradients and 2nd order Hessians.',
      highlightTarget: 'weights-inset',
    },
    {
      id: 'xgb-gain',
      symbolLatex: '\\text{Gain} = \\frac{1}{2} \\left[ \\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L + G_R)^2}{H_L + H_R + \\lambda} \\right] - \\gamma',
      label: 'Optimal Split Gain',
      description: 'Net gain after subtracting gamma tree pruning complexity cost.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: XGBoostParams, _seed: number): XGBoostState {
    const lr = Number(params.learningRate ?? 0.1);
    const lambda = Number(params.regLambda ?? 1.0);
    const gamma = Number(params.gammaGain ?? 0.1);
    const isReg = isRegressionDataset(dataset, params);
    const baseScore = isReg && dataset.train.length > 0
      ? dataset.train.reduce((acc, p) => acc + p.y, 0) / dataset.train.length
      : 0;

    return {
      trees: [],
      learningRate: lr,
      regLambda: lambda,
      gammaGain: gamma,
      epoch: 0,
      selectedTreeIndex: -1,
      baseScore,
      isRegression: isReg,
    };
  },

  step(state: XGBoostState, dataset: DatasetContainer, params: XGBoostParams): StepResult<XGBoostState> {
    const targetEstimators = Number(params.nEstimators ?? 20);
    const lr = Number(params.learningRate ?? 0.1);
    const lambda = Number(params.regLambda ?? 1.0);
    const gamma = Number(params.gammaGain ?? 0.1);
    const alpha = Number(params.regAlpha ?? 0.0);
    const maxDepth = Number(params.maxDepth ?? 3);
    const selectedTree = Number(params.selectedTree ?? -1);
    const isReg = state.isRegression ?? isRegressionDataset(dataset, params);
    const baseScore = state.baseScore ?? (isReg && dataset.train.length > 0
      ? dataset.train.reduce((acc, p) => acc + p.y, 0) / dataset.train.length
      : 0);

    let updatedTrees = state.trees;

    if (state.trees.length < targetEstimators) {
      const n = dataset.train.length;
      const g = new Float32Array(n);
      const h = new Float32Array(n);

      for (let i = 0; i < n; i++) {
        const pt = dataset.train[i];
        let F = baseScore;
        for (let t = 0; t < state.trees.length; t++) {
          F += lr * predictTree(state.trees[t], pt.x1, pt.x2);
        }
        if (isReg) {
          // L2 squared loss 0.5 * (F - y)^2: g_i = F - y, h_i = 1.0
          g[i] = F - pt.y;
          h[i] = 1.0;
        } else {
          const p = sigmoid(F);
          const y01 = pt.y === 1 ? 1.0 : 0.0;
          g[i] = p - y01;
          h[i] = Math.max(1e-4, p * (1 - p));
        }
      }

      let idGen = state.trees.length * 1000 + 1;
      const allowedFeatures = getAllowedFeatures(dataset, isReg);
      const newTree = buildLevelWiseTree(dataset.train, g, h, maxDepth, lambda, gamma, alpha, () => idGen++, allowedFeatures);
      updatedTrees = [...state.trees, newTree];
    } else if (state.trees.length > targetEstimators) {
      updatedTrees = state.trees.slice(0, targetEstimators);
    }

    const nextEpoch = state.epoch + 1;
    const nextState: XGBoostState = {
      trees: updatedTrees,
      learningRate: lr,
      regLambda: lambda,
      gammaGain: gamma,
      epoch: nextEpoch,
      selectedTreeIndex: selectedTree,
      baseScore,
      isRegression: isReg,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
      converged: updatedTrees.length >= targetEstimators,
    };
  },

  predict(input: Float32Array, state: XGBoostState, params: XGBoostParams): number {
    if (!state.trees || state.trees.length === 0 || state.epoch === 0) return 0;

    const sel = params.selectedTree !== undefined ? Number(params.selectedTree) : state.selectedTreeIndex;
    if (sel >= 0 && sel < state.trees.length) {
      return predictTree(state.trees[sel], input[0], input[1]);
    }

    let F = state.baseScore ?? 0;
    const lr = state.learningRate;
    for (let t = 0; t < state.trees.length; t++) {
      F += lr * predictTree(state.trees[t], input[0], input[1]);
    }
    return F;
  },

  loss(state: XGBoostState, dataset: DatasetContainer, _params: XGBoostParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;
    if (train.length === 0) return { trainLoss: 0, testLoss: 0 };

    const isReg = state.isRegression ?? false;

    if (!state.trees || state.trees.length === 0) {
      return {
        trainLoss: isReg ? 1.0 : 0.693,
        testLoss: isReg ? 1.0 : 0.693,
        trainAccuracy: isReg ? 0.0 : 50.0,
        testAccuracy: isReg ? 0.0 : 50.0,
        trainR2: isReg ? 0.0 : undefined,
        testR2: isReg ? 0.0 : undefined,
        custom: {
          'L2 Lambda': state.regLambda,
          'Gamma Gain': state.gammaGain,
          'Boosting Rounds': 0,
        },
      };
    }

    const computeF = (x1: number, x2: number) => {
      let F = state.baseScore ?? 0;
      for (let t = 0; t < state.trees.length; t++) {
        F += state.learningRate * predictTree(state.trees[t], x1, x2);
      }
      return F;
    };

    if (isReg) {
      let trMse = 0;
      let trMeanY = 0;
      for (let i = 0; i < train.length; i++) trMeanY += train[i].y;
      trMeanY /= Math.max(1, train.length);
      let trTss = 0;
      for (let i = 0; i < train.length; i++) {
        const F = computeF(train[i].x1, train[i].x2);
        const err = train[i].y - F;
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
          const F = computeF(test[i].x1, test[i].x2);
          const err = test[i].y - F;
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
          'L2 Lambda': state.regLambda,
          'Gamma Gain': state.gammaGain,
          'Boosting Rounds': state.trees.length,
        },
      };
    }

    let trCorrect = 0, trLoss = 0;
    for (let i = 0; i < train.length; i++) {
      const F = computeF(train[i].x1, train[i].x2);
      const p = sigmoid(F);
      const pred = F >= 0 ? 1 : -1;
      if (pred === train[i].y) trCorrect++;
      const y01 = train[i].y === 1 ? 1 : 0;
      trLoss += -(y01 * Math.log(Math.max(1e-7, p)) + (1 - y01) * Math.log(Math.max(1e-7, 1 - p)));
    }
    const trAcc = (trCorrect / train.length) * 100;
    trLoss /= train.length;

    let teCorrect = 0, teLoss = 0;
    if (test.length > 0) {
      for (let i = 0; i < test.length; i++) {
        const F = computeF(test[i].x1, test[i].x2);
        const p = sigmoid(F);
        const pred = F >= 0 ? 1 : -1;
        if (pred === test[i].y) teCorrect++;
        const y01 = test[i].y === 1 ? 1 : 0;
        teLoss += -(y01 * Math.log(Math.max(1e-7, p)) + (1 - y01) * Math.log(Math.max(1e-7, 1 - p)));
      }
      teLoss /= test.length;
    }
    const teAcc = test.length > 0 ? (teCorrect / test.length) * 100 : 0;

    return {
      trainLoss: trLoss,
      testLoss: teLoss,
      trainAccuracy: trAcc,
      testAccuracy: teAcc,
      custom: {
        'L2 Lambda': state.regLambda,
        'Gamma Gain': state.gammaGain,
        'Boosting Rounds': state.trees.length,
      },
    };
  },

  getVisualArtifacts(
    state: XGBoostState,
    _dataset: DatasetContainer,
    params: XGBoostParams
  ): VisualArtifact[] {
    if (!state.trees || state.trees.length === 0) {
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
      ];
    }

    const sel = params.selectedTree !== undefined ? Number(params.selectedTree) : state.selectedTreeIndex;
    const activeTree = (sel >= 0 && sel < state.trees.length)
      ? state.trees[sel]
      : state.trees[state.trees.length - 1];

    const splits = extractSplits(activeTree);
    const leaves = extractLeaves(activeTree);
    const allNodes = flattenTree(activeTree);

    return [
      {
        type: 'tree-partition',
        root: allNodes[0],
        allNodes,
        splits,
        leaves,
        totalTreesCount: state.trees.length,
        selectedTreeIndex: sel,
        treesRoots: state.trees.slice(0, 12),
      },
    ];
  },

  serializeSnapshot(state: XGBoostState): Float32Array {
    if (state.trees.length === 0) return new Float32Array(0);
    return serializeTreeToFloat32(state.trees[state.trees.length - 1]);
  },

  restoreSnapshot(buffer: Float32Array, state: XGBoostState): XGBoostState {
    if (!buffer || buffer.length === 0) {
      return { ...state, trees: [], epoch: 0 };
    }
    const restored = deserializeTreeFromFloat32(buffer);
    return {
      ...state,
      trees: [restored],
      epoch: 1,
    };
  },

  explain(
    state: XGBoostState,
    params: XGBoostParams,
    lastStep?: StepResult<XGBoostState>
  ): ExplanationPayload {
    const count = state.trees.length;
    const lambda = params.regLambda ?? 1.0;
    const gamma = params.gammaGain ?? 0.1;
    const isReg = state.isRegression ?? false;
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const mse = lastStep?.metrics.trainLoss?.toFixed(4) ?? '0.0000';

    return {
      summary: isReg
        ? `XGBoost constructed M = ${count} level-wise regression trees using 2nd-order Taylor expansion with λ = ${lambda} and γ = ${gamma}, achieving Train MSE = ${mse}.`
        : `XGBoost constructed M = ${count} level-wise balanced trees using 2nd-order Taylor expansion with λ = ${lambda} and γ = ${gamma}, reaching ${acc}% classification accuracy.`,
      coreAssumption:
        'Exact Newton-Raphson curvature optimization: second-order gradients (Hessians) provide accurate step-size scaling for robust regularized function expansion.',
      whenItFails:
        'When gamma is set excessively high (all splits pruned) or learning rate is too large without adequate L2 lambda regularization.',
      microLog: isReg
        ? [
            `Round [${count}]: Computed 1st order gradients g_i = F - y_i and 2nd order Hessians h_i = 1.0`,
            `Grew level-wise balanced regression tree with L2 penalty λ = ${lambda} and min gain γ = ${gamma}`,
            `Calculated optimal leaf weights w* = -G / (H + λ)`,
          ]
        : [
            `Round [${count}]: Computed 1st order gradients g_i and 2nd order Hessians h_i`,
            `Grew level-wise balanced tree with L2 penalty λ = ${lambda} and min gain γ = ${gamma}`,
            `Calculated optimal leaf weights w* = -G / (H + λ)`,
          ],
      mathLatex: {
        objective: '\\mathcal{L}^{(t)} \\approx \\sum_{i=1}^n \\left[ g_i f_t(\\mathbf{x}_i) + \\frac{1}{2} h_i f_t^2(\\mathbf{x}_i) \\right] + \\Omega(f_t)',
        lossOrUpdate: 'w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda}, \\quad \\Omega(f) = \\gamma T + \\frac{1}{2} \\lambda \\sum_{j=1}^T w_j^2',
        dualOrConstraint: '\\text{Gain} = \\frac{1}{2} \\left[ \\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L + G_R)^2}{H_L + H_R + \\lambda} \\right] - \\gamma',
      },
      codeSnippet: {
        sklearn: isReg
          ? `import xgboost as xgb\n\nmodel = xgb.XGBRegressor(n_estimators=${params.nEstimators ?? 20}, learning_rate=${params.learningRate ?? 0.1}, reg_lambda=${lambda}, gamma=${gamma}, max_depth=${params.maxDepth ?? 3})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`
          : `import xgboost as xgb\n\nmodel = xgb.XGBClassifier(n_estimators=${params.nEstimators ?? 20}, learning_rate=${params.learningRate ?? 0.1}, reg_lambda=${lambda}, gamma=${gamma}, max_depth=${params.maxDepth ?? 3})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`,
        pytorch: `# XGBoost Vectorized Newton Step\nimport torch\n\ndef xgb_leaf_weight(g, h, reg_lambda=${lambda}):\n    return -torch.sum(g) / (torch.sum(h) + reg_lambda)`,
      },
    };
  },
};
