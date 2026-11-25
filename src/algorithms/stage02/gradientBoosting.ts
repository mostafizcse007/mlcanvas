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
  findBestSplitForPoints,
  splitNode,
  predictTree,
  flattenTree,
  extractSplits,
  extractLeaves,
  serializeTreeToFloat32,
  deserializeTreeFromFloat32,
  isRegressionDataset,
  getAllowedFeatures,
} from './treeUtils';

export interface GradientBoostingParams {
  nEstimators?: number;
  learningRate?: number;
  maxDepth?: number;
  selectedTree?: number;
  task?: 'classification' | 'regression';
}

export interface GradientBoostingState {
  trees: TreeNode[];
  baseScore: number; // F_0 log-odds for classification or target mean for regression
  learningRate: number;
  epoch: number;
  selectedTreeIndex: number;
  residuals: Float32Array; // current pseudo-residuals r_i
  isRegression?: boolean;
}

function sigmoid(z: number): number {
  if (z > 15) return 1.0;
  if (z < -15) return 0.0;
  return 1.0 / (1.0 + Math.exp(-z));
}

function buildRegressionTreeOnResiduals(
  train: { x1: number; x2: number }[],
  residuals: Float32Array,
  maxDepth: number,
  idGen: () => number,
  allowedFeatures: (0 | 1)[] = [0, 1]
): TreeNode {
  const n = train.length;
  const pseudoPoints = train.map((pt, i) => ({ x1: pt.x1, x2: pt.x2, y: residuals[i] }));

  let sum = 0;
  for (let i = 0; i < n; i++) sum += residuals[i];
  const mean = n > 0 ? sum / n : 0;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    const d = residuals[i] - mean;
    varSum += d * d;
  }

  const root: TreeNode = {
    id: idGen(),
    depth: 0,
    isLeaf: true,
    samples: n,
    impurity: n > 0 ? varSum / n : 0,
    value: mean,
    region: [-6, 6, -6, 6],
  };

  const queue: { node: TreeNode; indices: number[] }[] = [{ node: root, indices: train.map((_, i) => i) }];

  while (queue.length > 0) {
    const { node, indices } = queue.shift()!;
    if (node.depth >= maxDepth || indices.length < 3 || node.impurity <= 1e-7) continue;

    const split = findBestSplitForPoints(pseudoPoints, indices, 'mse', 3, allowedFeatures);
    if (!split || split.gain <= 1e-6) continue;

    const { left, right } = splitNode(node, split, idGen);
    queue.push({ node: left, indices: split.leftIndices });
    queue.push({ node: right, indices: split.rightIndices });
  }

  return root;
}

export const gradientBoostingPlugin: AlgorithmPlugin<GradientBoostingState, GradientBoostingParams> = {
  id: 'gradient-boosting',
  name: 'Gradient Boosted Decision Trees (Vanilla GBM)',
  shortCode: 'GBDT',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Functional gradient descent fitting shallow regression trees to negative gradient pseudo-residuals with learning rate shrinkage.',
  paperReference: 'Friedman, Greedy Function Approximation: A Gradient Boosting Machine, Annals of Statistics, 2001',

  primaryParams: [
    {
      id: 'nEstimators',
      name: 'Stages (M)',
      type: 'slider',
      min: 1,
      max: 60,
      step: 1,
      default: 20,
      description: 'Number of sequential boosting regression trees fitted to residuals.',
    },
    {
      id: 'maxDepth',
      name: 'Max Depth',
      type: 'slider',
      min: 1,
      max: 6,
      step: 1,
      default: 3,
      description: 'Maximum depth allowed per weak regression tree.',
    },
    {
      id: 'learningRate',
      name: 'Shrinkage (η)',
      type: 'slider',
      min: 0.02,
      max: 1.0,
      step: 0.02,
      default: 0.1,
      description: 'Learning rate shrinkage factor multiplying each trees contribution.',
    },
  ],

  advancedParams: [
    {
      id: 'selectedTree',
      name: 'Inspect Tree #',
      type: 'slider',
      min: -1,
      max: 60,
      step: 1,
      default: -1,
      description: '-1 to inspect combined ensemble log-odds, or 0..M-1 for individual residual tree.',
    },
  ],

  symbolicLinks: [
    {
      id: 'gbm-residual',
      symbolLatex: 'r_{im} = -\\left[\\frac{\\partial L(y_i, F(\\mathbf{x}_i))}{\\partial F(\\mathbf{x}_i)}\\right]_{F_{m-1}} = y_i - \\sigma(F_{m-1}(\\mathbf{x}_i))',
      label: 'Pseudo-Residual Vector',
      description: 'Direction of steepest descent in function space for Bernoulli log-loss.',
      highlightTarget: 'normal-vector',
    },
    {
      id: 'gbm-shrinkage',
      symbolLatex: 'F_m(\\mathbf{x}) = F_{m-1}(\\mathbf{x}) + \\eta \\sum_{j=1}^{J_m} \\gamma_{jm} \\mathbb{I}(\\mathbf{x} \\in R_{jm})',
      label: 'Shrinkage Update',
      description: 'Regularized functional update scaled by shrinkage learning rate eta.',
      highlightTarget: 'weights-inset',
    },
  ],

  init(dataset: DatasetContainer, params: GradientBoostingParams, _seed: number): GradientBoostingState {
    const isRegression = isRegressionDataset(dataset, params);
    const n = dataset.train.length;
    let baseScore = 0;
    const residuals = new Float32Array(n);

    if (isRegression) {
      let sumY = 0;
      for (let i = 0; i < n; i++) sumY += dataset.train[i].y;
      baseScore = n > 0 ? sumY / n : 0;
      for (let i = 0; i < n; i++) {
        residuals[i] = dataset.train[i].y - baseScore;
      }
    } else {
      let countPos = 0;
      for (let i = 0; i < n; i++) {
        if (dataset.train[i].y === 1) countPos++;
      }
      const p0 = Math.max(0.01, Math.min(0.99, countPos / Math.max(1, n)));
      baseScore = Math.log(p0 / (1 - p0));
      for (let i = 0; i < n; i++) {
        const y01 = dataset.train[i].y === 1 ? 1.0 : 0.0;
        residuals[i] = y01 - p0;
      }
    }

    const lr = Number(params.learningRate ?? 0.1);

    return {
      trees: [],
      baseScore,
      learningRate: lr,
      epoch: 0,
      selectedTreeIndex: -1,
      residuals,
      isRegression,
    };
  },

  step(state: GradientBoostingState, dataset: DatasetContainer, params: GradientBoostingParams): StepResult<GradientBoostingState> {
    const isRegression = state.isRegression ?? isRegressionDataset(dataset, params);
    const targetEstimators = Number(params.nEstimators ?? 20);
    const lr = Number(params.learningRate ?? 0.1);
    const maxDepth = Number(params.maxDepth ?? 3);
    const selectedTree = Number(params.selectedTree ?? -1);

    let updatedTrees = state.trees;
    const n = dataset.train.length;
    const currentResiduals = new Float32Array(n);

    if (state.trees.length < targetEstimators) {
      // Compute current predictions F_m-1(x_i) and pseudo-residuals
      for (let i = 0; i < n; i++) {
        const pt = dataset.train[i];
        let F = state.baseScore;
        for (let t = 0; t < state.trees.length; t++) {
          F += lr * predictTree(state.trees[t], pt.x1, pt.x2);
        }
        if (isRegression) {
          currentResiduals[i] = pt.y - F;
        } else {
          const p = sigmoid(F);
          const y01 = pt.y === 1 ? 1.0 : 0.0;
          currentResiduals[i] = y01 - p;
        }
      }

      let idGen = state.trees.length * 1000 + 1;
      const allowedFeatures = getAllowedFeatures(dataset, isRegression);
      const newTree = buildRegressionTreeOnResiduals(dataset.train, currentResiduals, maxDepth, () => idGen++, allowedFeatures);
      updatedTrees = [...state.trees, newTree];
    } else if (state.trees.length > targetEstimators) {
      updatedTrees = state.trees.slice(0, targetEstimators);
      for (let i = 0; i < n; i++) currentResiduals[i] = state.residuals[i] ?? 0;
    } else {
      for (let i = 0; i < n; i++) currentResiduals[i] = state.residuals[i] ?? 0;
    }

    const nextEpoch = state.epoch + 1;
    const nextState: GradientBoostingState = {
      trees: updatedTrees,
      baseScore: state.baseScore,
      learningRate: lr,
      epoch: nextEpoch,
      selectedTreeIndex: selectedTree,
      residuals: currentResiduals,
      isRegression,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
      converged: updatedTrees.length >= targetEstimators,
    };
  },

  predict(input: Float32Array, state: GradientBoostingState, params: GradientBoostingParams): number {
    if (!state.trees || state.trees.length === 0 || state.epoch === 0) {
      return 0;
    }

    const sel = params.selectedTree !== undefined ? Number(params.selectedTree) : state.selectedTreeIndex;
    if (sel >= 0 && sel < state.trees.length) {
      return predictTree(state.trees[sel], input[0], input[1]);
    }

    let F = state.baseScore;
    const lr = state.learningRate;
    for (let t = 0; t < state.trees.length; t++) {
      F += lr * predictTree(state.trees[t], input[0], input[1]);
    }
    return F;
  },

  loss(state: GradientBoostingState, dataset: DatasetContainer, _params: GradientBoostingParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;
    if (train.length === 0) return { trainLoss: 0, testLoss: 0 };

    const isRegression = state.isRegression ?? isRegressionDataset(dataset);

    if (!state.trees || state.trees.length === 0) {
      return {
        trainLoss: isRegression ? 1.0 : 0.693,
        testLoss: isRegression ? 1.0 : 0.693,
        trainAccuracy: isRegression ? undefined : 50.0,
        testAccuracy: isRegression ? undefined : 50.0,
        custom: {
          'Shrinkage Rate': state.learningRate,
          'Base Score (F0)': Number(state.baseScore.toFixed(3)),
          'Fitted Trees': 0,
        },
      };
    }

    const computeF = (x1: number, x2: number) => {
      let F = state.baseScore;
      for (let t = 0; t < state.trees.length; t++) {
        F += state.learningRate * predictTree(state.trees[t], x1, x2);
      }
      return F;
    };

    if (isRegression) {
      let trainSE = 0;
      for (let i = 0; i < train.length; i++) {
        const p = train[i];
        const yHat = computeF(p.x1, p.x2);
        const err = p.y - yHat;
        trainSE += err * err;
      }
      const trainMSE = trainSE / train.length;

      let testMSE = 0;
      if (test.length > 0) {
        let testSE = 0;
        for (let i = 0; i < test.length; i++) {
          const p = test[i];
          const yHat = computeF(p.x1, p.x2);
          const err = p.y - yHat;
          testSE += err * err;
        }
        testMSE = testSE / test.length;
      }

      let meanY = 0;
      for (let i = 0; i < train.length; i++) meanY += train[i].y;
      meanY /= train.length;
      let ssTot = 0;
      for (let i = 0; i < train.length; i++) {
        const d = train[i].y - meanY;
        ssTot += d * d;
      }
      const r2 = ssTot > 1e-7 ? Math.max(-1, 1 - trainSE / ssTot) : 1;

      return {
        trainLoss: trainMSE,
        testLoss: testMSE,
        trainR2: r2,
        testR2: r2,
        custom: {
          'Shrinkage Rate': state.learningRate,
          'Base Score (F0)': Number(state.baseScore.toFixed(3)),
          'Fitted Trees': state.trees.length,
        },
      };
    }

    let trCorrect = 0;
    let trLogLoss = 0;
    for (let i = 0; i < train.length; i++) {
      const F = computeF(train[i].x1, train[i].x2);
      const p = sigmoid(F);
      const pred = F >= 0 ? 1 : -1;
      if (pred === train[i].y) trCorrect++;
      const y01 = train[i].y === 1 ? 1 : 0;
      trLogLoss += -(y01 * Math.log(Math.max(1e-7, p)) + (1 - y01) * Math.log(Math.max(1e-7, 1 - p)));
    }
    const trAcc = (trCorrect / train.length) * 100;
    trLogLoss /= train.length;

    let teCorrect = 0, teLogLoss = 0;
    if (test.length > 0) {
      for (let i = 0; i < test.length; i++) {
        const F = computeF(test[i].x1, test[i].x2);
        const p = sigmoid(F);
        const pred = F >= 0 ? 1 : -1;
        if (pred === test[i].y) teCorrect++;
        const y01 = test[i].y === 1 ? 1 : 0;
        teLogLoss += -(y01 * Math.log(Math.max(1e-7, p)) + (1 - y01) * Math.log(Math.max(1e-7, 1 - p)));
      }
      teLogLoss /= test.length;
    }
    const teAcc = test.length > 0 ? (teCorrect / test.length) * 100 : 0;

    return {
      trainLoss: trLogLoss,
      testLoss: teLogLoss,
      trainAccuracy: trAcc,
      testAccuracy: teAcc,
      custom: {
        'Shrinkage Rate': state.learningRate,
        'Base Log-Odds': Number(state.baseScore.toFixed(3)),
        'Fitted Trees': state.trees.length,
      },
    };
  },

  getVisualArtifacts(
    state: GradientBoostingState,
    dataset: DatasetContainer,
    params: GradientBoostingParams
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

    // Compute residual error bars for visualization
    const isReg = state.isRegression ?? false;
    const residualBars = dataset.train.slice(0, 80).map((pt, idx) => {
      const res = state.residuals[idx] ?? 0;
      const targetVal = isReg ? pt.y : (pt.y === 1 ? 1 : 0);
      return {
        id: pt.id,
        x1: pt.x1,
        x2: pt.x2,
        y: targetVal,
        yHat: targetVal - res,
        error: res,
      };
    });

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
      {
        type: 'residual-bars',
        residuals: residualBars,
      },
    ];
  },

  serializeSnapshot(state: GradientBoostingState): Float32Array {
    if (state.trees.length === 0) return new Float32Array(0);
    return serializeTreeToFloat32(state.trees[state.trees.length - 1]);
  },

  restoreSnapshot(buffer: Float32Array, state: GradientBoostingState): GradientBoostingState {
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
    state: GradientBoostingState,
    params: GradientBoostingParams,
    lastStep?: StepResult<GradientBoostingState>
  ): ExplanationPayload {
    const count = state.trees.length;
    const lr = params.learningRate ?? 0.1;
    const isReg = state.isRegression ?? (params.task === 'regression');
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const mse = lastStep?.metrics.trainLoss?.toFixed(4) ?? '0.0000';
    const r2 = lastStep?.metrics.trainR2 !== undefined ? lastStep.metrics.trainR2.toFixed(3) : '—';

    return {
      summary: isReg
        ? `Gradient Boosting Regressor built M = ${count} regression trees optimizing squared-error residuals with shrinkage η = ${lr}, achieving Train MSE = ${mse} (R² = ${r2}).`
        : `Gradient Boosting built M = ${count} regression trees optimizing pseudo-residuals with shrinkage η = ${lr}, achieving ${acc}% classification accuracy.`,
      coreAssumption:
        'Functional gradient descent: complex non-linear frontiers can be constructed as an additive expansion of shallow regression trees taking small steps along negative loss gradients.',
      whenItFails:
        'Without shrinkage (η = 1.0), the first tree over-corrects drastically on early noise; excessive depth causes rapid overfitting.',
      microLog: isReg
        ? [
            `Stage [${count}]: Evaluated squared-error residuals r_i = y_i - F(x_i)`,
            `Fitted weak regression tree on residuals with shrinkage step η = ${lr}`,
            `Residual errors whittled down across ${count} boosting rounds`,
          ]
        : [
            `Stage [${count}]: Evaluated Bernoulli pseudo-residuals r_i = y_i - σ(F(x_i))`,
            `Fitted weak regression tree on residuals with shrinkage step η = ${lr}`,
            `Pseudo-residual vectors whittled down across ${count} boosting rounds`,
          ],
      mathLatex: {
        objective: 'F_M(\\mathbf{x}) = F_0(\\mathbf{x}) + \\sum_{m=1}^M \\eta \\cdot h_m(\\mathbf{x})',
        lossOrUpdate: isReg
          ? 'r_{im} = -\\left[\\frac{\\partial L(y_i, F(\\mathbf{x}_i))}{\\partial F(\\mathbf{x}_i)}\\right]_{F_{m-1}} = y_i - F_{m-1}(\\mathbf{x}_i)'
          : 'r_{im} = -\\left[\\frac{\\partial L(y_i, F(\\mathbf{x}_i))}{\\partial F(\\mathbf{x}_i)}\\right]_{F_{m-1}} = y_i - \\sigma(F_{m-1}(\\mathbf{x}_i))',
        dualOrConstraint: isReg
          ? '\\gamma_{jm} = \\frac{1}{|R_{jm}|} \\sum_{\\mathbf{x}_i \\in R_{jm}} r_{im}'
          : '\\gamma_{jm} = \\frac{\\sum_{\\mathbf{x}_i \\in R_{jm}} r_{im}}{\\sum_{\\mathbf{x}_i \\in R_{jm}} p_{im}(1 - p_{im})}',
      },
      codeSnippet: {
        sklearn: isReg
          ? `from sklearn.ensemble import GradientBoostingRegressor\n\nmodel = GradientBoostingRegressor(n_estimators=${params.nEstimators ?? 20}, learning_rate=${params.learningRate ?? 0.1}, max_depth=${params.maxDepth ?? 3})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`
          : `from sklearn.ensemble import GradientBoostingClassifier\n\nmodel = GradientBoostingClassifier(n_estimators=${params.nEstimators ?? 20}, learning_rate=${params.learningRate ?? 0.1}, max_depth=${params.maxDepth ?? 3})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`,
        pytorch: `# Additive functional gradient inference in PyTorch\nimport torch\n\ndef gbm_predict(x, base_score, trees, eta=${lr}):\n    f = torch.full((x.shape[0],), base_score)\n    for tree in trees:\n        f += eta * tree(x)\n    return ${isReg ? 'f' : 'torch.where(f >= 0, 1, -1)'}`,
      },
    };
  },
};

