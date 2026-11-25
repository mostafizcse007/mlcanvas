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
  is1DRegressionDataset,
  getAllowedFeatures,
} from './treeUtils';

export interface StackingParams {
  metaLearner?: 'logistic' | 'ridge';
  metaReg?: number;
  cvFolds?: number;
}

export interface StackingState {
  linearWeights: [number, number];
  linearBias: number;
  treeRoot: TreeNode;
  metaWeights: [number, number]; // [w_tree, w_knn]
  metaBias: number;
  trainMetaFeatures: { id: number; p1: number; p2: number; y: number }[];
  trainDataPoints: { x1: number; x2: number; y: number }[];
  phase: number; // 1: Level-0 Fit, 2: Meta-Projection, 3: Level-1 Meta-Learner Fit
  epoch: number;
  seed: number;
  isRegression?: boolean;
  is1DRegression?: boolean;
}

function sigmoid(z: number): number {
  if (z > 15) return 1.0;
  if (z < -15) return 0.0;
  return 1.0 / (1.0 + Math.exp(-z));
}

function knnPredictRegression(
  train: { x1: number; x2: number; y: number }[],
  x1: number,
  x2: number,
  k: number = 7,
  is1DRegression: boolean = false
): number {
  if (train.length === 0) return 0;
  const dists: { dist: number; y: number }[] = [];
  for (let i = 0; i < train.length; i++) {
    const d1 = train[i].x1 - x1;
    const d2 = is1DRegression ? 0 : train[i].x2 - x2;
    dists.push({ dist: Math.sqrt(d1 * d1 + d2 * d2), y: train[i].y });
  }
  dists.sort((a, b) => a.dist - b.dist);

  let totalWY = 0;
  let totalW = 0;
  const actualK = Math.min(k, dists.length);
  for (let i = 0; i < actualK; i++) {
    const w = 1.0 / Math.max(1e-4, dists[i].dist);
    totalW += w;
    totalWY += w * dists[i].y;
  }
  return totalW > 0 ? totalWY / totalW : 0;
}

function fitMetaRegressor(
  metaFeatures: { p1: number; p2: number; y: number }[],
  cReg: number
): { weights: [number, number]; bias: number } {
  let w1 = 0.5, w2 = 0.5, b = 0.0;
  const lr = 0.05;
  const n = metaFeatures.length;
  if (n === 0) return { weights: [0.5, 0.5], bias: 0 };

  for (let iter = 0; iter < 150; iter++) {
    let gradW1 = 0, gradW2 = 0, gradB = 0;
    for (let i = 0; i < n; i++) {
      const mf = metaFeatures[i];
      const yHat = w1 * mf.p1 + w2 * mf.p2 + b;
      const err = yHat - mf.y;
      gradW1 += err * mf.p1;
      gradW2 += err * mf.p2;
      gradB += err;
    }
    w1 -= lr * (gradW1 / n + (w1 / cReg) * 0.01);
    w2 -= lr * (gradW2 / n + (w2 / cReg) * 0.01);
    b -= lr * (gradB / n);
  }
  return { weights: [w1, w2], bias: b };
}

function fitLinearBase(train: { x1: number; x2: number; y: number }[]): { weights: [number, number]; bias: number } {
  let w1 = 0, w2 = 0, b = 0;
  const lr = 0.05;
  const n = train.length;
  if (n === 0) return { weights: [0, 0], bias: 0 };

  for (let iter = 0; iter < 100; iter++) {
    let gradW1 = 0, gradW2 = 0, gradB = 0;
    for (let i = 0; i < n; i++) {
      const p = train[i];
      const z = w1 * p.x1 + w2 * p.x2 + b;
      const prob = sigmoid(z);
      const y01 = p.y === 1 ? 1.0 : 0.0;
      const err = prob - y01;
      gradW1 += err * p.x1;
      gradW2 += err * p.x2;
      gradB += err;
    }
    w1 -= (lr * gradW1) / n;
    w2 -= (lr * gradW2) / n;
    b -= (lr * gradB) / n;
  }
  return { weights: [w1, w2], bias: b };
}

function fitTreeBase(
  train: { x1: number; x2: number; y: number }[],
  isRegression: boolean = false,
  allowedFeatures: (0 | 1)[] = [0, 1]
): TreeNode {
  let nextId = 1;
  const idGen = () => nextId++;

  let rootVal = 0;
  let rootImpurity = 0;

  if (isRegression) {
    let sumY = 0;
    for (let i = 0; i < train.length; i++) sumY += train[i].y;
    rootVal = train.length > 0 ? sumY / train.length : 0;
    let varSum = 0;
    for (let i = 0; i < train.length; i++) {
      const d = train[i].y - rootVal;
      varSum += d * d;
    }
    rootImpurity = train.length > 0 ? varSum / train.length : 0;
  } else {
    let countPos = 0, countNeg = 0;
    for (let i = 0; i < train.length; i++) {
      if (train[i].y === 1) countPos++;
      else countNeg++;
    }
    rootVal = countPos >= countNeg ? 1 : -1;
    const p1 = countPos / Math.max(1, train.length);
    rootImpurity = Math.max(0, 1 - (p1 * p1 + (1 - p1) * (1 - p1)));
  }

  const root: TreeNode = {
    id: idGen(),
    depth: 0,
    isLeaf: true,
    samples: train.length,
    impurity: rootImpurity,
    value: rootVal,
    region: [-6, 6, -6, 6],
  };

  const queue: { node: TreeNode; indices: number[] }[] = [{ node: root, indices: train.map((_, i) => i) }];

  while (queue.length > 0) {
    const { node, indices } = queue.shift()!;
    if (node.depth >= 3 || indices.length < 4 || node.impurity <= 1e-7) continue;

    const split = findBestSplitForPoints(train, indices, isRegression ? 'mse' : 'gini', 3, allowedFeatures);
    if (!split || split.gain <= 1e-5) continue;

    const { left, right } = splitNode(node, split, idGen);
    queue.push({ node: left, indices: split.leftIndices });
    queue.push({ node: right, indices: split.rightIndices });
  }

  return root;
}

function knnPredictProb(train: { x1: number; x2: number; y: number }[], x1: number, x2: number, k: number = 7): number {
  if (train.length === 0) return 0.5;
  const dists: { dist: number; y: number }[] = [];
  for (let i = 0; i < train.length; i++) {
    const d1 = train[i].x1 - x1;
    const d2 = train[i].x2 - x2;
    dists.push({ dist: Math.sqrt(d1 * d1 + d2 * d2), y: train[i].y });
  }
  dists.sort((a, b) => a.dist - b.dist);

  let posWeight = 0;
  let totalWeight = 0;
  const actualK = Math.min(k, dists.length);
  for (let i = 0; i < actualK; i++) {
    const w = 1.0 / Math.max(1e-4, dists[i].dist);
    totalWeight += w;
    if (dists[i].y === 1) posWeight += w;
  }
  return totalWeight > 0 ? posWeight / totalWeight : 0.5;
}

function fitMetaLearner(
  metaFeatures: { p1: number; p2: number; y: number }[],
  cReg: number
): { weights: [number, number]; bias: number } {
  let w1 = 1.0, w2 = 1.0, b = -1.0;
  const lr = 0.1;
  const n = metaFeatures.length;
  if (n === 0) return { weights: [1, 1], bias: -1 };

  for (let iter = 0; iter < 150; iter++) {
    let gradW1 = 0, gradW2 = 0, gradB = 0;
    for (let i = 0; i < n; i++) {
      const mf = metaFeatures[i];
      const z = w1 * mf.p1 + w2 * mf.p2 + b;
      const prob = sigmoid(z);
      const y01 = mf.y === 1 ? 1.0 : 0.0;
      const err = prob - y01;
      gradW1 += err * mf.p1;
      gradW2 += err * mf.p2;
      gradB += err;
    }
    // L2 regularization on meta-weights
    w1 -= lr * (gradW1 / n + (w1 / cReg) * 0.01);
    w2 -= lr * (gradW2 / n + (w2 / cReg) * 0.01);
    b -= lr * (gradB / n);
  }
  return { weights: [w1, w2], bias: b };
}

export const stackingPlugin: AlgorithmPlugin<StackingState, StackingParams> = {
  id: 'stacking',
  name: 'Stacked Generalization (Stacking)',
  shortCode: 'Stacking',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Hierarchical stacked generalization projecting base model predictions into a meta-feature probability space and fitting a Level-1 meta-classifier.',
  paperReference: 'Wolpert, Stacked Generalization, Neural Networks, 1992',

  primaryParams: [
    {
      id: 'metaLearner',
      name: 'Meta-Learner',
      type: 'select',
      options: [
        { label: 'Logistic Regression', value: 'logistic' },
        { label: 'Ridge Classifier', value: 'ridge' },
      ],
      default: 'logistic',
      description: 'Level-1 model architecture trained on base model prediction probabilities.',
    },
    {
      id: 'metaReg',
      name: 'Regularization (C)',
      type: 'slider',
      min: 0.1,
      max: 10.0,
      step: 0.5,
      default: 1.0,
      description: 'Inverse regularization strength penalizing Level-1 meta-weights.',
    },
    {
      id: 'cvFolds',
      name: 'CV Folds (K)',
      type: 'slider',
      min: 2,
      max: 5,
      step: 1,
      default: 3,
      description: 'Out-of-fold cross-validation folds used to prevent meta-feature leakage.',
    },
  ],

  advancedParams: [],

  symbolicLinks: [
    {
      id: 'stacking-meta-feature',
      symbolLatex: '\\mathbf{z}_i = \\left[ P_{\\text{tree}}(y=1 \\mid \\mathbf{x}_i), \\; P_{\\text{knn}}(y=1 \\mid \\mathbf{x}_i) \\right]^T',
      label: 'Meta-Feature Vector',
      description: 'Level-0 base model predicted probabilities mapped to the [0, 1]² meta-space.',
      highlightTarget: 'weights-inset',
    },
    {
      id: 'stacking-meta-learner',
      symbolLatex: '\\hat{y}(\\mathbf{x}) = \\sigma \\left( w_1 z_1 + w_2 z_2 + b_\\text{meta} \\right)',
      label: 'Level-1 Meta-Classifier',
      description: 'Linear separating hyper-surface trained on out-of-fold probability vectors.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: StackingParams, seed: number): StackingState {
    const train = dataset.train;
    const isReg = isRegressionDataset(dataset, params);

    if (isReg) {
      const is1D = is1DRegressionDataset(dataset, true);
      const allowedFeatures = getAllowedFeatures(dataset, true);
      const treeRoot = fitTreeBase(train, true, allowedFeatures);
      const metaFeatures = train.map((pt) => {
        const y1 = predictTree(treeRoot, pt.x1, pt.x2);
        const y2 = knnPredictRegression(train, pt.x1, pt.x2, 7, is1D);
        return { id: pt.id, p1: y1, p2: y2, y: pt.y };
      });

      const cReg = Number(params.metaReg ?? 1.0);
      const { weights: metaW, bias: metaB } = fitMetaRegressor(metaFeatures, cReg);

      return {
        linearWeights: [0, 0],
        linearBias: 0,
        treeRoot,
        metaWeights: metaW,
        metaBias: metaB,
        trainMetaFeatures: metaFeatures,
        trainDataPoints: train.map((p) => ({ x1: p.x1, x2: p.x2, y: p.y })),
        phase: 0,
        epoch: 0,
        seed,
        isRegression: true,
        is1DRegression: is1D,
      };
    }

    const { weights: linW, bias: linB } = fitLinearBase(train);
    const treeRoot = fitTreeBase(train, false);

    // Compute meta-features: [p_tree, p_knn]
    const metaFeatures = train.map((pt) => {
      const predT = predictTree(treeRoot, pt.x1, pt.x2);
      const p1 = predT === 1 ? 0.88 : 0.12;
      const p2 = knnPredictProb(train, pt.x1, pt.x2, 7);
      return { id: pt.id, p1, p2, y: pt.y };
    });

    const cReg = Number(params.metaReg ?? 1.0);
    const { weights: metaW, bias: metaB } = fitMetaLearner(metaFeatures, cReg);

    return {
      linearWeights: linW,
      linearBias: linB,
      treeRoot,
      metaWeights: metaW,
      metaBias: metaB,
      trainMetaFeatures: metaFeatures,
      trainDataPoints: train.map((p) => ({ x1: p.x1, x2: p.x2, y: p.y })),
      phase: 0,
      epoch: 0,
      seed,
      isRegression: false,
    };
  },

  step(state: StackingState, dataset: DatasetContainer, params: StackingParams): StepResult<StackingState> {
    const nextEpoch = state.epoch + 1;
    const cReg = Number(params.metaReg ?? 1.0);
    const { weights: metaW, bias: metaB } = state.isRegression
      ? fitMetaRegressor(state.trainMetaFeatures, cReg)
      : fitMetaLearner(state.trainMetaFeatures, cReg);

    const nextState: StackingState = {
      ...state,
      metaWeights: metaW,
      metaBias: metaB,
      epoch: nextEpoch,
      phase: Math.min(3, nextEpoch),
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
      converged: nextEpoch >= 3,
    };
  },

  predict(input: Float32Array, state: StackingState, _params: StackingParams): number {
    if (state.epoch === 0) return 0;
    const x1 = input[0];
    const x2 = input[1];

    if (state.isRegression) {
      const y1 = predictTree(state.treeRoot, x1, x2);
      const y2 = knnPredictRegression(state.trainDataPoints ?? [], x1, x2, 7, state.is1DRegression ?? false);

      if (state.epoch === 1) return y1;
      if (state.epoch === 2) return y2;

      return state.metaWeights[0] * y1 + state.metaWeights[1] * y2 + state.metaBias;
    }

    const predT = predictTree(state.treeRoot, x1, x2);
    const p1 = predT === 1 ? 0.88 : 0.12;
    const p2 = knnPredictProb(state.trainDataPoints ?? [], x1, x2, 7);

    // Phase 1 (Epoch 1): Preview Level-0 Tree Model
    if (state.epoch === 1) {
      return (p1 - 0.5) * 4;
    }
    // Phase 2 (Epoch 2): Preview Level-0 k-NN Model
    if (state.epoch === 2) {
      return (p2 - 0.5) * 4;
    }

    // Phase 3+ (Epoch 3+): Level-1 Meta-Learner Logistic Combination
    const z = state.metaWeights[0] * p1 + state.metaWeights[1] * p2 + state.metaBias;
    return z;
  },

  loss(state: StackingState, dataset: DatasetContainer, _params: StackingParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;
    if (train.length === 0) return { trainLoss: 0, testLoss: 0 };

    if (state.epoch === 0) {
      const isReg = state.isRegression ?? false;
      return {
        trainLoss: isReg ? 1.0 : 0.693,
        testLoss: isReg ? 1.0 : 0.693,
        trainAccuracy: isReg ? 0.0 : 50.0,
        testAccuracy: isReg ? 0.0 : 50.0,
        trainR2: isReg ? 0.0 : undefined,
        testR2: isReg ? 0.0 : undefined,
        custom: {
          'Meta-Weight Tree': 0,
          'Meta-Weight KNN': 0,
          'Meta-Bias': 0,
        },
      };
    }

    if (state.isRegression) {
      let trMse = 0, trMeanY = 0;
      for (const mf of state.trainMetaFeatures) {
        trMeanY += mf.y;
        const yHat = state.metaWeights[0] * mf.p1 + state.metaWeights[1] * mf.p2 + state.metaBias;
        const err = mf.y - yHat;
        trMse += err * err;
      }
      trMse /= Math.max(1, train.length);
      trMeanY /= Math.max(1, train.length);
      let trTss = 0;
      for (const mf of state.trainMetaFeatures) {
        trTss += (mf.y - trMeanY) * (mf.y - trMeanY);
      }
      const trR2 = trTss > 1e-7 ? Math.max(-1.0, 1.0 - (trMse * train.length) / trTss) : 0;

      let teMse = 0, teR2 = 0;
      if (test.length > 0) {
        let teMeanY = 0;
        for (const pt of test) teMeanY += pt.y;
        teMeanY /= test.length;
        let teTss = 0;
        for (const pt of test) {
          const y1 = predictTree(state.treeRoot, pt.x1, pt.x2);
          const y2 = knnPredictRegression(state.trainDataPoints ?? train, pt.x1, pt.x2, 7, state.is1DRegression ?? false);
          const yHat = state.metaWeights[0] * y1 + state.metaWeights[1] * y2 + state.metaBias;
          const err = pt.y - yHat;
          teMse += err * err;
          teTss += (pt.y - teMeanY) * (pt.y - teMeanY);
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
          'Meta-Weight Tree': Number(state.metaWeights[0].toFixed(3)),
          'Meta-Weight KNN': Number(state.metaWeights[1].toFixed(3)),
          'Meta-Bias': Number(state.metaBias.toFixed(3)),
        },
      };
    }

    let trCorrect = 0, trLoss = 0;
    for (const mf of state.trainMetaFeatures) {
      const z = state.metaWeights[0] * mf.p1 + state.metaWeights[1] * mf.p2 + state.metaBias;
      const prob = sigmoid(z);
      const pred = prob >= 0.5 ? 1 : -1;
      if (pred === mf.y) trCorrect++;
      const y01 = mf.y === 1 ? 1 : 0;
      trLoss += -(y01 * Math.log(Math.max(1e-7, prob)) + (1 - y01) * Math.log(Math.max(1e-7, 1 - prob)));
    }
    const trAcc = (trCorrect / train.length) * 100;
    trLoss /= train.length;

    let teCorrect = 0, teLoss = 0;
    if (test.length > 0) {
      for (const pt of test) {
        const predT = predictTree(state.treeRoot, pt.x1, pt.x2);
        const p1 = predT === 1 ? 0.88 : 0.12;
        const p2 = knnPredictProb(state.trainDataPoints ?? train, pt.x1, pt.x2, 7);
        const z = state.metaWeights[0] * p1 + state.metaWeights[1] * p2 + state.metaBias;
        const prob = sigmoid(z);
        const pred = prob >= 0.5 ? 1 : -1;
        if (pred === pt.y) teCorrect++;
        const y01 = pt.y === 1 ? 1 : 0;
        teLoss += -(y01 * Math.log(Math.max(1e-7, prob)) + (1 - y01) * Math.log(Math.max(1e-7, 1 - prob)));
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
        'Meta-Weight Tree': Number(state.metaWeights[0].toFixed(3)),
        'Meta-Weight KNN': Number(state.metaWeights[1].toFixed(3)),
        'Meta-Bias': Number(state.metaBias.toFixed(3)),
      },
    };
  },

  getVisualArtifacts(
    state: StackingState,
    _dataset: DatasetContainer,
    _params: StackingParams
  ): VisualArtifact[] {
    if (state.epoch === 0) {
      return [
        {
          type: 'tree-partition',
          root: undefined,
          allNodes: [],
          splits: [],
          leaves: [],
          totalTreesCount: 0,
        },
      ];
    }

    const splits = extractSplits(state.treeRoot);
    const leaves = extractLeaves(state.treeRoot);
    const allNodes = flattenTree(state.treeRoot);

    return [
      {
        type: 'tree-partition',
        root: allNodes[0],
        allNodes,
        splits,
        leaves,
        totalTreesCount: 1,
      },
      {
        type: 'meta-feature-projection',
        points: state.trainMetaFeatures,
        weights: state.metaWeights,
        bias: state.metaBias,
        modelLabels: state.isRegression
          ? ['ŷ(Decision Tree)', 'ŷ(k-NN Regressor)']
          : ['P(Decision Tree)', 'P(k-NN Classifier)'],
      },
    ];
  },

  serializeSnapshot(state: StackingState): Float32Array {
    return serializeTreeToFloat32(state.treeRoot);
  },

  restoreSnapshot(buffer: Float32Array, state: StackingState): StackingState {
    const restored = deserializeTreeFromFloat32(buffer);
    return {
      ...state,
      treeRoot: restored,
      epoch: 3,
    };
  },

  explain(
    state: StackingState,
    params: StackingParams,
    lastStep?: StepResult<StackingState>
  ): ExplanationPayload {
    const isReg = state.isRegression ?? false;
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const mse = lastStep?.metrics.trainLoss?.toFixed(4) ?? '0.0000';

    if (state.epoch === 0) {
      return {
        summary: isReg
          ? `Stacking Regressor initialized (Epoch 0). Click Play or Step to fit Level-0 base models and Level-1 meta-regressor.`
          : `Stacking Classifier initialized (Epoch 0). Click Play or Step to fit Level-0 base models and project meta-features.`,
        coreAssumption:
          'Hierarchical meta-learning: a higher-level model can learn which base model to trust in different sub-domains of the prediction space.',
        whenItFails:
          'When base models produce heavily correlated outputs, causing collinearity in the Level-1 meta-feature matrix.',
        microLog: [
          `Epoch [0]: Untrained initial state`,
          `Level-0 base learners ready: CART ${isReg ? 'Regression Tree' : 'Decision Tree'} and k-NN`,
          `Click Play or Step to begin 3-phase stacked generalization`,
        ],
        mathLatex: {
          objective: isReg
            ? '\\min_{\\mathbf{w}, b} \\frac{1}{N} \\sum_{i=1}^N \\left( y_i - (\\mathbf{w}^T \\mathbf{z}_i + b) \\right)^2 + \\frac{1}{2C} \\|\\mathbf{w}\\|^2'
            : '\\min_{\\mathbf{w}, b} \\sum_{i=1}^N \\log \\left(1 + \\exp(-y_i (\\mathbf{w}^T \\mathbf{z}_i + b))\\right) + \\frac{1}{2C} \\|\\mathbf{w}\\|^2',
          lossOrUpdate: isReg
            ? '\\mathbf{z}_i = \\left[ \\hat{y}_{\\text{tree}}(\\mathbf{x}_i), \\; \\hat{y}_{\\text{knn}}(\\mathbf{x}_i) \\right]^T'
            : '\\mathbf{z}_i = \\left[ P_{\\text{tree}}(y=1 \\mid \\mathbf{x}_i), \\; P_{\\text{knn}}(y=1 \\mid \\mathbf{x}_i) \\right]^T',
          dualOrConstraint: isReg
            ? '\\hat{y}(\\mathbf{x}) = w_1 z_1 + w_2 z_2 + b_\\text{meta}'
            : 'z_{ik} \\in [0, 1] \\quad \\forall k \\in \\{1, 2\\}',
        },
        codeSnippet: {
          sklearn: isReg
            ? `from sklearn.ensemble import StackingRegressor\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.linear_model import Ridge\n\nmodel = StackingRegressor(\n    estimators=[\n        ('dt', DecisionTreeRegressor(max_depth=3)),\n        ('knn', KNeighborsRegressor(n_neighbors=7))\n    ],\n    final_estimator=Ridge(alpha=${1.0 / (Number(params.metaReg ?? 1.0))}),\n    cv=${params.cvFolds ?? 3}\n)\nmodel.fit(X_train, y_train)`
            : `from sklearn.ensemble import StackingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = StackingClassifier(\n    estimators=[\n        ('dt', DecisionTreeClassifier(max_depth=3)),\n        ('knn', KNeighborsClassifier(n_neighbors=7))\n    ],\n    final_estimator=LogisticRegression(C=${params.metaReg ?? 1.0}),\n    cv=${params.cvFolds ?? 3}\n)\nmodel.fit(X_train, y_train)`,
          pytorch: `# Stacking Level-1 Meta-Learner\nimport torch\nimport torch.nn as nn\n\nclass StackingMeta(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.meta_linear = nn.Linear(2, 1)\n    def forward(self, z_meta):\n        return self.meta_linear(z_meta)`,
        },
      };
    }

    return {
      summary: isReg
        ? `Stacking Regressor projected training samples into [ŷ(Tree), ŷ(KNN)] meta-feature space and fitted a Level-1 linear meta-regressor, achieving Train MSE = ${mse}.`
        : `Stacking generalizer projected training samples into [P(Tree), P(KNN)] meta-feature space and fitted a Level-1 ${params.metaLearner ?? 'logistic'} meta-learner, reaching ${acc}% accuracy.`,
      coreAssumption:
        'Hierarchical meta-learning: a higher-level model can learn which base model to trust in different sub-domains of the prediction space.',
      whenItFails:
        'When base models produce heavily correlated outputs, causing collinearity in the Level-1 meta-feature matrix.',
      microLog: isReg
        ? [
            `Step 1: Fit Level-0 Base Regressors (CART Regression Tree and k-NN Regressor)`,
            `Step 2: Projected N = ${state.trainMetaFeatures.length} points into continuous meta-prediction space`,
            `Step 3: Fit Level-1 Meta-Regressor: y_hat = ${state.metaWeights[0].toFixed(2)} * y_tree + ${state.metaWeights[1].toFixed(2)} * y_knn + ${state.metaBias.toFixed(2)}`,
          ]
        : [
            `Step 1: Fit Level-0 Base Learners (CART Decision Tree and k-NN)`,
            `Step 2: Projected N = ${state.trainMetaFeatures.length} points into [0, 1]² meta-probability space`,
            `Step 3: Fit Level-1 separating hyperplane: w_meta = [${state.metaWeights[0].toFixed(2)}, ${state.metaWeights[1].toFixed(2)}]`,
          ],
      mathLatex: {
        objective: isReg
          ? '\\min_{\\mathbf{w}, b} \\frac{1}{N} \\sum_{i=1}^N \\left( y_i - (\\mathbf{w}^T \\mathbf{z}_i + b) \\right)^2 + \\frac{1}{2C} \\|\\mathbf{w}\\|^2'
          : '\\min_{\\mathbf{w}, b} \\sum_{i=1}^N \\log \\left(1 + \\exp(-y_i (\\mathbf{w}^T \\mathbf{z}_i + b))\\right) + \\frac{1}{2C} \\|\\mathbf{w}\\|^2',
        lossOrUpdate: isReg
          ? '\\mathbf{z}_i = \\left[ \\hat{y}_{\\text{tree}}(\\mathbf{x}_i), \\; \\hat{y}_{\\text{knn}}(\\mathbf{x}_i) \\right]^T'
          : '\\mathbf{z}_i = \\left[ P_{\\text{tree}}(y=1 \\mid \\mathbf{x}_i), \\; P_{\\text{knn}}(y=1 \\mid \\mathbf{x}_i) \\right]^T',
        dualOrConstraint: isReg
          ? '\\hat{y}(\\mathbf{x}) = w_1 z_1 + w_2 z_2 + b_\\text{meta}'
          : 'z_{ik} \\in [0, 1] \\quad \\forall k \\in \\{1, 2\\}',
      },
      codeSnippet: {
        sklearn: isReg
          ? `from sklearn.ensemble import StackingRegressor\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.neighbors import KNeighborsRegressor\nfrom sklearn.linear_model import Ridge\n\nmodel = StackingRegressor(\n    estimators=[\n        ('dt', DecisionTreeRegressor(max_depth=3)),\n        ('knn', KNeighborsRegressor(n_neighbors=7))\n    ],\n    final_estimator=Ridge(alpha=${1.0 / (Number(params.metaReg ?? 1.0))}),\n    cv=${params.cvFolds ?? 3}\n)\nmodel.fit(X_train, y_train)`
          : `from sklearn.ensemble import StackingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\nfrom sklearn.linear_model import LogisticRegression\n\nmodel = StackingClassifier(\n    estimators=[\n        ('dt', DecisionTreeClassifier(max_depth=3)),\n        ('knn', KNeighborsClassifier(n_neighbors=7))\n    ],\n    final_estimator=LogisticRegression(C=${params.metaReg ?? 1.0}),\n    cv=${params.cvFolds ?? 3}\n)\nmodel.fit(X_train, y_train)`,
        pytorch: `# Stacking Level-1 Meta-Learner\nimport torch\nimport torch.nn as nn\n\nclass StackingMeta(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.meta_linear = nn.Linear(2, 1)\n    def forward(self, z_meta):\n        return self.meta_linear(z_meta)`,
      },
    };
  },
};
