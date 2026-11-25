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

export interface VotingParams {
  votingRule?: 'hard' | 'soft';
  weightLinear?: number;
  weightTree?: number;
  weightKnn?: number;
}

export interface VotingState {
  linearWeights: [number, number];
  linearBias: number;
  treeRoot: TreeNode;
  trainDataPoints: { x1: number; x2: number; y: number }[];
  epoch: number;
  linearAcc: number;
  treeAcc: number;
  knnAcc: number;
  isRegression?: boolean;
  is1DRegression?: boolean;
  linearMse?: number;
  treeMse?: number;
  knnMse?: number;
}

function sigmoid(z: number): number {
  if (z > 15) return 1.0;
  if (z < -15) return 0.0;
  return 1.0 / (1.0 + Math.exp(-z));
}

function fitSimpleLinearRegression(
  train: { x1: number; x2: number; y: number }[],
  is1DRegression: boolean = false
): { weights: [number, number]; bias: number } {
  let w1 = 0, w2 = 0, b = 0;
  const lr = 0.02;
  const n = train.length;
  if (n === 0) return { weights: [0, 0], bias: 0 };

  for (let iter = 0; iter < 150; iter++) {
    let gradW1 = 0, gradW2 = 0, gradB = 0;
    for (let i = 0; i < n; i++) {
      const p = train[i];
      const yHat = is1DRegression ? w1 * p.x1 + b : w1 * p.x1 + w2 * p.x2 + b;
      const err = yHat - p.y;
      gradW1 += err * p.x1;
      if (!is1DRegression) gradW2 += err * p.x2;
      gradB += err;
    }
    w1 -= (lr * gradW1) / n;
    if (!is1DRegression) w2 -= (lr * gradW2) / n;
    b -= (lr * gradB) / n;
  }
  return { weights: [w1, is1DRegression ? 0 : w2], bias: b };
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

function fitSimpleLogistic(train: { x1: number; x2: number; y: number }[]): { weights: [number, number]; bias: number } {
  let w1 = 0, w2 = 0, b = 0;
  const lr = 0.05;
  const n = train.length;
  if (n === 0) return { weights: [0, 0], bias: 0 };

  for (let iter = 0; iter < 120; iter++) {
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

function fitSimpleTree(
  train: { x1: number; x2: number; y: number }[],
  maxDepth: number = 3,
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
    if (node.depth >= maxDepth || indices.length < 4 || node.impurity <= 1e-7) continue;

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

export const votingClassifierPlugin: AlgorithmPlugin<VotingState, VotingParams> = {
  id: 'voting-classifier',
  name: 'Voting Classifier (Hard vs. Soft Ensemble)',
  shortCode: 'Voting',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Combines heterogeneous base classifiers (Logistic Regression, CART Decision Tree, k-Nearest Neighbors) via weighted majority (hard) or probability consensus (soft).',
  paperReference: 'Dietterich, Ensemble Methods in Machine Learning, 2000',

  primaryParams: [
    {
      id: 'votingRule',
      name: 'Voting Rule',
      type: 'select',
      options: [
        { label: 'Soft Voting (Weighted Probabilities)', value: 'soft' },
        { label: 'Hard Voting (Majority Class Count)', value: 'hard' },
      ],
      default: 'soft',
      description: 'Method of aggregating base predictions: average probability distribution vs discrete class vote tally.',
    },
    {
      id: 'weightLinear',
      name: 'Weight Linear (w₁)',
      type: 'slider',
      min: 0.1,
      max: 3.0,
      step: 0.1,
      default: 1.0,
      description: 'Voting weight assigned to base Logistic Regression model.',
    },
    {
      id: 'weightTree',
      name: 'Weight Tree (w₂)',
      type: 'slider',
      min: 0.1,
      max: 3.0,
      step: 0.1,
      default: 1.0,
      description: 'Voting weight assigned to base CART Decision Tree.',
    },
    {
      id: 'weightKnn',
      name: 'Weight KNN (w₃)',
      type: 'slider',
      min: 0.1,
      max: 3.0,
      step: 0.1,
      default: 1.0,
      description: 'Voting weight assigned to base k-Nearest Neighbors model.',
    },
  ],

  advancedParams: [],

  symbolicLinks: [
    {
      id: 'vote-soft',
      symbolLatex: '\\hat{y}_{\\text{soft}} = \\arg\\max_c \\sum_{m=1}^M w_m P_m(y = c \\mid \\mathbf{x})',
      label: 'Soft Probability Consensus',
      description: 'Continuous calibrated probability average weighted by model reliability.',
      highlightTarget: 'decision-hyperplane',
    },
    {
      id: 'vote-hard',
      symbolLatex: '\\hat{y}_{\\text{hard}} = \\arg\\max_c \\sum_{m=1}^M w_m \\mathbb{I}\\left(h_m(\\mathbf{x}) = c\\right)',
      label: 'Hard Majority Tally',
      description: 'Discrete majority voting tally ignoring internal class confidence.',
      highlightTarget: 'normal-vector',
    },
  ],

  init(dataset: DatasetContainer, params: VotingParams, _seed: number): VotingState {
    const train = dataset.train;
    const isReg = isRegressionDataset(dataset, params);

    if (isReg) {
      const is1D = is1DRegressionDataset(dataset, true);
      const allowedFeatures = getAllowedFeatures(dataset, true);
      const { weights, bias } = fitSimpleLinearRegression(train, is1D);
      const treeRoot = fitSimpleTree(train, 3, true, allowedFeatures);

      let linMse = 0, treeMse = 0, knnMse = 0;
      for (const p of train) {
        const yLin = is1D ? weights[0] * p.x1 + bias : weights[0] * p.x1 + weights[1] * p.x2 + bias;
        const eLin = p.y - yLin;
        linMse += eLin * eLin;

        const yTree = predictTree(treeRoot, p.x1, p.x2);
        const eTree = p.y - yTree;
        treeMse += eTree * eTree;

        const yKnn = knnPredictRegression(train, p.x1, p.x2, 7, is1D);
        const eKnn = p.y - yKnn;
        knnMse += eKnn * eKnn;
      }
      const n = Math.max(1, train.length);
      linMse /= n;
      treeMse /= n;
      knnMse /= n;

      return {
        linearWeights: weights,
        linearBias: bias,
        treeRoot,
        trainDataPoints: train,
        epoch: 0,
        linearAcc: 0,
        treeAcc: 0,
        knnAcc: 0,
        isRegression: true,
        is1DRegression: is1D,
        linearMse: linMse,
        treeMse: treeMse,
        knnMse: knnMse,
      };
    }

    const { weights, bias } = fitSimpleLogistic(train);
    const treeRoot = fitSimpleTree(train, 3, false);

    // Compute individual base accuracies
    let linCor = 0, treeCor = 0, knnCor = 0;
    for (const p of train) {
      const pLin = sigmoid(weights[0] * p.x1 + weights[1] * p.x2 + bias) >= 0.5 ? 1 : -1;
      if (pLin === p.y) linCor++;
      const pTree = predictTree(treeRoot, p.x1, p.x2);
      if (pTree === p.y) treeCor++;
      const pKnn = knnPredictProb(train, p.x1, p.x2, 7) >= 0.5 ? 1 : -1;
      if (pKnn === p.y) knnCor++;
    }
    const n = Math.max(1, train.length);

    return {
      linearWeights: weights,
      linearBias: bias,
      treeRoot,
      trainDataPoints: train,
      epoch: 0,
      linearAcc: (linCor / n) * 100,
      treeAcc: (treeCor / n) * 100,
      knnAcc: (knnCor / n) * 100,
      isRegression: false,
    };
  },

  step(state: VotingState, dataset: DatasetContainer, params: VotingParams): StepResult<VotingState> {
    const nextEpoch = state.epoch + 1;
    const nextState: VotingState = { ...state, epoch: nextEpoch };
    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
      converged: nextEpoch >= 3,
    };
  },

  predict(input: Float32Array, state: VotingState, params: VotingParams): number {
    if (state.epoch === 0) return 0;
    const x1 = input[0];
    const x2 = input[1];
    const wLin = Number(params.weightLinear ?? 1.0);
    const wTree = Number(params.weightTree ?? 1.0);
    const wKnn = Number(params.weightKnn ?? 1.0);

    if (state.isRegression) {
      const yLin = state.is1DRegression
        ? state.linearWeights[0] * x1 + state.linearBias
        : state.linearWeights[0] * x1 + state.linearWeights[1] * x2 + state.linearBias;
      const yTree = predictTree(state.treeRoot, x1, x2);
      const yKnn = knnPredictRegression(state.trainDataPoints, x1, x2, 7, state.is1DRegression ?? false);

      if (state.epoch === 1) return yLin;
      if (state.epoch === 2) return yTree;

      const totalW = wLin + wTree + wKnn;
      return totalW > 0 ? (wLin * yLin + wTree * yTree + wKnn * yKnn) / totalW : 0;
    }

    const rule = params.votingRule ?? 'soft';

    // Model 1: Logistic
    const probLin = sigmoid(state.linearWeights[0] * x1 + state.linearWeights[1] * x2 + state.linearBias);
    // Model 2: Tree
    const predTree = predictTree(state.treeRoot, x1, x2);
    const probTree = predTree === 1 ? 0.95 : 0.05;
    // Model 3: KNN
    const probKnn = knnPredictProb(state.trainDataPoints, x1, x2, 7);

    // Phase 1 (Epoch 1): Preview Model A (Logistic Regression)
    if (state.epoch === 1) {
      return (probLin - 0.5) * 4;
    }
    // Phase 2 (Epoch 2): Preview Model B (CART Decision Tree)
    if (state.epoch === 2) {
      return (probTree - 0.5) * 4;
    }

    // Phase 3+ (Epoch 3+): Full Weighted Consensus Voting
    if (rule === 'soft') {
      const totalW = wLin + wTree + wKnn;
      const combinedProb = (wLin * probLin + wTree * probTree + wKnn * probKnn) / totalW;
      return (combinedProb - 0.5) * 4;
    } else {
      let votePos = 0, voteNeg = 0;
      if (probLin >= 0.5) votePos += wLin; else voteNeg += wLin;
      if (predTree === 1) votePos += wTree; else voteNeg += wTree;
      if (probKnn >= 0.5) votePos += wKnn; else voteNeg += wKnn;
      return (votePos - voteNeg) / (wLin + wTree + wKnn || 1);
    }
  },

  loss(state: VotingState, dataset: DatasetContainer, params: VotingParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;
    if (train.length === 0) return { trainLoss: 0, testLoss: 0 };

    const inputBuf = new Float32Array(2);

    if (state.isRegression) {
      let trMse = 0, trMeanY = 0;
      for (let i = 0; i < train.length; i++) trMeanY += train[i].y;
      trMeanY /= Math.max(1, train.length);
      let trTss = 0;
      for (let i = 0; i < train.length; i++) {
        inputBuf[0] = train[i].x1;
        inputBuf[1] = train[i].x2;
        const pred = this.predict(inputBuf, state, params);
        const yHat = typeof pred === 'number' ? pred : pred[0];
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
          inputBuf[0] = test[i].x1;
          inputBuf[1] = test[i].x2;
          const pred = this.predict(inputBuf, state, params);
          const yHat = typeof pred === 'number' ? pred : pred[0];
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
          'Linear Model MSE': Number(state.linearMse?.toFixed(4) ?? 0),
          'Tree Model MSE': Number(state.treeMse?.toFixed(4) ?? 0),
          'KNN Model MSE': Number(state.knnMse?.toFixed(4) ?? 0),
        },
      };
    }

    let trCorrect = 0;
    for (let i = 0; i < train.length; i++) {
      inputBuf[0] = train[i].x1;
      inputBuf[1] = train[i].x2;
      const pred = this.predict(inputBuf, state, params);
      const predVal = typeof pred === 'number' ? pred : pred[0];
      const predLabel = predVal >= 0 ? 1 : -1;
      if (predLabel === train[i].y) trCorrect++;
    }
    const trAcc = (trCorrect / train.length) * 100;
    const trLoss = (train.length - trCorrect) / train.length;

    let teAcc = 0, teLoss = 0;
    if (test.length > 0) {
      let teCorrect = 0;
      for (let i = 0; i < test.length; i++) {
        inputBuf[0] = test[i].x1;
        inputBuf[1] = test[i].x2;
        const pred = this.predict(inputBuf, state, params);
        const predVal = typeof pred === 'number' ? pred : pred[0];
        const predLabel = predVal >= 0 ? 1 : -1;
        if (predLabel === test[i].y) teCorrect++;
      }
      teAcc = (teCorrect / test.length) * 100;
      teLoss = (test.length - teCorrect) / test.length;
    }

    return {
      trainLoss: trLoss,
      testLoss: teLoss,
      trainAccuracy: trAcc,
      testAccuracy: teAcc,
      custom: {
        'Linear Model Acc': `${state.linearAcc.toFixed(1)}%`,
        'Tree Model Acc': `${state.treeAcc.toFixed(1)}%`,
        'KNN Model Acc': `${state.knnAcc.toFixed(1)}%`,
        'Ensemble Acc': `${trAcc.toFixed(1)}%`,
      },
    };
  },

  getVisualArtifacts(
    state: VotingState,
    _dataset: DatasetContainer,
    params: VotingParams
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

    const isReg = state.isRegression ?? false;
    const baseModels = isReg
      ? [
          { name: 'Linear Regression (OLS)', color: '#0EA5E9', style: 'dashed' as const, accuracy: Number(state.linearMse?.toFixed(3) ?? 0), weight: Number(params.weightLinear ?? 1.0) },
          { name: 'Regression Tree (CART)', color: '#F59E0B', style: 'dashed' as const, accuracy: Number(state.treeMse?.toFixed(3) ?? 0), weight: Number(params.weightTree ?? 1.0) },
          { name: 'k-NN Regressor', color: '#EC4899', style: 'dashed' as const, accuracy: Number(state.knnMse?.toFixed(3) ?? 0), weight: Number(params.weightKnn ?? 1.0) },
        ]
      : [
          { name: 'Logistic Regression', color: '#0EA5E9', style: 'dashed' as const, accuracy: state.linearAcc, weight: Number(params.weightLinear ?? 1.0) },
          { name: 'Decision Tree (CART)', color: '#F59E0B', style: 'dashed' as const, accuracy: state.treeAcc, weight: Number(params.weightTree ?? 1.0) },
          { name: 'k-Nearest Neighbors', color: '#EC4899', style: 'dashed' as const, accuracy: state.knnAcc, weight: Number(params.weightKnn ?? 1.0) },
        ];

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
        type: 'voting-boundaries',
        baseModels,
      },
    ];
  },

  serializeSnapshot(state: VotingState): Float32Array {
    return serializeTreeToFloat32(state.treeRoot);
  },

  restoreSnapshot(buffer: Float32Array, state: VotingState): VotingState {
    const restored = deserializeTreeFromFloat32(buffer);
    return {
      ...state,
      treeRoot: restored,
      epoch: 1,
    };
  },

  explain(
    state: VotingState,
    params: VotingParams,
    lastStep?: StepResult<VotingState>
  ): ExplanationPayload {
    const rule = params.votingRule ?? 'soft';
    const isReg = state.isRegression ?? false;
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const mse = lastStep?.metrics.trainLoss?.toFixed(4) ?? '0.0000';

    return {
      summary: isReg
        ? `Voting Regressor combining [A] Linear (${state.linearMse?.toFixed(3) ?? '0.000'} MSE), [B] Tree (${state.treeMse?.toFixed(3) ?? '0.000'} MSE), and [C] KNN (${state.knnMse?.toFixed(3) ?? '0.000'} MSE) achieves ${mse} MSE.`
        : `Voting Classifier combining [A] Logistic (${state.linearAcc.toFixed(0)}%), [B] Tree (${state.treeAcc.toFixed(0)}%), and [C] KNN (${state.knnAcc.toFixed(0)}%) via ${rule} voting achieves ${acc}% ensemble accuracy.`,
      coreAssumption:
        'Model diversity hypothesis: different model families (linear, partition, instance-based) make independent, uncorrelated errors that cancel out upon aggregation.',
      whenItFails:
        'When one weak base learner is severely degraded and receives high weight, dragging down the consensus of the remaining accurate models.',
      microLog: isReg
        ? [
            state.is1DRegression
              ? `Fitted Model A (Linear Regression): y = ${state.linearWeights[0].toFixed(2)}x₁ + ${state.linearBias.toFixed(2)}`
              : `Fitted Model A (Linear Regression): y = ${state.linearWeights[0].toFixed(2)}x₁ + ${state.linearWeights[1].toFixed(2)}x₂ + ${state.linearBias.toFixed(2)}`,
            `Fitted Model B (CART Regression Tree) & Model C (k-Nearest Neighbors Regressor)`,
            state.epoch >= 3
              ? `Aggregated continuous predictions via weighted average with weights [${params.weightLinear ?? 1}, ${params.weightTree ?? 1}, ${params.weightKnn ?? 1}]`
              : state.epoch === 0
              ? `Ready to train: click Play or Step to evaluate base models & consensus`
              : `Step ${state.epoch}/3: Previewing base learner before weighted average consensus`,
          ]
        : [
            `Fitted Model A (Logistic Regression): w = [${state.linearWeights[0].toFixed(2)}, ${state.linearWeights[1].toFixed(2)}]`,
            `Fitted Model B (CART Decision Tree) & Model C (k-Nearest Neighbors)`,
            `Aggregated using ${rule} voting with weights [${params.weightLinear ?? 1}, ${params.weightTree ?? 1}, ${params.weightKnn ?? 1}]`,
          ],
      mathLatex: {
        objective: isReg
          ? '\\hat{y}(\\mathbf{x}) = \\frac{\\sum_{m=1}^M w_m \\hat{y}_m(\\mathbf{x})}{\\sum_{m=1}^M w_m}'
          : '\\hat{y}(\\mathbf{x}) = \\arg\\max_{c} \\sum_{m=1}^M w_m P_m(y = c \\mid \\mathbf{x})',
        lossOrUpdate: isReg
          ? '\\text{MSE}_{\\text{ensemble}} = \\frac{1}{N} \\sum_{i=1}^N \\left( y_i - \\hat{y}(\\mathbf{x}_i) \\right)^2'
          : 'P(\\text{all } M \\text{ err}) = \\prod_{m=1}^M \\epsilon_m \\ll \\min_m \\epsilon_m \\quad (\\text{if independent})',
        dualOrConstraint: 'w_1 + w_2 + w_3 = \\sum_{m=1}^3 w_m',
      },
      codeSnippet: {
        sklearn: isReg
          ? `from sklearn.ensemble import VotingRegressor\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.tree import DecisionTreeRegressor\nfrom sklearn.neighbors import KNeighborsRegressor\n\nmodel = VotingRegressor(\n    estimators=[\n        ('lr', LinearRegression()),\n        ('dt', DecisionTreeRegressor(max_depth=3)),\n        ('knn', KNeighborsRegressor(n_neighbors=7))\n    ],\n    weights=[${params.weightLinear ?? 1.0}, ${params.weightTree ?? 1.0}, ${params.weightKnn ?? 1.0}]\n)\nmodel.fit(X_train, y_train)`
          : `from sklearn.ensemble import VotingClassifier\nfrom sklearn.linear_model import LogisticRegression\nfrom sklearn.tree import DecisionTreeClassifier\nfrom sklearn.neighbors import KNeighborsClassifier\n\nmodel = VotingClassifier(\n    estimators=[\n        ('lr', LogisticRegression()),\n        ('dt', DecisionTreeClassifier(max_depth=3)),\n        ('knn', KNeighborsClassifier(n_neighbors=7))\n    ],\n    voting='${rule}',\n    weights=[${params.weightLinear ?? 1.0}, ${params.weightTree ?? 1.0}, ${params.weightKnn ?? 1.0}]\n)\nmodel.fit(X_train, y_train)`,
        pytorch: `# Weighted Ensemble Averaging\nimport torch\n\ndef ensemble_vote(p_linear, p_tree, p_knn, weights=[1, 1, 1]):\n    p_avg = (weights[0]*p_linear + weights[1]*p_tree + weights[2]*p_knn) / sum(weights)\n    return torch.where(p_avg >= 0.5, 1, -1)`,
      },
    };
  },
};
