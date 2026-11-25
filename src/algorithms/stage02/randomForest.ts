import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { mulberry32 } from '../../utils/prng';
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
  ImpurityCriterion,
  isRegressionDataset,
  getAllowedFeatures,
} from './treeUtils';

export interface RandomForestParams {
  nEstimators?: number;
  maxDepth?: number;
  mTry?: 'sqrt' | 'all' | '1-feature-only';
  featureSubspace?: 'sqrt' | 'all' | '1-feature-only';
  minSamplesSplit?: number;
  selectedTree?: number;
  task?: 'classification' | 'regression';
}

export interface RandomForestState {
  trees: TreeNode[];
  bootstrapIndices: number[][]; // treeIndex -> sample indices used
  epoch: number;
  selectedTreeIndex: number; // -1 for ensemble, 0..B-1 for specific tree
  seed: number;
  oobError: number;
  isRegression?: boolean;
}

function buildSingleRandomTree(
  train: { x1: number; x2: number; y: number }[],
  bootstrapIdxs: number[],
  maxDepth: number,
  mTry: 'sqrt' | 'all' | '1-feature-only',
  isRegression: boolean,
  rand: () => number,
  idGen: () => number,
  allowedFeatures: (0 | 1)[] = [0, 1]
): TreeNode {
  const rootSamples = bootstrapIdxs.length;

  let rootVal = 1;
  let rootImpurity = 0;
  let classCounts: [number, number, number] | undefined = undefined;

  if (isRegression) {
    let sum = 0;
    for (let i = 0; i < rootSamples; i++) sum += train[bootstrapIdxs[i]].y;
    rootVal = rootSamples > 0 ? sum / rootSamples : 0;
    let varSum = 0;
    for (let i = 0; i < rootSamples; i++) {
      const d = train[bootstrapIdxs[i]].y - rootVal;
      varSum += d * d;
    }
    rootImpurity = rootSamples > 0 ? varSum / rootSamples : 0;
  } else {
    let countPos = 0, countNeg = 0, countThird = 0;
    for (let i = 0; i < rootSamples; i++) {
      const y = train[bootstrapIdxs[i]].y;
      if (y === 1) countPos++;
      else if (y === 2) countThird++;
      else countNeg++;
    }
    let maxC = countPos;
    if (countNeg > maxC) { maxC = countNeg; rootVal = -1; }
    if (countThird > maxC) { rootVal = 2; }
    const p1 = countPos / rootSamples, p2 = countNeg / rootSamples, p3 = countThird / rootSamples;
    rootImpurity = Math.max(0, 1 - (p1 * p1 + p2 * p2 + p3 * p3));
    classCounts = [countPos, countNeg, countThird];
  }

  const root: TreeNode = {
    id: idGen(),
    depth: 0,
    isLeaf: true,
    samples: rootSamples,
    impurity: rootImpurity,
    value: rootVal,
    classCounts,
    region: [-6, 6, -6, 6],
  };

  const queue: { node: TreeNode; indices: number[] }[] = [{ node: root, indices: bootstrapIdxs }];
  const criterion: ImpurityCriterion = isRegression ? 'mse' : 'gini';

  while (queue.length > 0) {
    const { node, indices } = queue.shift()!;
    if (node.depth >= maxDepth || indices.length < 2 || node.impurity <= 1e-7) continue;

    // Determine feature subset for this split (strictly [0] on 1D regression)
    let featureSubset: (0 | 1)[] = allowedFeatures;
    if (allowedFeatures.length > 1 && (mTry === '1-feature-only' || mTry === 'sqrt')) {
      featureSubset = rand() < 0.5 ? [0] : [1];
    }

    const split = findBestSplitForPoints(train, indices, criterion, 2, featureSubset);
    if (!split || split.gain <= 1e-6) continue;

    const { left, right } = splitNode(node, split, idGen);
    queue.push({ node: left, indices: split.leftIndices });
    queue.push({ node: right, indices: split.rightIndices });
  }

  return root;
}

export function computeOOBMetrics(
  trees: TreeNode[],
  bootstrapIndices: number[][],
  train: { x1: number; x2: number; y: number }[],
  isRegression: boolean = false
): { oobError: number; oobAcc: number; oobMSE?: number; oobR2?: number } {
  const n = train.length;
  let oobEvaluated = 0;

  if (isRegression) {
    let oobSE = 0;
    let meanY = 0;
    for (let i = 0; i < n; i++) meanY += train[i].y;
    meanY = n > 0 ? meanY / n : 0;
    let ssTot = 0;

    for (let i = 0; i < n; i++) {
      const pt = train[i];
      let sumPred = 0;
      let count = 0;
      for (let t = 0; t < trees.length; t++) {
        const used = bootstrapIndices[t];
        let found = false;
        for (let j = 0; j < used.length; j++) {
          if (used[j] === i) { found = true; break; }
        }
        if (!found) {
          sumPred += predictTree(trees[t], pt.x1, pt.x2);
          count++;
        }
      }
      if (count > 0) {
        oobEvaluated++;
        const yHat = sumPred / count;
        const err = pt.y - yHat;
        oobSE += err * err;
        const d = pt.y - meanY;
        ssTot += d * d;
      }
    }

    const oobMSE = oobEvaluated > 0 ? oobSE / oobEvaluated : 0;
    const oobR2 = ssTot > 1e-7 ? Math.max(-1, 1 - oobSE / ssTot) : 1;
    return { oobError: oobMSE, oobAcc: oobR2 * 100, oobMSE, oobR2 };
  }

  let oobCorrect = 0;
  for (let i = 0; i < n; i++) {
    const pt = train[i];
    // Find all trees where sample i was NOT in bootstrap
    let votesPos = 0, votesNeg = 0, votesThird = 0;
    let treesConsulted = 0;

    for (let t = 0; t < trees.length; t++) {
      const used = bootstrapIndices[t];
      let found = false;
      for (let j = 0; j < used.length; j++) {
        if (used[j] === i) { found = true; break; }
      }
      if (!found) {
        treesConsulted++;
        const pred = predictTree(trees[t], pt.x1, pt.x2);
        if (pred === 1) votesPos++;
        else if (pred === 2) votesThird++;
        else votesNeg++;
      }
    }

    if (treesConsulted > 0) {
      oobEvaluated++;
      let maj = 1;
      let maxV = votesPos;
      if (votesNeg > maxV) { maxV = votesNeg; maj = -1; }
      if (votesThird > maxV) { maj = 2; }
      if (maj === pt.y) oobCorrect++;
    }
  }

  const oobAcc = oobEvaluated > 0 ? (oobCorrect / oobEvaluated) * 100 : 0;
  const oobError = oobEvaluated > 0 ? (oobEvaluated - oobCorrect) / oobEvaluated : 0;
  return { oobError, oobAcc };
}

export const randomForestPlugin: AlgorithmPlugin<RandomForestState, RandomForestParams> = {
  id: 'random-forest',
  name: 'Random Forest Ensemble',
  shortCode: 'RandForest',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Bootstrap aggregating (bagging) of decorrelated decision trees with randomized feature subspaces and Out-of-Bag (OOB) error estimation.',
  paperReference: 'Breiman, Random Forests, Machine Learning, 2001',

  primaryParams: [
    {
      id: 'nEstimators',
      name: 'Forest Size (B)',
      type: 'slider',
      min: 1,
      max: 50,
      step: 1,
      default: 20,
      description: 'Number of decorrelated decision trees to aggregate in the forest.',
    },
    {
      id: 'maxDepth',
      name: 'Max Depth',
      type: 'slider',
      min: 1,
      max: 8,
      step: 1,
      default: 5,
      description: 'Maximum depth allowed per individual base tree.',
    },
    {
      id: 'mTry',
      name: 'Feature Subspace (m_try)',
      type: 'select',
      options: [
        { label: 'Random Subspace (m = 1)', value: '1-feature-only' },
        { label: 'Square Root (m = √d)', value: 'sqrt' },
        { label: 'All Features (m = d)', value: 'all' },
      ],
      default: '1-feature-only',
      description: 'Number of random feature candidates considered at each split to decorrelate trees.',
    },
  ],

  advancedParams: [
    {
      id: 'selectedTree',
      name: 'Inspect Tree #',
      type: 'slider',
      min: -1,
      max: 50,
      step: 1,
      default: -1,
      description: '-1 to view combined ensemble, or 0..B-1 to inspect individual weak tree learner.',
    },
  ],

  symbolicLinks: [
    {
      id: 'rf-vote',
      symbolLatex: '\\hat{y}(\\mathbf{x}) = \\operatorname{mode} \\left\\{ T_b(\\mathbf{x}) \\right\\}_{b=1}^B',
      label: 'Majority Vote Aggregation',
      description: 'Ensemble consensus reducing variance by averaging decorrelated estimators.',
      highlightTarget: 'decision-hyperplane',
    },
    {
      id: 'rf-oob',
      symbolLatex: '\\text{OOB-Error} = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}\\left(y_i \\neq \\operatorname{mode}_{b: i \\notin S_b} T_b(\\mathbf{x}_i)\\right)',
      label: 'Out-of-Bag (OOB) Error',
      description: 'Unbiased generalization error estimated using instances withheld from bootstrap samples.',
      highlightTarget: 'normal-vector',
    },
  ],

  init(dataset: DatasetContainer, params: RandomForestParams, seed: number): RandomForestState {
    const isRegression = isRegressionDataset(dataset, params);
    return {
      trees: [],
      bootstrapIndices: [],
      epoch: 0,
      selectedTreeIndex: -1,
      seed,
      oobError: 0,
      isRegression,
    };
  },

  step(state: RandomForestState, dataset: DatasetContainer, params: RandomForestParams): StepResult<RandomForestState> {
    const isRegression = state.isRegression ?? isRegressionDataset(dataset, params);
    const targetEstimators = Number(params.nEstimators ?? 20);
    const maxDepth = Number(params.maxDepth ?? 5);
    const mTry = params.featureSubspace ?? params.mTry ?? '1-feature-only';
    const selectedTree = Number(params.selectedTree ?? -1);

    let updatedTrees = state.trees;
    let updatedBootstrap = state.bootstrapIndices;
    let oobError = state.oobError;

    if (state.trees.length < targetEstimators) {
      const rand = mulberry32(state.seed + state.trees.length * 997);
      const n = dataset.train.length;
      let idCounter = state.trees.length * 1000 + 1;

      // Draw bootstrap sample
      const boot: number[] = [];
      for (let i = 0; i < n; i++) {
        boot.push(Math.floor(rand() * n));
      }

      const allowedFeatures = getAllowedFeatures(dataset, isRegression);
      const newTree = buildSingleRandomTree(dataset.train, boot, maxDepth, mTry, isRegression, rand, () => idCounter++, allowedFeatures);
      updatedTrees = [...state.trees, newTree];
      updatedBootstrap = [...state.bootstrapIndices, boot];
      const oob = computeOOBMetrics(updatedTrees, updatedBootstrap, dataset.train, isRegression);
      oobError = oob.oobError;
    } else if (state.trees.length > targetEstimators) {
      updatedTrees = state.trees.slice(0, targetEstimators);
      updatedBootstrap = state.bootstrapIndices.slice(0, targetEstimators);
      const oob = computeOOBMetrics(updatedTrees, updatedBootstrap, dataset.train, isRegression);
      oobError = oob.oobError;
    }

    const nextEpoch = state.epoch + 1;
    const nextState: RandomForestState = {
      trees: updatedTrees,
      bootstrapIndices: updatedBootstrap,
      epoch: nextEpoch,
      selectedTreeIndex: selectedTree,
      seed: state.seed,
      oobError,
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

  predict(input: Float32Array, state: RandomForestState, params: RandomForestParams): number {
    if (!state.trees || state.trees.length === 0 || state.epoch === 0) return 0;

    const selectedIdx = params.selectedTree !== undefined ? Number(params.selectedTree) : state.selectedTreeIndex;
    if (selectedIdx >= 0 && selectedIdx < state.trees.length) {
      return predictTree(state.trees[selectedIdx], input[0], input[1]);
    }

    const isRegression = state.isRegression ?? false;
    if (isRegression) {
      let sum = 0;
      for (let i = 0; i < state.trees.length; i++) {
        sum += predictTree(state.trees[i], input[0], input[1]);
      }
      return sum / state.trees.length;
    }

    // Ensemble voting for classification
    let votePos = 0, voteNeg = 0, voteThird = 0;
    for (let i = 0; i < state.trees.length; i++) {
      const p = predictTree(state.trees[i], input[0], input[1]);
      if (p === 1) votePos++;
      else if (p === 2) voteThird++;
      else voteNeg++;
    }

    if (voteThird > votePos && voteThird > voteNeg) return 2;
    return (votePos - voteNeg) / (votePos + voteNeg || 1);
  },

  loss(state: RandomForestState, dataset: DatasetContainer, _params: RandomForestParams): TelemetryMetrics {
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
          'OOB Error': 0,
          'Forest Trees': 0,
        },
      };
    }

    if (isRegression) {
      let trainSE = 0;
      for (let i = 0; i < train.length; i++) {
        const p = train[i];
        let sum = 0;
        for (let t = 0; t < state.trees.length; t++) {
          sum += predictTree(state.trees[t], p.x1, p.x2);
        }
        const yHat = sum / state.trees.length;
        const err = p.y - yHat;
        trainSE += err * err;
      }
      const trainMSE = trainSE / train.length;

      let testMSE = 0;
      if (test.length > 0) {
        let testSE = 0;
        for (let i = 0; i < test.length; i++) {
          const p = test[i];
          let sum = 0;
          for (let t = 0; t < state.trees.length; t++) {
            sum += predictTree(state.trees[t], p.x1, p.x2);
          }
          const yHat = sum / state.trees.length;
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

      const oob = computeOOBMetrics(state.trees, state.bootstrapIndices, train, true);

      return {
        trainLoss: trainMSE,
        testLoss: testMSE,
        trainR2: r2,
        testR2: r2,
        custom: {
          'OOB MSE': Number((oob.oobMSE ?? 0).toFixed(4)),
          'OOB R²': Number((oob.oobR2 ?? 0).toFixed(4)),
          'Forest Trees': state.trees.length,
        },
      };
    }

    const predict = (x1: number, x2: number) => {
      let v1 = 0, v2 = 0, v3 = 0;
      for (let i = 0; i < state.trees.length; i++) {
        const p = predictTree(state.trees[i], x1, x2);
        if (p === 1) v1++;
        else if (p === 2) v3++;
        else v2++;
      }
      if (v3 > v1 && v3 > v2) return 2;
      return v1 >= v2 ? 1 : -1;
    };

    let trCorrect = 0;
    for (let i = 0; i < train.length; i++) {
      if (predict(train[i].x1, train[i].x2) === train[i].y) trCorrect++;
    }
    const trAcc = (trCorrect / train.length) * 100;
    const trLoss = (train.length - trCorrect) / train.length;

    let teAcc = 0;
    let teLoss = 0;
    if (test.length > 0) {
      let teCorrect = 0;
      for (let i = 0; i < test.length; i++) {
        if (predict(test[i].x1, test[i].x2) === test[i].y) teCorrect++;
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
        'OOB Error': Number(state.oobError.toFixed(4)),
        'OOB Accuracy': `${((1 - state.oobError) * 100).toFixed(1)}%`,
        'Forest Trees': state.trees.length,
      },
    };
  },

  getVisualArtifacts(
    state: RandomForestState,
    _dataset: DatasetContainer,
    params: RandomForestParams
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

    const selectedIdx = params.selectedTree !== undefined ? Number(params.selectedTree) : state.selectedTreeIndex;
    const activeTree = (selectedIdx >= 0 && selectedIdx < state.trees.length)
      ? state.trees[selectedIdx]
      : state.trees[state.trees.length - 1];

    const splits = extractSplits(activeTree);
    const leaves = extractLeaves(activeTree);
    const allNodes = flattenTree(activeTree);

    // Collect all trees splits for faint overlay
    const overlayTrees = state.trees.slice(0, 15).map((t, idx) => ({
      treeIndex: idx,
      splits: extractSplits(t).map((s) => ({
        featureIndex: s.featureIndex,
        threshold: s.threshold,
        range: s.range,
      })),
    }));

    return [
      {
        type: 'tree-partition',
        root: allNodes[0],
        allNodes,
        splits,
        leaves,
        totalTreesCount: state.trees.length,
        selectedTreeIndex: selectedIdx,
        treesRoots: state.trees.slice(0, 12),
      },
      {
        type: 'ensemble-overlay',
        trees: overlayTrees,
      },
    ];
  },

  serializeSnapshot(state: RandomForestState): Float32Array {
    if (state.trees.length === 0) return new Float32Array(0);
    return serializeTreeToFloat32(state.trees[state.trees.length - 1]);
  },

  restoreSnapshot(buffer: Float32Array, state: RandomForestState): RandomForestState {
    if (!buffer || buffer.length === 0) {
      return { ...state, trees: [], bootstrapIndices: [], epoch: 0 };
    }
    const restored = deserializeTreeFromFloat32(buffer);
    return {
      ...state,
      trees: [restored],
      epoch: 1,
    };
  },

  explain(
    state: RandomForestState,
    params: RandomForestParams,
    lastStep?: StepResult<RandomForestState>
  ): ExplanationPayload {
    const treeCount = state.trees.length;
    const isRegression = state.isRegression ?? false;
    const oobAcc = ((1 - state.oobError) * 100).toFixed(1);
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const perfSummary = isRegression
      ? `MSE ${lastStep?.metrics.trainLoss.toFixed(4) ?? '0.0000'} (R² ${lastStep?.metrics.trainR2 !== undefined ? lastStep.metrics.trainR2.toFixed(3) : '1.000'}) with OOB MSE ${state.oobError.toFixed(4)}`
      : `${acc}% train accuracy with ${oobAcc}% Out-of-Bag (OOB) validation accuracy`;

    return {
      summary: `Random Forest ensemble of B = ${treeCount} trees achieves ${perfSummary}.`,
      coreAssumption:
        'Variance reduction via bagging & subspace decorrelation: averaging B weakly correlated trees shrinks variance by Var(Tree)/B while preserving low bias.',
      whenItFails:
        'When the number of relevant features is exceedingly sparse (m_try misses the only informative feature) or base trees overfit on heavy label noise.',
      microLog: [
        `Forest Step: Built tree #${treeCount} using bootstrap with replacement (N = ${state.bootstrapIndices[0]?.length ?? 0})`,
        `Random feature subspace: m_try = ${params.mTry ?? '1-feature-only'} | Tree Max Depth = ${params.maxDepth ?? 5}`,
        isRegression
          ? `Current Out-of-Bag MSE: ${state.oobError.toFixed(4)} | Inspected: ${params.selectedTree ?? -1 === -1 ? 'Ensemble Average' : `Tree #${params.selectedTree}`}`
          : `Current Out-of-Bag Error: ${(state.oobError * 100).toFixed(1)}% | Inspected: ${params.selectedTree ?? -1 === -1 ? 'Ensemble Combined' : `Tree #${params.selectedTree}`}`,
      ],
      mathLatex: {
        objective: isRegression
          ? '\\hat{y}(\\mathbf{x}) = \\frac{1}{B} \\sum_{b=1}^B T_b(\\mathbf{x})'
          : '\\hat{y}(\\mathbf{x}) = \\operatorname{mode} \\left\\{ T_b(\\mathbf{x}) \\right\\}_{b=1}^B',
        lossOrUpdate: '\\text{Var}(\\bar{T}) = \\rho \\sigma^2 + \\frac{1 - \\rho}{B} \\sigma^2, \\quad \\rho = \\text{pairwise correlation}',
        dualOrConstraint: isRegression
          ? '\\text{OOB-MSE} = \\frac{1}{N} \\sum_{i=1}^N \\left( y_i - \\frac{1}{|B_i|} \\sum_{b \\in B_i} T_b(\\mathbf{x}_i) \\right)^2'
          : '\\text{OOB} = \\frac{1}{N} \\sum_{i=1}^N \\mathbb{I}\\left(y_i \\neq \\hat{y}_{\\text{OOB}}(\\mathbf{x}_i)\\right)',
      },
      codeSnippet: {
        sklearn: isRegression
          ? `from sklearn.ensemble import RandomForestRegressor\n\nmodel = RandomForestRegressor(n_estimators=${params.nEstimators ?? 20}, max_depth=${params.maxDepth ?? 5}, max_features='${params.mTry === '1-feature-only' ? 1 : 'sqrt'}', oob_score=True)\nmodel.fit(X_train, y_train)\nprint(f"OOB R2: {model.oob_score_:.4f}")`
          : `from sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier(n_estimators=${params.nEstimators ?? 20}, max_depth=${params.maxDepth ?? 5}, max_features='${params.mTry === '1-feature-only' ? 1 : 'sqrt'}', oob_score=True)\nmodel.fit(X_train, y_train)\nprint(f"OOB Score: {model.oob_score_:.4f}")`,
        pytorch: `# Forest Inference in PyTorch\nimport torch\n\ndef forest_predict(x, all_tree_predictions):\n    # all_tree_predictions: [B, N_samples]\n    return torch.mean(all_tree_predictions, dim=0)`,
      },
    };
  },
};
