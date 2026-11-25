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
  ImpurityCriterion,
  computeImpurity,
  findBestSplitForPoints,
  splitNode,
  predictTree,
  predictTreeSoft,
  predictTreePositiveProb,
  flattenTree,
  extractSplits,
  extractLeaves,
  pruneCostComplexity,
  serializeTreeToFloat32,
  deserializeTreeFromFloat32,
  computeBoundingBox,
  SplitCandidate,
  isRegressionDataset,
  getAllowedFeatures,
} from './treeUtils';

export interface DecisionTreeParams {
  maxDepth?: number;
  criterion?: ImpurityCriterion;
  minSamplesSplit?: number;
  ccpAlpha?: number;
  task?: 'classification' | 'regression';
  threshold?: number;
}

export interface DecisionTreeState {
  root: TreeNode;
  epoch: number;
  nodeIdCounter: number;
  leafPointMap: Map<number, number[]>; // leafId -> sample point indices
  lastCandidateSplit?: {
    featureIndex: 0 | 1;
    threshold: number;
    gain: number;
    criterion: string;
  };
  converged: boolean;
  isRegression?: boolean;
}

export function computeTreeLoss(
  root: TreeNode,
  dataset: DatasetContainer,
  isRegression: boolean,
  threshold: number = 0.5
): TelemetryMetrics {
  const train = dataset.train;
  const test = dataset.test;

  if (train.length === 0) {
    return { trainLoss: 0, testLoss: 0 };
  }

  if (isRegression) {
    let trainSE = 0;
    for (let i = 0; i < train.length; i++) {
      const p = train[i];
      const yHat = predictTree(root, p.x1, p.x2);
      const err = p.y - yHat;
      trainSE += err * err;
    }
    const trainMSE = trainSE / train.length;

    let testMSE = 0;
    if (test.length > 0) {
      let testSE = 0;
      for (let i = 0; i < test.length; i++) {
        const p = test[i];
        const yHat = predictTree(root, p.x1, p.x2);
        const err = p.y - yHat;
        testSE += err * err;
      }
      testMSE = testSE / test.length;
    }

    // Variance for R2
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
    };
  } else {
    // Classification 0-1 loss & accuracy
    const classifyPoint = (x1: number, x2: number): number => {
      const hard = predictTree(root, x1, x2);
      if (hard === 2) return 2;
      const prob = predictTreePositiveProb(root, x1, x2);
      return prob >= threshold ? 1 : -1;
    };

    let trainCorrect = 0;
    for (let i = 0; i < train.length; i++) {
      const p = train[i];
      const pred = classifyPoint(p.x1, p.x2);
      if (pred === p.y) {
        trainCorrect++;
      }
    }
    const trainAcc = (trainCorrect / train.length) * 100;
    const trainLoss = (train.length - trainCorrect) / train.length;

    let testAcc = 0;
    let testLoss = 0;
    if (test.length > 0) {
      let testCorrect = 0;
      for (let i = 0; i < test.length; i++) {
        const p = test[i];
        const pred = classifyPoint(p.x1, p.x2);
        if (pred === p.y) {
          testCorrect++;
        }
      }
      testAcc = (testCorrect / test.length) * 100;
      testLoss = (test.length - testCorrect) / test.length;
    }

    return {
      trainLoss,
      testLoss,
      trainAccuracy: trainAcc,
      testAccuracy: testAcc,
    };
  }
}

export const decisionTreePlugin: AlgorithmPlugin<DecisionTreeState, DecisionTreeParams> = {
  id: 'decision-tree',
  name: 'Decision Tree (CART Classifier & Regressor)',
  shortCode: 'CART Tree',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Recursive greedy binary feature split with exact Gini/Entropy impurity reduction and cost-complexity pruning.',
  paperReference: 'Breiman, Friedman, Olshen & Stone, Classification and Regression Trees (CART), 1984',

  primaryParams: [
    {
      id: 'maxDepth',
      name: 'Max Depth',
      type: 'slider',
      min: 1,
      max: 10,
      step: 1,
      default: 4,
      description: 'Maximum depth limit of the binary tree to prevent high-variance memorization.',
    },
    {
      id: 'criterion',
      name: 'Split Criterion',
      type: 'select',
      options: [
        { label: 'Gini Impurity (1 - sum p²)', value: 'gini' },
        { label: 'Information Gain (Entropy)', value: 'entropy' },
        { label: 'Mean Squared Error (MSE)', value: 'mse' },
      ],
      default: 'gini',
      description: 'Mathematical objective measuring node impurity reduction.',
    },
    {
      id: 'minSamplesSplit',
      name: 'Min Samples Split',
      type: 'slider',
      min: 2,
      max: 20,
      step: 1,
      default: 2,
      description: 'Minimum number of training samples required to evaluate an internal node split.',
    },
    {
      id: 'ccpAlpha',
      name: 'Pruning (α_ccp)',
      type: 'slider',
      min: 0.0,
      max: 0.15,
      step: 0.005,
      default: 0.0,
      description: 'Minimal cost-complexity pruning parameter penalizing excessive leaf count.',
    },
  ],

  advancedParams: [
    {
      id: 'task',
      name: 'Task Type',
      type: 'select',
      options: [
        { label: 'Classification', value: 'classification' },
        { label: 'Regression', value: 'regression' },
      ],
      default: 'classification',
      description: 'Toggle between discrete majority voting and continuous leaf mean prediction.',
    },
  ],

  symbolicLinks: [
    {
      id: 'gini',
      symbolLatex: 'I_G(t) = 1 - \\sum_{k=1}^K p_k^2',
      label: 'Gini Impurity',
      description: 'Expected error rate if a random sample is labeled according to the class distribution.',
      highlightTarget: 'decision-hyperplane',
    },
    {
      id: 'delta-gain',
      symbolLatex: '\\Delta I = I(t) - \\frac{N_L}{N} I(t_L) - \\frac{N_R}{N} I(t_R)',
      label: 'Split Information Gain',
      description: 'Exact impurity reduction achieved by partitioning feature space along axis threshold t.',
      highlightTarget: 'normal-vector',
    },
    {
      id: 'prune-cost',
      symbolLatex: 'R_\\alpha(T) = R(T) + \\alpha |T|',
      label: 'Cost-Complexity Pruning',
      description: 'Penalized tree risk trading empirical loss against structural complexity.',
      highlightTarget: 'weights-inset',
    },
  ],

  init(dataset: DatasetContainer, params: DecisionTreeParams, _seed: number): DecisionTreeState {
    const isRegression = isRegressionDataset(dataset, params);
    const criterion: ImpurityCriterion = isRegression ? 'mse' : (params.criterion || 'gini');
    const targets = dataset.train.map((p) => p.y);
    const parent = computeImpurity(targets, criterion);

    let idCounter = 1;
    const root: TreeNode = {
      id: idCounter++,
      depth: 0,
      isLeaf: true,
      samples: dataset.train.length,
      impurity: parent.impurity,
      value: parent.value,
      classCounts: parent.classCounts,
      region: computeBoundingBox(dataset.train),
    };

    const leafPointMap = new Map<number, number[]>();
    leafPointMap.set(root.id, dataset.train.map((_, i) => i));

    return {
      root,
      epoch: 0,
      nodeIdCounter: idCounter,
      leafPointMap,
      converged: false,
      isRegression,
    };
  },

  step(state: DecisionTreeState, dataset: DatasetContainer, params: DecisionTreeParams): StepResult<DecisionTreeState> {
    const isRegression = state.isRegression ?? isRegressionDataset(dataset, params);
    const criterion: ImpurityCriterion = isRegression ? 'mse' : (params.criterion || 'gini');
    const maxDepth = Number(params.maxDepth ?? 4);
    const minSamplesSplit = Number(params.minSamplesSplit ?? 2);
    const ccpAlpha = Number(params.ccpAlpha ?? 0);

    const train = dataset.train;
    let bestGain = -Infinity;
    let targetLeaf: TreeNode | null = null;
    let bestSplit: SplitCandidate | null = null;

    const allowedFeatures = getAllowedFeatures(dataset, isRegression);
    const threshold = Number(params.threshold ?? 0.5);

    // Find all leaves that can be split
    const leaves = extractLeaves(state.root);
    for (const lf of leaves) {
      // Find pointer to actual node
      let leafNode: TreeNode | null = null;
      function findNode(n: TreeNode) {
        if (n.id === lf.id) leafNode = n;
        else {
          if (n.left) findNode(n.left);
          if (n.right) findNode(n.right);
        }
      }
      findNode(state.root);
      if (!leafNode) continue;
      const ln = leafNode as TreeNode;

      if (ln.depth >= maxDepth) continue;
      const pointIndices = state.leafPointMap.get(ln.id) || [];
      if (pointIndices.length < minSamplesSplit) continue;

      const split = findBestSplitForPoints(train, pointIndices, criterion, minSamplesSplit, allowedFeatures);
      if (split && split.gain > bestGain) {
        bestGain = split.gain;
        targetLeaf = ln;
        bestSplit = split;
      }
    }

    if (!targetLeaf || !bestSplit) {
      const metrics = computeTreeLoss(state.root, dataset, isRegression, threshold);
      const nextEpoch = state.converged ? state.epoch : state.epoch + 1;
      return {
        state: { ...state, epoch: nextEpoch, converged: true },
        epoch: nextEpoch,
        metrics,
        converged: true,
      };
    }

    // Perform split on target leaf
    let nextId = state.nodeIdCounter;
    const { left, right } = splitNode(targetLeaf, bestSplit, () => nextId++);

    state.leafPointMap.delete(targetLeaf.id);
    state.leafPointMap.set(left.id, bestSplit.leftIndices);
    state.leafPointMap.set(right.id, bestSplit.rightIndices);

    // Apply pruning if enabled
    if (ccpAlpha > 0) {
      pruneCostComplexity(state.root, ccpAlpha);
    }

    const nextEpoch = state.epoch + 1;
    const metrics = computeTreeLoss(state.root, dataset, isRegression, threshold);

    const nextState: DecisionTreeState = {
      root: state.root,
      epoch: nextEpoch,
      nodeIdCounter: nextId,
      leafPointMap: state.leafPointMap,
      lastCandidateSplit: {
        featureIndex: bestSplit.featureIndex,
        threshold: bestSplit.threshold,
        gain: bestSplit.gain,
        criterion: criterion.toUpperCase(),
      },
      converged: false,
      isRegression,
    };

    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
      converged: false,
    };
  },

  predict(input: Float32Array, state: DecisionTreeState, params: DecisionTreeParams): number {
    if (state.epoch === 0) return 0;
    const isRegression = state.isRegression ?? (params.task === 'regression' || params.criterion === 'mse');
    if (isRegression) {
      return predictTree(state.root, input[0], input[1]);
    }
    if (params.threshold !== undefined) {
      const hard = predictTree(state.root, input[0], input[1]);
      if (hard === 2) return 2;
      const prob = predictTreePositiveProb(state.root, input[0], input[1]);
      return (prob - Number(params.threshold)) * 2;
    }
    return predictTreeSoft(state.root, input[0], input[1]);
  },

  loss(state: DecisionTreeState, dataset: DatasetContainer, params: DecisionTreeParams): TelemetryMetrics {
    const isRegression = state.isRegression ?? isRegressionDataset(dataset, params);
    const threshold = Number(params.threshold ?? 0.5);
    return computeTreeLoss(state.root, dataset, isRegression, threshold);
  },

  getVisualArtifacts(
    state: DecisionTreeState,
    _dataset: DatasetContainer,
    _params: DecisionTreeParams
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

    const splits = extractSplits(state.root);
    const leaves = extractLeaves(state.root);
    const allNodes = flattenTree(state.root);

    return [
      {
        type: 'tree-partition',
        root: allNodes[0],
        allNodes,
        splits,
        leaves,
        totalTreesCount: 1,
        candidateSplit: state.lastCandidateSplit,
      },
    ];
  },

  serializeSnapshot(state: DecisionTreeState): Float32Array {
    return serializeTreeToFloat32(state.root);
  },

  restoreSnapshot(buffer: Float32Array, state: DecisionTreeState): DecisionTreeState {
    const restoredRoot = deserializeTreeFromFloat32(buffer);
    return {
      ...state,
      root: restoredRoot,
      epoch: Math.max(0, extractSplits(restoredRoot).length),
    };
  },

  explain(
    state: DecisionTreeState,
    params: DecisionTreeParams,
    lastStep?: StepResult<DecisionTreeState>
  ): ExplanationPayload {
    const splits = extractSplits(state.root);
    const leaves = extractLeaves(state.root);
    const isReg = state.isRegression ?? false;
    const criterion = isReg ? 'MSE' : (params.criterion || 'gini').toUpperCase();
    const acc = isReg
      ? `MSE ${lastStep?.metrics.trainLoss.toFixed(4) ?? '0.0000'} (R² ${lastStep?.metrics.trainR2 !== undefined ? lastStep.metrics.trainR2.toFixed(3) : '—'})`
      : `${lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0'}%`;

    return {
      summary: `CART Decision Tree built ${splits.length} axis-aligned orthogonal partitions with ${leaves.length} leaves, currently achieving ${acc} performance under ${criterion} criterion.`,
      coreAssumption:
        'Hierarchical piecewise constant partition: the underlying function or decision space can be decomposed into axis-parallel hyper-rectangles.',
      whenItFails:
        'Highly sensitive to small data perturbations (high variance) and struggling with diagonal or rotated decision boundaries, causing staircasing artifacts.',
      microLog: [
        `Step [${state.epoch}] - Evaluated orthogonal splits along features X₁ and X₂`,
        state.lastCandidateSplit
          ? `Selected split X${state.lastCandidateSplit.featureIndex + 1} < ${state.lastCandidateSplit.threshold.toFixed(2)} with ΔGain = ${state.lastCandidateSplit.gain.toFixed(4)}`
          : `Tree initialized or converged at depth limit`,
        `Current leaf count: ${leaves.length} partitions | Active Pruning: α = ${params.ccpAlpha ?? 0}`,
      ],
      mathLatex: {
        objective: isReg
          ? '\\text{MSE}(t) = \\frac{1}{N_t}\\sum_{i \\in t} (y_i - \\bar{y}_t)^2'
          : 'I_G(t) = 1 - \\sum_{k=1}^K p_k^2, \\quad I_E(t) = -\\sum_{k=1}^K p_k \\log_2 p_k',
        lossOrUpdate: isReg
          ? '\\Delta \\text{MSE} = \\text{MSE}(t) - \\frac{N_L}{N}\\text{MSE}(t_L) - \\frac{N_R}{N}\\text{MSE}(t_R)'
          : '\\Delta I(s, t) = I(t) - \\frac{N_L}{N} I(t_L) - \\frac{N_R}{N} I(t_R)',
        dualOrConstraint: 'R_\\alpha(T) = R(T) + \\alpha |T|, \\quad \\alpha \\in [0, 0.15]',
      },
      codeSnippet: {
        sklearn: isReg
          ? `from sklearn.tree import DecisionTreeRegressor\n\nmodel = DecisionTreeRegressor(max_depth=${params.maxDepth ?? 4}, criterion='squared_error', min_samples_split=${params.minSamplesSplit ?? 2}, ccp_alpha=${params.ccpAlpha ?? 0.0})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`
          : `from sklearn.tree import DecisionTreeClassifier\n\nmodel = DecisionTreeClassifier(max_depth=${params.maxDepth ?? 4}, criterion='${params.criterion ?? 'gini'}', min_samples_split=${params.minSamplesSplit ?? 2}, ccp_alpha=${params.ccpAlpha ?? 0.0})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`,
        pytorch: `# Tree-structured routing inference\nimport torch\n\ndef tree_forward(x, thresholds, feature_idx, left_children, right_children, leaf_values):\n    node = 0\n    while left_children[node] != -1:\n        feat = feature_idx[node]\n        node = left_children[node] if x[feat] <= thresholds[node] else right_children[node]\n    return leaf_values[node]`,
      },
    };
  },
};
