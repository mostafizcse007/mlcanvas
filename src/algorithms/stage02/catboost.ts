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
  predictTree,
  flattenTree,
  extractLeaves,
  serializeTreeToFloat32,
  deserializeTreeFromFloat32,
  isRegressionDataset,
  getAllowedFeatures,
} from './treeUtils';

export interface CatBoostParams {
  depth?: number;
  learningRate?: number;
  l2LeafReg?: number;
  categoricalMode?: 'continuous' | 'discrete-buckets';
  nEstimators?: number;
  selectedTree?: number;
}

export interface ObliviousSplit {
  depth: number;
  featureIndex: 0 | 1;
  threshold: number;
}

export interface CatBoostState {
  trees: TreeNode[];
  obliviousSplitsPerTree: ObliviousSplit[][];
  learningRate: number;
  depth: number;
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

function buildObliviousTree(
  train: { x1: number; x2: number }[],
  residuals: Float32Array,
  depth: number,
  l2Reg: number,
  idGen: () => number,
  allowedFeatures: (0 | 1)[] = [0, 1]
): { root: TreeNode; splits: ObliviousSplit[] } {
  const n = train.length;
  const splits: ObliviousSplit[] = [];

  // Bitmask leaf assignments for all samples
  // At depth d, there are 2^d leaves, sample belongs to leaf index determined by bitmask
  const sampleLeafMask = new Uint32Array(n);

  for (let d = 0; d < depth; d++) {
    const numLeaves = 1 << d;
    let bestGain = -Infinity;
    let bestFIdx: 0 | 1 = 0;
    let bestThresh = 0;

    for (const fIdx of allowedFeatures) {
      // Find candidate thresholds
      const vals = train.map((p) => fIdx === 0 ? p.x1 : p.x2);
      vals.sort((a, b) => a - b);
      const step = Math.max(1, Math.floor(n / 20));

      for (let i = step; i < n; i += step) {
        if (vals[i] === vals[i - step]) continue;
        const thresh = (vals[i] + vals[i - step]) / 2;

        // Evaluate gain across ALL 2^d current leaves simultaneously
        let totalGain = 0;
        for (let l = 0; l < numLeaves; l++) {
          let sumRLeft = 0, countLeft = 0;
          let sumRRight = 0, countRight = 0;

          for (let s = 0; s < n; s++) {
            if (sampleLeafMask[s] === l) {
              const v = fIdx === 0 ? train[s].x1 : train[s].x2;
              if (v <= thresh) {
                sumRLeft += residuals[s];
                countLeft++;
              } else {
                sumRRight += residuals[s];
                countRight++;
              }
            }
          }

          if (countLeft > 0) totalGain += (sumRLeft * sumRLeft) / (countLeft + l2Reg);
          if (countRight > 0) totalGain += (sumRRight * sumRRight) / (countRight + l2Reg);
        }

        if (totalGain > bestGain) {
          bestGain = totalGain;
          bestFIdx = fIdx;
          bestThresh = thresh;
        }
      }
    }

    splits.push({ depth: d, featureIndex: bestFIdx, threshold: bestThresh });

    // Update bitmask for next level
    for (let s = 0; s < n; s++) {
      const v = bestFIdx === 0 ? train[s].x1 : train[s].x2;
      if (v > bestThresh) {
        sampleLeafMask[s] |= (1 << d);
      }
    }
  }

  // Compute leaf values for the 2^depth leaves
  const finalLeavesCount = 1 << depth;
  const leafValues = new Float64Array(finalLeavesCount);
  const leafCounts = new Uint32Array(finalLeavesCount);
  const leafResidualSums = new Float64Array(finalLeavesCount);

  for (let s = 0; s < n; s++) {
    const mask = sampleLeafMask[s];
    leafCounts[mask]++;
    leafResidualSums[mask] += residuals[s];
  }

  for (let l = 0; l < finalLeavesCount; l++) {
    leafValues[l] = leafResidualSums[l] / (leafCounts[l] + l2Reg);
  }

  // Construct standard TreeNode binary hierarchy from the oblivious splits
  function constructNode(d: number, maskPrefix: number, region: [number, number, number, number]): TreeNode {
    if (d === depth) {
      return {
        id: idGen(),
        depth: d,
        isLeaf: true,
        samples: leafCounts[maskPrefix],
        impurity: 0,
        value: leafValues[maskPrefix],
        region,
      };
    }

    const sp = splits[d];
    const [xMin, xMax, yMin, yMax] = region;
    let lRegion: [number, number, number, number];
    let rRegion: [number, number, number, number];

    if (sp.featureIndex === 0) {
      lRegion = [xMin, sp.threshold, yMin, yMax];
      rRegion = [sp.threshold, xMax, yMin, yMax];
    } else {
      lRegion = [xMin, xMax, yMin, sp.threshold];
      rRegion = [xMin, xMax, sp.threshold, yMax];
    }

    const leftNode = constructNode(d + 1, maskPrefix, lRegion);
    const rightNode = constructNode(d + 1, maskPrefix | (1 << d), rRegion);

    return {
      id: idGen(),
      depth: d,
      isLeaf: false,
      featureIndex: sp.featureIndex,
      threshold: sp.threshold,
      samples: leftNode.samples + rightNode.samples,
      impurity: 0,
      value: (leftNode.value + rightNode.value) / 2,
      left: leftNode,
      right: rightNode,
      region,
    };
  }

  const root = constructNode(0, 0, [-6, 6, -6, 6]);
  return { root, splits };
}

export const catboostPlugin: AlgorithmPlugin<CatBoostState, CatBoostParams> = {
  id: 'catboost',
  name: 'CatBoost (Symmetric Oblivious Trees)',
  shortCode: 'CatBoost',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Oblivious symmetric decision trees sharing identical feature and split thresholds across each depth layer, producing full-span plaid grid partitions.',
  paperReference: 'Prokhorenkova et al., CatBoost: Unbiased Boosting with Categorical Features, NeurIPS, 2018',

  primaryParams: [
    {
      id: 'nEstimators',
      name: 'Rounds (M)',
      type: 'slider',
      min: 1,
      max: 60,
      step: 1,
      default: 20,
      description: 'Number of sequential oblivious trees.',
    },
    {
      id: 'depth',
      name: 'Tree Depth',
      type: 'slider',
      min: 1,
      max: 6,
      step: 1,
      default: 3,
      description: 'Symmetric depth of each oblivious tree. Yields exactly 2^depth leaves per tree.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.02,
      max: 1.0,
      step: 0.02,
      default: 0.1,
      description: 'Shrinkage rate scaling leaf values.',
    },
    {
      id: 'l2LeafReg',
      name: 'L2 Leaf Reg (λ)',
      type: 'slider',
      min: 0.5,
      max: 10.0,
      step: 0.5,
      default: 3.0,
      description: 'L2 regularization preventing extreme oblivious leaf weights.',
    },
  ],

  advancedParams: [
    {
      id: 'categoricalMode',
      name: 'Categorical Mode',
      type: 'select',
      options: [
        { label: 'Continuous Manifold', value: 'continuous' },
        { label: 'Discrete Target Buckets', value: 'discrete-buckets' },
      ],
      default: 'continuous',
      description: 'Simulate ordered target encoding statistics on discrete feature buckets.',
    },
    {
      id: 'selectedTree',
      name: 'Inspect Tree #',
      type: 'slider',
      min: -1,
      max: 60,
      step: 1,
      default: -1,
      description: '-1 for combined ensemble, or 0..M-1 for individual plaid grid tree.',
    },
  ],

  symbolicLinks: [
    {
      id: 'cb-oblivious',
      symbolLatex: '\\text{Leaf}(\\mathbf{x}) = \\sum_{d=0}^{D-1} 2^d \\cdot \\mathbb{I}\\left(x_{j_d} > t_d\\right)',
      label: 'Oblivious Bitmask Indexing',
      description: 'Symmetric oblivious tree evaluation is equivalent to a single parallel bitwise vector lookup.',
      highlightTarget: 'decision-hyperplane',
    },
    {
      id: 'cb-target-stat',
      symbolLatex: '\\hat{x}_{k}^i = \\frac{\\sum_{j < i: x_j = x_i} y_j + a \\cdot p}{\\sum_{j < i: x_j = x_i} 1 + a}',
      label: 'Ordered Target Statistics',
      description: 'Avoids target leakage by calculating categorical expectations strictly over historical permutation orders.',
      highlightTarget: 'normal-vector',
    },
  ],

  init(dataset: DatasetContainer, params: CatBoostParams, _seed: number): CatBoostState {
    const depth = Number(params.depth ?? 3);
    const lr = Number(params.learningRate ?? 0.1);
    const isReg = isRegressionDataset(dataset, params);
    const baseScore = isReg && dataset.train.length > 0
      ? dataset.train.reduce((acc, p) => acc + p.y, 0) / dataset.train.length
      : 0;

    return {
      trees: [],
      obliviousSplitsPerTree: [],
      learningRate: lr,
      depth,
      epoch: 0,
      selectedTreeIndex: -1,
      baseScore,
      isRegression: isReg,
    };
  },

  step(state: CatBoostState, dataset: DatasetContainer, params: CatBoostParams): StepResult<CatBoostState> {
    const targetEstimators = Number(params.nEstimators ?? 20);
    const depth = Number(params.depth ?? 3);
    const lr = Number(params.learningRate ?? 0.1);
    const l2Reg = Number(params.l2LeafReg ?? 3.0);
    const selectedTree = Number(params.selectedTree ?? -1);
    const isReg = state.isRegression ?? isRegressionDataset(dataset, params);
    const baseScore = state.baseScore ?? (isReg && dataset.train.length > 0
      ? dataset.train.reduce((acc, p) => acc + p.y, 0) / dataset.train.length
      : 0);

    let updatedTrees = state.trees;
    let updatedSplits = state.obliviousSplitsPerTree;

    if (state.trees.length < targetEstimators) {
      const n = dataset.train.length;
      const residuals = new Float32Array(n);

      for (let i = 0; i < n; i++) {
        const pt = dataset.train[i];
        let F = baseScore;
        for (let t = 0; t < state.trees.length; t++) {
          F += lr * predictTree(state.trees[t], pt.x1, pt.x2);
        }
        if (isReg) {
          residuals[i] = pt.y - F;
        } else {
          const p = sigmoid(F);
          const y01 = pt.y === 1 ? 1.0 : 0.0;
          residuals[i] = y01 - p;
        }
      }

      let idGen = state.trees.length * 1000 + 1;
      const allowedFeatures = getAllowedFeatures(dataset, isReg);
      const { root, splits } = buildObliviousTree(dataset.train, residuals, depth, l2Reg, () => idGen++, allowedFeatures);

      updatedTrees = [...state.trees, root];
      updatedSplits = [...state.obliviousSplitsPerTree, splits];
    } else if (state.trees.length > targetEstimators) {
      updatedTrees = state.trees.slice(0, targetEstimators);
      updatedSplits = state.obliviousSplitsPerTree.slice(0, targetEstimators);
    }

    const nextEpoch = state.epoch + 1;
    const nextState: CatBoostState = {
      trees: updatedTrees,
      obliviousSplitsPerTree: updatedSplits,
      learningRate: lr,
      depth,
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

  predict(input: Float32Array, state: CatBoostState, params: CatBoostParams): number {
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

  loss(state: CatBoostState, dataset: DatasetContainer, _params: CatBoostParams): TelemetryMetrics {
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
          'Oblivious Depth': state.depth,
          'Leaves per Tree': 1 << state.depth,
          'Fitted Trees': 0,
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
          'Oblivious Depth': state.depth,
          'Leaves per Tree': 1 << state.depth,
          'Fitted Trees': state.trees.length,
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
        'Oblivious Depth': state.depth,
        'Leaves per Tree': 1 << state.depth,
        'Fitted Trees': state.trees.length,
      },
    };
  },

  getVisualArtifacts(
    state: CatBoostState,
    _dataset: DatasetContainer,
    params: CatBoostParams
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
    const inspectedIdx = (sel >= 0 && sel < state.trees.length) ? sel : (state.trees.length - 1);
    const activeTree = state.trees[inspectedIdx];

    const leaves = extractLeaves(activeTree);
    const allNodes = flattenTree(activeTree);
    const splits = state.obliviousSplitsPerTree[inspectedIdx] || [];

    // CatBoost splits are full-span plaid gridlines across the entire canvas [-6, 6]
    const plaidSplits = splits.map((s) => ({
      featureIndex: s.featureIndex,
      threshold: s.threshold,
      range: [-6, 6] as [number, number],
      depth: s.depth,
      label: `${s.featureIndex === 0 ? 'X₁' : 'X₂'} < ${s.threshold.toFixed(2)}`,
      isPlaid: true,
    }));

    return [
      {
        type: 'tree-partition',
        root: allNodes[0],
        allNodes,
        splits: plaidSplits,
        leaves,
        totalTreesCount: state.trees.length,
        selectedTreeIndex: sel,
        treesRoots: state.trees.slice(0, 12),
      },
    ];
  },

  serializeSnapshot(state: CatBoostState): Float32Array {
    if (state.trees.length === 0) return new Float32Array(0);
    return serializeTreeToFloat32(state.trees[state.trees.length - 1]);
  },

  restoreSnapshot(buffer: Float32Array, state: CatBoostState): CatBoostState {
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
    state: CatBoostState,
    params: CatBoostParams,
    lastStep?: StepResult<CatBoostState>
  ): ExplanationPayload {
    const count = state.trees.length;
    const depth = params.depth ?? 3;
    const numLeaves = 1 << depth;
    const isReg = state.isRegression ?? false;
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const mse = lastStep?.metrics.trainLoss?.toFixed(4) ?? '0.0000';

    return {
      summary: isReg
        ? `CatBoost built M = ${count} symmetric oblivious regression trees (depth = ${depth}, ${numLeaves} leaves per tree), achieving Train MSE = ${mse} with full-span plaid partitioning.`
        : `CatBoost built M = ${count} symmetric oblivious trees (depth = ${depth}, ${numLeaves} leaves per tree), achieving ${acc}% classification accuracy with full-span plaid partitioning.`,
      coreAssumption:
        'Symmetric oblivious tree structure acts as a regularizer preventing overfitting on continuous and categorical data while allowing extremely fast SIMD bitmask inference.',
      whenItFails:
        'When the target decision boundary requires highly localized non-symmetric partitions, where global full-span splits waste capacity on uninformative sub-regions.',
      microLog: isReg
        ? [
            `Constructed symmetric oblivious tree #${count} fitting continuous residuals`,
            `Every node at depth d shares identical feature index and threshold`,
            `Partitioned 2D feature manifold into complete 2^${depth} = ${numLeaves} plaid grid cells`,
          ]
        : [
            `Constructed symmetric tree #${count} with shared split per depth layer`,
            `Every node at depth d shares identical feature index and threshold`,
            `Partitioned 2D feature manifold into complete 2^${depth} = ${numLeaves} plaid grid cells`,
          ],
      mathLatex: {
        objective: 'F_M(\\mathbf{x}) = \\sum_{m=1}^M \\eta \\cdot \\text{leaf}_{m}\\left(\\sum_{d=0}^{D-1} 2^d \\mathbb{I}(x_{j_{md}} > t_{md})\\right)',
        lossOrUpdate: 'w_l = \\frac{\\sum_{i: \\text{mask}(i)=l} r_i}{|\\{i: \\text{mask}(i)=l\\}| + \\lambda}',
        dualOrConstraint: '\\text{node}(d, l) \\text{ shares } (j_d, t_d) \\quad \\forall l \\in \\{0, \\dots, 2^d - 1\\}',
      },
      codeSnippet: {
        sklearn: isReg
          ? `from catboost import CatBoostRegressor\n\nmodel = CatBoostRegressor(iterations=${params.nEstimators ?? 20}, depth=${depth}, learning_rate=${params.learningRate ?? 0.1}, l2_leaf_reg=${params.l2LeafReg ?? 3.0}, verbose=0)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`
          : `from catboost import CatBoostClassifier\n\nmodel = CatBoostClassifier(iterations=${params.nEstimators ?? 20}, depth=${depth}, learning_rate=${params.learningRate ?? 0.1}, l2_leaf_reg=${params.l2LeafReg ?? 3.0}, verbose=0)\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`,
        pytorch: `# Oblivious Tree Bitmask Lookup in PyTorch\nimport torch\n\ndef oblivious_predict(x, splits, leaf_values):\n    mask = torch.zeros(x.shape[0], dtype=torch.int64)\n    for d, (feat, thresh) in enumerate(splits):\n        mask |= (x[:, feat] > thresh).to(torch.int64) << d\n    return leaf_values[mask]`,
      },
    };
  },
};

