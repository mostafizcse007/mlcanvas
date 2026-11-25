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

export interface LightGBMParams {
  numLeaves?: number;
  nBins?: number;
  learningRate?: number;
  minDataInLeaf?: number;
  nEstimators?: number;
  selectedTree?: number;
}

export interface LightGBMState {
  trees: TreeNode[];
  binsX1: number[];
  binsX2: number[];
  learningRate: number;
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

function computeHistogramBins(points: number[], nBins: number): number[] {
  if (points.length === 0) return [];
  let min = Infinity, max = -Infinity;
  for (const p of points) {
    if (p < min) min = p;
    if (p > max) max = p;
  }
  if (min === max) return [min];

  const binStep = (max - min) / nBins;
  const cuts: number[] = [];
  for (let b = 1; b < nBins; b++) {
    cuts.push(min + b * binStep);
  }
  return cuts;
}

function buildLeafWiseTree(
  train: { x1: number; x2: number }[],
  g: Float32Array,
  h: Float32Array,
  numLeaves: number,
  minDataInLeaf: number,
  binsX1: number[],
  binsX2: number[],
  idGen: () => number,
  allowedFeatures: (0 | 1)[] = [0, 1]
): TreeNode {
  const n = train.length;
  let gTotal = 0, hTotal = 0;
  for (let i = 0; i < n; i++) {
    gTotal += g[i];
    hTotal += h[i];
  }
  const rootWeight = -gTotal / (hTotal + 1.0);

  const root: TreeNode = {
    id: idGen(),
    depth: 0,
    isLeaf: true,
    samples: n,
    impurity: (gTotal * gTotal) / (hTotal + 1.0),
    value: rootWeight,
    region: [-6, 6, -6, 6],
  };

  interface LeafCandidate {
    node: TreeNode;
    indices: number[];
    bestSplit: SplitCandidate | null;
  }

  function findHistogramSplit(indices: number[], _node?: TreeNode): SplitCandidate | null {
    if (indices.length < 2 * minDataInLeaf) return null;

    let gNode = 0, hNode = 0;
    for (const idx of indices) {
      gNode += g[idx];
      hNode += h[idx];
    }
    const simNode = (gNode * gNode) / (hNode + 1.0);

    let bestGain = -Infinity;
    let bestSplit: SplitCandidate | null = null;

    for (const fIdx of allowedFeatures) {
      const cuts = fIdx === 0 ? binsX1 : binsX2;

      for (const cut of cuts) {
        let gL = 0, hL = 0, countL = 0;
        let gR = 0, hR = 0, countR = 0;
        const leftIdxs: number[] = [];
        const rightIdxs: number[] = [];

        for (const idx of indices) {
          const val = fIdx === 0 ? train[idx].x1 : train[idx].x2;
          if (val <= cut) {
            leftIdxs.push(idx);
            gL += g[idx];
            hL += h[idx];
            countL++;
          } else {
            rightIdxs.push(idx);
            gR += g[idx];
            hR += h[idx];
            countR++;
          }
        }

        if (countL < minDataInLeaf || countR < minDataInLeaf) continue;

        const simL = (gL * gL) / (hL + 1.0);
        const simR = (gR * gR) / (hR + 1.0);
        const gain = 0.5 * (simL + simR - simNode);

        if (gain > bestGain && gain > 1e-6) {
          bestGain = gain;
          bestSplit = {
            featureIndex: fIdx,
            threshold: cut,
            gain,
            leftIndices: leftIdxs,
            rightIndices: rightIdxs,
            leftImpurity: simL,
            rightImpurity: simR,
            leftValue: -gL / (hL + 1.0),
            rightValue: -gR / (hR + 1.0),
          };
        }
      }
    }
    return bestSplit;
  }

  const initialSplit = findHistogramSplit(train.map((_, i) => i), root);
  const activeLeaves: LeafCandidate[] = [
    { node: root, indices: train.map((_, i) => i), bestSplit: initialSplit },
  ];

  let currentLeafCount = 1;

  // Best-First Leaf-Wise Growth until numLeaves is reached
  while (currentLeafCount < numLeaves && activeLeaves.length > 0) {
    // Find leaf candidate with the maximum split gain
    let bestCandidateIdx = -1;
    let maxGain = -Infinity;

    for (let i = 0; i < activeLeaves.length; i++) {
      const cand = activeLeaves[i];
      if (cand.bestSplit && cand.bestSplit.gain > maxGain) {
        maxGain = cand.bestSplit.gain;
        bestCandidateIdx = i;
      }
    }

    if (bestCandidateIdx === -1 || maxGain <= 1e-6) break;

    const cand = activeLeaves.splice(bestCandidateIdx, 1)[0];
    const split = cand.bestSplit!;

    const { left, right } = splitNode(cand.node, split, idGen);
    currentLeafCount++; // 1 leaf replaced with 2 = net +1 leaf

    const splitL = findHistogramSplit(split.leftIndices, left);
    const splitR = findHistogramSplit(split.rightIndices, right);

    activeLeaves.push({ node: left, indices: split.leftIndices, bestSplit: splitL });
    activeLeaves.push({ node: right, indices: split.rightIndices, bestSplit: splitR });
  }

  return root;
}

export const lightgbmPlugin: AlgorithmPlugin<LightGBMState, LightGBMParams> = {
  id: 'lightgbm',
  name: 'LightGBM (Leaf-Wise Best-First Boosting)',
  shortCode: 'LightGBM',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Fast histogram-binned gradient boosting with leaf-wise (best-first) tree growth producing asymmetric deep cascading partitions.',
  paperReference: 'Ke et al., LightGBM: A Highly Efficient Gradient Boosting Decision Tree, NeurIPS, 2017',

  primaryParams: [
    {
      id: 'nEstimators',
      name: 'Rounds (M)',
      type: 'slider',
      min: 1,
      max: 60,
      step: 1,
      default: 20,
      description: 'Total number of sequential boosting rounds.',
    },
    {
      id: 'numLeaves',
      name: 'Num Leaves',
      type: 'slider',
      min: 4,
      max: 31,
      step: 1,
      default: 7,
      description: 'Maximum number of leaves per tree. Leaf-wise growth splits the single best leaf regardless of depth.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.02,
      max: 1.0,
      step: 0.02,
      default: 0.1,
      description: 'Shrinkage rate applied to leaf weights.',
    },
    {
      id: 'nBins',
      name: 'Histogram Bins',
      type: 'slider',
      min: 8,
      max: 64,
      step: 4,
      default: 16,
      description: 'Number of discrete histogram buckets used to discretize continuous features.',
    },
  ],

  advancedParams: [
    {
      id: 'minDataInLeaf',
      name: 'Min Data in Leaf',
      type: 'slider',
      min: 1,
      max: 20,
      step: 1,
      default: 5,
      description: 'Minimal sample threshold required to form a valid leaf partition.',
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
      id: 'lgb-leaf-wise',
      symbolLatex: '\\text{Leaf}^* = \\arg\\max_{\\text{leaf } l} \\left( \\Delta L_l \\right)',
      label: 'Best-First Leaf Selection',
      description: 'LightGBM greedily splits the single leaf maximizing global loss reduction, yielding asymmetric topology.',
      highlightTarget: 'decision-hyperplane',
    },
    {
      id: 'lgb-histogram',
      symbolLatex: 'H_k(b) = \\sum_{i \\in \\text{bin}(b)} h_i, \\quad G_k(b) = \\sum_{i \\in \\text{bin}(b)} g_i',
      label: 'Histogram Bin Aggregation',
      description: 'Continuous coordinates are discretized into fixed bins to accelerate split evaluation from O(N) to O(Bins).',
      highlightTarget: 'normal-vector',
    },
  ],

  init(dataset: DatasetContainer, params: LightGBMParams, _seed: number): LightGBMState {
    const nBins = Number(params.nBins ?? 16);
    const lr = Number(params.learningRate ?? 0.1);
    const isReg = isRegressionDataset(dataset, params);
    const baseScore = isReg && dataset.train.length > 0
      ? dataset.train.reduce((acc, p) => acc + p.y, 0) / dataset.train.length
      : 0;

    const x1Vals = dataset.train.map((p) => p.x1);
    const x2Vals = dataset.train.map((p) => p.x2);
    const binsX1 = computeHistogramBins(x1Vals, nBins);
    const binsX2 = computeHistogramBins(x2Vals, nBins);

    return {
      trees: [],
      binsX1,
      binsX2,
      learningRate: lr,
      epoch: 0,
      selectedTreeIndex: -1,
      baseScore,
      isRegression: isReg,
    };
  },

  step(state: LightGBMState, dataset: DatasetContainer, params: LightGBMParams): StepResult<LightGBMState> {
    const targetEstimators = Number(params.nEstimators ?? 20);
    const lr = Number(params.learningRate ?? 0.1);
    const numLeaves = Number(params.numLeaves ?? 7);
    const minDataInLeaf = Number(params.minDataInLeaf ?? 5);
    const selectedTree = Number(params.selectedTree ?? -1);
    const isReg = state.isRegression ?? isRegressionDataset(dataset, params);
    const baseScore = state.baseScore ?? (isReg && dataset.train.length > 0
      ? dataset.train.reduce((acc, p) => acc + p.y, 0) / dataset.train.length
      : 0);

    let updatedTrees = state.trees;

    // Dynamically build next tree if below target
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
      const newTree = buildLeafWiseTree(dataset.train, g, h, numLeaves, minDataInLeaf, state.binsX1, state.binsX2, () => idGen++, allowedFeatures);
      updatedTrees = [...state.trees, newTree];
    } else if (state.trees.length > targetEstimators) {
      updatedTrees = state.trees.slice(0, targetEstimators);
    }

    const nextEpoch = state.epoch + 1;
    const nextState: LightGBMState = {
      trees: updatedTrees,
      binsX1: state.binsX1,
      binsX2: state.binsX2,
      learningRate: lr,
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

  predict(input: Float32Array, state: LightGBMState, params: LightGBMParams): number {
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

  loss(state: LightGBMState, dataset: DatasetContainer, _params: LightGBMParams): TelemetryMetrics {
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
          'Leaf Budget': _params.numLeaves ?? 7,
          'Histogram Bins': state.binsX1.length,
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
        'Leaf Budget': _params.numLeaves ?? 7,
        'Histogram Bins': state.binsX1.length,
        'Fitted Trees': state.trees.length,
      },
    };
  },

  getVisualArtifacts(
    state: LightGBMState,
    _dataset: DatasetContainer,
    params: LightGBMParams
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
      {
        type: 'histogram-grid',
        binsX1: state.binsX1,
        binsX2: state.binsX2,
      },
    ];
  },

  serializeSnapshot(state: LightGBMState): Float32Array {
    if (!state.trees || state.trees.length === 0) return new Float32Array(0);
    return serializeTreeToFloat32(state.trees[state.trees.length - 1]);
  },

  restoreSnapshot(buffer: Float32Array, state: LightGBMState): LightGBMState {
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
    state: LightGBMState,
    params: LightGBMParams,
    lastStep?: StepResult<LightGBMState>
  ): ExplanationPayload {
    const count = state.trees.length;
    const leaves = params.numLeaves ?? 7;
    const bins = state.binsX1.length;
    const isReg = state.isRegression ?? false;
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const mse = lastStep?.metrics.trainLoss?.toFixed(4) ?? '0.0000';

    return {
      summary: isReg
        ? `LightGBM constructed M = ${count} asymmetric leaf-wise regression trees with ${leaves} leaves across ${bins} histogram bins, achieving Train MSE = ${mse}.`
        : `LightGBM constructed M = ${count} asymmetric leaf-wise trees with ${leaves} leaves across ${bins} histogram bins, achieving ${acc}% classification accuracy.`,
      coreAssumption:
        'Leaf-wise (best-first) growth achieves lower loss than level-wise depth-first growth under the same split budget by directly expanding the maximum loss-reduction leaf.',
      whenItFails:
        'When num_leaves is unconstrained on small datasets without adequate min_data_in_leaf, leaf-wise growth forms deep single-branch overfit fingers.',
      microLog: isReg
        ? [
            `Discretized continuous features into ${bins} quantized histogram cutpoints`,
            `Greedily split max-variance-reduction leaf up to leaf budget of ${leaves}`,
            `Formed asymmetric cascading tree topology fitting continuous residuals with η = ${params.learningRate ?? 0.1}`,
          ]
        : [
            `Discretized features into ${bins} quantized histogram cutpoints`,
            `Greedily split max-gain leaf up to leaf budget of ${leaves}`,
            `Formed asymmetric cascading tree topology with shrinkage η = ${params.learningRate ?? 0.1}`,
          ],
      mathLatex: {
        objective: '\\Delta L_{\\text{leaf}} = \\frac{1}{2} \\left[ \\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L + G_R)^2}{H_L + H_R + \\lambda} \\right]',
        lossOrUpdate: 'l^* = \\arg\\max_{l \\in \\text{leaves}} \\Delta L_l, \\quad |\\text{leaves}| \\le \\text{num\\_leaves}',
        dualOrConstraint: 'x_{ij} \\mapsto \\text{bin}(x_{ij}) \\in \\{1, \\dots, K\\}, \\quad K \\ll N',
      },
      codeSnippet: {
        sklearn: isReg
          ? `import lightgbm as lgb\n\nmodel = lgb.LGBMRegressor(num_leaves=${leaves}, max_bin=${bins}, learning_rate=${params.learningRate ?? 0.1}, min_child_samples=${params.minDataInLeaf ?? 5}, n_estimators=${params.nEstimators ?? 20})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`
          : `import lightgbm as lgb\n\nmodel = lgb.LGBMClassifier(num_leaves=${leaves}, max_bin=${bins}, learning_rate=${params.learningRate ?? 0.1}, min_child_samples=${params.minDataInLeaf ?? 5}, n_estimators=${params.nEstimators ?? 20})\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`,
        pytorch: `# Histogram bin mapping in PyTorch\nimport torch\n\ndef histogram_bin_index(x, bin_cuts):\n    return torch.bucketize(x, torch.tensor(bin_cuts))`,
      },
    };
  },
};
