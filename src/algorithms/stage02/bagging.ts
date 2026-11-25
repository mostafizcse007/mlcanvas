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

export interface BaggingParams {
  nEstimators?: number;
  sampleRatio?: number;
  maxDepth?: number;
  selectedTree?: number;
  task?: 'classification' | 'regression';
}

export interface BaggingState {
  trees: TreeNode[];
  pointSampleCounts: number[][]; // treeIndex -> array of counts for each point id
  epoch: number;
  selectedTreeIndex: number;
  seed: number;
  oobError: number;
  isRegression?: boolean;
}

function buildBaggedTree(
  train: { id: number; x1: number; x2: number; y: number }[],
  bootstrapIdxs: number[],
  maxDepth: number,
  isRegression: boolean,
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

    // Full feature set [0, 1] used in standard Bagging
    const split = findBestSplitForPoints(train, indices, criterion, 2, allowedFeatures);
    if (!split || split.gain <= 1e-6) continue;

    const { left, right } = splitNode(node, split, idGen);
    queue.push({ node: left, indices: split.leftIndices });
    queue.push({ node: right, indices: split.rightIndices });
  }

  return root;
}

export const baggingPlugin: AlgorithmPlugin<BaggingState, BaggingParams> = {
  id: 'bagging',
  name: 'Bagging (Bootstrap Aggregation)',
  shortCode: 'Bagging',
  stage: 'tree-partition',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Bootstrap Aggregating fitting unpruned deep trees to random with-replacement resamples, showcasing Out-of-Bag (OOB) isolation and sample multiplicity badges.',
  paperReference: 'Breiman, Bagging Predictors, Machine Learning, 1996',

  primaryParams: [
    {
      id: 'nEstimators',
      name: 'Estimators (B)',
      type: 'slider',
      min: 1,
      max: 50,
      step: 1,
      default: 15,
      description: 'Number of independent bootstrap base trees in the ensemble.',
    },
    {
      id: 'maxDepth',
      name: 'Max Depth',
      type: 'slider',
      min: 1,
      max: 8,
      step: 1,
      default: 6,
      description: 'Maximum depth allowed per base tree (deep trees have high variance, reduced by bagging).',
    },
    {
      id: 'sampleRatio',
      name: 'Sample Ratio',
      type: 'slider',
      min: 0.3,
      max: 1.0,
      step: 0.05,
      default: 0.8,
      description: 'Fraction of dataset drawn with replacement for each bootstrap sample.',
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
      description: 'Inspect specific tree sample badges (2x, 3x) vs Out-of-Bag (hollow dashed) points.',
    },
  ],

  symbolicLinks: [
    {
      id: 'bootstrap-prob',
      symbolLatex: 'P(i \\in S_b) = 1 - \\left(1 - \\frac{1}{N}\\right)^N \\approx 1 - \\frac{1}{e} \\approx 63.2\\%',
      label: 'Bootstrap In-Bag Probability',
      description: 'Each bootstrap sample with replacement includes ~63.2% unique instances, leaving ~36.8% as Out-of-Bag.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: BaggingParams, seed: number): BaggingState {
    const isRegression = isRegressionDataset(dataset, params);
    return {
      trees: [],
      pointSampleCounts: [],
      epoch: 0,
      selectedTreeIndex: -1,
      seed,
      oobError: 0,
      isRegression,
    };
  },

  step(state: BaggingState, dataset: DatasetContainer, params: BaggingParams): StepResult<BaggingState> {
    const isRegression = state.isRegression ?? isRegressionDataset(dataset, params);
    const targetEstimators = Number(params.nEstimators ?? 15);
    const sampleRatio = Number(params.sampleRatio ?? 0.8);
    const maxDepth = Number(params.maxDepth ?? 6);
    const selectedTree = Number(params.selectedTree ?? -1);

    let updatedTrees = state.trees;
    let updatedCounts = state.pointSampleCounts;
    let oobError = state.oobError;
    const n = dataset.train.length;

    if (state.trees.length < targetEstimators) {
      const rand = mulberry32(state.seed + state.trees.length * 827);
      const sampleSize = Math.max(10, Math.floor(n * sampleRatio));

      const boot: number[] = [];
      const counts = new Array(n).fill(0);
      for (let i = 0; i < sampleSize; i++) {
        const idx = Math.floor(rand() * n);
        boot.push(idx);
        counts[idx]++;
      }

      let idCounter = state.trees.length * 1000 + 1;
      const allowedFeatures = getAllowedFeatures(dataset, isRegression);
      const newTree = buildBaggedTree(dataset.train, boot, maxDepth, isRegression, () => idCounter++, allowedFeatures);

      updatedTrees = [...state.trees, newTree];
      updatedCounts = [...state.pointSampleCounts, counts];

      // Compute OOB Error
      let oobEval = 0;
      if (isRegression) {
        let oobSE = 0;
        let meanY = 0;
        for (let i = 0; i < n; i++) meanY += dataset.train[i].y;
        meanY = n > 0 ? meanY / n : 0;
        let ssTot = 0;

        for (let i = 0; i < n; i++) {
          const pt = dataset.train[i];
          let sumPred = 0;
          let count = 0;
          for (let t = 0; t < updatedTrees.length; t++) {
            if (updatedCounts[t][i] === 0) {
              sumPred += predictTree(updatedTrees[t], pt.x1, pt.x2);
              count++;
            }
          }
          if (count > 0) {
            oobEval++;
            const yHat = sumPred / count;
            const err = pt.y - yHat;
            oobSE += err * err;
            const d = pt.y - meanY;
            ssTot += d * d;
          }
        }
        oobError = oobEval > 0 ? oobSE / oobEval : 0;
      } else {
        let oobCorrect = 0;
        for (let i = 0; i < n; i++) {
          const pt = dataset.train[i];
          let vPos = 0, vNeg = 0, vThird = 0;
          let consulted = 0;
          for (let t = 0; t < updatedTrees.length; t++) {
            if (updatedCounts[t][i] === 0) {
              consulted++;
              const pred = predictTree(updatedTrees[t], pt.x1, pt.x2);
              if (pred === 1) vPos++;
              else if (pred === 2) vThird++;
              else vNeg++;
            }
          }
          if (consulted > 0) {
            oobEval++;
            let maj = 1;
            let maxV = vPos;
            if (vNeg > maxV) { maxV = vNeg; maj = -1; }
            if (vThird > maxV) { maj = 2; }
            if (maj === pt.y) oobCorrect++;
          }
        }
        oobError = oobEval > 0 ? (oobEval - oobCorrect) / oobEval : 0;
      }
    } else if (state.trees.length > targetEstimators) {
      updatedTrees = state.trees.slice(0, targetEstimators);
      updatedCounts = state.pointSampleCounts.slice(0, targetEstimators);
    }

    const nextEpoch = state.epoch + 1;
    const nextState: BaggingState = {
      trees: updatedTrees,
      pointSampleCounts: updatedCounts,
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

  predict(input: Float32Array, state: BaggingState, params: BaggingParams): number {
    if (!state.trees || state.trees.length === 0 || state.epoch === 0) return 0;

    const sel = params.selectedTree !== undefined ? Number(params.selectedTree) : state.selectedTreeIndex;
    if (sel >= 0 && sel < state.trees.length) {
      return predictTree(state.trees[sel], input[0], input[1]);
    }

    const isRegression = state.isRegression ?? false;
    if (isRegression) {
      let sum = 0;
      for (let i = 0; i < state.trees.length; i++) {
        sum += predictTree(state.trees[i], input[0], input[1]);
      }
      return sum / state.trees.length;
    }

    let vPos = 0, vNeg = 0, vThird = 0;
    for (let i = 0; i < state.trees.length; i++) {
      const pred = predictTree(state.trees[i], input[0], input[1]);
      if (pred === 1) vPos++;
      else if (pred === 2) vThird++;
      else vNeg++;
    }

    if (vThird > vPos && vThird > vNeg) return 2;
    return (vPos - vNeg) / (vPos + vNeg || 1);
  },

  loss(state: BaggingState, dataset: DatasetContainer, _params: BaggingParams): TelemetryMetrics {
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
          'Bagged Trees': 0,
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

      return {
        trainLoss: trainMSE,
        testLoss: testMSE,
        trainR2: r2,
        testR2: r2,
        custom: {
          'OOB MSE': Number(state.oobError.toFixed(4)),
          'Bagged Trees': state.trees.length,
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

    let teAcc = 0, teLoss = 0;
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
        'Bagged Trees': state.trees.length,
      },
    };
  },

  getVisualArtifacts(
    state: BaggingState,
    dataset: DatasetContainer,
    params: BaggingParams
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
    const tree = state.trees[inspectedIdx];

    const splits = extractSplits(tree);
    const leaves = extractLeaves(tree);
    const allNodes = flattenTree(tree);

    // Compute bagging samples for inspected tree
    const counts = state.pointSampleCounts[inspectedIdx] || [];
    const samples = dataset.train.map((p, idx) => ({
      id: p.id,
      count: counts[idx] || 0,
      isOOB: (counts[idx] || 0) === 0,
    }));

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
        type: 'bagging-samples',
        samples,
        oobError: state.oobError,
      },
    ];
  },

  serializeSnapshot(state: BaggingState): Float32Array {
    if (state.trees.length === 0) return new Float32Array(0);
    return serializeTreeToFloat32(state.trees[state.trees.length - 1]);
  },

  restoreSnapshot(buffer: Float32Array, state: BaggingState): BaggingState {
    if (!buffer || buffer.length === 0) {
      return { ...state, trees: [], pointSampleCounts: [], epoch: 0 };
    }
    const restored = deserializeTreeFromFloat32(buffer);
    return {
      ...state,
      trees: [restored],
      epoch: 1,
    };
  },

  explain(
    state: BaggingState,
    params: BaggingParams,
    lastStep?: StepResult<BaggingState>
  ): ExplanationPayload {
    const treeCount = state.trees.length;
    const isRegression = state.isRegression ?? false;
    const oobAcc = ((1 - state.oobError) * 100).toFixed(1);
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';
    const perfSummary = isRegression
      ? `MSE ${lastStep?.metrics.trainLoss.toFixed(4) ?? '0.0000'} (R² ${lastStep?.metrics.trainR2 !== undefined ? lastStep.metrics.trainR2.toFixed(3) : '1.000'}) with OOB MSE ${state.oobError.toFixed(4)}`
      : `${acc}% accuracy with ${oobAcc}% Out-of-Bag validation accuracy`;

    return {
      summary: `Bagging ensemble aggregating B = ${treeCount} full-feature decision trees achieves ${perfSummary}.`,
      coreAssumption:
        'Variance reduction on unstable high-variance base learners: bagging dramatically reduces estimator variance without inflating bias.',
      whenItFails:
        'When base models have high bias (like linear models or stumps) where averaging provides little to no error reduction.',
      microLog: [
        `Fitted base tree #${treeCount} on bootstrap sample (ratio = ${params.sampleRatio ?? 0.8})`,
        `Identified ~36.8% Out-of-Bag samples for tree #${treeCount}`,
        isRegression
          ? `Current Out-of-Bag MSE: ${state.oobError.toFixed(4)} across ${treeCount} trees`
          : `Current ensemble vote consensus across ${treeCount} trees`,
      ],
      mathLatex: {
        objective: '\\hat{f}_{\\text{bag}}(\\mathbf{x}) = \\frac{1}{B} \\sum_{b=1}^B f_b(\\mathbf{x})',
        lossOrUpdate: 'E\\left[(\\hat{f}_{\\text{bag}}(\\mathbf{x}) - f(\\mathbf{x}))^2\\right] = \\text{Bias}^2 + \\frac{1}{B} \\text{Var} + \\frac{B-1}{B} \\text{Cov}',
        dualOrConstraint: 'P(i \\notin S_b) = \\left(1 - \\frac{1}{N}\\right)^N \\to \\frac{1}{e} \\approx 36.8\\%',
      },
      codeSnippet: {
        sklearn: isRegression
          ? `from sklearn.ensemble import BaggingRegressor\nfrom sklearn.tree import DecisionTreeRegressor\n\nmodel = BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=${params.maxDepth ?? 6}), n_estimators=${params.nEstimators ?? 15}, max_samples=${params.sampleRatio ?? 0.8}, oob_score=True)\nmodel.fit(X_train, y_train)\nprint(f"OOB Score: {model.oob_score_:.4f}")`
          : `from sklearn.ensemble import BaggingClassifier\nfrom sklearn.tree import DecisionTreeClassifier\n\nmodel = BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=${params.maxDepth ?? 6}), n_estimators=${params.nEstimators ?? 15}, max_samples=${params.sampleRatio ?? 0.8}, oob_score=True)\nmodel.fit(X_train, y_train)\nprint(f"OOB Score: {model.oob_score_:.4f}")`,
        pytorch: `# Bagging Majority Consensus\nimport torch\n\ndef bagging_predict(all_preds):\n    return torch.mean(all_preds, dim=0) if is_regression else torch.mode(all_preds, dim=0).values`,
      },
    };
  },
};

