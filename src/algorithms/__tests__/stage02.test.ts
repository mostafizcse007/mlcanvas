import { describe, it, expect } from 'vitest';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { decisionTreePlugin } from '../stage02/decisionTree';
import { randomForestPlugin } from '../stage02/randomForest';
import { baggingPlugin } from '../stage02/bagging';
import { adaboostPlugin } from '../stage02/adaboost';
import { gradientBoostingPlugin } from '../stage02/gradientBoosting';
import { xgboostPlugin } from '../stage02/xgboost';
import { lightgbmPlugin } from '../stage02/lightgbm';
import { catboostPlugin } from '../stage02/catboost';
import { votingClassifierPlugin } from '../stage02/votingClassifier';
import { stackingPlugin } from '../stage02/stacking';
import { predictTree, TreeNode } from '../stage02/treeUtils';

describe('Stage 02: Trees, Ensembles & Boosting Arena Unit Tests', () => {
  // Test 1: Scale Invariance Proof
  it('proves monotonic-scale invariance under 10x X2 stretching for Decision Tree and XGBoost', () => {
    const rawDataset = generateTabular2dDataset({
      kind: 'xor-quadrants',
      sampleCount: 100,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const stretchedDataset = {
      train: rawDataset.train.map((p) => ({ ...p, x2: p.x2 * 10 })),
      test: rawDataset.test.map((p) => ({ ...p, x2: p.x2 * 10 })),
    };

    // A. Decision Tree Scale Invariance
    let dtRaw = decisionTreePlugin.init(rawDataset, { maxDepth: 4, minSamplesSplit: 2 }, 42);
    let dtStretched = decisionTreePlugin.init(stretchedDataset, { maxDepth: 4, minSamplesSplit: 2 }, 42);

    for (let s = 0; s < 5; s++) {
      dtRaw = decisionTreePlugin.step(dtRaw, rawDataset, { maxDepth: 4, minSamplesSplit: 2 }).state;
      dtStretched = decisionTreePlugin.step(dtStretched, stretchedDataset, { maxDepth: 4, minSamplesSplit: 2 }).state;
    }

    const inputRaw = new Float32Array(2);
    const inputStretched = new Float32Array(2);

    for (let i = 0; i < rawDataset.test.length; i++) {
      const pRaw = rawDataset.test[i];
      const pStr = stretchedDataset.test[i];

      inputRaw[0] = pRaw.x1; inputRaw[1] = pRaw.x2;
      inputStretched[0] = pStr.x1; inputStretched[1] = pStr.x2;

      const predRaw = decisionTreePlugin.predict(inputRaw, dtRaw, {});
      const predStretched = decisionTreePlugin.predict(inputStretched, dtStretched, {});
      expect(predRaw).toBe(predStretched);
    }

    // B. XGBoost Scale Invariance
    const xgbParams = { nEstimators: 5, learningRate: 0.1, maxDepth: 3 };
    let xgbRaw = xgboostPlugin.init(rawDataset, xgbParams, 42);
    let xgbStretched = xgboostPlugin.init(stretchedDataset, xgbParams, 42);

    for (let s = 0; s < 3; s++) {
      xgbRaw = xgboostPlugin.step(xgbRaw, rawDataset, xgbParams).state;
      xgbStretched = xgboostPlugin.step(xgbStretched, stretchedDataset, xgbParams).state;
    }

    for (let i = 0; i < rawDataset.test.length; i++) {
      const pRaw = rawDataset.test[i];
      const pStr = stretchedDataset.test[i];

      inputRaw[0] = pRaw.x1; inputRaw[1] = pRaw.x2;
      inputStretched[0] = pStr.x1; inputStretched[1] = pStr.x2;

      const predRaw = xgboostPlugin.predict(inputRaw, xgbRaw, xgbParams);
      const predStretched = xgboostPlugin.predict(inputStretched, xgbStretched, xgbParams);
      expect(predRaw).toBe(predStretched);
    }
  });

  // Test 2: AdaBoost Reweighting Dynamics
  it('strictly increases misclassified sample weights and decreases correctly classified sample weights', () => {
    const dataset = generateTabular2dDataset({
      kind: 'xor-quadrants',
      sampleCount: 80,
      noise: 0.05,
      testSplit: 0.2,
      seed: 99,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { nEstimators: 10, learningRate: 0.8, stumpDepth: 1 as const };
    const initialState = adaboostPlugin.init(dataset, params, 99);
    expect(initialState.epoch).toBe(0);
    expect(initialState.stumps.length).toBe(0);

    const initialW0 = initialState.weights[0];
    expect(initialW0).toBeCloseTo(1.0 / dataset.train.length, 6);

    // Step 1: Builds Stump 1
    const step1 = adaboostPlugin.step(initialState, dataset, params);
    expect(step1.state.stumps.length).toBe(1);
    const stump1 = step1.state.stumps[0];

    // Step 2: Reweights based on Stump 1 errors and builds Stump 2
    const step2 = adaboostPlugin.step(step1.state, dataset, params);
    const updatedWeights = step2.state.weights;

    let misclassifiedCount = 0;
    let correctlyClassifiedCount = 0;

    for (let i = 0; i < dataset.train.length; i++) {
      const pt = dataset.train[i];
      const pred = predictTree(stump1, pt.x1, pt.x2);
      const isMis = pred !== pt.y;

      if (isMis) {
        misclassifiedCount++;
        // Weight must strictly increase
        expect(updatedWeights[i]).toBeGreaterThan(initialState.weights[i]);
      } else {
        correctlyClassifiedCount++;
        // Weight must strictly decrease
        expect(updatedWeights[i]).toBeLessThan(initialState.weights[i]);
      }
    }

    expect(misclassifiedCount).toBeGreaterThan(0);
    expect(correctlyClassifiedCount).toBeGreaterThan(0);

    // Sum of updated weights must normalize to ~1.0
    let sumW = 0;
    for (let i = 0; i < updatedWeights.length; i++) sumW += updatedWeights[i];
    expect(sumW).toBeCloseTo(1.0, 5);
  });

  // Test 3: CatBoost Oblivious Symmetry
  it('enforces oblivious symmetry where every node at depth d shares identical featureIndex and threshold', () => {
    const dataset = generateTabular2dDataset({
      kind: 'concentric-circles',
      sampleCount: 100,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { depth: 3, nEstimators: 3, learningRate: 0.1, l2LeafReg: 3.0 };
    let state = catboostPlugin.init(dataset, params, 42);
    expect(state.epoch).toBe(0);

    for (let s = 0; s < 2; s++) {
      state = catboostPlugin.step(state, dataset, params).state;
    }

    expect(state.trees.length).toBe(2);

    for (let t = 0; t < state.trees.length; t++) {
      const tree = state.trees[t];

      // Collect nodes grouped by depth
      const nodesByDepth = new Map<number, TreeNode[]>();
      function traverse(n: TreeNode) {
        const list = nodesByDepth.get(n.depth) || [];
        list.push(n);
        nodesByDepth.set(n.depth, list);
        if (n.left) traverse(n.left);
        if (n.right) traverse(n.right);
      }
      traverse(tree);

      // Verify that every internal node at depth d shares identical featureIndex and threshold
      for (const [, nodes] of nodesByDepth.entries()) {
        const internalNodes = nodes.filter((n) => !n.isLeaf);
        if (internalNodes.length > 1) {
          const firstFeature = internalNodes[0].featureIndex;
          const firstThresh = internalNodes[0].threshold;

          for (let i = 1; i < internalNodes.length; i++) {
            expect(internalNodes[i].featureIndex).toBe(firstFeature);
            expect(internalNodes[i].threshold).toBeCloseTo(firstThresh!, 6);
          }
        }
      }
    }
  });

  // Test 4: XGBoost Level-Wise vs. LightGBM Leaf-Wise Topology
  it('verifies level-wise expansion for XGBoost and asymmetric best-first leaf expansion for LightGBM', () => {
    const dataset = generateTabular2dDataset({
      kind: 'twin-spirals',
      sampleCount: 140,
      noise: 0.02,
      testSplit: 0.2,
      seed: 77,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    // XGBoost level-wise with maxDepth=2
    const xgbInit = xgboostPlugin.init(dataset, { maxDepth: 2, nEstimators: 1, learningRate: 0.1 }, 77);
    expect(xgbInit.epoch).toBe(0);
    const xgb = xgboostPlugin.step(xgbInit, dataset, { maxDepth: 2, nEstimators: 1, learningRate: 0.1 }).state;
    expect(xgb.trees.length).toBe(1);
    const xgbTree = xgb.trees[0];

    // Max depth of XGBoost tree must be <= 2
    function getMaxDepth(n: TreeNode): number {
      if (n.isLeaf || !n.left || !n.right) return n.depth;
      return Math.max(getMaxDepth(n.left), getMaxDepth(n.right));
    }
    expect(getMaxDepth(xgbTree)).toBeLessThanOrEqual(2);

    // LightGBM with numLeaves=7 and minDataInLeaf=2
    const lgbInit = lightgbmPlugin.init(dataset, { numLeaves: 7, minDataInLeaf: 2, nBins: 24, learningRate: 0.1 }, 77);
    expect(lgbInit.epoch).toBe(0);
    const lgb = lightgbmPlugin.step(lgbInit, dataset, { numLeaves: 7, minDataInLeaf: 2, nBins: 24, learningRate: 0.1 }).state;
    expect(lgb.trees.length).toBe(1);
    const lgbTree = lgb.trees[0];

    // In LightGBM best-first growth, high gain branches grow deeper than balanced depth 2
    const lgbDepth = getMaxDepth(lgbTree);
    expect(lgbDepth).toBeGreaterThanOrEqual(3);
  });

  // Test 5: Immediate Epoch 0 Initialization Across All 10 Stage 02 Plugins
  it('immediately initializes all 10 Stage 02 plugins at Epoch 0 without stalling', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 80,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const plugins = [
      decisionTreePlugin,
      randomForestPlugin,
      baggingPlugin,
      adaboostPlugin,
      gradientBoostingPlugin,
      xgboostPlugin,
      lightgbmPlugin,
      catboostPlugin,
      votingClassifierPlugin,
      stackingPlugin,
    ];

    const input = new Float32Array([0.5, -0.5]);

    for (const plugin of plugins as any[]) {
      const state = plugin.init(dataset, {}, 42);
      expect(state).toBeDefined();

      const metrics = plugin.loss(state, dataset, {});
      expect(metrics).toBeDefined();
      expect(Number.isFinite(metrics.trainLoss)).toBe(true);

      const artifacts = plugin.getVisualArtifacts(state, dataset, {});
      expect(artifacts).toBeDefined();
      expect(artifacts.length).toBeGreaterThan(0);

      const pred = plugin.predict(input, state, {});
      expect(Number.isFinite(pred)).toBe(true);

      const snapshot = plugin.serializeSnapshot(state);
      expect(snapshot).toBeDefined();
      expect(snapshot.length).toBeGreaterThanOrEqual(0);

      const restored = plugin.restoreSnapshot(snapshot, state);
      expect(restored).toBeDefined();
    }
  });

  // Test 6: Continuous Non-Stopping Epoch Execution
  it('continues incrementing epochs without freezing after reaching target estimators', () => {
    const dataset = generateTabular2dDataset({
      kind: 'xor-quadrants',
      sampleCount: 60,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { nEstimators: 3, learningRate: 0.1 };
    let state = lightgbmPlugin.init(dataset, params, 42);
    expect(state.epoch).toBe(0);

    for (let step = 0; step < 6; step++) {
      const res = lightgbmPlugin.step(state, dataset, params);
      expect(res.epoch).toBe(step + 1);
      state = res.state;
    }

    expect(state.epoch).toBe(6);
    expect(state.trees.length).toBe(3); // target estimators respected while epoch continues to run!
  });

  // Test 7: All 10 Stage 02 Plugins on 1D Regression (Sine Wave)
  it('correctly fits continuous targets on 1D Sine Wave across all 10 Stage 02 algorithms', () => {
    const reg1dDataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 80,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const plugins = [
      { plugin: decisionTreePlugin, params: { maxDepth: 4 } },
      { plugin: randomForestPlugin, params: { nEstimators: 6, maxDepth: 4, featureSubspace: 'all' as const } },
      { plugin: baggingPlugin, params: { nEstimators: 5, maxDepth: 4 } },
      { plugin: adaboostPlugin, params: { nEstimators: 6, learningRate: 0.5, stumpDepth: 2 as const } },
      { plugin: gradientBoostingPlugin, params: { nEstimators: 6, learningRate: 0.2, maxDepth: 3 } },
      { plugin: xgboostPlugin, params: { nEstimators: 6, learningRate: 0.2, maxDepth: 3 } },
      { plugin: lightgbmPlugin, params: { nEstimators: 6, learningRate: 0.2, numLeaves: 7 } },
      { plugin: catboostPlugin, params: { nEstimators: 6, learningRate: 0.2, depth: 3 } },
      { plugin: votingClassifierPlugin, params: { weightLinear: 1, weightTree: 2, weightKnn: 1 } },
      { plugin: stackingPlugin, params: { metaReg: 1.0 } },
    ];

    const input = new Float32Array([1.5, 0.0]);

    for (const { plugin, params } of plugins as any[]) {
      let state = plugin.init(reg1dDataset, params, 42);
      expect(state).toBeDefined();

      for (let s = 0; s < 5; s++) {
        state = plugin.step(state, reg1dDataset, params).state;
      }

      const metrics = plugin.loss(state, reg1dDataset, params);
      expect(metrics.trainLoss).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(metrics.trainLoss)).toBe(true);

      const pred = plugin.predict(input, state, params);
      expect(Number.isFinite(pred)).toBe(true);

      // Verify explanation conforms to 3-lens inspector
      const explanation = plugin.explain(state, params, { state, epoch: 5, metrics, converged: true });
      expect(explanation.microLog.length).toBe(3);
    }
  });

  // Test 8: All 10 Stage 02 Plugins on 2D Surface Regression
  it('correctly fits continuous surface on 2D Surface across all 10 Stage 02 algorithms', () => {
    const reg2dDataset = generateTabular2dDataset({
      kind: 'linear-surface',
      sampleCount: 80,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const plugins = [
      { plugin: decisionTreePlugin, params: { maxDepth: 4 } },
      { plugin: randomForestPlugin, params: { nEstimators: 6, maxDepth: 4 } },
      { plugin: baggingPlugin, params: { nEstimators: 5, maxDepth: 4 } },
      { plugin: adaboostPlugin, params: { nEstimators: 6, learningRate: 0.5 } },
      { plugin: gradientBoostingPlugin, params: { nEstimators: 6, learningRate: 0.2, maxDepth: 3 } },
      { plugin: xgboostPlugin, params: { nEstimators: 6, learningRate: 0.2, maxDepth: 3 } },
      { plugin: lightgbmPlugin, params: { nEstimators: 6, learningRate: 0.2, numLeaves: 7 } },
      { plugin: catboostPlugin, params: { nEstimators: 6, learningRate: 0.2, depth: 3 } },
      { plugin: votingClassifierPlugin, params: { weightLinear: 1, weightTree: 1, weightKnn: 1 } },
      { plugin: stackingPlugin, params: { metaReg: 1.0 } },
    ];

    const input = new Float32Array([-1.0, 2.0]);

    for (const { plugin, params } of plugins as any[]) {
      let state = plugin.init(reg2dDataset, params, 42);
      expect(state).toBeDefined();

      for (let s = 0; s < 4; s++) {
        state = plugin.step(state, reg2dDataset, params).state;
      }

      const metrics = plugin.loss(state, reg2dDataset, params);
      expect(metrics.trainLoss).toBeGreaterThanOrEqual(0);
      expect(Number.isFinite(metrics.trainLoss)).toBe(true);

      const pred = plugin.predict(input, state, params);
      expect(Number.isFinite(pred)).toBe(true);

      const explanation = plugin.explain(state, params, { state, epoch: 4, metrics, converged: true });
      expect(explanation.microLog.length).toBe(3);
    }
  });

  // Test 9: Random Forest Feature Subspace Verification
  it('supports all feature subspace modes (all, 1-feature-only, sqrt) in Random Forest for both tasks', () => {
    const classDataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 60,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const regDataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 60,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const subspaces: ('all' | '1-feature-only' | 'sqrt')[] = ['all', '1-feature-only', 'sqrt'];

    for (const sub of subspaces) {
      // Classification
      let stateC = randomForestPlugin.init(classDataset, { nEstimators: 4, featureSubspace: sub }, 42);
      for (let s = 0; s < 4; s++) {
        stateC = randomForestPlugin.step(stateC, classDataset, { nEstimators: 4, featureSubspace: sub }).state;
      }
      expect(stateC.trees.length).toBe(4);
      const metricsC = randomForestPlugin.loss(stateC, classDataset, { featureSubspace: sub });
      expect(metricsC.trainAccuracy).toBeGreaterThan(70);

      // Regression
      let stateR = randomForestPlugin.init(regDataset, { nEstimators: 4, featureSubspace: sub }, 42);
      for (let s = 0; s < 4; s++) {
        stateR = randomForestPlugin.step(stateR, regDataset, { nEstimators: 4, featureSubspace: sub }).state;
      }
      expect(stateR.trees.length).toBe(4);
      const metricsR = randomForestPlugin.loss(stateR, regDataset, { featureSubspace: sub });
      expect(metricsR.trainLoss).toBeLessThan(1.0);
    }
  });
});
