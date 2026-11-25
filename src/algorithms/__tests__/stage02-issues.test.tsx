// @vitest-environment jsdom
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { DatasetContainer, TreeNodeData } from '../../types/algorithm';
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
import { Stage02Canvas } from '../../rendering/Stage02Canvas';
import { DirtyManager } from '../../rendering/dirty-manager';
import { TopControlBar } from '../../ui/TopControlBar';
import { InteractiveStageContainer } from '../../ui/InteractiveStageContainer';

// Mock HTMLCanvasElement.getContext for jsdom component tests
if (typeof HTMLCanvasElement !== 'undefined') {
  HTMLCanvasElement.prototype.getContext = (() => ({
    fillRect: () => {},
    clearRect: () => {},
    getImageData: (_x: number, _y: number, w: number, h: number) => ({
      data: new Uint8ClampedArray(w * h * 4),
    }),
    putImageData: () => {},
    createImageData: (w: number, h: number) => ({
      data: new Uint8ClampedArray(w * h * 4),
      width: w,
      height: h,
    }),
    setTransform: () => {},
    drawImage: () => {},
    save: () => {},
    restore: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    closePath: () => {},
    stroke: () => {},
    fill: () => {},
    arc: () => {},
    rect: () => {},
    clip: () => {},
    scale: () => {},
    translate: () => {},
    setLineDash: () => {},
    fillText: () => {},
    strokeRect: () => {},
  })) as any;
}

if (typeof ResizeObserver === 'undefined') {
  (globalThis as any).ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
}

describe('Stage 02 Exhaustive Bug Fixes (ISSUE-001 through ISSUE-006)', () => {
  // ISSUE-001: 1D Regression Target Leakage (X2 === y)
  it('ISSUE-001: prevents X2 target leakage across all 10 Stage 02 solvers on 1D dataset [(0,0),(1,0),(2,10),(3,10)]', () => {
    const dataset1D: DatasetContainer = {
      train: [
        { id: 1, x1: 0, x2: 0, y: 0, isTest: false },
        { id: 2, x1: 1, x2: 0, y: 0, isTest: false },
        { id: 3, x1: 2, x2: 10, y: 10, isTest: false },
        { id: 4, x1: 3, x2: 10, y: 10, isTest: false },
      ],
      test: [
        { id: 5, x1: 0.5, x2: 0, y: 0, isTest: true },
        { id: 6, x1: 2.5, x2: 10, y: 10, isTest: true },
      ],
    };

    // (b) Decision Tree splits strictly at X1 < 1.5 with ΔMSE = 25.0 and leaves ŷ=0, ŷ=10
    let dtState = decisionTreePlugin.init(dataset1D, { task: 'regression', criterion: 'mse', maxDepth: 3 }, 42);
    dtState = decisionTreePlugin.step(dtState, dataset1D, { task: 'regression', criterion: 'mse', maxDepth: 3 }).state;
    expect(dtState.root.featureIndex).toBe(0);
    expect(dtState.root.threshold).toBeCloseTo(1.5, 6);
    expect(dtState.root.gain).toBeCloseTo(25.0, 6);
    expect(dtState.root.left?.value).toBeCloseTo(0, 6);
    expect(dtState.root.right?.value).toBeCloseTo(10, 6);

    // (a) All 10 Stage 02 solvers: zero tree nodes ever split on featureIndex === 1
    const allSolvers = [
      { plugin: decisionTreePlugin, params: { task: 'regression', criterion: 'mse', maxDepth: 3 }, steps: 3 },
      { plugin: randomForestPlugin, params: { task: 'regression', nEstimators: 4, maxDepth: 3, featureSubspace: '1-feature-only' }, steps: 4 },
      { plugin: baggingPlugin, params: { task: 'regression', nEstimators: 4, maxDepth: 3 }, steps: 4 },
      { plugin: adaboostPlugin, params: { nEstimators: 4, stumpDepth: 1 }, steps: 4 },
      { plugin: gradientBoostingPlugin, params: { task: 'regression', nEstimators: 4, maxDepth: 2 }, steps: 4 },
      { plugin: xgboostPlugin, params: { nEstimators: 4, maxDepth: 2 }, steps: 4 },
      { plugin: lightgbmPlugin, params: { nEstimators: 4, numLeaves: 4, minDataInLeaf: 1 }, steps: 4 },
      { plugin: catboostPlugin, params: { nEstimators: 4, depth: 2 }, steps: 4 },
      { plugin: votingClassifierPlugin, params: { weightLinear: 1, weightTree: 1, weightKnn: 1 }, steps: 3 },
      { plugin: stackingPlugin, params: { metaReg: 1.0 }, steps: 3 },
    ];

    for (const { plugin, params, steps } of allSolvers as any[]) {
      let st = plugin.init(dataset1D, params, 42);
      for (let i = 0; i < steps; i++) {
        st = plugin.step(st, dataset1D, params).state;
      }
      const artifacts = plugin.getVisualArtifacts(st, dataset1D, params);
      const treeArt = artifacts.find((a: any) => a.type === 'tree-partition') as any;
      if (treeArt && Array.isArray(treeArt.allNodes)) {
        for (const nd of treeArt.allNodes as TreeNodeData[]) {
          if (!nd.isLeaf) {
            expect(nd.featureIndex).toBe(0);
          }
        }
      }
      if (treeArt && Array.isArray(treeArt.splits)) {
        for (const sp of treeArt.splits) {
          expect(sp.featureIndex).toBe(0);
        }
      }
    }

    // (c) Voting OLS fits y as a function of X1 alone (never y = 1.00*x2)
    let votingState = votingClassifierPlugin.init(dataset1D, {}, 42);
    votingState = votingClassifierPlugin.step(votingState, dataset1D, {}).state;
    votingState = votingClassifierPlugin.step(votingState, dataset1D, {}).state;
    votingState = votingClassifierPlugin.step(votingState, dataset1D, {}).state;
    expect(votingState.linearWeights[1]).toBe(0);
    expect(votingState.linearWeights[0]).toBeGreaterThan(0);
  });

  // ISSUE-002: Boosting Convergence at M & Signed Leaf Weights
  it('ISSUE-002: boosting algorithms converge at exactly M rounds and Stage02Canvas renders small positive weights (+0.05) without Class -1', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 40,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const M = 5;
    const boosters = [
      adaboostPlugin,
      gradientBoostingPlugin,
      xgboostPlugin,
      lightgbmPlugin,
      catboostPlugin,
    ];

    for (const plugin of boosters as any[]) {
      let state = plugin.init(dataset, { nEstimators: M }, 42);
      for (let round = 1; round <= M; round++) {
        const res = plugin.step(state, dataset, { nEstimators: M });
        state = res.state;
        expect(res.epoch).toBe(round);
        expect(res.converged).toBe(round === M);
      }
    }

    // Component check: small positive weight (+0.05) on XGBoost leaf renders w=+0.05, NOT Class -1
    const mockNodes: TreeNodeData[] = [
      {
        id: 1,
        depth: 0,
        isLeaf: false,
        featureIndex: 0,
        threshold: 0.0,
        leftId: 2,
        rightId: 3,
        samples: 40,
        impurity: 1.0,
        value: 0,
        gain: 0.5,
        region: [-6, 6, -6, 6],
      },
      {
        id: 2,
        depth: 1,
        isLeaf: true,
        samples: 20,
        impurity: 0,
        value: 0.05,
        region: [-6, 0, -6, 6],
      },
      {
        id: 3,
        depth: 1,
        isLeaf: true,
        samples: 20,
        impurity: 0,
        value: -0.12,
        region: [0, 6, -6, 6],
      },
    ];

    render(
      <Stage02Canvas
        theme="paper"
        trainData={dataset.train}
        testData={dataset.test}
        artifacts={[
          {
            type: 'tree-partition',
            root: mockNodes[0],
            allNodes: mockNodes,
            splits: [{ featureIndex: 0, threshold: 0, range: [-6, 6], depth: 0 }],
            leaves: [],
            totalTreesCount: 1,
          },
        ]}
        gridResolution={32}
        dirtyManager={new DirtyManager()}
        algorithm={xgboostPlugin}
        datasetTaskType="classification"
        pointerMode="add-drag"
        activeClass={1}
        onAddPoint={() => {}}
        onUpdatePoint={() => {}}
        onDeletePoint={() => {}}
        predictValueAtPoint={() => 0}
      />
    );

    expect(screen.getByText('w=+0.05')).toBeTruthy();
    expect(screen.getByText('w=-0.12')).toBeTruthy();
    expect(screen.queryByText('Class -1')).toBeNull();
  });

  // ISSUE-006: Voting Regressor MSE, Stacking Epoch 0, AdaBoost Split-Axis Label Consistency
  it('ISSUE-006: verifies Voting Regressor weighted-average MSE, Stacking Epoch 0 untrained state, and AdaBoost axis consistency', () => {
    const reg2d = generateTabular2dDataset({
      kind: 'linear-surface',
      sampleCount: 60,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    // 1. Voting Regressor
    let vState = votingClassifierPlugin.init(reg2d, { weightLinear: 1, weightTree: 1, weightKnn: 1 }, 42);
    for (let s = 0; s < 3; s++) {
      vState = votingClassifierPlugin.step(vState, reg2d, { weightLinear: 1, weightTree: 1, weightKnn: 1 }).state;
    }
    const vMetrics = votingClassifierPlugin.loss(vState, reg2d, { weightLinear: 1, weightTree: 1, weightKnn: 1 });
    // Ensemble MSE must differ from CART MSE alone (and be finite)
    expect(vMetrics.trainLoss).not.toBeCloseTo(vState.treeMse!, 4);
    expect(vMetrics.trainLoss).toBeLessThan(vState.treeMse! + 0.5);

    // 2. Stacking Epoch 0
    const clsDataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 40,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });
    const stackInit = stackingPlugin.init(clsDataset, {}, 42);
    const stackLoss0 = stackingPlugin.loss(stackInit, clsDataset, {});
    expect(stackLoss0.trainAccuracy).toBe(50);
    expect(stackLoss0.trainLoss).toBeCloseTo(0.693, 3);
    const stackExp0 = stackingPlugin.explain(stackInit, {});
    expect(stackExp0.summary).toContain('Epoch 0');

    // 3. AdaBoost Split-Axis Label Consistency on Concentric Circles & Twin Spirals
    for (const kind of ['concentric-circles', 'twin-spirals'] as const) {
      const ds = generateTabular2dDataset({
        kind,
        sampleCount: 80,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });
      let adaState = adaboostPlugin.init(ds, { nEstimators: 5, stumpDepth: 2 }, 42);
      for (let i = 0; i < 5; i++) {
        adaState = adaboostPlugin.step(adaState, ds, { nEstimators: 5, stumpDepth: 2 }).state;
      }
      const arts = adaboostPlugin.getVisualArtifacts(adaState, ds, { selectedTree: -1 });
      const treeArt = arts.find((a) => a.type === 'tree-partition') as Extract<typeof arts[number], { type: 'tree-partition' }>;
      const internalNodes = (treeArt.allNodes ?? []).filter((n) => !n.isLeaf);
      expect(treeArt.splits.length).toBe(internalNodes.length);
      for (let i = 0; i < internalNodes.length; i++) {
        const expectedPrefix = internalNodes[i].featureIndex === 0 ? 'X₁' : 'X₂';
        expect(treeArt.splits[i].featureIndex).toBe(internalNodes[i].featureIndex);
        expect(treeArt.splits[i].label?.startsWith(expectedPrefix)).toBe(true);
      }
    }
  });

  // ISSUE-003: Decision Tree Inspector Gain Prefix (ΔG, IG, ΔMSE)
  it('ISSUE-003: maps Decision Tree internal node gain prefix to ΔG (gini), IG (entropy), and ΔMSE (mse)', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 20,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const nodes: TreeNodeData[] = [
      {
        id: 1,
        depth: 0,
        isLeaf: false,
        featureIndex: 0,
        threshold: 0.25,
        leftId: 2,
        rightId: 3,
        samples: 20,
        impurity: 0.5,
        value: 1,
        gain: 0.42,
        region: [-6, 6, -6, 6],
      },
      { id: 2, depth: 1, isLeaf: true, samples: 10, impurity: 0, value: 1, region: [-6, 0.25, -6, 6] },
      { id: 3, depth: 1, isLeaf: true, samples: 10, impurity: 0, value: -1, region: [0.25, 6, -6, 6] },
    ];

    const renderWithCriterion = (criterion: 'gini' | 'entropy' | 'mse', taskType: 'classification' | 'regression') =>
      render(
        <Stage02Canvas
          theme="paper"
          trainData={dataset.train}
          testData={dataset.test}
          artifacts={[{ type: 'tree-partition', root: nodes[0], allNodes: nodes, splits: [], leaves: [], totalTreesCount: 1 }]}
          gridResolution={32}
          dirtyManager={new DirtyManager()}
          algorithm={decisionTreePlugin}
          datasetTaskType={taskType}
          hyperparams={{ criterion }}
          pointerMode="add-drag"
          activeClass={1}
          onAddPoint={() => {}}
          onUpdatePoint={() => {}}
          onDeletePoint={() => {}}
          predictValueAtPoint={() => 0}
        />
      );

    const r1 = renderWithCriterion('gini', 'classification');
    expect(screen.getByText(/ΔG 0\.42/)).toBeTruthy();
    r1.unmount();

    const r2 = renderWithCriterion('entropy', 'classification');
    expect(screen.getByText(/IG 0\.42/)).toBeTruthy();
    r2.unmount();

    const r3 = renderWithCriterion('mse', 'regression');
    expect(screen.getByText(/ΔMSE 0\.42/)).toBeTruthy();
    r3.unmount();
  });

  // ISSUE-004: Decision Tree & GBDT Regression Mode UI/State Sync
  it('ISSUE-004: updates TopControlBar options, SymbolicStrip formulas, and GBDT Core Summary in lockstep for Regression', () => {
    const regDataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 40,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    // 1. TopControlBar hides Gini/Entropy in Decision Tree Regression mode
    render(
      <TopControlBar
        isRunning={false}
        epoch={0}
        currentAlgorithm={decisionTreePlugin}
        datasetTaskType="regression"
        hyperparams={{ task: 'regression', criterion: 'mse' }}
        theme="paper"
        reducedMotion={false}
        onTogglePlay={() => {}}
        onReset={() => {}}
        onStep={() => {}}
        onOpenAlgorithmModal={() => {}}
        onUpdateHyperparam={() => {}}
        onToggleTheme={() => {}}
        onToggleReducedMotion={() => {}}
        onCopyShareLink={() => {}}
        onExportPng={() => {}}
      />
    );
    const critSelect = screen.getByLabelText(/Split Criterion/i) as HTMLSelectElement;
    expect(Array.from(critSelect.options).map((o) => o.value)).toEqual(['mse']);

    // 2. InteractiveStageContainer renders MSE SymbolicStrip for Decision Tree Regression
    const { container } = render(
      <InteractiveStageContainer
        algorithm={decisionTreePlugin}
        theme="paper"
        datasetTaskType="regression"
        trainData={regDataset.train}
        testData={regDataset.test}
        artifacts={[]}
        gridValues={null}
        gridResolution={32}
        dirtyManager={new DirtyManager()}
        circuitBreakerTripped={false}
        pointerMode="add-drag"
        activeClass={1}
        hyperparams={{ task: 'regression', criterion: 'mse' }}
        onUpdateHyperparam={() => {}}
        onAddPoint={() => {}}
        onUpdatePoint={() => {}}
        onDeletePoint={() => {}}
        onClampAndRecover={() => {}}
        predictValueAtPoint={() => 0}
      />
    );
    expect(container.textContent).toContain('MSE');

    // 3. GBDT Regression explain() reports MSE/R2 and squared-error residual r_i = y_i - F(x_i)
    let gbmState = gradientBoostingPlugin.init(regDataset, { task: 'regression', nEstimators: 3 }, 42);
    const gbmStep = gradientBoostingPlugin.step(gbmState, regDataset, { task: 'regression', nEstimators: 3 });
    const gbmExp = gradientBoostingPlugin.explain(gbmStep.state, { task: 'regression', nEstimators: 3 }, gbmStep);
    expect(gbmExp.summary).toContain('Train MSE');
    expect(gbmExp.summary).not.toContain('classification accuracy');
    expect(gbmExp.microLog[0]).toContain('y_i - F(x_i)');
  });

  // ISSUE-005: Decision Threshold (τ) Slider on Decision Tree
  it('ISSUE-005: Decision Tree threshold (τ) across [0.05, 0.95] reclassifies impure leaves and updates Accuracy', () => {
    // 4 points in a single leaf (maxDepth: 1, minSamplesSplit: 10 prevents splitting): 3 positive, 1 negative -> p_leaf = 0.75
    const impureDataset: DatasetContainer = {
      train: [
        { id: 1, x1: 0, x2: 0, y: 1, isTest: false },
        { id: 2, x1: 0.1, x2: 0, y: 1, isTest: false },
        { id: 3, x1: 0.2, x2: 0, y: 1, isTest: false },
        { id: 4, x1: 0.3, x2: 0, y: -1, isTest: false },
      ],
      test: [],
    };

    const dtState = decisionTreePlugin.init(impureDataset, { maxDepth: 1, minSamplesSplit: 10 }, 42);
    const stepped = decisionTreePlugin.step(dtState, impureDataset, { maxDepth: 1, minSamplesSplit: 10 }).state;

    // At tau = 0.50: p_leaf = 0.75 >= 0.50 -> predicts +1 -> 3/4 correct = 75%
    const metricsLow = decisionTreePlugin.loss(stepped, impureDataset, { threshold: 0.5 });
    expect(metricsLow.trainAccuracy).toBeCloseTo(75, 5);

    // At tau = 0.85: p_leaf = 0.75 < 0.85 -> predicts -1 -> 1/4 correct = 25%
    const metricsHigh = decisionTreePlugin.loss(stepped, impureDataset, { threshold: 0.85 });
    expect(metricsHigh.trainAccuracy).toBeCloseTo(25, 5);

    const predLow = decisionTreePlugin.predict(new Float32Array([0, 0]), stepped, { threshold: 0.5 });
    const predHigh = decisionTreePlugin.predict(new Float32Array([0, 0]), stepped, { threshold: 0.85 });
    expect(predLow).toBeGreaterThan(0);
    expect(predHigh).toBeLessThan(0);
  });
});
