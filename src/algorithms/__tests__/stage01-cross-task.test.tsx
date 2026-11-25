import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  binarizeTarget,
  binarizeTarget01,
  isContinuousRegressionTargets,
  isMultiClassClassificationDataset,
} from '../stage01/crossTask';
import { logisticRegressionPlugin } from '../stage01/logisticRegression';
import { perceptronPlugin } from '../stage01/perceptron';
import { knnPlugin, knnPredictWithDataset } from '../stage01/knn';
import { naiveBayesPlugin } from '../stage01/naiveBayes';
import { linearSvmPlugin } from '../stage01/linearSvm';
import { kernelSvmPlugin } from '../stage01/kernelSvm';
import {
  generateTabular2dDataset,
  CLASSIFICATION_DATASETS,
  REGRESSION_DATASETS,
} from '../../datasets/tabular2d';
import { DatasetContainer } from '../../types/algorithm';
import { TelemetrySidebar } from '../../ui/TelemetrySidebar';
import { InteractiveStageContainer } from '../../ui/InteractiveStageContainer';
import { Stage01Canvas } from '../../rendering/Stage01Canvas';
import { DirtyManager } from '../../rendering/dirty-manager';
import { HistoryRingBuffer } from '../../utils/ringBuffer';

function makeSyntheticContinuousDataset(): DatasetContainer {
  const train = [
    // Negative targets [-2, -0.25]
    { id: 1, x1: -2.0, x2: -2.0, y: -2.0 },
    { id: 2, x1: -1.6, x2: -1.8, y: -1.7 },
    { id: 3, x1: -1.4, x2: -1.2, y: -1.3 },
    { id: 4, x1: -1.0, x2: -1.1, y: -1.05 },
    { id: 5, x1: -0.8, x2: -0.7, y: -0.75 },
    { id: 6, x1: -0.5, x2: -0.6, y: -0.55 },
    { id: 7, x1: -0.3, x2: -0.2, y: -0.25 },
    { id: 8, x1: -1.8, x2: -1.5, y: -1.65 },
    // Exact zero boundary sample (must binarize to +1)
    { id: 9, x1: 0.0, x2: 0.0, y: 0.0 },
    // Positive targets [+0.25, +2.0] including exact +2.0
    { id: 10, x1: 0.3, x2: 0.2, y: 0.25 },
    { id: 11, x1: 0.6, x2: 0.5, y: 0.55 },
    { id: 12, x1: 0.9, x2: 0.8, y: 0.85 },
    { id: 13, x1: 1.1, x2: 1.2, y: 1.15 },
    { id: 14, x1: 1.5, x2: 1.4, y: 1.45 },
    { id: 15, x1: 1.7, x2: 1.8, y: 1.75 },
    { id: 16, x1: 2.0, x2: 2.0, y: 2.0 },
  ];

  const test = [
    { id: 101, x1: -1.5, x2: -1.4, y: -1.45 },
    { id: 102, x1: -0.6, x2: -0.5, y: -0.55 },
    { id: 103, x1: 0.0, x2: 0.0, y: 0.0 },
    { id: 104, x1: 0.7, x2: 0.8, y: 0.75 },
    { id: 105, x1: 1.8, x2: 1.9, y: 1.85 },
  ];

  return { train, test };
}

describe('Stage 01 Cross-Task Mode & Dual-Mode k-NN', () => {
  describe('Shared Cross-Task target helpers', () => {
    it('binarizes continuous targets cleanly around threshold zero including exact y = 0', () => {
      expect(binarizeTarget(-2.0)).toBe(-1);
      expect(binarizeTarget(-0.0001)).toBe(-1);
      expect(binarizeTarget(0)).toBe(1);
      expect(binarizeTarget(-0)).toBe(1);
      expect(binarizeTarget(0.0001)).toBe(1);
      expect(binarizeTarget(1)).toBe(1);
      expect(binarizeTarget(2)).toBe(1);

      expect(binarizeTarget01(-2.0)).toBe(0);
      expect(binarizeTarget01(-0.0001)).toBe(0);
      expect(binarizeTarget01(0)).toBe(1);
      expect(binarizeTarget01(2.0)).toBe(1);
    });

    it('distinguishes continuous regression datasets from discrete 2-class and 3-class datasets', () => {
      const continuous = makeSyntheticContinuousDataset();
      expect(isContinuousRegressionTargets(continuous)).toBe(true);
      // Even though synthetic continuous data contains y = 2.0, it must NOT enter 3-class softmax mode
      expect(isMultiClassClassificationDataset(continuous)).toBe(false);

      const binary = generateTabular2dDataset({
        kind: 'linear-blobs',
        sampleCount: 40,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });
      expect(isContinuousRegressionTargets(binary)).toBe(false);
      expect(isMultiClassClassificationDataset(binary)).toBe(false);

      const threeClass = generateTabular2dDataset({
        kind: 'linear-blobs',
        sampleCount: 60,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 3,
        scaling: 'raw',
        stretchX2: false,
      });
      expect(isContinuousRegressionTargets(threeClass)).toBe(false);
      expect(isMultiClassClassificationDataset(threeClass)).toBe(true);
    });
  });

  describe('Synthetic continuous targets spanning [-2, +2]', () => {
    it('Logistic Regression populates both +1 and -1 classes and achieves > 70% accuracy', () => {
      const dataset = makeSyntheticContinuousDataset();
      const params = { learningRate: 0.15, cPenalty: 1.0, threshold: 0.5 };
      const initState = logisticRegressionPlugin.init(dataset, params, 42);

      expect(initState.isMultiClass).toBe(false);

      let lastRes = logisticRegressionPlugin.step(initState, dataset, params);
      for (let i = 1; i < 30; i++) {
        lastRes = logisticRegressionPlugin.step(lastRes.state, dataset, params);
      }

      const preds = dataset.train.map((p) =>
        Number(logisticRegressionPlugin.predict(new Float32Array([p.x1, p.x2]), lastRes.state, params)) >= 0 ? 1 : -1
      );
      expect(preds).toContain(1);
      expect(preds).toContain(-1);
      expect(Math.abs(lastRes.state.bias)).toBeLessThan(2.0);
      expect(lastRes.metrics.trainAccuracy).toBeGreaterThan(70);
      expect(lastRes.metrics.testAccuracy).toBeGreaterThan(70);
      expect(lastRes.metrics.trainLoss).toBeLessThan(0.5);
    });

    it('Rosenblatt Perceptron separates [-2, +2] trends without single-class collapse', () => {
      const dataset = makeSyntheticContinuousDataset();
      const params = { learningRate: 0.1, featureMapping: 'linear' as const };
      const initState = perceptronPlugin.init(dataset, params, 42);

      let lastRes = perceptronPlugin.step(initState, dataset, params);
      for (let i = 1; i < 30; i++) {
        lastRes = perceptronPlugin.step(lastRes.state, dataset, params);
      }

      const preds = dataset.train.map((p) =>
        Number(perceptronPlugin.predict(new Float32Array([p.x1, p.x2]), lastRes.state, params)) >= 0 ? 1 : -1
      );
      expect(preds).toContain(1);
      expect(preds).toContain(-1);
      expect(lastRes.metrics.trainAccuracy).toBeGreaterThan(70);
      expect(lastRes.metrics.testAccuracy).toBeGreaterThan(70);
      const predNeg = Number(perceptronPlugin.predict(new Float32Array([-2.0, -2.0]), lastRes.state, params));
      const predPos = Number(perceptronPlugin.predict(new Float32Array([2.0, 2.0]), lastRes.state, params));
      expect(predNeg).toBeLessThan(0);
      expect(predPos).toBeGreaterThan(0);
    });

    it('Naive Bayes populates both classes, priors sum to 1.0, and achieves > 70% accuracy', () => {
      const dataset = makeSyntheticContinuousDataset();
      const params = { variant: 'gaussian' as const, laplaceAlpha: 1.0 };
      const initState = naiveBayesPlugin.init(dataset, params, 42);
      const stepRes = naiveBayesPlugin.step(initState, dataset, params);

      const posClass = stepRes.state.classes.find((c) => c.label === 1)!;
      const negClass = stepRes.state.classes.find((c) => c.label === -1)!;

      expect(posClass.count).toBeGreaterThan(0);
      expect(negClass.count).toBeGreaterThan(0);
      expect(posClass.count + negClass.count).toBe(dataset.train.length);
      expect(posClass.prior + negClass.prior).toBeCloseTo(1.0, 6);
      expect(posClass.meanX1).not.toBeCloseTo(negClass.meanX1, 2);

      const preds = dataset.train.map((p) =>
        Number(naiveBayesPlugin.predict(new Float32Array([p.x1, p.x2]), stepRes.state, params)) >= 0 ? 1 : -1
      );
      expect(preds).toContain(1);
      expect(preds).toContain(-1);
      expect(stepRes.metrics.trainLoss).not.toBeCloseTo(Math.LN2, 2);
      expect(stepRes.metrics.trainAccuracy).toBeGreaterThan(70);
      expect(stepRes.metrics.testAccuracy).toBeGreaterThan(70);
    });

    it('k-NN performs mean regression on [-2, +2] targets with valid R², RMSE, and > 70% accuracy', () => {
      const dataset = makeSyntheticContinuousDataset();
      const params = { neighbors: 3, metric: 'euclidean' as const, weighting: 'uniform' as const };
      const initState = knnPlugin.init(dataset, params, 42);
      const stepRes = knnPlugin.step(initState, dataset, params);

      expect(stepRes.state.isRegression).toBe(true);
      expect(stepRes.metrics.trainR2).toBeDefined();
      expect(stepRes.metrics.trainR2).toBeGreaterThan(0.7);
      expect(stepRes.metrics.trainLoss).toBeLessThan(0.5);
      expect(stepRes.metrics.trainAccuracy).toBeGreaterThan(70);
      expect(stepRes.metrics.testAccuracy).toBeGreaterThan(70);

      const predLow = knnPredictWithDataset(new Float32Array([-1.8, -1.8]), stepRes.state, dataset);
      const predHigh = knnPredictWithDataset(new Float32Array([1.8, 1.8]), stepRes.state, dataset);
      expect(predLow).toBeLessThan(-0.8);
      expect(predHigh).toBeGreaterThan(0.8);

      const directPred = Number(knnPlugin.predict(new Float32Array([1.8, 1.8]), stepRes.state, params));
      expect(directPred).toBeCloseTo(predHigh, 5);
    });

    it('Linear SVM and Kernel SVM separate [-2, +2] continuous trends with > 70% accuracy', () => {
      const dataset = makeSyntheticContinuousDataset();

      const svmInit = linearSvmPlugin.init(dataset, { learningRate: 0.05, cPenalty: 1.0 }, 42);
      let svmRes = linearSvmPlugin.step(svmInit, dataset, { learningRate: 0.05, cPenalty: 1.0 });
      for (let i = 1; i < 20; i++) {
        svmRes = linearSvmPlugin.step(svmRes.state, dataset, { learningRate: 0.05, cPenalty: 1.0 });
      }
      expect(svmRes.metrics.trainAccuracy).toBeGreaterThan(70);
      expect(svmRes.metrics.testAccuracy).toBeGreaterThan(70);

      const ksvmInit = kernelSvmPlugin.init(dataset, { kernel: 'rbf', gamma: 0.5, cPenalty: 1.0 }, 42);
      let ksvmRes = kernelSvmPlugin.step(ksvmInit, dataset, { kernel: 'rbf', gamma: 0.5, cPenalty: 1.0 });
      for (let i = 1; i < 15; i++) {
        ksvmRes = kernelSvmPlugin.step(ksvmRes.state, dataset, { kernel: 'rbf', gamma: 0.5, cPenalty: 1.0 });
      }
      expect(ksvmRes.metrics.trainAccuracy).toBeGreaterThan(70);
      expect(ksvmRes.metrics.testAccuracy).toBeGreaterThan(70);
    });
  });

  describe('All 4 Stage 01 Regression datasets across all classifiers + k-NN', () => {
    for (const kind of REGRESSION_DATASETS) {
      it(`produces valid boundaries/curves and metrics on ${kind}`, () => {
        const dataset = generateTabular2dDataset({
          kind,
          sampleCount: 120,
          noise: 0.08,
          testSplit: 0.2,
          seed: 42,
          numClasses: 2,
          scaling: 'raw',
          stretchX2: false,
        });

        // 1. Logistic Regression
        const logInit = logisticRegressionPlugin.init(dataset, { learningRate: 0.15, cPenalty: 1.0 }, 42);
        let logRes = logisticRegressionPlugin.step(logInit, dataset, { learningRate: 0.15, cPenalty: 1.0 });
        for (let i = 0; i < 20; i++) {
          logRes = logisticRegressionPlugin.step(logRes.state, dataset, { learningRate: 0.15, cPenalty: 1.0 });
        }
        expect(logRes.state.isMultiClass).toBe(false);
        expect(logRes.metrics.trainAccuracy).toBeGreaterThan(75);
        expect(logRes.metrics.testAccuracy).toBeGreaterThan(70);

        // 2. Perceptron
        const perInit = perceptronPlugin.init(dataset, { learningRate: 0.1 }, 42);
        let perRes = perceptronPlugin.step(perInit, dataset, { learningRate: 0.1 });
        for (let i = 0; i < 20; i++) {
          perRes = perceptronPlugin.step(perRes.state, dataset, { learningRate: 0.1 });
        }
        expect(perRes.metrics.trainAccuracy).toBeGreaterThan(75);
        expect(perRes.metrics.testAccuracy).toBeGreaterThan(70);

        // 3. Naive Bayes
        const nbInit = naiveBayesPlugin.init(dataset, { laplaceAlpha: 1.0 }, 42);
        const nbRes = naiveBayesPlugin.step(nbInit, dataset, { laplaceAlpha: 1.0 });
        const priorSum = nbRes.state.classes.reduce((acc, c) => acc + c.prior, 0);
        expect(priorSum).toBeCloseTo(1.0, 6);
        expect(nbRes.metrics.trainAccuracy).toBeGreaterThan(75);
        expect(nbRes.metrics.testAccuracy).toBeGreaterThan(70);

        // 4. k-NN (Native 1D or 2D Regression)
        const knnInit = knnPlugin.init(dataset, { neighbors: 5, metric: 'euclidean', weighting: 'distance' }, 42);
        const knnRes = knnPlugin.step(knnInit, dataset, { neighbors: 5, metric: 'euclidean', weighting: 'distance' });
        expect(knnRes.state.isRegression).toBe(true);
        expect(knnRes.metrics.trainR2).toBeGreaterThan(0.75);
        expect(knnRes.metrics.testR2).toBeGreaterThan(0.65);
        expect(knnRes.metrics.trainAccuracy).toBeGreaterThan(75);

        const knnArts = knnPlugin.getVisualArtifacts(knnRes.state, dataset, {});
        if (kind === 'linear-surface') {
          expect(knnRes.state.is1D).toBe(false);
        } else {
          expect(knnRes.state.is1D).toBe(true);
          const curve = knnArts.find((a) => a.type === 'curve-fit') as any;
          expect(curve).toBeDefined();
          expect(curve.points.length).toBeGreaterThan(20);
          expect(knnArts.some((a) => a.type === 'residual-bars')).toBe(true);
          // Zero x2 leakage in 1D k-NN prediction
          const pA = knnPredictWithDataset(new Float32Array([1.5, 50]), knnRes.state, dataset);
          const pB = knnPredictWithDataset(new Float32Array([1.5, -50]), knnRes.state, dataset);
          expect(pA).toBeCloseTo(pB, 6);
        }
      });
    }

    it('never triggers 3-class mode when switching to Regression dataset even with numClasses: 3', () => {
      const regDataset = generateTabular2dDataset({
        kind: 'sine-wave',
        sampleCount: 80,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 3,
        scaling: 'raw',
        stretchX2: false,
      });

      expect(isContinuousRegressionTargets(regDataset)).toBe(true);
      expect(regDataset.train.every((p) => Math.abs(p.x2 - p.y) < 1e-6)).toBe(true);
    });
  });

  describe('Zero regressions on Classification datasets', () => {
    for (const kind of CLASSIFICATION_DATASETS) {
      it(`keeps classification behavior intact on ${kind}`, () => {
        const dataset = generateTabular2dDataset({
          kind,
          sampleCount: 160,
          noise: 0.05,
          testSplit: 0.2,
          seed: 42,
          numClasses: 2,
          scaling: 'raw',
          stretchX2: false,
        });

        const knnParams = kind === 'twin-spirals' ? { neighbors: 3 } : { neighbors: 5 };
        const knnInit = knnPlugin.init(dataset, knnParams, 42);
        const knnRes = knnPlugin.step(knnInit, dataset, knnParams);
        expect(knnRes.state.isRegression).toBe(false);
        expect(knnRes.metrics.trainR2).toBeUndefined();
        expect(knnRes.metrics.trainAccuracy).toBeGreaterThan(70);
        const knnArts = knnPlugin.getVisualArtifacts(knnRes.state, dataset, {});
        expect(knnArts.some((a) => a.type === 'curve-fit')).toBe(false);

        const nbInit = naiveBayesPlugin.init(dataset, {}, 42);
        const nbRes = naiveBayesPlugin.step(nbInit, dataset, {});
        expect(nbRes.state.classes[0].prior + nbRes.state.classes[1].prior).toBeCloseTo(1.0, 6);
        expect(Number.isFinite(nbRes.metrics.trainAccuracy!)).toBe(true);
      });
    }
  });

  describe('TelemetrySidebar and Stage01Canvas Cross-Task UI rendering', () => {
    let prevDoc: unknown;
    beforeAll(() => {
      prevDoc = (globalThis as { document?: unknown }).document;
      (globalThis as { document?: unknown }).document = {
        createElement: () => ({ width: 0, height: 0, getContext: () => null }),
      };
    });
    afterAll(() => {
      (globalThis as { document?: unknown }).document = prevDoc;
    });

    it('displays Accuracy (not R²) and Decision Threshold for Cross-Task classifiers, and R²/RMSE for k-NN regression', () => {
      const history = new HistoryRingBuffer(10, 4);

      for (const algo of [logisticRegressionPlugin, perceptronPlugin, naiveBayesPlugin]) {
        const html = renderToStaticMarkup(
          <TelemetrySidebar
            algorithm={algo}
            datasetTaskType="regression"
            metrics={{
              trainLoss: 0.25,
              testLoss: 0.3,
              trainAccuracy: 91.5,
              testAccuracy: 88.0,
            }}
            historyBuffer={history}
            theme="paper"
            decisionThreshold={algo.id === 'logistic-regression' ? 0.5 : 0}
            onUpdateThreshold={() => {}}
          />
        );
        expect(html).toContain('Accuracy');
        expect(html).toContain('91.5%');
        expect(html).not.toContain('R² Score');
        expect(html).toContain('Decision Threshold (τ)');
      }

      const knnHtml = renderToStaticMarkup(
        <TelemetrySidebar
          algorithm={knnPlugin}
          datasetTaskType="regression"
          metrics={{
            trainLoss: 0.16,
            testLoss: 0.2,
            trainR2: 0.912,
            testR2: 0.89,
            trainAccuracy: 94.0,
          }}
          historyBuffer={history}
          theme="paper"
          decisionThreshold={0}
          onUpdateThreshold={() => {}}
        />
      );
      expect(knnHtml).toContain('R² Score');
      expect(knnHtml).toContain('0.912');
      expect(knnHtml).toContain('RMSE');
      expect(knnHtml).toContain('0.400');
      expect(knnHtml).not.toContain('Decision Threshold (τ)');
    });

    it('renders 2D confidence legend & discretize toggle for Cross-Task classifiers on 1D datasets, and 1D curve legend for k-NN regression', () => {
      const sineDataset = generateTabular2dDataset({
        kind: 'sine-wave',
        sampleCount: 40,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });
      const noop = () => {};

      const logCanvasHtml = renderToStaticMarkup(
        <Stage01Canvas
          algorithm={logisticRegressionPlugin}
          taskType="classification"
          datasetTaskType="regression"
          theme="paper"
          trainData={sineDataset.train}
          testData={sineDataset.test}
          artifacts={[]}
          gridResolution={96}
          dirtyManager={new DirtyManager()}
          pointerMode="add-drag"
          activeClass={1}
          onAddPoint={noop}
          onUpdatePoint={noop}
          onDeletePoint={noop}
          predictValueAtPoint={() => 0}
        />
      );
      expect(logCanvasHtml).toContain('Data &amp; boundary confidence:');
      expect(logCanvasHtml).toContain('Discretize output');

      const knnCanvasHtml = renderToStaticMarkup(
        <Stage01Canvas
          algorithm={knnPlugin}
          taskType="classification"
          datasetTaskType="regression"
          theme="paper"
          trainData={sineDataset.train}
          testData={sineDataset.test}
          artifacts={[]}
          gridResolution={96}
          dirtyManager={new DirtyManager()}
          pointerMode="add-drag"
          activeClass={1}
          onAddPoint={noop}
          onUpdatePoint={noop}
          onDeletePoint={noop}
          predictValueAtPoint={() => 0}
        />
      );
      expect(knnCanvasHtml).toContain('Continuous 1D response curve');
      expect(knnCanvasHtml).not.toContain('Discretize output');

      // Pre-flight validator does NOT warn on binarizable continuous data or native k-NN regression
      const containerHtml = renderToStaticMarkup(
        <InteractiveStageContainer
          algorithm={naiveBayesPlugin}
          theme="paper"
          datasetTaskType="regression"
          isCrossTaskMismatch={true}
          crossTaskMessage="Cross-Task Mode: Classifier is binarizing continuous targets around threshold zero."
          trainData={sineDataset.train}
          testData={sineDataset.test}
          artifacts={[]}
          gridResolution={96}
          dirtyManager={new DirtyManager()}
          circuitBreakerTripped={false}
          pointerMode="add-drag"
          activeClass={1}
          onAddPoint={noop}
          onUpdatePoint={noop}
          onDeletePoint={noop}
          onClampAndRecover={noop}
          predictValueAtPoint={() => 0}
        />
      );
      expect(containerHtml).toContain('Cross-Task Mode');
      expect(containerHtml).not.toContain('Less than 2 classes present');
    });
  });
});
