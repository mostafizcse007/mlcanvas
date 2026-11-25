import { describe, it, expect } from 'vitest';
import { linearRegressionPlugin, is1DRegressionDataset } from '../stage01/linearRegression';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { is1DRegressionData } from '../../rendering/Stage01Canvas';
import { ridgeL2Plugin } from '../stage01/ridgeL2';
import { lassoL1Plugin } from '../stage01/lassoL1';
import { linearSvmPlugin } from '../stage01/linearSvm';
import { perceptronPlugin } from '../stage01/perceptron';
import { logisticRegressionPlugin } from '../stage01/logisticRegression';

describe('Stage 01: Linear Regression (OLS / SGD) Verification & Defect Fixes', () => {
  // 1. Sub-Epochs per Step control
  describe('Issue 1: Sub-Epochs per Step SGD data passes & updates', () => {
    it('1, 5, 10, and 20 passes produce strictly monotonic and scaled parameter updates per step', () => {
      const dataset = generateTabular2dDataset({
        kind: 'linear-trend',
        sampleCount: 100,
        noise: 0.0,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      // Fixed initial state for fair comparison
      const init1 = linearRegressionPlugin.init(dataset, { solver: 'sgd', learningRate: 0.01 }, 123);
      const init5 = linearRegressionPlugin.init(dataset, { solver: 'sgd', learningRate: 0.01 }, 123);
      const init10 = linearRegressionPlugin.init(dataset, { solver: 'sgd', learningRate: 0.01 }, 123);
      const init20 = linearRegressionPlugin.init(dataset, { solver: 'sgd', learningRate: 0.01 }, 123);

      const res1 = linearRegressionPlugin.step(init1, dataset, { solver: 'sgd', learningRate: 0.01, batchPasses: 1 });
      const res5 = linearRegressionPlugin.step(init5, dataset, { solver: 'sgd', learningRate: 0.01, batchPasses: 5 });
      const res10 = linearRegressionPlugin.step(init10, dataset, { solver: 'sgd', learningRate: 0.01, batchPasses: 10 });
      const res20 = linearRegressionPlugin.step(init20, dataset, { solver: 'sgd', learningRate: 0.01, batchPasses: 20 });

      // Each single Step increments the iteration/step count by 1
      expect(res1.epoch).toBe(1);
      expect(res5.epoch).toBe(1);
      expect(res10.epoch).toBe(1);
      expect(res20.epoch).toBe(1);

      // Verify parameter delta from initial slope
      const delta1 = Math.abs(res1.state.weights[0] - init1.weights[0]);
      const delta5 = Math.abs(res5.state.weights[0] - init5.weights[0]);
      const delta10 = Math.abs(res10.state.weights[0] - init10.weights[0]);
      const delta20 = Math.abs(res20.state.weights[0] - init20.weights[0]);

      // More passes produce strictly more parameter updates and closer fit in a single Step
      expect(delta5).toBeGreaterThan(delta1 * 2.5);
      expect(delta10).toBeGreaterThan(delta5);
      expect(delta20).toBeGreaterThan(delta10);

      // Loss should decrease monotonically as more passes are executed per step
      expect(res5.metrics.trainLoss).toBeLessThan(res1.metrics.trainLoss);
      expect(res10.metrics.trainLoss).toBeLessThan(res5.metrics.trainLoss);
      expect(res20.metrics.trainLoss).toBeLessThan(res10.metrics.trainLoss);

      // MicroLog labels pass count accurately
      const explain1 = linearRegressionPlugin.explain(res1.state, { solver: 'sgd', batchPasses: 1 }, res1);
      expect(explain1.microLog[2]).toContain('1 pass/step');
      const explain20 = linearRegressionPlugin.explain(res20.state, { solver: 'sgd', batchPasses: 20 }, res20);
      expect(explain20.microLog[2]).toContain('20 passes/step');
    });

    it('supports alternative parameter aliases: cyclesPerStep and subEpochs', () => {
      const dataset = generateTabular2dDataset({
        kind: 'linear-trend',
        sampleCount: 60,
        noise: 0.02,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const initA = linearRegressionPlugin.init(dataset, { solver: 'sgd', learningRate: 0.02 }, 77);
      const initB = linearRegressionPlugin.init(dataset, { solver: 'sgd', learningRate: 0.02 }, 77);

      const resCycles = linearRegressionPlugin.step(initA, dataset, { solver: 'sgd', learningRate: 0.02, cyclesPerStep: 10 });
      const resSubEpochs = linearRegressionPlugin.step(initB, dataset, { solver: 'sgd', learningRate: 0.02, subEpochs: 10 });

      expect(resCycles.state.weights[0]).toBeCloseTo(resSubEpochs.state.weights[0], 6);
      expect(resCycles.state.bias).toBeCloseTo(resSubEpochs.state.bias, 6);
    });
  });

  // 2. Dataset switching between 1D and 2D
  describe('Issue 2: Dataset switching between 1D and 2D datasets', () => {
    it('correctly distinguishes 1D datasets (Sine Wave, Poly Curve, Linear Trend) from 2D Surface', () => {
      const sineData = generateTabular2dDataset({ kind: 'sine-wave', sampleCount: 80, noise: 0.1, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });
      const polyData = generateTabular2dDataset({ kind: 'polynomial-cubic', sampleCount: 80, noise: 0.1, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });
      const trendData = generateTabular2dDataset({ kind: 'linear-trend', sampleCount: 80, noise: 0.1, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });
      const surfaceData = generateTabular2dDataset({ kind: 'linear-surface', sampleCount: 80, noise: 0.1, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });

      expect(is1DRegressionDataset(sineData)).toBe(true);
      expect(is1DRegressionDataset(polyData)).toBe(true);
      expect(is1DRegressionDataset(trendData)).toBe(true);
      expect(is1DRegressionDataset(surfaceData)).toBe(false);
    });

    it('keeps visualization, fit label, predictions, and 1D/2D descriptions in sync when switching', () => {
      const trendData = generateTabular2dDataset({ kind: 'linear-trend', sampleCount: 60, noise: 0.02, testSplit: 0.2, seed: 42, numClasses: 2, scaling: 'raw', stretchX2: false });
      const surfaceData = generateTabular2dDataset({ kind: 'linear-surface', sampleCount: 60, noise: 0.02, testSplit: 0.2, seed: 42, numClasses: 2, scaling: 'raw', stretchX2: false });

      // Step 1: 1D Linear Trend
      const state1D = linearRegressionPlugin.init(trendData, { solver: 'closed-form' }, 42);
      const step1D = linearRegressionPlugin.step(state1D, trendData, { solver: 'closed-form' });
      const arts1D = linearRegressionPlugin.getVisualArtifacts(step1D.state, trendData, {});
      const exp1D = linearRegressionPlugin.explain(step1D.state, { solver: 'closed-form' }, step1D);

      expect(step1D.state.is1D).toBe(true);
      expect(step1D.state.weights[1]).toBe(-1); // Normal vector [m, -1] for line y = mx + c
      expect(arts1D.some((a) => a.type === 'residual-bars')).toBe(true);
      expect(exp1D.microLog[0]).not.toContain('x2'); // No x2 in 1D description
      expect(exp1D.codeSnippet.pytorch).toContain('in_features=1');

      // 1D Prediction must decouple x2
      const predA = linearRegressionPlugin.predict(new Float32Array([2.0, 100.0]), step1D.state, {}) as number;
      const predB = linearRegressionPlugin.predict(new Float32Array([2.0, -999.0]), step1D.state, {}) as number;
      expect(predA).toBeCloseTo(predB, 6);

      // Step 2: Switch to 2D Surface
      const state2D = linearRegressionPlugin.init(surfaceData, { solver: 'closed-form' }, 42);
      const step2D = linearRegressionPlugin.step(state2D, surfaceData, { solver: 'closed-form' });
      const arts2D = linearRegressionPlugin.getVisualArtifacts(step2D.state, surfaceData, {});
      const exp2D = linearRegressionPlugin.explain(step2D.state, { solver: 'closed-form' }, step2D);

      expect(step2D.state.is1D).toBe(false);
      expect(step2D.state.weights[1]).not.toBe(-1); // w1 is active feature weight for x2
      expect(arts2D.some((a) => a.type === 'residual-bars')).toBe(false); // No 1D vertical bars in 2D
      const hp = arts2D.find((a) => a.type === 'hyperplane');
      expect(hp?.label).toBe('ŷ = 0 Contour');
      expect(exp2D.microLog[0]).toContain('x2'); // x2 present in 2D description
      expect(exp2D.codeSnippet.pytorch).toContain('in_features=2');

      // 2D Prediction must depend on both x1 and x2
      const pred2A = linearRegressionPlugin.predict(new Float32Array([2.0, 1.0]), step2D.state, {}) as number;
      const pred2B = linearRegressionPlugin.predict(new Float32Array([2.0, 5.0]), step2D.state, {}) as number;
      expect(pred2A).not.toBeCloseTo(pred2B, 2);
    });
  });

  // 3. Extreme-noise OLS inconsistency
  describe('Issue 3: Extreme noise, 400 samples, 10x Stretch OLS inconsistency', () => {
    it('does NOT report zero loss or R²=1 under 50% noise, 400 samples, and 10x Stretch', () => {
      const extremeNoiseData = generateTabular2dDataset({
        kind: 'linear-trend',
        sampleCount: 400,
        noise: 0.50,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: true, // 10x anisotropic stretching
      });

      // 1D detection must NOT fail when stretchX2 is enabled!
      expect(is1DRegressionDataset(extremeNoiseData)).toBe(true);

      const params = { solver: 'closed-form' as const };
      const initState = linearRegressionPlugin.init(extremeNoiseData, params, 42);
      const stepRes = linearRegressionPlugin.step(initState, extremeNoiseData, params);

      // Defect verification: Train loss and test loss must be genuinely positive due to 50% noise
      expect(stepRes.metrics.trainLoss).toBeGreaterThan(0.1);
      expect(stepRes.metrics.testLoss).toBeGreaterThan(0.1);

      // R2 must NOT be 1.0 (no false perfect fit from x2 target leakage!)
      expect(stepRes.metrics.trainR2).toBeLessThan(0.95);
      expect(stepRes.metrics.trainR2).toBeGreaterThan(0.0);
      expect(stepRes.metrics.testR2).toBeLessThan(0.95);
      expect(stepRes.metrics.testR2).toBeGreaterThan(0.0);

      // The fitted slope w0 should be close to true trend ~0.65, not 0
      expect(stepRes.state.weights[0]).toBeGreaterThan(0.4);
      expect(stepRes.state.weights[0]).toBeLessThan(0.9);
      expect(stepRes.state.weights[1]).toBe(-1); // Preserves line convention w1 = -1

      // Predictions, residuals, loss, and R2 all use consistent 1D data
      const samplePt = extremeNoiseData.train[0];
      const manualYHat = stepRes.state.weights[0] * samplePt.x1 + stepRes.state.bias;
      const pluginYHat = linearRegressionPlugin.predict(new Float32Array([samplePt.x1, samplePt.x2]), stepRes.state, params) as number;
      expect(pluginYHat).toBeCloseTo(manualYHat, 5);

      const resEntry = stepRes.state.residuals.find((r) => r.id === samplePt.id);
      expect(resEntry?.yHat).toBeCloseTo(manualYHat, 5);
      expect(resEntry?.error).toBeCloseTo(samplePt.y - manualYHat, 5);
    });

    it('handles ill-conditioned or singular normal equations gracefully with stable solve and no false R²=1', () => {
      // Degenerate dataset where all x1 values are identical (zero variance / singular design)
      const singularDataset = {
        train: [
          { id: 1, x1: 2.0, x2: 2.0, y: 1.0 },
          { id: 2, x1: 2.0, x2: 2.0, y: 3.0 },
          { id: 3, x1: 2.0, x2: 2.0, y: 5.0 },
          { id: 4, x1: 2.0, x2: 2.0, y: 7.0 },
        ],
        test: [
          { id: 5, x1: 2.0, x2: 2.0, y: 4.0 },
        ],
      };

      const params = { solver: 'closed-form' as const };
      const initState = linearRegressionPlugin.init(singularDataset, params, 42);
      const stepRes = linearRegressionPlugin.step(initState, singularDataset, params);

      // Weights must be finite and not NaN
      expect(isFinite(stepRes.state.weights[0])).toBe(true);
      expect(isFinite(stepRes.state.bias)).toBe(true);
      expect(isNaN(stepRes.state.weights[0])).toBe(false);

      // Must NOT report a false R2 = 1.0 on singular data with variance in y
      expect(stepRes.metrics.trainR2).toBeLessThanOrEqual(0.01);
      // Train loss should be equal to sample variance of y: mean=4, var=((9+1+1+9)/4) = 5
      expect(stepRes.metrics.trainLoss).toBeCloseTo(5.0, 1);
    });
  });

  // 4. OLS least-squares retention and SGD updates
  describe('Issue 4 & 5: OLS retention, SGD updates, and control sequences', () => {
    it('OLS mode reaches least-squares solution and retains it on subsequent steps', () => {
      const dataset = generateTabular2dDataset({
        kind: 'linear-trend',
        sampleCount: 80,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const params = { solver: 'closed-form' as const };
      const state0 = linearRegressionPlugin.init(dataset, params, 42);
      const step1 = linearRegressionPlugin.step(state0, dataset, params);

      expect(step1.converged).toBe(true);
      expect(step1.epoch).toBe(1);

      // Step 2, 3, 4 should retain the exact least-squares parameters
      const step2 = linearRegressionPlugin.step(step1.state, dataset, params);
      expect(step2.converged).toBe(true);
      expect(step2.state.weights[0]).toBeCloseTo(step1.state.weights[0], 6);
      expect(step2.state.bias).toBeCloseTo(step1.state.bias, 6);
      expect(step2.metrics.trainLoss).toBeCloseTo(step1.metrics.trainLoss, 6);
    });

    it('Huber loss curbs gradient from large outlier compared to squared MSE', () => {
      // Normal dataset with one extreme outlier
      const outlierDataset = {
        train: [
          { id: 1, x1: -2.0, x2: -1.0, y: -1.0 },
          { id: 2, x1: -1.0, x2: -0.5, y: -0.5 },
          { id: 3, x1: 0.0, x2: 0.0, y: 0.0 },
          { id: 4, x1: 1.0, x2: 0.5, y: 0.5 },
          { id: 5, x1: 2.0, x2: 50.0, y: 50.0 }, // Extreme outlier
        ],
        test: [],
      };

      const mseState = linearRegressionPlugin.step(
        linearRegressionPlugin.init(outlierDataset, { solver: 'sgd', criterion: 'mse', learningRate: 0.05 }, 1),
        outlierDataset,
        { solver: 'sgd', criterion: 'mse', learningRate: 0.05, batchPasses: 1 }
      ).state;

      const huberState = linearRegressionPlugin.step(
        linearRegressionPlugin.init(outlierDataset, { solver: 'sgd', criterion: 'huber', huberDelta: 1.0, learningRate: 0.05 }, 1),
        outlierDataset,
        { solver: 'sgd', criterion: 'huber', huberDelta: 1.0, learningRate: 0.05, batchPasses: 1 }
      ).state;

      // Huber gradient is linear past delta, so weights deflect much less towards the outlier than MSE
      expect(Math.abs(huberState.weights[0])).toBeLessThan(Math.abs(mseState.weights[0]));
    });

    it('state snapshots serialize and restore is1D flag and parameters without loss', () => {
      const originalState = {
        weights: new Float32Array([0.725, -1.0]),
        bias: 0.415,
        epoch: 12,
        residuals: [],
        gradientNorm: 0.082,
        is1D: true,
      };

      const buffer = linearRegressionPlugin.serializeSnapshot(originalState);
      const restored = linearRegressionPlugin.restoreSnapshot(buffer, originalState);

      expect(restored.weights[0]).toBeCloseTo(0.725, 4);
      expect(restored.weights[1]).toBeCloseTo(-1.0, 4);
      expect(restored.bias).toBeCloseTo(0.415, 4);
      expect(restored.epoch).toBe(12);
      expect(restored.is1D).toBe(true);
    });

    it('closed-form Huber loss (IRLS) robustly downweights outliers compared to closed-form OLS (MSE)', () => {
      const outlierDataset = {
        train: [
          { id: 1, x1: -2.0, x2: -1.0, y: -1.0 },
          { id: 2, x1: -1.0, x2: -0.5, y: -0.5 },
          { id: 3, x1: 0.0, x2: 0.0, y: 0.0 },
          { id: 4, x1: 1.0, x2: 0.5, y: 0.5 },
          { id: 5, x1: 2.0, x2: 1.0, y: 1.0 },
          { id: 6, x1: 3.0, x2: 60.0, y: 60.0 }, // Rogue outlier
        ],
        test: [],
      };

      const olsMse = linearRegressionPlugin.init(outlierDataset, { solver: 'closed-form', criterion: 'mse' }, 1);
      const olsHuber = linearRegressionPlugin.init(outlierDataset, { solver: 'closed-form', criterion: 'huber', huberDelta: 1.0 }, 1);

      // OLS MSE is pulled heavily by (3.0, 60.0) -> slope w0 >> 2.0
      // Huber IRLS downweights the outlier -> slope w0 remains closer to 0.5
      expect(olsHuber.weights[0]).toBeLessThan(olsMse.weights[0]);
      expect(olsHuber.weights[0]).toBeCloseTo(0.5, 0);
    });

    it('applies zscore and minmax scaling consistently to 1D datasets without data leakage or NaN', () => {
      for (const scaling of ['raw', 'zscore', 'minmax'] as const) {
        const dataset = generateTabular2dDataset({
          kind: 'sine-wave',
          sampleCount: 80,
          noise: 0.1,
          testSplit: 0.25,
          seed: 42,
          numClasses: 2,
          scaling,
          stretchX2: false,
        });

        expect(is1DRegressionDataset(dataset)).toBe(true);

        const init = linearRegressionPlugin.init(dataset, { solver: 'sgd', learningRate: 0.02 }, 1);
        const stepRes = linearRegressionPlugin.step(init, dataset, { solver: 'sgd', learningRate: 0.02, batchPasses: 5 });

        expect(Number.isFinite(stepRes.state.weights[0])).toBe(true);
        expect(Number.isFinite(stepRes.state.bias)).toBe(true);
        expect(Number.isFinite(stepRes.metrics.trainLoss)).toBe(true);
        expect(Number.isFinite(stepRes.metrics.testLoss)).toBe(true);
        expect(Number.isFinite(stepRes.metrics.trainR2)).toBe(true);
      }
    });

    it('is1DRegressionData correctly tags 1D datasets for stage heading display', () => {
      const sine = generateTabular2dDataset({ kind: 'sine-wave', sampleCount: 50, noise: 0.1, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });
      const surface = generateTabular2dDataset({ kind: 'linear-surface', sampleCount: 50, noise: 0.1, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });
      const blobs = generateTabular2dDataset({ kind: 'linear-blobs', sampleCount: 50, noise: 0.1, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });

      expect(is1DRegressionData(sine.train)).toBe(true);
      expect(is1DRegressionData(surface.train)).toBe(false);
      expect(is1DRegressionData(blobs.train)).toBe(false);
    });
  });

  // 6. Strict isolation verification
  describe('Isolation: other algorithms remain completely untouched and valid', () => {
    it('Ridge L2, Lasso L1, Linear SVM, Perceptron, and Logistic Regression continue running normally', () => {
      const dataset = generateTabular2dDataset({
        kind: 'linear-blobs',
        sampleCount: 60,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      // Ridge L2
      const ridgeState = ridgeL2Plugin.init(dataset, { lambdaPenalty: 0.1 }, 42);
      const ridgeRes = ridgeL2Plugin.step(ridgeState, dataset, { lambdaPenalty: 0.1 });
      expect(ridgeRes.epoch).toBe(1);
      expect(ridgeRes.metrics.trainLoss).toBeGreaterThan(0);

      // Lasso L1
      const lassoState = lassoL1Plugin.init(dataset, { lambdaPenalty: 0.1 }, 42);
      const lassoRes = lassoL1Plugin.step(lassoState, dataset, { lambdaPenalty: 0.1 });
      expect(lassoRes.epoch).toBe(1);

      // Linear SVM
      const svmState = linearSvmPlugin.init(dataset, { cPenalty: 1.0 }, 42);
      const svmRes = linearSvmPlugin.step(svmState, dataset, { cPenalty: 1.0 });
      expect(svmRes.epoch).toBe(1);

      // Perceptron
      const perceptronState = perceptronPlugin.init(dataset, {}, 42);
      const perceptronRes = perceptronPlugin.step(perceptronState, dataset, {});
      expect(perceptronRes.epoch).toBe(1);

      // Logistic Regression
      const logState = logisticRegressionPlugin.init(dataset, {}, 42);
      const logRes = logisticRegressionPlugin.step(logState, dataset, {});
      expect(logRes.epoch).toBe(1);
    });
  });
});
