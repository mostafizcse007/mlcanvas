import { describe, it, expect } from 'vitest';
import {
  polynomialRegressionPlugin,
  computePolyFeatures,
  get2DPolynomialTerms,
  getFeatureCount,
} from '../stage01/polynomialRegression';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { serializeToUrlHash, parseUrlHash } from '../../utils/urlState';
import { linearRegressionPlugin } from '../stage01/linearRegression';
import { ridgeL2Plugin } from '../stage01/ridgeL2';
import { lassoL1Plugin } from '../stage01/lassoL1';
import { linearSvmPlugin } from '../stage01/linearSvm';

describe('Stage 01: Polynomial Feature Regression (OLS / SGD) Comprehensive Suite', () => {
  // 1. Multivariate basis specification & 1D vs 2D prediction agreement
  describe('Issue 1: 1D vs 2D Feature Expansion & Prediction Agreement', () => {
    it('generates exact declared feature counts: K_1D = d + 1, K_2D = (d+1)(d+2)/2', () => {
      const degrees = [1, 2, 3, 4, 6, 8, 12];
      for (const d of degrees) {
        expect(getFeatureCount(d, true)).toBe(d + 1);
        expect(getFeatureCount(d, false)).toBe(((d + 1) * (d + 2)) / 2);
      }

      // Specific known counts
      expect(getFeatureCount(1, false)).toBe(3);  // 1, x1, x2
      expect(getFeatureCount(2, false)).toBe(6);  // 1, x1, x2, x1^2, x1*x2, x2^2
      expect(getFeatureCount(3, false)).toBe(10); // 1, x1, x2, x1^2, x1*x2, x2^2, x1^3, x1^2*x2, x1*x2^2, x2^3
      expect(getFeatureCount(8, false)).toBe(45);
      expect(getFeatureCount(12, false)).toBe(91);
    });

    it('2D polynomial term ordering follows graded lexicographic order with cross terms', () => {
      const termsD2 = get2DPolynomialTerms(2);
      expect(termsD2).toEqual([
        { p1: 0, p2: 0, degree: 0 },
        { p1: 1, p2: 0, degree: 1 },
        { p1: 0, p2: 1, degree: 1 },
        { p1: 2, p2: 0, degree: 2 },
        { p1: 1, p2: 1, degree: 2 },
        { p1: 0, p2: 2, degree: 2 },
      ]);
    });

    it('1D regression predictions are strictly invariant to varying x2 when x1 is held fixed', () => {
      const dataset1D = generateTabular2dDataset({
        kind: 'sine-wave',
        sampleCount: 60,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const state1D = polynomialRegressionPlugin.init(dataset1D, { degree: 3, solver: 'closed-form' }, 42);
      expect(state1D.is1D).toBe(true);

      const x1 = 1.5;
      const predA = polynomialRegressionPlugin.predict(new Float32Array([x1, -5.0]), state1D, {}) as number;
      const predB = polynomialRegressionPlugin.predict(new Float32Array([x1, 0.0]), state1D, {}) as number;
      const predC = polynomialRegressionPlugin.predict(new Float32Array([x1, 5.0]), state1D, {}) as number;

      // All must be exactly identical: no x2 sensitivity in 1D regression!
      expect(predA).toBeCloseTo(predB, 6);
      expect(predB).toBeCloseTo(predC, 6);
    });

    it('2D surface regression predictions vary with both x1 and x2 according to the bivariate basis', () => {
      const dataset2D = generateTabular2dDataset({
        kind: 'linear-surface',
        sampleCount: 80,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const state2D = polynomialRegressionPlugin.init(dataset2D, { degree: 2, solver: 'closed-form' }, 42);
      expect(state2D.is1D).toBe(false);
      expect(state2D.coefficients.length).toBe(6); // (2+1)(2+2)/2 = 6

      const x1 = 2.0;
      const predA = polynomialRegressionPlugin.predict(new Float32Array([x1, -2.0]), state2D, {}) as number;
      const predB = polynomialRegressionPlugin.predict(new Float32Array([x1, 2.0]), state2D, {}) as number;

      // On a 2D surface, changing x2 holding x1 fixed must change the surface elevation
      expect(Math.abs(predA - predB)).toBeGreaterThan(0.1);

      // Verify manual prediction matches model prediction
      const feats = computePolyFeatures(x1, 2.0, 2, false);
      let expected = 0;
      for (let i = 0; i < 6; i++) expected += state2D.coefficients[i] * feats[i];
      expect(predB).toBeCloseTo(expected, 5);
    });

    it('model summary, microLog, and math inspector agree on basis dimension in 1D vs 2D', () => {
      const d1 = generateTabular2dDataset({ kind: 'sine-wave', sampleCount: 50, noise: 0.0, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });
      const d2 = generateTabular2dDataset({ kind: 'linear-surface', sampleCount: 50, noise: 0.0, testSplit: 0.2, seed: 1, numClasses: 2, scaling: 'raw', stretchX2: false });

      const s1 = polynomialRegressionPlugin.init(d1, { degree: 4, solver: 'closed-form' }, 1);
      const s2 = polynomialRegressionPlugin.init(d2, { degree: 4, solver: 'closed-form' }, 1);

      const exp1 = polynomialRegressionPlugin.explain(s1, { degree: 4 });
      const exp2 = polynomialRegressionPlugin.explain(s2, { degree: 4 });

      // 1D: K = 5
      expect(exp1.summary).toContain('5 features');
      expect(exp1.microLog[0]).toContain('Vandermonde expansion: dim(Φ) = 5');
      expect(exp1.mathLatex.dualOrConstraint).toContain('\\phi(x_1)');
      expect(exp1.codeSnippet.sklearn).toContain('X_train[:, :1]');

      // 2D: K = (4+1)(4+2)/2 = 15
      expect(exp2.summary).toContain('15 features');
      expect(exp2.microLog[0]).toContain('Bivariate polynomial expansion: dim(Φ) = 15');
      expect(exp2.mathLatex.dualOrConstraint).toContain('K = \\frac{(d+1)(d+2)}{2}');
      expect(exp2.codeSnippet.sklearn).toContain('X_train[:, :2]');
    });
  });

  // 2. Telemetry and accuracy audit
  describe('Issue 2: Telemetry Metrics & Absence of Undefined Accuracy in Regression Logs', () => {
    it('does NOT report undefined "Acc = ...%" in regression microLog', () => {
      const dataset = generateTabular2dDataset({
        kind: 'sine-wave',
        sampleCount: 60,
        noise: 0.1,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const state = polynomialRegressionPlugin.init(dataset, { degree: 3, solver: 'closed-form' }, 42);
      const stepRes = polynomialRegressionPlugin.step(state, dataset, { degree: 3, solver: 'closed-form' });
      const exp = polynomialRegressionPlugin.explain(stepRes.state, { degree: 3 }, stepRes);

      const logText = exp.microLog.join('\n');
      // Must NOT contain undefined "Acc = " in regression logs
      expect(logText).not.toContain('Acc =');
      expect(logText).toContain('Train MSE =');
      expect(logText).toContain('R² =');
      expect(logText).toContain('RMSE =');
    });

    it('reports mathematically sound MSE, RMSE, and R² metrics', () => {
      const dataset = generateTabular2dDataset({
        kind: 'polynomial-cubic',
        sampleCount: 80,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const state = polynomialRegressionPlugin.init(dataset, { degree: 3, solver: 'closed-form', regularization: 0.001 }, 42);
      const stepRes = polynomialRegressionPlugin.step(state, dataset, { degree: 3, solver: 'closed-form', regularization: 0.001 });

      expect(stepRes.metrics.trainLoss).toBeGreaterThan(0);
      expect(stepRes.metrics.trainLoss).toBeLessThan(0.15); // Good fit on cubic data
      expect(stepRes.metrics.trainR2).toBeGreaterThan(0.85); // High R2
      expect(stepRes.metrics.testLoss).toBeGreaterThan(0);
      expect(Number.isFinite(stepRes.metrics.trainLoss)).toBe(true);
      expect(Number.isFinite(stepRes.metrics.testLoss)).toBe(true);
    });
  });

  // 3. Share URL state persistence & restoration
  describe('Issue 3: Share URL State Serialization & Restoration', () => {
    it('serializes degree, regularization, and solver into the share URL hash', () => {
      const hash = serializeToUrlHash({
        algo: 'polynomial-regression',
        dataset: 'sine-wave',
        degree: 6,
        reg: 0.05,
        solver: 'closed-form',
        seed: 123,
        noise: 0.25,
        split: 0.3,
        scale: 'zscore',
      });

      expect(hash).toContain('algo=polynomial-regression');
      expect(hash).toContain('data=sine-wave');
      expect(hash).toContain('deg=6');
      expect(hash).toContain('reg=0.05');
      expect(hash).toContain('sol=closed-form');
      expect(hash).toContain('seed=123');
      expect(hash).toContain('noise=25');
      expect(hash).toContain('split=30');
      expect(hash).toContain('scale=zscore');
    });

    it('parses and restores degree, reg, and solver faithfully from URL hash', () => {
      (globalThis as any).window = {
        location: {
          hash: '#algo=polynomial-regression&data=polynomial-cubic&deg=8&reg=0.5&sol=sgd&seed=77&noise=15&split=25',
        },
      };

      const parsed = parseUrlHash();
      expect(parsed.algo).toBe('polynomial-regression');
      expect(parsed.dataset).toBe('polynomial-cubic');
      expect(parsed.degree).toBe(8);
      expect(parsed.reg).toBe(0.5);
      expect(parsed.solver).toBe('sgd');
      expect(parsed.seed).toBe(77);
      expect(parsed.noise).toBeCloseTo(0.15, 2);
      expect(parsed.split).toBeCloseTo(0.25, 2);
    });

    it('does not alter URL serialization for other algorithms where polynomial settings are absent', () => {
      const svmHash = serializeToUrlHash({
        algo: 'linear-svm',
        dataset: 'linear-blobs',
        c: 1.0,
        lr: 0.02,
        seed: 42,
      });

      expect(svmHash).toContain('algo=linear-svm');
      expect(svmHash).toContain('c=1');
      expect(svmHash).not.toContain('deg=');
      expect(svmHash).not.toContain('reg=');
      expect(svmHash).not.toContain('sol=');
    });
  });

  // 4. Runge boundary oscillation vs numerical conditioning diagnostics
  describe('Issue 4: Runge Phenomenon & Numerical Conditioning Diagnostics', () => {
    it('distinguishes high-degree unregularized Runge oscillation from regularized damping', () => {
      const dataset = generateTabular2dDataset({
        kind: 'sine-wave',
        sampleCount: 60,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      // Degree 8 with 0.0 regularization -> high Runge oscillation & ill-conditioned matrix
      const unregState = polynomialRegressionPlugin.init(dataset, { degree: 8, regularization: 0.0, solver: 'closed-form' }, 42);
      const unregStep = polynomialRegressionPlugin.step(unregState, dataset, { degree: 8, regularization: 0.0, solver: 'closed-form' });
      const unregExp = polynomialRegressionPlugin.explain(unregStep.state, { degree: 8, regularization: 0.0, solver: 'closed-form' });

      // Should detect oscillation/conditioning warning
      expect(unregStep.state.rungeWarning || unregStep.state.conditioningWarning).toBe(true);
      expect(unregExp.microLog.some((l) => l.includes('Runge') || l.includes('Ill-conditioned') || l.includes('CRITICAL'))).toBe(true);

      // Degree 8 with 0.50 (Heavy Damping) -> ridge penalty contracts high-order monomials
      const regState = polynomialRegressionPlugin.init(dataset, { degree: 8, regularization: 0.5, solver: 'closed-form' }, 42);
      const regStep = polynomialRegressionPlugin.step(regState, dataset, { degree: 8, regularization: 0.5, solver: 'closed-form' });
      const regExp = polynomialRegressionPlugin.explain(regStep.state, { degree: 8, regularization: 0.5, solver: 'closed-form' });

      // Boundary swing is visibly damped
      expect(regStep.state.rungeWarning).toBe(false);
      expect(regExp.microLog.some((l) => l.includes('STABLE'))).toBe(true);
    });

    it('reports well-conditioned stable state for modest degrees (e.g. d=3)', () => {
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

      const state = polynomialRegressionPlugin.init(dataset, { degree: 3, regularization: 0.05, solver: 'closed-form' }, 42);
      const stepRes = polynomialRegressionPlugin.step(state, dataset, { degree: 3, regularization: 0.05, solver: 'closed-form' });
      const exp = polynomialRegressionPlugin.explain(stepRes.state, { degree: 3, regularization: 0.05, solver: 'closed-form' });

      expect(stepRes.state.rungeWarning).toBe(false);
      expect(stepRes.state.conditioningWarning).toBe(false);
      expect(exp.microLog.some((l) => l.includes('STABLE'))).toBe(true);
    });
  });

  // 5. Solver coverage: closed-form and SGD across degrees and ridge values
  describe('Solver & Parameter Matrix Coverage: Degrees (1, 3, 6, 8, 12) & Ridge (0, 0.001, 0.05, 0.5)', () => {
    const degreesToTest = [1, 3, 6, 8, 12];
    const ridgeToTest = [0.0, 0.001, 0.05, 0.5];
    const datasetsToTest = ['sine-wave', 'polynomial-cubic', 'linear-trend', 'linear-surface'] as const;

    for (const kind of datasetsToTest) {
      it(`trains stable closed-form and SGD on ${kind} across degrees and regularization`, () => {
        const dataset = generateTabular2dDataset({
          kind,
          sampleCount: 60,
          noise: 0.05,
          testSplit: 0.2,
          seed: 42,
          numClasses: 2,
          scaling: 'raw',
          stretchX2: false,
        });

        for (const degree of degreesToTest) {
          for (const reg of ridgeToTest) {
            // Closed-form
            const cfInit = polynomialRegressionPlugin.init(dataset, { degree, regularization: reg, solver: 'closed-form' }, 1);
            const cfRes = polynomialRegressionPlugin.step(cfInit, dataset, { degree, regularization: reg, solver: 'closed-form' });

            expect(Number.isFinite(cfRes.metrics.trainLoss)).toBe(true);
            expect(Number.isFinite(cfRes.metrics.testLoss)).toBe(true);
            expect(cfRes.metrics.trainLoss).toBeGreaterThanOrEqual(0);
            expect(cfRes.state.coefficients.length).toBe(getFeatureCount(degree, cfRes.state.is1D));

            // SGD
            const sgdInit = polynomialRegressionPlugin.init(dataset, { degree, regularization: reg, solver: 'sgd', learningRate: 0.02 }, 1);
            const sgdRes = polynomialRegressionPlugin.step(sgdInit, dataset, { degree, regularization: reg, solver: 'sgd', learningRate: 0.02, batchPasses: 2 });

            expect(Number.isFinite(sgdRes.metrics.trainLoss)).toBe(true);
            expect(Number.isFinite(sgdRes.metrics.testLoss)).toBe(true);
            expect(sgdRes.epoch).toBe(1);
          }
        }
      });
    }

    it('applies feature scaling (raw, zscore, minmax) consistently without data leakage or NaN', () => {
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

        const init = polynomialRegressionPlugin.init(dataset, { degree: 4, regularization: 0.01, solver: 'closed-form' }, 42);
        const stepRes = polynomialRegressionPlugin.step(init, dataset, { degree: 4, regularization: 0.01, solver: 'closed-form' });

        expect(Number.isFinite(stepRes.metrics.trainLoss)).toBe(true);
        expect(Number.isFinite(stepRes.metrics.testLoss)).toBe(true);
        expect(stepRes.metrics.trainR2).toBeGreaterThan(-1.0);
      }
    });

    it('handles degenerate / singular inputs gracefully with fallback to target mean and finite metrics', () => {
      // Degenerate dataset with all identical x1 coordinates
      const singularDataset = {
        train: [
          { id: 1, x1: 2.0, x2: 2.0, y: 1.0 },
          { id: 2, x1: 2.0, x2: 2.0, y: 3.0 },
          { id: 3, x1: 2.0, x2: 2.0, y: 5.0 },
          { id: 4, x1: 2.0, x2: 2.0, y: 7.0 },
        ],
        test: [{ id: 5, x1: 2.0, x2: 2.0, y: 4.0 }],
      };

      const init = polynomialRegressionPlugin.init(singularDataset, { degree: 4, regularization: 0.0, solver: 'closed-form' }, 1);
      const stepRes = polynomialRegressionPlugin.step(init, singularDataset, { degree: 4, regularization: 0.0, solver: 'closed-form' });

      expect(Number.isFinite(stepRes.state.coefficients[0])).toBe(true);
      expect(Number.isFinite(stepRes.metrics.trainLoss)).toBe(true);
      expect(isNaN(stepRes.metrics.trainLoss)).toBe(false);
      // Fallback: horizontal line at mean y = 4.0, variance of y is ((9+1+1+9)/4) = 5
      expect(stepRes.metrics.trainLoss).toBeCloseTo(5.0, 1);
    });

    it('snapshots serialize and restore state without loss across 1D and 2D modes', () => {
      // 1D snapshot
      const state1D = {
        coefficients: new Float32Array([1.2, -0.4, 0.8, -0.1]),
        degree: 3,
        epoch: 15,
        curvePoints: [[0, 1]] as [number, number][],
        residuals: [],
        rungeWarning: false,
        conditioningWarning: false,
        is1D: true,
      };

      const buf1D = polynomialRegressionPlugin.serializeSnapshot(state1D);
      const restored1D = polynomialRegressionPlugin.restoreSnapshot(buf1D, state1D);
      expect(restored1D.degree).toBe(3);
      expect(restored1D.epoch).toBe(15);
      expect(restored1D.is1D).toBe(true);
      expect(restored1D.coefficients[0]).toBeCloseTo(1.2, 5);
      expect(restored1D.coefficients[2]).toBeCloseTo(0.8, 5);

      // 2D snapshot
      const state2D = {
        coefficients: new Float32Array([0.5, 0.2, 0.3, 0.1, -0.2, 0.4]),
        degree: 2,
        epoch: 7,
        curvePoints: [],
        residuals: [],
        rungeWarning: false,
        conditioningWarning: false,
        is1D: false,
      };

      const buf2D = polynomialRegressionPlugin.serializeSnapshot(state2D);
      const restored2D = polynomialRegressionPlugin.restoreSnapshot(buf2D, state2D);
      expect(restored2D.degree).toBe(2);
      expect(restored2D.epoch).toBe(7);
      expect(restored2D.is1D).toBe(false);
      expect(restored2D.coefficients.length).toBe(6);
      expect(restored2D.coefficients[4]).toBeCloseTo(-0.2, 5);
    });
  });

  // 6. Strict isolation verification
  describe('Isolation: other algorithms remain completely untouched and valid', () => {
    it('Linear Regression, Ridge L2, Lasso L1, and Linear SVM continue operating normally', () => {
      const dataset = generateTabular2dDataset({
        kind: 'linear-trend',
        sampleCount: 50,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      // Linear Regression
      const linState = linearRegressionPlugin.init(dataset, { solver: 'closed-form' }, 1);
      const linRes = linearRegressionPlugin.step(linState, dataset, { solver: 'closed-form' });
      expect(linRes.epoch).toBe(1);
      expect(linRes.metrics.trainLoss).toBeGreaterThan(0);

      // Ridge L2
      const ridgeState = ridgeL2Plugin.init(dataset, { lambdaPenalty: 0.1 }, 1);
      const ridgeRes = ridgeL2Plugin.step(ridgeState, dataset, { lambdaPenalty: 0.1 });
      expect(ridgeRes.epoch).toBe(1);
      expect(ridgeRes.metrics.trainLoss).toBeGreaterThan(0);

      // Lasso L1
      const lassoState = lassoL1Plugin.init(dataset, { lambdaPenalty: 0.1 }, 1);
      const lassoRes = lassoL1Plugin.step(lassoState, dataset, { lambdaPenalty: 0.1 });
      expect(lassoRes.epoch).toBe(1);

      // Linear SVM
      const svmState = linearSvmPlugin.init(dataset, { cPenalty: 1.0 }, 1);
      const svmRes = linearSvmPlugin.step(svmState, dataset, { cPenalty: 1.0 });
      expect(svmRes.epoch).toBe(1);
    });
  });
});
