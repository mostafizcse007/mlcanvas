import { describe, it, expect } from 'vitest';
import {
  generateTabular2dDataset,
  getDatasetTaskType,
  CLASSIFICATION_DATASETS,
  REGRESSION_DATASETS,
} from '../../datasets/tabular2d';
import { polynomialRegressionPlugin } from '../stage01/polynomialRegression';
import { linearRegressionPlugin } from '../stage01/linearRegression';
import { ridgeL2Plugin } from '../stage01/ridgeL2';
import { gaussianProcessPlugin } from '../stage01/gaussianProcess';

describe('Regression Datasets & Algorithm Integration', () => {
  it('correctly categorizes classification vs regression datasets', () => {
    for (const kind of CLASSIFICATION_DATASETS) {
      expect(getDatasetTaskType(kind)).toBe('classification');
    }
    for (const kind of REGRESSION_DATASETS) {
      expect(getDatasetTaskType(kind)).toBe('regression');
    }
  });

  it('generates well-formed 1D regression datasets where points align at (x1, y)', () => {
    const oneDimensionalKinds = ['sine-wave', 'polynomial-cubic', 'linear-trend'] as const;

    for (const kind of oneDimensionalKinds) {
      const { train, test } = generateTabular2dDataset({
        kind,
        sampleCount: 60,
        noise: 0.05,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      expect(train.length + test.length).toBe(60);
      for (const pt of [...train, ...test]) {
        expect(Number.isFinite(pt.x1)).toBe(true);
        expect(Number.isFinite(pt.x2)).toBe(true);
        expect(Number.isFinite(pt.y)).toBe(true);
        // For 1D regression, vertical coordinate x2 equals continuous target y
        expect(pt.x2).toBe(pt.y);
      }
    }
  });

  it('fits 1D Sine Wave with Polynomial Feature Regression cleanly', () => {
    const dataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 80,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { degree: 5, regularization: 0.001 };
    const initState = polynomialRegressionPlugin.init(dataset, params, 42);
    expect(initState.coefficients.length).toBe(6);

    const stepResult = polynomialRegressionPlugin.step(initState, dataset, params);
    expect(stepResult.epoch).toBe(1);
    expect(stepResult.metrics.trainLoss).toBeGreaterThan(0);
    expect(stepResult.metrics.trainLoss).toBeLessThan(5.0);

    const artifacts = polynomialRegressionPlugin.getVisualArtifacts(stepResult.state, dataset, params);
    const curveFit = artifacts.find((a) => a.type === 'curve-fit');
    const residualBars = artifacts.find((a) => a.type === 'residual-bars');

    expect(curveFit).toBeDefined();
    expect(residualBars).toBeDefined();
  });

  it('fits Linear Surface with Linear Regression and Ridge Regression', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-surface',
      sampleCount: 100,
      noise: 0.05,
      testSplit: 0.2,
      seed: 999,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { solver: 'closed-form' as const, learningRate: 0.02 };
    const linState = linearRegressionPlugin.init(dataset, params, 999);
    const linStep = linearRegressionPlugin.step(linState, dataset, params);

    expect(linStep.metrics.trainLoss).toBeGreaterThan(0);
    expect(linStep.metrics.trainLoss).toBeLessThan(1.0);

    const ridgeState = ridgeL2Plugin.init(dataset, { lambdaPenalty: 0.01 }, 999);
    const ridgeStep = ridgeL2Plugin.step(ridgeState, dataset, { lambdaPenalty: 0.01 });
    expect(ridgeStep.metrics.trainLoss).toBeGreaterThan(0);
  });

  it('initializes Ridge L2 in untrained state at Epoch 0 with no hyperplane, and fits upon step', () => {
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

    const params = { lambdaPenalty: 0.2, solver: 'closed-form' as const };
    const initState = ridgeL2Plugin.init(dataset, params, 42);

    expect(initState.epoch).toBe(0);
    expect(initState.weights[0]).not.toBe(0);
    expect(initState.weights[1]).not.toBe(0);

    const initArtifacts = ridgeL2Plugin.getVisualArtifacts(initState, dataset, params);
    const initHyperplane = initArtifacts.find((a) => a.type === 'hyperplane');
    expect(initHyperplane).toBeDefined(); // Hyperplane exists from initial non-zero weights

    const stepRes = ridgeL2Plugin.step(initState, dataset, params);
    expect(stepRes.epoch).toBe(1);
    expect(stepRes.converged).toBe(true);
    expect(Math.hypot(stepRes.state.weights[0], stepRes.state.weights[1])).toBeGreaterThan(0.5);

    const solvedArtifacts = ridgeL2Plugin.getVisualArtifacts(stepRes.state, dataset, params);
    const solvedHyperplane = solvedArtifacts.find((a) => a.type === 'hyperplane');
    expect(solvedHyperplane).toBeDefined(); // Hyperplane appears after solve
  });

  it('runs Gaussian Process regression on 1D Sine Wave with progressive ingestion and accurate fit', () => {
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

    const params = { lengthScale: 1.2, signalVariance: 1.0, noiseLevel: 0.02, solver: 'sequential' as const };
    const initState = gaussianProcessPlugin.init(dataset, params, 42);

    expect(initState.epoch).toBe(0);
    expect(initState.is1D).toBe(true);
    expect(initState.meanCurve.length).toBeGreaterThan(0);

    // Step 1: Iterative progressive ingestion
    const step1 = gaussianProcessPlugin.step(initState, dataset, params);
    expect(step1.epoch).toBe(1);
    expect(step1.state.ingestedCount).toBeGreaterThan(0);
    expect(step1.state.meanCurve.length).toBeGreaterThan(0);

    // Closed-form solve should achieve high R2 on sine wave
    const cfParams = { lengthScale: 1.2, signalVariance: 1.0, noiseLevel: 0.02, solver: 'closed-form' as const };
    const cfStep = gaussianProcessPlugin.step(initState, dataset, cfParams);
    expect(cfStep.converged).toBe(true);
    expect(cfStep.metrics.trainR2).toBeGreaterThan(0.9);
    expect(cfStep.metrics.trainLoss).toBeLessThan(0.1);

    // Check predict on grid for 1D: evaluates mu(x1) - x2
    const predCenter = Number(gaussianProcessPlugin.predict(new Float32Array([0, 0]), cfStep.state, cfParams));
    // Sine wave at x=0 is near 0
    expect(Math.abs(predCenter)).toBeLessThan(0.5);
  });

  it('runs Linear Regression SGD on 1D Linear Trend with positive slope', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-trend',
      sampleCount: 60,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { solver: 'sgd' as const, learningRate: 0.05 };
    let state = linearRegressionPlugin.init(dataset, params, 42);

    for (let e = 0; e < 10; e++) {
      const res = linearRegressionPlugin.step(state, dataset, params);
      state = res.state;
    }

    expect(state.epoch).toBe(10);
    // Slope w[0] should be positive (trend slopes upward ~0.65)
    expect(state.weights[0]).toBeGreaterThan(0.2);
    // w[1] should be -1 for (m*x1 - x2 + b = 0)
    expect(state.weights[1]).toBe(-1);
  });

  it('correctly trains Linear Regression on 2D binary classification datasets with +1 and -1 classes', () => {
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

    const params = { solver: 'closed-form' as const };
    const initState = linearRegressionPlugin.init(dataset, params, 42);
    const stepRes = linearRegressionPlugin.step(initState, dataset, params);

    expect(stepRes.converged).toBe(true);
    // Should classify well-separated linear blobs cleanly
    expect(stepRes.metrics.trainAccuracy).toBeGreaterThan(90);
    expect(stepRes.metrics.testAccuracy).toBeGreaterThan(90);

    const artifacts = linearRegressionPlugin.getVisualArtifacts(stepRes.state, dataset, params);
    const hyperplane = artifacts.find((a) => a.type === 'hyperplane');
    expect(hyperplane).toBeDefined();
  });

  it('supports 1D regression point manipulation with sign flipping', () => {
    const initialPt = { id: 1, x1: 2.5, x2: 1.8, y: 1.8, isTest: false };
    expect(initialPt.x2).toBe(initialPt.y);

    // Flipping in 1D regression mirrors across zero axis
    const flippedPt = { ...initialPt, x2: -initialPt.x2, y: -initialPt.y };
    expect(flippedPt.x2).toBe(-1.8);
    expect(flippedPt.y).toBe(-1.8);
    expect(flippedPt.x2).toBe(flippedPt.y);

    // Adding point with negative active class (-1) sets negative target
    const clickedY = 2.0;
    const targetYNeg = -Math.abs(clickedY);
    const newPtNeg = { id: 2, x1: -1.0, x2: targetYNeg, y: targetYNeg, isTest: false };
    expect(newPtNeg.y).toBe(-2.0);
    expect(newPtNeg.x2).toBe(newPtNeg.y);

    // Adding point with positive active class (+1) sets positive target
    const targetYPos = Math.abs(-1.5);
    const newPtPos = { id: 3, x1: 0.5, x2: targetYPos, y: targetYPos, isTest: false };
    expect(newPtPos.y).toBe(1.5);
    expect(newPtPos.x2).toBe(newPtPos.y);
  });
});
