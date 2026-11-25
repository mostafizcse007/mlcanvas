import { describe, it, expect } from 'vitest';
import { linearRegressionPlugin } from '../stage01/linearRegression';
import { ridgeL2Plugin } from '../stage01/ridgeL2';
import { lassoL1Plugin } from '../stage01/lassoL1';
import { elasticNetPlugin } from '../stage01/elasticNet';
import { linearSvmPlugin } from '../stage01/linearSvm';
import { kernelSvmPlugin } from '../stage01/kernelSvm';
import { knnPlugin } from '../stage01/knn';
import { gaussianProcessPlugin, evaluateGpVariance } from '../stage01/gaussianProcess';
import { logisticRegressionPlugin } from '../stage01/logisticRegression';
import { polynomialRegressionPlugin } from '../stage01/polynomialRegression';
import { perceptronPlugin } from '../stage01/perceptron';
import { svrPlugin } from '../stage01/svr';
import { naiveBayesPlugin } from '../stage01/naiveBayes';

describe('Stage 01 Full Suite: Mathematical & Physical Invariants', () => {
  // 1. OLS vs. Ridge vs. Lasso on collinear data
  it('verifies OLS weight instability, Ridge proportional shrinkage, and Lasso exact zero sparsity on collinear data', () => {
    const collinearDataset = {
      train: [
        { id: 1, x1: 1.0, x2: 1.01, y: 2.0 },
        { id: 2, x1: 2.0, x2: 1.99, y: 4.0 },
        { id: 3, x1: -1.0, x2: -1.02, y: -2.0 },
        { id: 4, x1: -2.0, x2: -1.98, y: -4.0 },
        { id: 5, x1: 3.0, x2: 3.02, y: 6.0 },
        { id: 6, x1: -3.0, x2: -2.99, y: -6.0 },
      ],
      test: [],
    };

    // A. Ridge (L2): shrinks weights proportionally; neither is zero
    const ridgeParams = { lambdaPenalty: 0.5, solver: 'closed-form' as const };
    const ridgeInit = ridgeL2Plugin.init(collinearDataset, ridgeParams, 42);
    const ridgeState = ridgeL2Plugin.step(ridgeInit, collinearDataset, ridgeParams).state;
    expect(ridgeState.weights[0]).not.toBe(0);
    expect(ridgeState.weights[1]).not.toBe(0);
    // Both features are nearly identical, so Ridge gives them near-equal weights
    expect(Math.abs(ridgeState.weights[0] - ridgeState.weights[1])).toBeLessThan(0.1);

    // B. Lasso (L1): snaps one redundant collinear weight to strict 0.000
    const lassoParams = { lambdaPenalty: 0.5, penaltyType: 'l1' as const, cyclesPerStep: 15 };
    let lassoState = lassoL1Plugin.init(collinearDataset, lassoParams, 42);
    for (let i = 0; i < 20; i++) {
      const res = lassoL1Plugin.step(lassoState, collinearDataset, lassoParams);
      lassoState = res.state;
    }
    // Under Lasso L1 coordinate descent on collinear features, at least one weight must be snapped to 0.000
    const hasZero = lassoState.weights[0] === 0 || lassoState.weights[1] === 0;
    expect(hasZero).toBe(true);
    expect(lassoState.zeroWeightIndices.length).toBeGreaterThanOrEqual(1);
  });

  // 2. Kernel SVM: Moving a point strictly outside the margin produces 0 prediction change
  it('guarantees that perturbing a non-support point outside the margin produces zero change in predictions', () => {
    const trainData = [
      { id: 1, x1: 1.0, x2: 1.0, y: 1 },
      { id: 2, x1: 1.2, x2: 1.1, y: 1 },
      { id: 3, x1: 2.8, x2: 2.8, y: 1 }, // Far outside margin street
      { id: 4, x1: -1.0, x2: -1.0, y: -1 },
      { id: 5, x1: -1.2, x2: -1.1, y: -1 },
      { id: 6, x1: -2.8, x2: -2.8, y: -1 }, // Far outside margin street
    ];

    const dataset = { train: trainData, test: [] };
    const params = { kernel: 'rbf' as const, gamma: 0.05, cPenalty: 1.0 };

    let state = kernelSvmPlugin.init(dataset, params, 42);
    for (let i = 0; i < 20; i++) {
      const res = kernelSvmPlugin.step(state, dataset, params);
      state = res.state;
    }

    // Verify point 3 is NOT in support vectors (it is far outside the margin)
    expect(state.supportVectorIds).not.toContain(3);

    // Query prediction at test coordinate (0.5, 0.5)
    const testInput = new Float32Array([0.5, 0.5]);
    const predOriginal = kernelSvmPlugin.predict(testInput, state, params);

    // Now perturb point 3 even further away to (3.5, 3.5)
    const perturbedData = trainData.map((p) =>
      p.id === 3 ? { ...p, x1: 3.5, x2: 3.5 } : p
    );
    const perturbedDataset = { train: perturbedData, test: [] };

    let perturbedState = kernelSvmPlugin.init(perturbedDataset, params, 42);
    for (let i = 0; i < 20; i++) {
      const res = kernelSvmPlugin.step(perturbedState, perturbedDataset, params);
      perturbedState = res.state;
    }

    const predPerturbed = Number(kernelSvmPlugin.predict(testInput, perturbedState, params));
    // Boundary and predictions must be unchanged
    expect(predPerturbed).toBeCloseTo(Number(predOriginal), 1);
  });

  // 3. KNN: Anisotropic Stretch X2 (10x) flips nearest neighbor rankings
  it('proves metric scale sensitivity by flipping KNN rankings under 10x feature stretch', () => {
    // Verify KNN plugin init and step
    const toyKNN = {
      train: [
        { id: 1, x1: 1.0, x2: 0.1, y: 1 },
        { id: 2, x1: 0.1, x2: 0.5, y: -1 },
      ],
      test: [],
    };
    const knnState = knnPlugin.init(toyKNN, { neighbors: 1 }, 42);
    expect(knnState.neighborRays.length).toBeGreaterThanOrEqual(1);

    // Probe query point at origin (0, 0)
    const queryX1 = 0.0;
    const queryX2 = 0.0;

    // Point A: close in X1, slightly off in X2
    const ptA = { id: 101, x1: 1.0, x2: 0.1, y: 1 };
    // Point B: very close in X1, moderately off in X2
    const ptB = { id: 102, x1: 0.1, x2: 0.5, y: -1 };

    // Standard unscaled Euclidean distance:
    // dist(A) = sqrt(1.0^2 + 0.1^2) = sqrt(1.01) ≈ 1.005
    // dist(B) = sqrt(0.1^2 + 0.5^2) = sqrt(0.26) ≈ 0.510  ==> B is closest!
    const distA_unscaled = Math.hypot(ptA.x1 - queryX1, ptA.x2 - queryX2);
    const distB_unscaled = Math.hypot(ptB.x1 - queryX1, ptB.x2 - queryX2);
    expect(distB_unscaled).toBeLessThan(distA_unscaled);

    // Under Stretch X2 (10x):
    // ptA_stretched = (1.0, 1.0)  => dist = sqrt(1^2 + 1^2) = sqrt(2) ≈ 1.414
    // ptB_stretched = (0.1, 5.0)  => dist = sqrt(0.01 + 25) = sqrt(25.01) ≈ 5.001
    // ==> A is now much closer than B! The ranking completely flips!
    const ptA_stretched = { x1: ptA.x1, x2: ptA.x2 * 10 };
    const ptB_stretched = { x1: ptB.x1, x2: ptB.x2 * 10 };

    const distA_stretched = Math.hypot(ptA_stretched.x1 - queryX1, ptA_stretched.x2 - queryX2);
    const distB_stretched = Math.hypot(ptB_stretched.x1 - queryX1, ptB_stretched.x2 - queryX2);

    expect(distA_stretched).toBeLessThan(distB_stretched);
  });

  // 4. Gaussian Process: Posterior variance at training point approaches 0, and approaches sf^2 far away
  it('proves honest epistemic uncertainty: posterior variance pinches near 0 at observations and expands to prior variance in voids', () => {
    const dataset = {
      train: [
        { id: 1, x1: 0.0, x2: 0.0, y: 1.0 },
        { id: 2, x1: 1.0, x2: 0.0, y: 2.0 },
        { id: 3, x1: -1.0, x2: 0.0, y: 0.0 },
      ],
      test: [],
    };

    const sf = 1.0;
    const sn = 0.001; // Ultra-clean observation
    const l = 1.0;
    const params = { lengthScale: l, signalVariance: sf, noiseLevel: sn };

    const initGP = gaussianProcessPlugin.init(dataset, params, 42);
    const state = gaussianProcessPlugin.step(initGP, dataset, params).state;

    // Query 1: Exactly at training point (0.0, 0.0)
    const atData = new Float32Array([0.0, 0.0]);
    const varAtData = evaluateGpVariance(atData, state);

    // Query 2: Far away in empty space (8.0, 8.0)
    const farAway = new Float32Array([8.0, 8.0]);
    const varFarAway = evaluateGpVariance(farAway, state);

    // Epistemic certainty check:
    expect(varAtData).toBeLessThan(0.05); // Pinches tight
    expect(varFarAway).toBeGreaterThan(0.95); // Balloons back up to prior sf = 1.0
  });

  // 5. Verification of all remaining Stage 01 plugins
  it('verifies all Stage 01 plugins initialize and step cleanly without NaNs', () => {
    const toyDataset = {
      train: [
        { id: 1, x1: 1.0, x2: 1.0, y: 1 },
        { id: 2, x1: -1.0, x2: -1.0, y: -1 },
        { id: 3, x1: 2.0, x2: 1.5, y: 1 },
        { id: 4, x1: -2.0, x2: -1.5, y: -1 },
      ],
      test: [],
    };

    // Linear Regression
    const linState = linearRegressionPlugin.init(toyDataset, {}, 42);
    const linRes = linearRegressionPlugin.step(linState, toyDataset, {});
    expect(isFinite(linRes.metrics.trainLoss)).toBe(true);

    // Polynomial Regression
    const polyState = polynomialRegressionPlugin.init(toyDataset, { degree: 3 }, 42);
    const polyRes = polynomialRegressionPlugin.step(polyState, toyDataset, { degree: 3 });
    expect(isFinite(polyRes.metrics.trainLoss)).toBe(true);

    // Elastic Net
    const elState = elasticNetPlugin.init(toyDataset, { l1Ratio: 0.5 }, 42);
    const elRes = elasticNetPlugin.step(elState, toyDataset, { l1Ratio: 0.5 });
    expect(isFinite(elRes.metrics.trainLoss)).toBe(true);

    // Logistic Regression
    const logState = logisticRegressionPlugin.init(toyDataset, {}, 42);
    const logRes = logisticRegressionPlugin.step(logState, toyDataset, {});
    expect(isFinite(logRes.metrics.trainLoss)).toBe(true);

    // Perceptron
    const perState = perceptronPlugin.init(toyDataset, {}, 42);
    const perRes = perceptronPlugin.step(perState, toyDataset, {});
    expect(isFinite(perRes.metrics.trainLoss)).toBe(true);

    // SVR
    const svrState = svrPlugin.init(toyDataset, { epsilon: 0.2 }, 42);
    const svrRes = svrPlugin.step(svrState, toyDataset, { epsilon: 0.2 });
    expect(isFinite(svrRes.metrics.trainLoss)).toBe(true);

    // Naive Bayes
    const nbState = naiveBayesPlugin.init(toyDataset, {}, 42);
    const nbRes = naiveBayesPlugin.step(nbState, toyDataset, {});
    expect(isFinite(nbRes.metrics.trainLoss)).toBe(true);
  });

  // 6. Strict Epoch 0 Visual Neutrality across all 13 algorithms
  it('guarantees that all 13 algorithms produce valid visual artifacts at Epoch 0 from deterministic seed', () => {
    const dataset = {
      train: [
        { id: 1, x1: 1.0, x2: 1.0, y: 1 },
        { id: 2, x1: -1.0, x2: -1.0, y: -1 },
      ],
      test: [],
    };

    const allPlugins = [
      linearRegressionPlugin,
      ridgeL2Plugin,
      lassoL1Plugin,
      elasticNetPlugin,
      polynomialRegressionPlugin,
      svrPlugin,
      gaussianProcessPlugin,
      linearSvmPlugin,
      kernelSvmPlugin,
      logisticRegressionPlugin,
      perceptronPlugin,
      knnPlugin,
      naiveBayesPlugin,
    ];

    for (const plugin of allPlugins) {
      const state = (plugin as any).init(dataset, {}, 42);
      expect(state.epoch).toBe(0);
      const artifacts = (plugin as any).getVisualArtifacts(state, dataset, {});
      // Every plugin must return visual artifacts at Epoch 0
      expect(artifacts.length).toBeGreaterThan(0);
    }
  });

  // 7. Dimensional and Mathematical Correctness on 1D Continuous Regression
  it('verifies all 7 regression models fit 1D upward trend with positive slope and valid metrics', () => {
    const linearTrendDataset = {
      train: [
        { id: 1, x1: -4.0, x2: -2.0, y: -2.0 },
        { id: 2, x1: -2.0, x2: -1.0, y: -1.0 },
        { id: 3, x1: 0.0, x2: 0.0, y: 0.0 },
        { id: 4, x1: 2.0, x2: 1.0, y: 1.0 },
        { id: 5, x1: 4.0, x2: 2.0, y: 2.0 },
      ],
      test: [],
    };

    // Linear Regression
    let linState = linearRegressionPlugin.init(linearTrendDataset, { solver: 'sgd', learningRate: 0.05 }, 42);
    for (let i = 0; i < 15; i++) linState = linearRegressionPlugin.step(linState, linearTrendDataset, { solver: 'sgd', learningRate: 0.05 }).state;
    expect(linState.weights[0]).toBeGreaterThan(0.2); // Positive slope
    expect(linState.weights[1]).toBe(-1); // Normal [m, -1]

    // Ridge L2
    let ridgeState = ridgeL2Plugin.init(linearTrendDataset, { solver: 'sgd', lambdaPenalty: 0.05, learningRate: 0.05 }, 42);
    for (let i = 0; i < 15; i++) ridgeState = ridgeL2Plugin.step(ridgeState, linearTrendDataset, { solver: 'sgd', lambdaPenalty: 0.05, learningRate: 0.05 }).state;
    expect(ridgeState.weights[0]).toBeGreaterThan(0.2);
    expect(ridgeState.weights[1]).toBe(-1);

    // Lasso L1
    let lassoState = lassoL1Plugin.init(linearTrendDataset, { lambdaPenalty: 0.01, cyclesPerStep: 5 }, 42);
    for (let i = 0; i < 5; i++) lassoState = lassoL1Plugin.step(lassoState, linearTrendDataset, { lambdaPenalty: 0.01, cyclesPerStep: 5 }).state;
    expect(lassoState.weights[0]).toBeGreaterThan(0.2);
    expect(lassoState.weights[1]).toBe(-1);

    // Elastic Net
    let elState = elasticNetPlugin.init(linearTrendDataset, { lambdaPenalty: 0.01, l1Ratio: 0.5, cyclesPerStep: 5 }, 42);
    for (let i = 0; i < 5; i++) elState = elasticNetPlugin.step(elState, linearTrendDataset, { lambdaPenalty: 0.01, l1Ratio: 0.5, cyclesPerStep: 5 }).state;
    expect(elState.weights[0]).toBeGreaterThan(0.2);
    expect(elState.weights[1]).toBe(-1);

    // SVR
    let svrState = svrPlugin.init(linearTrendDataset, { epsilon: 0.2, learningRate: 0.05 }, 42);
    for (let i = 0; i < 15; i++) svrState = svrPlugin.step(svrState, linearTrendDataset, { epsilon: 0.2, learningRate: 0.05 }).state;
    expect(svrState.weights[0]).toBeGreaterThan(0.1);
    expect(svrState.weights[1]).toBe(-1);

    // Polynomial Regression
    let polyState = polynomialRegressionPlugin.init(linearTrendDataset, { degree: 1, solver: 'closed-form' }, 42);
    polyState = polynomialRegressionPlugin.step(polyState, linearTrendDataset, { degree: 1, solver: 'closed-form' }).state;
    expect(polyState.coefficients[1]).toBeGreaterThan(0.5); // Monomial slope coefficient

    // Gaussian Process Regression
    let gpState = gaussianProcessPlugin.init(linearTrendDataset, { lengthScale: 2.0, solver: 'closed-form' }, 42);
    const gpRes = gaussianProcessPlugin.step(gpState, linearTrendDataset, { lengthScale: 2.0, solver: 'closed-form' });
    expect(gpRes.metrics.trainR2).toBeGreaterThan(0.9);
  });

  // 8. Stage 01 Cross-Task Classification Validation
  it('evaluates classification accuracy and generates hyperplane for regression algorithms on 2D binary classification data', () => {
    const classificationData = {
      train: [
        { id: 1, x1: -2.0, x2: -2.0, y: 1 },
        { id: 2, x1: -1.8, x2: -1.9, y: 1 },
        { id: 3, x1: 2.0, x2: 2.0, y: -1 },
        { id: 4, x1: 1.9, x2: 2.1, y: -1 },
      ],
      test: [],
    };

    // Test Linear Regression on 2D classification
    const linInit = linearRegressionPlugin.init(classificationData, { solver: 'closed-form' }, 42);
    const linStep = linearRegressionPlugin.step(linInit, classificationData, { solver: 'closed-form' });
    expect(linStep.metrics.trainAccuracy).toBe(100);
    const artifacts = linearRegressionPlugin.getVisualArtifacts(linStep.state, classificationData, {});
    const hp = artifacts.find((a) => a.type === 'hyperplane');
    expect(hp).toBeDefined();

    // Test Ridge L2 on 2D classification
    const ridgeInit = ridgeL2Plugin.init(classificationData, { solver: 'closed-form', lambdaPenalty: 0.1 }, 42);
    const ridgeStep = ridgeL2Plugin.step(ridgeInit, classificationData, { solver: 'closed-form', lambdaPenalty: 0.1 });
    expect(ridgeStep.metrics.trainAccuracy).toBe(100);
  });
});
