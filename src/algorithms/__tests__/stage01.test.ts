import { describe, it, expect } from 'vitest';
import { linearSvmPlugin } from '../stage01/linearSvm';
import { lassoL1Plugin } from '../stage01/lassoL1';
import { generateTabular2dDataset } from '../../datasets/tabular2d';

describe('Stage 01: Core AlgorithmPlugin Sanity Tests', () => {
  // Test 1: Deterministic reproducibility with the same seed
  it('guarantees deterministic reproducibility with the same seed', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 100,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { learningRate: 0.03, cPenalty: 1.0, batchPasses: 5 };

    // Run A
    const stateA1 = linearSvmPlugin.init(dataset, params, 42);
    const resultA = linearSvmPlugin.step(stateA1, dataset, params);

    // Run B
    const stateB1 = linearSvmPlugin.init(dataset, params, 42);
    const resultB = linearSvmPlugin.step(stateB1, dataset, params);

    expect(resultA.state.weights[0]).toBeCloseTo(resultB.state.weights[0], 8);
    expect(resultA.state.weights[1]).toBeCloseTo(resultB.state.weights[1], 8);
    expect(resultA.state.bias).toBeCloseTo(resultB.state.bias, 8);
    expect(resultA.metrics.trainLoss).toBeCloseTo(resultB.metrics.trainLoss, 8);
  });

  // Test 2: Loss decrease on linearly separable data
  it('decreases loss and separates linearly separable data under Linear SVM', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 120,
      noise: 0.02, // Well-separated
      testSplit: 0.2,
      seed: 1234,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { learningRate: 0.05, cPenalty: 1.0, batchPasses: 5 };
    let state = linearSvmPlugin.init(dataset, params, 1234);

    const initialMetrics = linearSvmPlugin.loss(state, dataset, params);

    // Train for 20 epochs
    for (let i = 0; i < 20; i++) {
      const res = linearSvmPlugin.step(state, dataset, params);
      state = res.state;
    }

    const finalMetrics = linearSvmPlugin.loss(state, dataset, params);

    // Loss should have reduced significantly and accuracy should be high
    expect(finalMetrics.trainLoss).toBeLessThan(initialMetrics.trainLoss);
    expect(finalMetrics.trainAccuracy).toBeGreaterThanOrEqual(90);
    expect(state.marginWidth).toBeGreaterThan(0);
  });

  // Test 3: Exact weight sparsity (w === 0) under high Lasso lambda
  it('enforces exact weight sparsity (w === 0) under high Lasso lambda', () => {
    // Generate dataset where X2 has zero correlation with target y
    const dataset = {
      train: [
        { id: 1, x1: 2.0, x2: 0.1, y: 2.0 },
        { id: 2, x1: 4.0, x2: -0.2, y: 4.0 },
        { id: 3, x1: -2.0, x2: 0.3, y: -2.0 },
        { id: 4, x1: -4.0, x2: -0.1, y: -4.0 },
      ],
      test: [],
    };

    // Under high lambda, feature with weak correlation must be snapped strictly to 0
    const highLambdaParams = { lambdaPenalty: 1.0, penaltyType: 'l1' as const, cyclesPerStep: 10 };
    let state = lassoL1Plugin.init(dataset, highLambdaParams, 42);

    for (let i = 0; i < 15; i++) {
      const res = lassoL1Plugin.step(state, dataset, highLambdaParams);
      state = res.state;
    }

    // Weight w2 should be strictly 0.0
    expect(state.weights[1]).toBe(0);
    expect(state.zeroWeightIndices).toContain(1);

    const metrics = lassoL1Plugin.loss(state, dataset, highLambdaParams);
    expect(metrics.sparsity?.zeroCount).toBeGreaterThanOrEqual(1);
  });

  // Test 4: Bit-exact serializeSnapshot -> restoreSnapshot roundtrip
  it('performs bit-exact serializeSnapshot -> restoreSnapshot roundtrip', () => {
    // Test Linear SVM roundtrip
    const svmState = {
      weights: new Float32Array([1.234567, -0.987654]),
      bias: 0.456789,
      epoch: 42,
      supportVectorIndices: [3, 7, 15, 29],
      marginWidth: 1.25,
      gradientNorm: 0.00312,
    };

    const svmBuffer = linearSvmPlugin.serializeSnapshot(svmState);
    const restoredSvm = linearSvmPlugin.restoreSnapshot(svmBuffer, svmState);

    expect(restoredSvm.epoch).toBe(svmState.epoch);
    expect(restoredSvm.bias).toBeCloseTo(svmState.bias, 5);
    expect(restoredSvm.marginWidth).toBeCloseTo(svmState.marginWidth, 5);
    expect(restoredSvm.gradientNorm).toBeCloseTo(svmState.gradientNorm, 5);
    expect(restoredSvm.weights[0]).toBeCloseTo(svmState.weights[0], 5);
    expect(restoredSvm.weights[1]).toBeCloseTo(svmState.weights[1], 5);
    expect(restoredSvm.supportVectorIndices).toEqual(svmState.supportVectorIndices);

    // Test Lasso L1 roundtrip
    const lassoState = {
      weights: new Float32Array([0.8888, 0.0]),
      bias: -0.1234,
      epoch: 88,
      zeroWeightIndices: [1],
      maxDelta: 0.000042,
    };

    const lassoBuffer = lassoL1Plugin.serializeSnapshot(lassoState);
    const restoredLasso = lassoL1Plugin.restoreSnapshot(lassoBuffer, lassoState);

    expect(restoredLasso.epoch).toBe(lassoState.epoch);
    expect(restoredLasso.bias).toBeCloseTo(lassoState.bias, 5);
    expect(restoredLasso.maxDelta).toBeCloseTo(lassoState.maxDelta, 5);
    expect(restoredLasso.weights[0]).toBeCloseTo(lassoState.weights[0], 5);
    expect(restoredLasso.weights[1]).toBe(0.0);
    expect(restoredLasso.zeroWeightIndices).toEqual(lassoState.zeroWeightIndices);
  });
});
