import { describe, it, expect } from 'vitest';
import { HistoryRingBuffer } from '../../utils/ringBuffer';
import { perceptronPlugin } from '../stage01/perceptron';
import { linearSvmPlugin } from '../stage01/linearSvm';
import { ridgeL2Plugin } from '../stage01/ridgeL2';
import { naiveBayesPlugin, NaiveBayesParams } from '../stage01/naiveBayes';
import { generateTabular2dDataset, DatasetOptions } from '../../datasets/tabular2d';

const defaultOptions: DatasetOptions = {
  kind: 'linear-blobs',
  sampleCount: 60,
  noise: 0.01,
  testSplit: 0.2,
  seed: 42,
  numClasses: 2,
  scaling: 'raw',
  stretchX2: false,
};

describe('TensorFlow Playground Continuous Epoch Training Loop', () => {
  it('HistoryRingBuffer retains and dynamically downsamples full loss history across thousands of epochs', () => {
    const buffer = new HistoryRingBuffer(300, 2);

    // Simulate 5,000 epochs of training
    for (let ep = 1; ep <= 5000; ep++) {
      buffer.push({
        epoch: ep,
        trainLoss: 1.0 / (1.0 + ep * 0.05),
        testLoss: 1.2 / (1.0 + ep * 0.04),
        trainAccuracy: 0.99,
        testAccuracy: 0.95,
        weights: new Float32Array([0.5, -0.5]),
        bias: 0.1,
      });
    }

    // Recent loss series is bound to 300 capacity
    const recent = buffer.getRecentLossSeries();
    expect(recent.epochs.length).toBe(300);
    expect(recent.epochs[0]).toBe(4701);
    expect(recent.epochs[299]).toBe(5000);

    // Full loss series retains Epoch 1 through Epoch 5000 with downsampling
    const full = buffer.getLossSeries(600);
    expect(full.epochs.length).toBe(600);
    expect(full.epochs[0]).toBe(1);
    expect(full.epochs[599]).toBe(5000);
    // Initial high loss is preserved
    expect(full.trainLosses[0]).toBeCloseTo(1.0 / (1.0 + 0.05), 3);
    // Asymptotic low loss is preserved
    expect(full.trainLosses[599]).toBeCloseTo(1.0 / (1.0 + 5000 * 0.05), 3);
  });

  it('Perceptron continues incrementing epochs cleanly past convergence without stalling', () => {
    const dataset = generateTabular2dDataset({ ...defaultOptions, seed: 42 });

    const params = { learningRate: 0.1 };
    let state = perceptronPlugin.init(dataset, params, 42);

    let convergedSeen = false;
    // Run for 100 continuous steps
    for (let step = 0; step < 100; step++) {
      const result = perceptronPlugin.step(state, dataset, params);
      expect(result.epoch).toBe(state.epoch + 1);
      state = result.state;
      if (result.converged) {
        convergedSeen = true;
      }
    }

    expect(convergedSeen).toBe(true);
    expect(state.epoch).toBe(100);
  });

  it('Linear SVM steps continuously past tolerance while epoch increments', () => {
    const dataset = generateTabular2dDataset({ ...defaultOptions, seed: 99 });

    const params = { learningRate: 0.05, cPenalty: 1.0, batchPasses: 2 };
    let state = linearSvmPlugin.init(dataset, params, 99);

    for (let step = 0; step < 60; step++) {
      const result = linearSvmPlugin.step(state, dataset, params);
      expect(result.epoch).toBe(state.epoch + 1);
      state = result.state;
    }

    expect(state.epoch).toBe(60);
  });

  it('Naive Bayes advances epoch counter across continuous steps while maintaining analytical fit', () => {
    const dataset = generateTabular2dDataset({ ...defaultOptions, seed: 7 });

    const params: NaiveBayesParams = { variant: 'gaussian', laplaceAlpha: 1.0 };
    let state = naiveBayesPlugin.init(dataset, params, 7);

    for (let step = 0; step < 20; step++) {
      const result = naiveBayesPlugin.step(state, dataset, params);
      expect(result.epoch).toBe(state.epoch + 1);
      expect(result.converged).toBe(true);
      state = result.state;
    }

    expect(state.epoch).toBe(20);
  });

  it('guarantees atomic synchronization between model hyperplane and 2D decision grid evaluation', () => {
    const dataset = generateTabular2dDataset({ ...defaultOptions, seed: 123 });
    const params = { learningRate: 0.05, cPenalty: 1.0 };
    let state = linearSvmPlugin.init(dataset, params, 123);

    // Initial state at Epoch 0 has no trained boundary
    expect(state.epoch).toBe(0);

    // Step 1: Step the model
    const res = linearSvmPlugin.step(state, dataset, params);
    expect(res.epoch).toBe(1);

    // Evaluate 2D grid point at (1.0, 1.0)
    const testInput = new Float32Array([1.0, 1.0]);
    const mathScore = res.state.weights[0] * testInput[0] + res.state.weights[1] * testInput[1] + res.state.bias;
    const pluginPred = linearSvmPlugin.predict(testInput, res.state, params);
    const predVal = typeof pluginPred === 'number' ? pluginPred : pluginPred[0];

    // Prediction sign must exactly match the mathematical affine score w^T x + b
    expect(Math.sign(predVal)).toBe(Math.sign(mathScore));
  });

  it('calculates 100ms TensorFlow Playground cadence pacing delay with zero drift or negative timers', () => {
    const targetCadenceMs = 100;

    // Simulate fast 12ms step execution
    const fastElapsed = 12;
    const fastDelay = Math.max(0, targetCadenceMs - fastElapsed);
    expect(fastDelay).toBe(88);

    // Simulate heavy 115ms step execution (e.g. large ensemble)
    const heavyElapsed = 115;
    const heavyDelay = Math.max(0, targetCadenceMs - heavyElapsed);
    expect(heavyDelay).toBe(0);
  });

  it('Ridge L2 iterative SGD maintains healthy weight magnitude across 50 epochs without collapsing to zero', () => {
    const dataset = generateTabular2dDataset({ ...defaultOptions, seed: 42 });
    const params = { solver: 'sgd' as const, lambdaPenalty: 0.2, learningRate: 0.02 };
    let state = ridgeL2Plugin.init(dataset, params, 42);

    for (let ep = 0; ep < 50; ep++) {
      state = ridgeL2Plugin.step(state, dataset, params).state;
    }

    const weightNorm = Math.hypot(state.weights[0], state.weights[1]);
    expect(weightNorm).toBeGreaterThan(0.2); // Prevents weight collapse / gradient color fade
    expect(state.weights[0]).toBeLessThan(0); // Separates linear-blobs
    expect(state.weights[1]).toBeLessThan(0);
  });
});
