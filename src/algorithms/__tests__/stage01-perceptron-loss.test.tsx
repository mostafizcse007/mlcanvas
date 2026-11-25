import { describe, it, expect } from 'vitest';
import { perceptronPlugin } from '../stage01/perceptron';
import { generateTabular2dDataset, DatasetKind } from '../../datasets/tabular2d';

describe('Stage 01 — Rosenblatt Perceptron Epoch 0 Evaluation & Unified Loss Units', () => {
  it('evaluates Epoch 0 misclassifications honestly and never claims 100% convergence when errors > 0', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 160,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { learningRate: 0.1 };
    const initState = perceptronPlugin.init(dataset, params, 42);
    const initMetrics = perceptronPlugin.loss(initState, dataset, params);
    const explanation = perceptronPlugin.explain(initState, params, {
      state: initState,
      epoch: 0,
      metrics: initMetrics,
      converged: false,
    });

    expect(initState.epoch).toBe(0);
    // Random initial hyperplane misclassifies roughly half of opposite-quadrant blobs
    expect(initState.mistakesCount).toBeGreaterThan(0);
    expect(initMetrics.trainLoss).toBeGreaterThan(0);
    expect(initMetrics.testLoss).toBeGreaterThan(0);
    expect(initMetrics.trainAccuracy!).toBeLessThan(100);
    expect(explanation.microLog.join(' ')).not.toContain('Zero mistakes');
    expect(explanation.microLog.join(' ')).not.toContain('100% convergence');
    expect(explanation.microLog[0]).toContain(`${initState.mistakesCount}`);
  });

  it('uses identical mean 0-1 misclassification rate units in [0, 1] for Train Loss and Test Loss on non-separable datasets', () => {
    for (const kind of ['concentric-circles', 'xor-quadrants', 'twin-spirals'] as DatasetKind[]) {
      const dataset = generateTabular2dDataset({
        kind,
        sampleCount: 160,
        noise: 0.1,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const params = { learningRate: 0.1 };
      let state = perceptronPlugin.init(dataset, params, 42);
      let metrics = perceptronPlugin.loss(state, dataset, params);

      for (let e = 0; e < 10; e++) {
        const res = perceptronPlugin.step(state, dataset, params);
        state = res.state;
        metrics = res.metrics;
      }

      // Both trainLoss and testLoss MUST lie strictly in [0, 1] (never raw count 63 vs percentage 71.875)
      expect(metrics.trainLoss).toBeGreaterThan(0.1);
      expect(metrics.trainLoss).toBeLessThanOrEqual(1.0);
      expect(metrics.testLoss).toBeGreaterThan(0.1);
      expect(metrics.testLoss).toBeLessThanOrEqual(1.0);

      // Exact mathematical identity: trainLoss == 1 - trainAccuracy / 100, testLoss == 1 - testAccuracy / 100
      expect(metrics.trainLoss).toBeCloseTo(1 - metrics.trainAccuracy! / 100, 10);
      expect(metrics.testLoss).toBeCloseTo(1 - metrics.testAccuracy! / 100, 10);

      const explanation = perceptronPlugin.explain(state, params, {
        state,
        epoch: state.epoch,
        metrics,
      });
      expect(explanation.microLog.join(' ')).not.toContain('Zero mistakes');
    }
  });

  it('reaches 0.0000 trainLoss and zero-mistake convergence cleanly on linearly separable blobs', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 160,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { learningRate: 0.1 };
    let state = perceptronPlugin.init(dataset, params, 42);
    let lastStep = perceptronPlugin.step(state, dataset, params);
    state = lastStep.state;
    for (let i = 0; i < 40 && !lastStep.converged; i++) {
      lastStep = perceptronPlugin.step(state, dataset, params);
      state = lastStep.state;
    }

    expect(lastStep.converged).toBe(true);
    expect(state.mistakesCount).toBe(0);
    expect(lastStep.metrics.trainLoss).toBe(0);
    expect(lastStep.metrics.trainAccuracy).toBe(100);
    expect(lastStep.metrics.testLoss).toBeCloseTo(1 - lastStep.metrics.testAccuracy! / 100, 10);

    const explanation = perceptronPlugin.explain(state, params, lastStep);
    expect(explanation.microLog[1]).toContain('Zero mistakes! Separating hyperplane has achieved 100% convergence.');
  });
});
