import { describe, it, expect } from 'vitest';
import { serializeToUrlHash, parseUrlHash } from '../../utils/urlState';
import { polynomialRegressionPlugin } from '../stage01/polynomialRegression';
import { linearRegressionPlugin } from '../stage01/linearRegression';
import { ridgeL2Plugin } from '../stage01/ridgeL2';
import { generateTabular2dDataset } from '../../datasets/tabular2d';

describe('Telemetry & URL State Sanitization', () => {
  it('never serializes NaN into the URL address bar', () => {
    const serialized = serializeToUrlHash({
      algo: 'polynomial-regression',
      c: NaN as any,
      lambda: NaN as any,
      lr: NaN as any,
      seed: 42,
    });

    expect(serialized).not.toContain('NaN');
    expect(serialized).not.toContain('c=');
    expect(serialized).not.toContain('lam=');
    expect(serialized).not.toContain('lr=');
    expect(serialized).toContain('algo=polynomial-regression');
    expect(serialized).toContain('seed=42');
  });

  it('safely rejects NaN when parsing URL hash', () => {
    // Mock window hash
    (globalThis as any).window = {
      location: {
        hash: '#algo=polynomial-regression&c=NaN&lam=NaN&lr=NaN&seed=99',
      },
    };

    const parsed = parseUrlHash();
    expect(parsed.algo).toBe('polynomial-regression');
    expect(parsed.seed).toBe(99);
    expect(parsed.c).toBeUndefined();
    expect(parsed.lambda).toBeUndefined();
    expect(parsed.lr).toBeUndefined();
  });

  it('computes accuracy and R² for polynomial regression on 2-class datasets', () => {
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

    const params = { degree: 4, regularization: 0.01 };
    const state = polynomialRegressionPlugin.init(dataset, params, 42);
    const metrics = polynomialRegressionPlugin.loss(state, dataset, params);

    expect(metrics.trainLoss).toBeGreaterThan(0);
    expect(metrics.testLoss).toBeGreaterThan(0);
    expect(metrics.trainAccuracy).toBeDefined();
    expect(metrics.trainAccuracy).toBeGreaterThanOrEqual(0);
    expect(metrics.trainAccuracy).toBeLessThanOrEqual(100);
    expect(metrics.trainR2).toBeDefined();

    // Verify explain() syncs with lastStep metrics
    const stepResult = {
      state,
      epoch: 1,
      metrics,
    };
    const explanation = polynomialRegressionPlugin.explain(state, params, stepResult);
    expect(explanation.microLog[1]).toContain(metrics.trainLoss.toFixed(4));
    expect(explanation.microLog[1]).toContain(metrics.testLoss.toFixed(4));
  });

  it('computes accuracy and R² for linear and ridge regression on 2-class datasets', () => {
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

    const linParams = { criterion: 'mse' as const, huberDelta: 1.0, solver: 'closed-form' as const };
    const linState = linearRegressionPlugin.init(dataset, linParams, 42);
    const linMetrics = linearRegressionPlugin.loss(linState, dataset, linParams);

    expect(linMetrics.trainAccuracy).toBeDefined();
    expect(linMetrics.trainR2).toBeDefined();

    const ridgeParams = { lambdaPenalty: 0.1, solver: 'closed-form' as const };
    const ridgeState = ridgeL2Plugin.init(dataset, ridgeParams, 42);
    const ridgeMetrics = ridgeL2Plugin.loss(ridgeState, dataset, ridgeParams);

    expect(ridgeMetrics.trainAccuracy).toBeDefined();
    expect(ridgeMetrics.trainR2).toBeDefined();
  });
});
