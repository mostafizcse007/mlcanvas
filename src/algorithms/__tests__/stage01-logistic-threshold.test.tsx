import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  logisticRegressionPlugin,
  clampThreshold,
  LOGREG_MIN_THRESHOLD,
  LOGREG_MAX_THRESHOLD,
} from '../stage01/logisticRegression';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { sigmoid } from '../../utils/math';
import { TopControlBar } from '../../ui/TopControlBar';
import { TelemetrySidebar } from '../../ui/TelemetrySidebar';
import { HistoryRingBuffer } from '../../utils/ringBuffer';

function computeExpectedAccuracy(
  points: { x1: number; x2: number; y: number }[],
  weights: Float32Array,
  bias: number,
  tau: number
): number {
  let correct = 0;
  for (const pt of points) {
    const prob = sigmoid(weights[0] * pt.x1 + weights[1] * pt.x2 + bias);
    const pred = prob >= tau ? 1 : -1;
    const target = pt.y >= 0 ? 1 : -1;
    if (pred === target) correct++;
  }
  return points.length > 0 ? (correct / points.length) * 100 : 100;
}

describe('Stage 01 — Logistic Regression Decision Threshold (τ) Synchronization & Inference Sweep', () => {
  it('clamps τ to [0.01, 0.99] so logit ln(τ/(1-τ)) never produces Infinity or NaN', () => {
    expect(LOGREG_MIN_THRESHOLD).toBe(0.01);
    expect(LOGREG_MAX_THRESHOLD).toBe(0.99);

    expect(clampThreshold(0)).toBe(0.01);
    expect(clampThreshold(-0.5)).toBe(0.01);
    expect(clampThreshold(1)).toBe(0.99);
    expect(clampThreshold(1.5)).toBe(0.99);
    expect(clampThreshold(Number.NaN)).toBe(0.5);
    expect(clampThreshold(Number.POSITIVE_INFINITY)).toBe(0.5);
    expect(clampThreshold(0.91)).toBe(0.91);

    const thresholdParam = logisticRegressionPlugin.primaryParams.find((p) => p.id === 'threshold');
    expect(thresholdParam).toBeDefined();
    expect(thresholdParam?.min).toBe(0.01);
    expect(thresholdParam?.max).toBe(0.99);
    expect(thresholdParam?.step).toBe(0.01);
    expect(thresholdParam?.default).toBe(0.5);

    const { train, test } = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 160,
      noise: 0.15,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });
    const dataset = { train, test };
    let state = logisticRegressionPlugin.init(dataset, { learningRate: 0.05, cPenalty: 1.0, threshold: 0.5 }, 42);
    for (let i = 0; i < 40; i++) {
      state = logisticRegressionPlugin.step(state, dataset, { learningRate: 0.05, cPenalty: 1.0, threshold: 0.5 }).state;
    }

    for (const extremeTau of [0, -1, 1, 2, Number.NaN]) {
      const params = { learningRate: 0.05, cPenalty: 1.0, threshold: extremeTau };
      const metrics = logisticRegressionPlugin.loss(state, dataset, params);
      expect(Number.isFinite(metrics.trainLoss)).toBe(true);
      expect(Number.isFinite(metrics.testLoss)).toBe(true);
      expect(Number.isFinite(metrics.trainAccuracy ?? NaN)).toBe(true);
      expect(Number.isFinite(metrics.testAccuracy ?? NaN)).toBe(true);

      const artifacts = logisticRegressionPlugin.getVisualArtifacts(state, dataset, params);
      const hyperplane = artifacts.find((a) => a.type === 'hyperplane');
      expect(hyperplane).toBeDefined();
      if (hyperplane?.type === 'hyperplane') {
        expect(Number.isFinite(hyperplane.bias)).toBe(true);
        expect(Number.isFinite(hyperplane.normal[0])).toBe(true);
        expect(Number.isFinite(hyperplane.normal[1])).toBe(true);
      }

      const pred = logisticRegressionPlugin.predict(new Float32Array([0.2, -0.3]), state, params);
      expect(Number.isFinite(pred)).toBe(true);

      const explanation = logisticRegressionPlugin.explain(state, params);
      expect(explanation.microLog).not.toContain('NaN');
      expect(explanation.microLog).not.toContain('Infinity');
    }
  });

  it('sweeping τ updates Accuracy and shifts the boundary without altering weights, log-loss, or epoch', () => {
    // Use moderate noise so points populate probability band (0.05, 0.95) and τ shifts measurably affect Accuracy
    const { train, test } = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 160,
      noise: 0.35,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });
    const dataset = { train, test };
    const baseParams = { learningRate: 0.05, cPenalty: 1.0, threshold: 0.5 };
    let state = logisticRegressionPlugin.init(dataset, baseParams, 42);

    for (let i = 0; i < 80; i++) {
      state = logisticRegressionPlugin.step(state, dataset, baseParams).state;
    }

    expect(state.epoch).toBe(80);
    const snapshotWeights = new Float32Array(state.weights!);
    const snapshotBias = state.bias;
    const snapshotEpoch = state.epoch;
    const baseMetrics = logisticRegressionPlugin.loss(state, dataset, baseParams);

    const trainAccuracies = new Map<number, number>();

    for (const tau of [0.01, 0.1, 0.25, 0.5, 0.75, 0.91, 0.95, 0.99]) {
      const updatedState = { ...state, threshold: clampThreshold(tau) };
      const params = { ...baseParams, threshold: tau };

      const metrics = logisticRegressionPlugin.loss(updatedState, dataset, params);

      // Inference-time τ must NEVER change trained weights, bias, epoch, or cross-entropy log-loss
      expect(updatedState.weights![0]).toBe(snapshotWeights[0]);
      expect(updatedState.weights![1]).toBe(snapshotWeights[1]);
      expect(updatedState.bias).toBe(snapshotBias);
      expect(updatedState.epoch).toBe(snapshotEpoch);
      expect(metrics.trainLoss).toBeCloseTo(baseMetrics.trainLoss, 10);
      expect(metrics.testLoss).toBeCloseTo(baseMetrics.testLoss, 10);

      // Accuracy must match σ(wᵀx + b) >= τ exactly
      const expectedTrainAcc = computeExpectedAccuracy(train, snapshotWeights, snapshotBias, tau);
      const expectedTestAcc = computeExpectedAccuracy(test, snapshotWeights, snapshotBias, tau);
      expect(metrics.trainAccuracy).toBeCloseTo(expectedTrainAcc, 10);
      expect(metrics.testAccuracy).toBeCloseTo(expectedTestAcc, 10);
      trainAccuracies.set(tau, metrics.trainAccuracy!);

      // Hyperplane equation wᵀx + b_eff = 0 with b_eff = b - ln(τ / (1 - τ))
      const artifacts = logisticRegressionPlugin.getVisualArtifacts(updatedState, dataset, params);
      const hyperplane = artifacts.find((a) => a.type === 'hyperplane');
      expect(hyperplane).toBeDefined();
      if (hyperplane?.type === 'hyperplane') {
        const expectedEffBias = snapshotBias - Math.log(tau / (1 - tau));
        expect(hyperplane.normal[0]).toBe(snapshotWeights[0]);
        expect(hyperplane.normal[1]).toBe(snapshotWeights[1]);
        expect(hyperplane.bias).toBeCloseTo(expectedEffBias, 8);
        expect(hyperplane.label).toBe(`P = ${tau.toFixed(2)}`);
      }

      // Predict centers at 0 right on the σ(wᵀx + b) == τ iso-probability contour
      const predOnBoundary = logisticRegressionPlugin.predict(
        new Float32Array([
          (Math.log(tau / (1 - tau)) - snapshotBias) / snapshotWeights[0],
          0,
        ]),
        updatedState,
        params
      );
      expect(predOnBoundary).toBeCloseTo(0, 6);
    }

    // Balanced optimal threshold 0.50 must outperform extreme imbalanced cutoff 0.99 on balanced blobs
    expect(trainAccuracies.get(0.5)!).toBeGreaterThan(trainAccuracies.get(0.99)!);
    expect(trainAccuracies.get(0.5)!).toBeGreaterThan(trainAccuracies.get(0.01)!);
  });

  it('keeps TopControlBar and TelemetrySidebar synchronized to one threshold value and [0.01, 0.99] bounds', () => {
    const synchronizedTau = 0.91;
    const metrics = {
      trainLoss: 0.1421,
      testLoss: 0.1583,
      trainAccuracy: 87.5,
      testAccuracy: 84.4,
    };

    const topHtml = renderToStaticMarkup(
      <TopControlBar
        isRunning={false}
        epoch={81}
        isConverged={true}
        currentAlgorithm={logisticRegressionPlugin}
        hyperparams={{ learningRate: 0.05, cPenalty: 1.0, threshold: synchronizedTau }}
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

    const sidebarHtml = renderToStaticMarkup(
      <TelemetrySidebar
        algorithm={logisticRegressionPlugin}
        datasetTaskType="classification"
        metrics={metrics}
        historyBuffer={new HistoryRingBuffer(16, 16)}
        theme="paper"
        decisionThreshold={synchronizedTau}
        onUpdateThreshold={() => {}}
        isConverged={true}
        currentEpoch={81}
      />
    );

    // Header slider displays synchronized 0.91 readout and [0.01, 0.99] range
    expect(topHtml).toContain('id="top-param-threshold"');
    expect(topHtml).toContain('min="0.01"');
    expect(topHtml).toContain('max="0.99"');
    expect(topHtml).toContain('step="0.01"');
    expect(topHtml).toContain('value="0.91"');
    expect(topHtml).toContain('>0.91</span>');
    expect(topHtml).toContain('000,081');

    // Right-panel slider displays identical 0.91 readout and [0.01, 0.99] range
    expect(sidebarHtml).toContain('id="decision-threshold-slider-2"');
    expect(sidebarHtml).toContain('min="0.01"');
    expect(sidebarHtml).toContain('max="0.99"');
    expect(sidebarHtml).toContain('step="0.01"');
    expect(sidebarHtml).toContain('value="0.91"');
    expect(sidebarHtml).toContain('>0.91</span>');
  });
});
