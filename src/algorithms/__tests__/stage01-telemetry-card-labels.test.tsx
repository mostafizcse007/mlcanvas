import { describe, it, expect } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { knnPlugin } from '../stage01/knn';
import { gaussianProcessPlugin } from '../stage01/gaussianProcess';
import { linearSvmPlugin } from '../stage01/linearSvm';
import { perceptronPlugin } from '../stage01/perceptron';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { TelemetrySidebar } from '../../ui/TelemetrySidebar';
import { HistoryRingBuffer } from '../../utils/ringBuffer';

function makeClassificationDataset() {
  return generateTabular2dDataset({
    kind: 'linear-blobs',
    sampleCount: 160,
    noise: 0.1,
    testSplit: 0.2,
    seed: 42,
    numClasses: 2,
    scaling: 'raw',
    stretchX2: false,
  });
}

describe('Stage 01 — 4th Telemetry Card Labels (k-NN, GPR, Linear SVM)', () => {
  it('displays PROBE RADIUS for k-NN classification matching k-NN boundingRadius and never Margin 2/||w||', () => {
    const dataset = makeClassificationDataset();
    const history = new HistoryRingBuffer(10, 4);

    for (const k of [5, 1]) {
      const params = { neighbors: k, metric: 'euclidean' as const, weighting: 'uniform' as const };
      const state = knnPlugin.init(dataset, params, 42);
      const stepRes = knnPlugin.step(state, dataset, params);
      const artifacts = knnPlugin.getVisualArtifacts(stepRes.state, dataset, params);
      const probeArtifact = artifacts.find((a) => a.type === 'knn-probe-rays');

      expect(stepRes.metrics.marginWidth).toBeUndefined();
      expect(probeArtifact?.type).toBe('knn-probe-rays');
      if (probeArtifact?.type === 'knn-probe-rays') {
        expect(stepRes.metrics.custom?.probeRadius).toBeCloseTo(probeArtifact.radius, 8);
      }

      const html = renderToStaticMarkup(
        <TelemetrySidebar
          algorithm={knnPlugin}
          datasetTaskType="classification"
          metrics={stepRes.metrics}
          historyBuffer={history}
          theme="paper"
          decisionThreshold={0}
          onUpdateThreshold={() => {}}
        />
      );

      expect(html).toContain('Probe Radius');
      expect(html).toContain(stepRes.state.boundingRadius.toFixed(3));
      expect(html).not.toContain('Margin');
      expect(html).not.toContain('2/||w||');
    }
  });

  it('displays MEAN STD σ for GPR classification from empirical posterior variance (not hardcoded 2.000 Margin)', () => {
    const dataset = makeClassificationDataset();
    const history = new HistoryRingBuffer(10, 4);
    const params = { lengthScale: 1.2, signalVariance: 1.0, noiseLevel: 0.02, solver: 'closed-form' as const };

    const initState = gaussianProcessPlugin.init(dataset, params, 42);
    const initMetrics = gaussianProcessPlugin.loss(initState, dataset, params);
    const stepRes = gaussianProcessPlugin.step(initState, dataset, params);

    expect(stepRes.metrics.marginWidth).toBeUndefined();
    expect(typeof stepRes.metrics.custom?.meanStd).toBe('number');
    const initStd = Number(initMetrics.custom?.meanStd);
    const convStd = Number(stepRes.metrics.custom?.meanStd);

    // Epistemic mean posterior std drops after ingesting full training data and is strictly not hardcoded 2.000
    expect(convStd).toBeGreaterThan(0);
    expect(convStd).toBeLessThan(initStd);
    expect(convStd.toFixed(3)).not.toBe('2.000');

    const html = renderToStaticMarkup(
      <TelemetrySidebar
        algorithm={gaussianProcessPlugin}
        datasetTaskType="classification"
        metrics={stepRes.metrics}
        historyBuffer={history}
        theme="paper"
        decisionThreshold={0}
        onUpdateThreshold={() => {}}
      />
    );

    expect(html).toContain('Mean Std');
    expect(html).toContain('σ');
    expect(html).toContain(convStd.toFixed(3));
    expect(html).not.toContain('Margin');
    expect(html).not.toContain('2/||w||');
  });

  it('preserves MARGIN 2/||w|| strictly for Linear SVM and never shows it on non-Linear-SVM classifiers', () => {
    const dataset = makeClassificationDataset();
    const history = new HistoryRingBuffer(10, 4);
    const params = { cPenalty: 1.0, learningRate: 0.01 };

    let state = linearSvmPlugin.init(dataset, params, 42);
    for (let i = 0; i < 15; i++) {
      state = linearSvmPlugin.step(state, dataset, params).state;
    }
    const metrics = linearSvmPlugin.loss(state, dataset, params);
    expect(metrics.marginWidth).toBeDefined();

    const svmHtml = renderToStaticMarkup(
      <TelemetrySidebar
        algorithm={linearSvmPlugin}
        datasetTaskType="classification"
        metrics={metrics}
        historyBuffer={history}
        theme="paper"
        decisionThreshold={0}
        onUpdateThreshold={() => {}}
      />
    );

    expect(svmHtml).toContain('Margin');
    expect(svmHtml).toContain('2/||w||');
    expect(svmHtml).toContain(metrics.marginWidth!.toFixed(3));

    // Even if a metric payload carries marginWidth on a non-Linear-SVM classifier, Margin 2/||w|| stays restricted to Linear SVM
    const perceptronHtml = renderToStaticMarkup(
      <TelemetrySidebar
        algorithm={perceptronPlugin}
        datasetTaskType="classification"
        metrics={{ trainLoss: 0.1, testLoss: 0.1, trainAccuracy: 95, marginWidth: 1.5 }}
        historyBuffer={history}
        theme="paper"
        decisionThreshold={0}
        onUpdateThreshold={() => {}}
      />
    );

    expect(perceptronHtml).not.toContain('Margin');
    expect(perceptronHtml).not.toContain('2/||w||');
  });
});
