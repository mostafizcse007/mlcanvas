import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { kernelSvmPlugin } from '../stage01/kernelSvm';
import { generateTabular2dDataset, DatasetKind } from '../../datasets/tabular2d';
import { TelemetrySidebar } from '../../ui/TelemetrySidebar';
import { Stage01Canvas } from '../../rendering/Stage01Canvas';
import { HistoryRingBuffer } from '../../utils/ringBuffer';
import { DirtyManager } from '../../rendering/dirty-manager';

const ONE_D_REGRESSION_KINDS: DatasetKind[] = [
  'sine-wave',
  'polynomial-cubic',
  'linear-trend',
];

describe('Stage 01 — Kernel SVM Cross-Task Mode on 1D Regression Datasets', () => {
  let prevDoc: unknown;
  beforeAll(() => {
    prevDoc = (globalThis as { document?: unknown }).document;
    (globalThis as { document?: unknown }).document = {
      createElement: () => ({ width: 0, height: 0, getContext: () => null }),
    };
  });
  afterAll(() => {
    (globalThis as { document?: unknown }).document = prevDoc;
  });

  it.each(ONE_D_REGRESSION_KINDS)(
    'separates binarized y >= 0 vs y < 0, emits decision-contour + SV rings, and syncs Accuracy with Core Summary on %s',
    (kind) => {
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

      const params = { kernel: 'rbf' as const, gamma: 0.5, cPenalty: 1.0 };
      let state = kernelSvmPlugin.init(dataset, params, 42);

      // Even at Epoch 0, Kernel SVM emits both decision-contour and support-vector-rings
      const initArts = kernelSvmPlugin.getVisualArtifacts(state, dataset, params);
      expect(initArts.some((a) => a.type === 'decision-contour')).toBe(true);
      expect(initArts.some((a) => a.type === 'support-vector-rings')).toBe(true);

      let lastStep = kernelSvmPlugin.step(state, dataset, params);
      state = lastStep.state;
      for (let i = 1; i < 15; i++) {
        lastStep = kernelSvmPlugin.step(state, dataset, params);
        state = lastStep.state;
      }

      // Strong separation across upper (y >= 0) vs lower (y < 0) half-planes
      expect(lastStep.metrics.trainAccuracy).toBeGreaterThan(85);
      expect(lastStep.metrics.testAccuracy).toBeGreaterThan(85);
      expect(lastStep.metrics.trainR2).toBeUndefined();

      const upperPred = Number(kernelSvmPlugin.predict(new Float32Array([0, 2.0]), state, params));
      const lowerPred = Number(kernelSvmPlugin.predict(new Float32Array([0, -2.0]), state, params));
      expect(upperPred).toBeGreaterThan(0);
      expect(lowerPred).toBeLessThan(0);

      const artifacts = kernelSvmPlugin.getVisualArtifacts(state, dataset, params);
      const contour = artifacts.find((a) => a.type === 'decision-contour');
      expect(contour).toBeDefined();
      if (contour?.type === 'decision-contour') {
        expect(contour.level).toBe(0);
        expect(contour.showMargins).toBe(true);
      }
      expect(artifacts.some((a) => a.type === 'support-vector-rings')).toBe(true);

      // Explanation Core Summary percentage must match Telemetry Accuracy card exactly
      const explanation = kernelSvmPlugin.explain(state, params, lastStep);
      const expectedAccStr = `${lastStep.metrics.trainAccuracy!.toFixed(1)}%`;
      expect(explanation.summary).toContain(expectedAccStr);

      const sidebarHtml = renderToStaticMarkup(
        <TelemetrySidebar
          algorithm={kernelSvmPlugin}
          datasetTaskType="regression"
          metrics={lastStep.metrics}
          historyBuffer={new HistoryRingBuffer(16, 16)}
          theme="paper"
          decisionThreshold={0}
          onUpdateThreshold={() => {}}
          explanation={explanation}
          currentEpoch={state.epoch}
        />
      );
      expect(sidebarHtml).toContain('Accuracy');
      expect(sidebarHtml).toContain(expectedAccStr);
      expect(sidebarHtml).not.toContain('R² Score');

      // Stage01Canvas enters Cross-Task 2D heatmap + decision boundary mode
      const noop = () => {};
      const canvasHtml = renderToStaticMarkup(
        <Stage01Canvas
          algorithm={kernelSvmPlugin}
          taskType="classification"
          datasetTaskType="regression"
          theme="paper"
          trainData={dataset.train}
          testData={dataset.test}
          artifacts={artifacts}
          gridResolution={96}
          dirtyManager={new DirtyManager()}
          pointerMode="add-drag"
          activeClass={1}
          onAddPoint={noop}
          onUpdatePoint={noop}
          onDeletePoint={noop}
          predictValueAtPoint={(x1, x2) => Number(kernelSvmPlugin.predict(new Float32Array([x1, x2]), state, params))}
        />
      );
      expect(canvasHtml).toContain('Data &amp; boundary confidence:');
      expect(canvasHtml).toContain('Discretize output');
    }
  );
});
