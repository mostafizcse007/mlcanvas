import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { svrPlugin } from '../stage01/svr';
import { gaussianProcessPlugin } from '../stage01/gaussianProcess';
import {
  generateTabular2dDataset,
  getDatasetTaskType,
  CLASSIFICATION_DATASETS,
  DatasetKind,
} from '../../datasets/tabular2d';
import { Stage01Canvas } from '../../rendering/Stage01Canvas';
import { InteractiveStageContainer } from '../../ui/InteractiveStageContainer';
import { DirtyManager } from '../../rendering/dirty-manager';

const ONE_D_REGRESSION_DATASETS: DatasetKind[] = [
  'sine-wave',
  'polynomial-cubic',
  'linear-trend',
];

const TWO_D_DATASETS: DatasetKind[] = [
  'linear-surface',
  ...CLASSIFICATION_DATASETS,
];

function makeDataset(
  kind: DatasetKind,
  scaling: 'raw' | 'zscore' | 'minmax' = 'raw',
  stretchX2: boolean = false
) {
  return generateTabular2dDataset({
    kind,
    sampleCount: 80,
    noise: 0.1,
    testSplit: 0.2,
    seed: 42,
    numClasses: 2,
    scaling,
    stretchX2,
  });
}

describe('Stage 01 — SVR & Gaussian Process 1D vs 2D Overlay Gating Across All 8 Datasets', () => {
  describe('SVR (svrPlugin) 1D vs 2D artifact and prediction behavior', () => {
    it.each(ONE_D_REGRESSION_DATASETS)(
      'emits 1D ε-tube, curve-fit, SV rings, and vertical residual-bars on %s (including scaled modes)',
      (kind) => {
        for (const [scaling, stretch] of [
          ['raw', false],
          ['zscore', false],
          ['minmax', false],
          ['raw', true],
        ] as const) {
          const dataset = makeDataset(kind, scaling, stretch);
          const params = { epsilon: 0.4, cPenalty: 1.0, learningRate: 0.02 };
          const initState = svrPlugin.init(dataset, params, 42);
          const stepRes = svrPlugin.step(initState, dataset, params);

          for (const st of [initState, stepRes.state]) {
            expect(st.is1D).toBe(true);
            const arts = svrPlugin.getVisualArtifacts(st, dataset, params);
            const types = arts.map((a) => a.type);

            expect(types).toContain('epsilon-tube');
            expect(types).toContain('curve-fit');
            expect(types).toContain('support-vector-rings');
            expect(types).toContain('residual-bars');
            expect(types).not.toContain('hyperplane');
          }
        }
      }
    );

    it.each(TWO_D_DATASETS)(
      'never emits 1D ε-tube, curve-fit, or residual-bars on 2D dataset %s (emits 2D hyperplane + SV rings only)',
      (kind) => {
        const dataset = makeDataset(kind);
        const params = { epsilon: 0.4, cPenalty: 1.0, learningRate: 0.02 };
        const initState = svrPlugin.init(dataset, params, 42);
        let state = initState;
        for (let i = 0; i < 15; i++) {
          state = svrPlugin.step(state, dataset, params).state;
        }

        for (const st of [initState, state]) {
          expect(st.is1D).toBe(false);
          expect(st.curvePoints).toHaveLength(0);
          expect(st.upperTubePoints).toHaveLength(0);
          expect(st.lowerTubePoints).toHaveLength(0);

          const arts = svrPlugin.getVisualArtifacts(st, dataset, params);
          const types = arts.map((a) => a.type);

          expect(types).not.toContain('epsilon-tube');
          expect(types).not.toContain('curve-fit');
          expect(types).not.toContain('residual-bars');
          expect(types).toContain('hyperplane');
          expect(types).toContain('support-vector-rings');
        }

        // Predictions depend on both x1 and x2 (no 1D collapse)
        const predPosX2 = svrPlugin.predict(new Float32Array([1.0, 2.5]), state, params);
        const predNegX2 = svrPlugin.predict(new Float32Array([1.0, -2.5]), state, params);
        expect(predPosX2).not.toBe(predNegX2);

        // Snapshot restore preserves is1D === false
        const buf = svrPlugin.serializeSnapshot(state);
        const restored = svrPlugin.restoreSnapshot(buf, { ...state, is1D: undefined });
        expect(restored.is1D).toBe(false);
      }
    );
  });

  describe('Gaussian Process Regression (gaussianProcessPlugin) 1D vs 2D artifact and prediction behavior', () => {
    it.each(ONE_D_REGRESSION_DATASETS)(
      'emits 1D μ(x*) ± 2σ(x*) uncertainty-band and curve-fit on %s (including scaled modes)',
      (kind) => {
        for (const [scaling, stretch] of [
          ['raw', false],
          ['zscore', false],
          ['minmax', false],
          ['raw', true],
        ] as const) {
          const dataset = makeDataset(kind, scaling, stretch);
          const params = { lengthScale: 1.2, signalVariance: 1.0, noiseLevel: 0.02, solver: 'closed-form' as const };
          const initState = gaussianProcessPlugin.init(dataset, params, 42);
          const stepRes = gaussianProcessPlugin.step(initState, dataset, params);

          for (const st of [initState, stepRes.state]) {
            expect(st.is1D).toBe(true);
            expect(st.meanCurve.length).toBeGreaterThan(10);
            expect(st.upperBand.length).toBeGreaterThan(10);
            expect(st.lowerBand.length).toBeGreaterThan(10);

            const arts = gaussianProcessPlugin.getVisualArtifacts(st, dataset, params);
            const types = arts.map((a) => a.type);

            expect(types).toContain('uncertainty-band');
            expect(types).toContain('curve-fit');
            expect(types).not.toContain('uncertainty-surface');
          }
        }
      }
    );

    it.each(TWO_D_DATASETS)(
      'never emits 1D uncertainty-band, posterior sample curves, or 1D curve-fit on 2D dataset %s',
      (kind) => {
        const dataset = makeDataset(kind);
        const params = { lengthScale: 1.2, signalVariance: 1.0, noiseLevel: 0.02, solver: 'closed-form' as const };
        const initState = gaussianProcessPlugin.init(dataset, params, 42);
        const stepRes = gaussianProcessPlugin.step(initState, dataset, params);

        for (const st of [initState, stepRes.state]) {
          expect(st.is1D).toBe(false);
          expect(st.meanCurve).toHaveLength(0);
          expect(st.upperBand).toHaveLength(0);
          expect(st.lowerBand).toHaveLength(0);
          expect(st.samples).toHaveLength(0);

          const arts = gaussianProcessPlugin.getVisualArtifacts(st, dataset, params);
          const types = arts.map((a) => a.type);

          expect(types).not.toContain('uncertainty-band');
          expect(types).not.toContain('curve-fit');
          expect(types).not.toContain('epsilon-tube');
          expect(types).not.toContain('residual-bars');
          expect(types).toContain('uncertainty-surface');
        }

        // 2D prediction varies across x2 when training data spans (x1, x2)
        const predUpper = Number(gaussianProcessPlugin.predict(new Float32Array([0.0, 2.5]), stepRes.state, params));
        const predLower = Number(gaussianProcessPlugin.predict(new Float32Array([0.0, -2.5]), stepRes.state, params));
        expect(Number.isFinite(predUpper)).toBe(true);
        expect(Number.isFinite(predLower)).toBe(true);
        if (kind !== 'concentric-circles' && kind !== 'xor-quadrants') {
          expect(Math.abs(predUpper - predLower)).toBeGreaterThan(1e-3);
        }

        // Snapshot restore preserves is1D === false
        const buf = gaussianProcessPlugin.serializeSnapshot(stepRes.state);
        const restored = gaussianProcessPlugin.restoreSnapshot(buf, { ...stepRes.state, is1D: true });
        expect(restored.is1D).toBe(false);
      }
    );
  });

  describe('Stage01Canvas & InteractiveStageContainer strict is1DRegression UI gating', () => {
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

    it('enables 1D curve legend ONLY on the 3 1D regression datasets and enforces 2D mode on 2D Surface + 4 Classification datasets', () => {
      const noop = () => {};

      for (const algo of [svrPlugin, gaussianProcessPlugin]) {
        for (const kind of ONE_D_REGRESSION_DATASETS) {
          const dataset = makeDataset(kind);
          const canvasHtml = renderToStaticMarkup(
            <Stage01Canvas
              algorithm={algo}
              taskType="regression"
              datasetTaskType={getDatasetTaskType(kind)}
              theme="paper"
              trainData={dataset.train}
              testData={dataset.test}
              artifacts={[]}
              gridResolution={96}
              dirtyManager={new DirtyManager()}
              pointerMode="add-drag"
              activeClass={1}
              onAddPoint={noop}
              onUpdatePoint={noop}
              onDeletePoint={noop}
              predictValueAtPoint={() => 0}
            />
          );
          expect(canvasHtml).toContain('Continuous 1D response curve');
          expect(canvasHtml).not.toContain('Discretize output');
        }

        for (const kind of TWO_D_DATASETS) {
          const dataset = makeDataset(kind);
          const datasetTaskType = getDatasetTaskType(kind);
          // Even if stale 1D artifacts are passed, classification / 2D surface must remain in 2D mode
          const stale1DArtifacts = [
            {
              type: 'epsilon-tube' as const,
              pointsUpper: [[0, 1], [1, 2]] as [number, number][],
              pointsLower: [[0, -1], [1, 0]] as [number, number][],
              epsilon: 0.4,
            },
            {
              type: 'uncertainty-band' as const,
              meanCurve: [[0, 0], [1, 1]] as [number, number][],
              upperBand: [[0, 1], [1, 2]] as [number, number][],
              lowerBand: [[0, -1], [1, 0]] as [number, number][],
            },
          ];

          const canvasHtml = renderToStaticMarkup(
            <Stage01Canvas
              algorithm={algo}
              taskType="regression"
              datasetTaskType={datasetTaskType}
              theme="paper"
              trainData={dataset.train}
              testData={dataset.test}
              artifacts={stale1DArtifacts}
              gridResolution={96}
              dirtyManager={new DirtyManager()}
              pointerMode="add-drag"
              activeClass={1}
              onAddPoint={noop}
              onUpdatePoint={noop}
              onDeletePoint={noop}
              predictValueAtPoint={() => 0}
            />
          );
          expect(canvasHtml).not.toContain('Continuous 1D response curve');
          expect(canvasHtml).toContain('Discretize output');
          if (kind === 'linear-surface') {
            expect(canvasHtml).toContain('Predicted surface elevation:');
          } else {
            expect(canvasHtml).toContain('Data &amp; boundary confidence:');
          }

          const containerHtml = renderToStaticMarkup(
            <InteractiveStageContainer
              algorithm={algo}
              theme="paper"
              datasetTaskType={datasetTaskType}
              trainData={dataset.train}
              testData={dataset.test}
              artifacts={[]}
              gridResolution={96}
              dirtyManager={new DirtyManager()}
              circuitBreakerTripped={false}
              pointerMode="add-drag"
              activeClass={1}
              onAddPoint={noop}
              onUpdatePoint={noop}
              onDeletePoint={noop}
              onClampAndRecover={noop}
              predictValueAtPoint={() => 0}
            />
          );
          expect(containerHtml).toContain('Continuous 2D Manifold');
          expect(containerHtml).not.toContain('Continuous 1D Regression Curve');
        }
      }
    });
  });
});
