import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import { DataPoint2D } from '../../datasets/tabular2d';
import { DirtyManager } from '../dirty-manager';
import {
  getVisibleDataPoints,
  hitTestDataPoints,
  renderDataPoints2D,
} from '../pointUtils';
import { Stage01Canvas } from '../Stage01Canvas';
import { Stage02Canvas } from '../Stage02Canvas';
import { Stage04Canvas } from '../Stage04Canvas';
import { InteractiveStageContainer } from '../../ui/InteractiveStageContainer';
import { linearSvmPlugin } from '../../algorithms/stage01/linearSvm';
import { decisionTreePlugin } from '../../algorithms/stage02/decisionTree';
import { mlpPlugin } from '../../algorithms/stage04/mlp';
import { autoencoderPlugin } from '../../algorithms/stage04/autoencoder';
import { ganPlugin } from '../../algorithms/stage04/gan';

interface RecordedArc {
  x: number;
  y: number;
  radius: number;
  fillStyle?: string;
  strokeStyle?: string;
  lineWidth?: number;
}

function createMockCanvasContext() {
  const arcs: RecordedArc[] = [];
  let currentArc: RecordedArc | null = null;
  const ctx: Record<string, unknown> = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    font: '',
    textAlign: 'left',
    textBaseline: 'alphabetic',
    beginPath: () => {
      currentArc = null;
    },
    arc: (x: number, y: number, radius: number) => {
      currentArc = { x, y, radius };
      arcs.push(currentArc);
    },
    fill: () => {
      if (currentArc) {
        currentArc.fillStyle = String(ctx.fillStyle);
      }
    },
    stroke: () => {
      if (currentArc) {
        currentArc.strokeStyle = String(ctx.strokeStyle);
        currentArc.lineWidth = Number(ctx.lineWidth);
      }
    },
    save: () => {},
    restore: () => {},
    setLineDash: () => {},
    fillText: () => {},
    clearRect: () => {},
    fillRect: () => {},
    strokeRect: () => {},
    scale: () => {},
  };

  return {
    ctx: ctx as unknown as CanvasRenderingContext2D,
    arcs,
  };
}

const sampleTrain2D: DataPoint2D[] = [
  { id: 1, x1: -2, x2: -2, y: -1, isTest: false },
  { id: 2, x1: 2, x2: 2, y: 1, isTest: false },
];

const sampleTest2D: DataPoint2D[] = [
  { id: 101, x1: -3, x2: 3, y: -1, isTest: true },
  { id: 102, x1: 3, x2: -3, y: 1, isTest: true },
];

const sampleTrain1D: DataPoint2D[] = [
  { id: 11, x1: -2, x2: -1, y: -1, isTest: false },
  { id: 12, x1: 2, x2: 1, y: 1, isTest: false },
];

const sampleTest1D: DataPoint2D[] = [
  { id: 201, x1: 0, x2: 0.5, y: 0.5, isTest: true },
];

const noop = () => {};
const predictZero = () => 0;

let prevDocument: unknown;

beforeAll(() => {
  prevDocument = (globalThis as { document?: unknown }).document;
  (globalThis as { document?: unknown }).document = {
    createElement: () => ({
      width: 0,
      height: 0,
      getContext: () => null,
    }),
  };
});

afterAll(() => {
  (globalThis as { document?: unknown }).document = prevDocument;
});

describe('Global showTestData visibility, hit-testing, and component binding', () => {
  it('omits testData and any isTest points when showTestData is false', () => {
    const visibleAll = getVisibleDataPoints(sampleTrain2D, sampleTest2D, true);
    expect(visibleAll.map((p) => p.id)).toEqual([1, 2, 101, 102]);

    const visibleTrainOnly = getVisibleDataPoints(sampleTrain2D, sampleTest2D, false);
    expect(visibleTrainOnly.map((p) => p.id)).toEqual([1, 2]);

    // Also filters out any test-flagged point inside trainData and omits testData lacking explicit isTest
    const mixedTrain: DataPoint2D[] = [
      ...sampleTrain1D,
      { id: 999, x1: 1, x2: 0.5, y: 0.5, isTest: true },
    ];
    const unflaggedTest: DataPoint2D[] = [{ id: 202, x1: -1, x2: -0.5, y: -0.5 }];

    expect(getVisibleDataPoints(mixedTrain, unflaggedTest, false).map((p) => p.id)).toEqual([11, 12]);
    const shown1D = getVisibleDataPoints(sampleTrain1D, unflaggedTest, true);
    expect(shown1D).toHaveLength(3);
    expect(shown1D[2].isTest).toBe(true);
  });

  it('skips pointer hit-testing for test points when showTestData is false', () => {
    const width = 120;
    const height = 120;
    const toPxX = (x1: number, w: number) => ((x1 + 6) / 12) * w;
    const toPxY = (x2: number, h: number) => h - ((x2 + 6) / 12) * h;

    // Coordinates of test point id=101 (-3, 3) -> px=30, py=30
    const testPx = toPxX(-3, width);
    const testPy = toPxY(3, height);

    const visibleWhenShown = getVisibleDataPoints(sampleTrain2D, sampleTest2D, true);
    const hitShown = hitTestDataPoints(visibleWhenShown, {
      px: testPx,
      py: testPy,
      width,
      height,
      hitRadiusPx: 10,
      toPxX,
      toPxY,
    });
    expect(hitShown?.id).toBe(101);

    const visibleWhenHidden = getVisibleDataPoints(sampleTrain2D, sampleTest2D, false);
    const hitHidden = hitTestDataPoints(visibleWhenHidden, {
      px: testPx,
      py: testPy,
      width,
      height,
      hitRadiusPx: 10,
      toPxX,
      toPxY,
    });
    expect(hitHidden).toBeNull();
  });

  it('draws ringed test points when showTestData is true and omits them when false (Stages 01/02 & 04 styles, 2D & 1D)', () => {
    const width = 240;
    const height = 240;
    const toPxX = (x1: number, w: number) => ((x1 + 6) / 12) * w;
    const toPxY = (x2: number, h: number) => h - ((x2 + 6) / 12) * h;

    // Stage 01 / 02 style (with inner micro-dot for test points) on 2D data
    const shown01 = createMockCanvasContext();
    renderDataPoints2D(shown01.ctx, {
      trainData: sampleTrain2D,
      testData: sampleTest2D,
      showTestData: true,
      width,
      height,
      toPxX,
      toPxY,
      getColor: (pt) => (pt.y > 0 ? '#D94E34' : '#1E40AF'),
      isCarbon: false,
    });
    // 2 train arcs + 2*(outer ring + inner micro-dot) = 6 arcs
    expect(shown01.arcs).toHaveLength(6);
    expect(shown01.arcs.some((a) => a.x === toPxX(-3, width) && a.y === toPxY(3, height))).toBe(true);

    const hidden01 = createMockCanvasContext();
    renderDataPoints2D(hidden01.ctx, {
      trainData: sampleTrain2D,
      testData: sampleTest2D,
      showTestData: false,
      width,
      height,
      toPxX,
      toPxY,
      getColor: (pt) => (pt.y > 0 ? '#D94E34' : '#1E40AF'),
      isCarbon: false,
    });
    // Only 2 train arcs, 0 test arcs
    expect(hidden01.arcs).toHaveLength(2);
    expect(hidden01.arcs.some((a) => a.x === toPxX(-3, width) && a.y === toPxY(3, height))).toBe(false);

    // Stage 04 style (innerDotRadius: 0) on 1D regression data
    const shown04 = createMockCanvasContext();
    renderDataPoints2D(shown04.ctx, {
      trainData: sampleTrain1D,
      testData: sampleTest1D,
      showTestData: true,
      width,
      height,
      toPxX,
      toPxY,
      getColor: () => '#D94E34',
      isCarbon: false,
      trainRadius: 4.5,
      testRadius: 4.0,
      testStrokeWidth: 1.5,
      innerDotRadius: 0,
    });
    expect(shown04.arcs).toHaveLength(3);
    expect(shown04.arcs.some((a) => a.x === toPxX(0, width) && a.y === toPxY(0.5, height))).toBe(true);

    const hidden04 = createMockCanvasContext();
    renderDataPoints2D(hidden04.ctx, {
      trainData: sampleTrain1D,
      testData: sampleTest1D,
      showTestData: false,
      width,
      height,
      toPxX,
      toPxY,
      getColor: () => '#D94E34',
      isCarbon: false,
      trainRadius: 4.5,
      testRadius: 4.0,
      testStrokeWidth: 1.5,
      innerDotRadius: 0,
    });
    expect(hidden04.arcs).toHaveLength(2);
    expect(hidden04.arcs.some((a) => a.x === toPxX(0, width) && a.y === toPxY(0.5, height))).toBe(false);
  });

  it('binds showTestData prop to the "Show test data" checkbox across Stages 01, 02, and 04 without altering model state', () => {
    const extractCheckboxChecked = (html: string): boolean => {
      const match = html.match(/<input[^>]*type="checkbox"[^>]*\/><span>Show test data<\/span>/);
      expect(match).not.toBeNull();
      return /\bchecked(?:="")?/.test(match![0]);
    };

    const dirty01 = new DirtyManager();
    const s01Checked = renderToStaticMarkup(
      <InteractiveStageContainer
        algorithm={linearSvmPlugin}
        theme="paper"
        trainData={sampleTrain2D}
        testData={sampleTest2D}
        artifacts={[]}
        gridResolution={96}
        dirtyManager={dirty01}
        circuitBreakerTripped={false}
        pointerMode="add-drag"
        activeClass={1}
        showTestData={true}
        onToggleShowTestData={noop}
        onAddPoint={noop}
        onUpdatePoint={noop}
        onDeletePoint={noop}
        onClampAndRecover={noop}
        predictValueAtPoint={predictZero}
      />
    );
    const s01Unchecked = renderToStaticMarkup(
      <Stage01Canvas
        algorithm={linearSvmPlugin}
        theme="paper"
        trainData={sampleTrain2D}
        testData={sampleTest2D}
        artifacts={[]}
        gridResolution={96}
        dirtyManager={dirty01}
        pointerMode="add-drag"
        activeClass={1}
        showTestData={false}
        onToggleShowTestData={noop}
        onAddPoint={noop}
        onUpdatePoint={noop}
        onDeletePoint={noop}
        predictValueAtPoint={predictZero}
      />
    );
    expect(extractCheckboxChecked(s01Checked)).toBe(true);
    expect(extractCheckboxChecked(s01Unchecked)).toBe(false);

    const dirty02 = new DirtyManager();
    const s02Checked = renderToStaticMarkup(
      <Stage02Canvas
        algorithm={decisionTreePlugin}
        theme="paper"
        trainData={sampleTrain2D}
        testData={sampleTest2D}
        artifacts={[]}
        gridResolution={96}
        dirtyManager={dirty02}
        pointerMode="add-drag"
        activeClass={1}
        showTestData={true}
        onToggleShowTestData={noop}
        onAddPoint={noop}
        onUpdatePoint={noop}
        onDeletePoint={noop}
        predictValueAtPoint={predictZero}
      />
    );
    const s02Unchecked = renderToStaticMarkup(
      <Stage02Canvas
        algorithm={decisionTreePlugin}
        theme="paper"
        trainData={sampleTrain2D}
        testData={sampleTest2D}
        artifacts={[]}
        gridResolution={96}
        dirtyManager={dirty02}
        pointerMode="add-drag"
        activeClass={1}
        showTestData={false}
        onToggleShowTestData={noop}
        onAddPoint={noop}
        onUpdatePoint={noop}
        onDeletePoint={noop}
        predictValueAtPoint={predictZero}
      />
    );
    expect(extractCheckboxChecked(s02Checked)).toBe(true);
    expect(extractCheckboxChecked(s02Unchecked)).toBe(false);

    for (const algo of [mlpPlugin, autoencoderPlugin, ganPlugin]) {
      const dirty04 = new DirtyManager();
      const s04Checked = renderToStaticMarkup(
        <Stage04Canvas
          algorithm={algo}
          theme="paper"
          trainData={sampleTrain2D}
          testData={sampleTest2D}
          artifacts={[]}
          gridResolution={96}
          dirtyManager={dirty04}
          pointerMode="add-drag"
          activeClass={1}
          showTestData={true}
          onToggleShowTestData={noop}
          onAddPoint={noop}
          onUpdatePoint={noop}
          onDeletePoint={noop}
          predictValueAtPoint={predictZero}
        />
      );
      const s04Unchecked = renderToStaticMarkup(
        <Stage04Canvas
          algorithm={algo}
          theme="paper"
          trainData={sampleTrain2D}
          testData={sampleTest2D}
          artifacts={[]}
          gridResolution={96}
          dirtyManager={dirty04}
          pointerMode="add-drag"
          activeClass={1}
          showTestData={false}
          onToggleShowTestData={noop}
          onAddPoint={noop}
          onUpdatePoint={noop}
          onDeletePoint={noop}
          predictValueAtPoint={predictZero}
        />
      );
      expect(extractCheckboxChecked(s04Checked)).toBe(true);
      expect(extractCheckboxChecked(s04Unchecked)).toBe(false);
    }

    // Verify redraw signaling & uninterrupted test-loss computation on full dataset
    const dirty = new DirtyManager();
    expect(dirty.checkAndClear()).toBe(true);
    expect(dirty.checkAndClear()).toBe(false);
    dirty.markDirty();
    expect(dirty.checkAndClear()).toBe(true);

    const dataset = { train: sampleTrain2D, test: sampleTest2D };
    const init = linearSvmPlugin.init(dataset, { learningRate: 0.05, cPenalty: 1.0 }, 42);
    const step1 = linearSvmPlugin.step(init, dataset, { learningRate: 0.05, cPenalty: 1.0 });
    expect(step1.epoch).toBe(1);
    expect(Number.isFinite(step1.metrics.testLoss)).toBe(true);
  });
});
