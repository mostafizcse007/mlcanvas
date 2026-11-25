// @vitest-environment jsdom
import { describe, it, expect, afterEach, vi, beforeAll } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { TelemetrySidebar } from '../TelemetrySidebar';
import { Stage04Canvas } from '../../rendering/Stage04Canvas';
import { mlpPlugin } from '../../algorithms/stage04/mlp';
import { ganPlugin } from '../../algorithms/stage04/gan';
import { HistoryRingBuffer } from '../../utils/ringBuffer';
import { DirtyManager } from '../../rendering/dirty-manager';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { AppWorkbenchShell } from '../AppWorkbenchShell';

beforeAll(() => {
  // Mock ResizeObserver and canvas getContext for jsdom
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as any;
  HTMLCanvasElement.prototype.getContext = (() =>
    new Proxy(
      {
        createImageData: (w: number, h: number) => ({
          width: w,
          height: h,
          data: new Uint8ClampedArray(w * h * 4),
        }),
        measureText: () => ({ width: 0 }),
      },
      {
        get(target, prop) {
          if (prop in target) return (target as any)[prop];
          return () => {};
        },
      }
    )) as any;
});

afterEach(() => {
  cleanup();
});

describe('Stage 04 UI & State Regression Tests (ISSUE-001, ISSUE-002, ISSUE-003)', () => {
  it('STAGE-04-ISSUE-002: hides Decision Threshold (τ) in Regression mode and shows it in Classification mode on MLP', () => {
    const buf = new HistoryRingBuffer(300, 16);
    const noop = () => {};

    const { rerender } = render(
      <TelemetrySidebar
        algorithm={mlpPlugin}
        datasetTaskType="regression"
        metrics={{ trainLoss: 0.1, testLoss: 0.12, trainR2: 0.92 }}
        historyBuffer={buf}
        theme="paper"
        decisionThreshold={0}
        onUpdateThreshold={noop}
      />
    );
    expect(screen.queryByLabelText('Decision Threshold')).toBeNull();

    rerender(
      <TelemetrySidebar
        algorithm={mlpPlugin}
        datasetTaskType="classification"
        metrics={{ trainLoss: 0.1, testLoss: 0.12, trainAccuracy: 95 }}
        historyBuffer={buf}
        theme="paper"
        decisionThreshold={0}
        onUpdateThreshold={noop}
      />
    );
    expect(screen.getByLabelText('Decision Threshold')).not.toBeNull();
  });

  it('STAGE-04-ISSUE-002: disables/masks X2-dependent input features on 1D regression datasets in Stage04Canvas', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-trend',
      sampleCount: 40,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });
    const dm = new DirtyManager();
    const onUpdateHyperparam = vi.fn();

    render(
      <Stage04Canvas
        theme="paper"
        trainData={dataset.train}
        testData={dataset.test}
        artifacts={[]}
        gridResolution={64}
        dirtyManager={dm}
        algorithm={mlpPlugin}
        datasetTaskType="regression"
        hyperparams={{ numLayers: 2, layer1: 4, layer2: 2, features: 'x1,x2' }}
        pointerMode="add-drag"
        activeClass={1}
        onAddPoint={() => {}}
        onUpdatePoint={() => {}}
        onDeletePoint={() => {}}
        predictValueAtPoint={() => 0}
        onUpdateHyperparam={onUpdateHyperparam}
      />
    );

    const x2DisabledNode = screen.getByTitle(/^x2 disabled on 1D regression/i);
    expect(x2DisabledNode.getAttribute('aria-disabled')).toBe('true');
    fireEvent.click(x2DisabledNode);
    expect(onUpdateHyperparam).not.toHaveBeenCalled();
  });

  it('STAGE-04-ISSUE-003: renders space between G(z) and z ~ N(0, I) in GAN header and 2·ln(2) ≈ 1.386 in Arena Status', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 40,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });
    const dm = new DirtyManager();
    const buf = new HistoryRingBuffer(300, 16);

    render(
      <div>
        <Stage04Canvas
          theme="paper"
          trainData={dataset.train}
          testData={dataset.test}
          artifacts={[]}
          gridResolution={64}
          dirtyManager={dm}
          algorithm={ganPlugin}
          datasetTaskType="classification"
          hyperparams={{}}
          pointerMode="add-drag"
          activeClass={1}
          onAddPoint={() => {}}
          onUpdatePoint={() => {}}
          onDeletePoint={() => {}}
          predictValueAtPoint={() => 0}
        />
        <TelemetrySidebar
          algorithm={ganPlugin}
          datasetTaskType="classification"
          metrics={{ trainLoss: 2 * Math.LN2, testLoss: Math.LN2 }}
          historyBuffer={buf}
          theme="paper"
          decisionThreshold={0}
          onUpdateThreshold={() => {}}
        />
      </div>
    );

    expect(screen.getByText(/Latent Noise Mesh Warp G\(z\)/).parentElement?.textContent).toContain(
      'Latent Noise Mesh Warp G(z) z ~ N(0, I)'
    );
    expect(screen.getByText(/Minimax Equilibrium: L_D\* = 2·ln\(2\) ≈ 1\.386/)).not.toBeNull();
  });

  it('STAGE-04-ISSUE-001: clears lossHistory (ring buffer) to length <= 1 when MLP topology is mutated', async () => {
    const postedMessages: any[] = [];
    class MockWorker {
      onmessage: ((e: MessageEvent) => void) | null = null;
      onerror: ((e: any) => void) | null = null;
      postMessage(msg: any) {
        postedMessages.push(msg);
        if (msg.type === 'INIT_ALGORITHM' || msg.type === 'UPDATE_HYPERPARAMS') {
          this.onmessage?.({
            data: {
              type: 'STEP_RESULT',
              result: {
                state: { epoch: 0 },
                epoch: 0,
                metrics: { trainLoss: 0.5, testLoss: 0.5 },
                converged: false,
              },
              artifacts: [],
              explanation: {
                summary: '',
                coreAssumption: '',
                whenItFails: '',
                microLog: [],
                mathLatex: { objective: '', lossOrUpdate: '', dualOrConstraint: '' },
                codeSnippet: { sklearn: '', pytorch: '' },
              },
            },
          } as MessageEvent);
        } else if (msg.type === 'STEP') {
          this.onmessage?.({
            data: {
              type: 'STEP_RESULT',
              result: {
                state: { epoch: 1 },
                epoch: 1,
                metrics: { trainLoss: 0.4, testLoss: 0.4 },
                converged: false,
              },
              artifacts: [],
              explanation: {
                summary: '',
                coreAssumption: '',
                whenItFails: '',
                microLog: [],
                mathLatex: { objective: '', lossOrUpdate: '', dualOrConstraint: '' },
                codeSnippet: { sklearn: '', pytorch: '' },
              },
            },
          } as MessageEvent);
        }
      }
      terminate() {}
    }

    (globalThis as any).Worker = MockWorker;
    window.location.hash = '#algo=mlp';

    render(<AppWorkbenchShell />);

    // Simulate stepping 5 times to populate lossHistory
    const stepBtn = screen.getByTitle(/Step \+1 Iteration/i);
    for (let i = 0; i < 5; i++) {
      fireEvent.click(stepBtn);
    }

    // Scrubber max should be >= 4 before topology mutation
    const scrubberBefore = screen.getByLabelText('Timeline Scrubber') as HTMLInputElement;
    expect(Number(scrubberBefore.max)).toBeGreaterThanOrEqual(4);

    // Mutate MLP topology by clicking Add Hidden Layer (+)
    const addLayerBtn = await screen.findByTitle('Add Hidden Layer');
    fireEvent.click(addLayerBtn);

    // After topology mutation, lossHistory / scrubber max must be cleared to <= 1 (max === 0 means count <= 1)
    const scrubberAfter = screen.getByLabelText('Timeline Scrubber') as HTMLInputElement;
    expect(Number(scrubberAfter.max)).toBeLessThanOrEqual(1);
  });
});
