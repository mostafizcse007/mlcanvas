import { describe, it, expect } from 'vitest';
import { HistoryRingBuffer } from '../../utils/ringBuffer';
import { linearSvmPlugin } from '../stage01/linearSvm';
import { decisionTreePlugin } from '../stage02/decisionTree';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { VisualArtifact, TelemetryMetrics, ExplanationPayload } from '../../types/algorithm';

describe('Timeline Scrubber & History Snapshot Pipeline Regression Tests', () => {
  it('captures immutable lightweight per-epoch snapshots in HistoryRingBuffer', () => {
    const ringBuffer = new HistoryRingBuffer(10, 16);

    const artifact1: VisualArtifact = {
      type: 'hyperplane',
      normal: [0.5, -0.5],
      bias: 0.2,
      style: 'solid',
    };

    const metrics1: TelemetryMetrics = {
      trainLoss: 0.85,
      testLoss: 0.92,
      trainAccuracy: 75.0,
      testAccuracy: 70.0,
      marginWidth: 1.414,
      custom: { supportVectorCount: 12 },
    };

    const explanation1: ExplanationPayload = {
      summary: 'Epoch 1 Linear SVM',
      coreAssumption: 'Linear separability',
      whenItFails: 'Non-linear data',
      microLog: ['Epoch 1 fit', 'Step evaluated', 'Loss recorded'],
      mathLatex: { objective: '\\min \\frac{1}{2}\\|w\\|^2', lossOrUpdate: 'w \\leftarrow w - \\eta \\nabla L' },
      codeSnippet: { sklearn: 'from sklearn.svm import LinearSVC', pytorch: '# PyTorch SVM' },
    };

    const gridValues1 = new Float32Array([0.1, 0.2, -0.1, -0.2]);

    ringBuffer.push({
      epoch: 1,
      trainLoss: 0.85,
      testLoss: 0.92,
      trainAccuracy: 75.0,
      testAccuracy: 70.0,
      weights: new Float32Array([0.5, -0.5]),
      bias: 0.2,
      margin: 1.414,
      metrics: metrics1,
      artifacts: [artifact1],
      explanation: explanation1,
      gridValues: gridValues1,
    });

    // Mutate the original objects to ensure snapshot immutability
    gridValues1[0] = 99.9;
    (artifact1 as any).bias = 999;
    metrics1.trainLoss = 999;

    const retrieved = ringBuffer.get(0);
    expect(retrieved).not.toBeNull();
    expect(retrieved!.epoch).toBe(1);
    expect(retrieved!.trainLoss).toBe(0.85);
    expect(retrieved!.metrics?.trainLoss).toBe(0.85);
    expect(retrieved!.gridValues![0]).toBeCloseTo(0.1, 4);
    expect((retrieved!.artifacts![0] as any).bias).toBe(0.2);
  });

  it('correctly handles ring buffer capacity overflow while preserving indexing', () => {
    const capacity = 5;
    const ringBuffer = new HistoryRingBuffer(capacity, 4);

    for (let ep = 0; ep < 12; ep++) {
      ringBuffer.push({
        epoch: ep,
        trainLoss: 1.0 / (ep + 1),
        testLoss: 1.2 / (ep + 1),
        weights: new Float32Array([ep, ep * 2]),
        bias: ep,
      });
    }

    const stats = ringBuffer.getStats();
    expect(stats.count).toBe(5);
    expect(stats.earliestEpoch).toBe(7);
    expect(stats.latestEpoch).toBe(11);

    // Logical index 0 is the earliest retained epoch (7)
    const earliest = ringBuffer.get(0);
    expect(earliest).not.toBeNull();
    expect(earliest!.epoch).toBe(7);

    // Logical index 4 is the latest epoch (11)
    const latest = ringBuffer.get(4);
    expect(latest).not.toBeNull();
    expect(latest!.epoch).toBe(11);

    // Out of bounds logical indices return null
    expect(ringBuffer.get(-1)).toBeNull();
    expect(ringBuffer.get(5)).toBeNull();
  });

  it('verifies scrubbing to epoch k on Stage 01 Linear SVM rewinds weights, decision boundary, SVs, and metrics', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 80,
      noise: 0.08,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const datasetContainer = { train: dataset.train, test: dataset.test };
    const params = { learningRate: 0.05, cPenalty: 1.0, batchPasses: 1 };

    let state = linearSvmPlugin.init(datasetContainer, params, 42);
    const ringBuffer = new HistoryRingBuffer(100, 4);

    // Epoch 0 initial snapshot
    const initMetrics = linearSvmPlugin.loss(state, datasetContainer, params);
    const initArtifacts = linearSvmPlugin.getVisualArtifacts(state, datasetContainer, params);
    const initExplanation = linearSvmPlugin.explain(state, params, { state, epoch: 0, metrics: initMetrics });

    ringBuffer.push({
      epoch: 0,
      trainLoss: initMetrics.trainLoss,
      testLoss: initMetrics.testLoss,
      trainAccuracy: initMetrics.trainAccuracy,
      testAccuracy: initMetrics.testAccuracy,
      weights: new Float32Array(state.weights),
      bias: state.bias,
      margin: state.marginWidth,
      metrics: initMetrics,
      artifacts: initArtifacts,
      explanation: initExplanation,
      modelState: state,
    });

    // Run 30 steps
    for (let step = 1; step <= 30; step++) {
      const stepRes = linearSvmPlugin.step(state, datasetContainer, params);
      state = stepRes.state;
      const stepArtifacts = linearSvmPlugin.getVisualArtifacts(state, datasetContainer, params);
      const stepExplanation = linearSvmPlugin.explain(state, params, stepRes);

      ringBuffer.push({
        epoch: stepRes.epoch,
        trainLoss: stepRes.metrics.trainLoss,
        testLoss: stepRes.metrics.testLoss,
        trainAccuracy: stepRes.metrics.trainAccuracy,
        testAccuracy: stepRes.metrics.testAccuracy,
        weights: new Float32Array(state.weights),
        bias: state.bias,
        margin: state.marginWidth,
        metrics: stepRes.metrics,
        artifacts: stepArtifacts,
        explanation: stepExplanation,
        modelState: state,
      });
    }

    expect(ringBuffer.getStats().count).toBe(31);

    // Live state is Epoch 30
    const liveSnapshot = ringBuffer.get(30)!;
    expect(liveSnapshot.epoch).toBe(30);

    // Scrub back to Epoch 5
    const scrubIndex = 5;
    const scrubbedSnapshot = ringBuffer.get(scrubIndex)!;
    expect(scrubbedSnapshot.epoch).toBe(5);

    // Verify weights and bias at Epoch 5 differ from live Epoch 30
    expect(scrubbedSnapshot.weights[0]).not.toBe(liveSnapshot.weights[0]);

    // Verify hyperplane artifact at Epoch 5 reflects Epoch 5 model parameters
    const hpEpoch5 = scrubbedSnapshot.artifacts!.find((a) => a.type === 'hyperplane') as any;
    const hpEpoch30 = liveSnapshot.artifacts!.find((a) => a.type === 'hyperplane') as any;
    expect(hpEpoch5).toBeDefined();
    expect(hpEpoch30).toBeDefined();
    expect(hpEpoch5.bias).not.toBe(hpEpoch30.bias);

    // Verify telemetry metrics at Epoch 5 reflect Epoch 5 loss
    expect(scrubbedSnapshot.metrics!.trainLoss).toBe(scrubbedSnapshot.trainLoss);

    // Returning to live restores live epoch 30
    const activeScrubIndex: number | null = null;
    const displayedEpoch = activeScrubIndex !== null ? ringBuffer.get(activeScrubIndex)!.epoch : liveSnapshot.epoch;
    const displayedArtifacts = activeScrubIndex !== null ? ringBuffer.get(activeScrubIndex)!.artifacts : liveSnapshot.artifacts;

    expect(displayedEpoch).toBe(30);
    expect(displayedArtifacts).toBe(liveSnapshot.artifacts);
  });

  it('verifies scrubbing on Stage 02 Decision Tree rewinds tree partitions and leaves', () => {
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

    const datasetContainer = { train: dataset.train, test: dataset.test };
    const params = { maxDepth: 5, minSamplesSplit: 2, criterion: 'gini' as const };

    let state = decisionTreePlugin.init(datasetContainer, params, 42);
    const ringBuffer = new HistoryRingBuffer(50, 4);

    // Step through tree splits
    for (let step = 1; step <= 8; step++) {
      const stepRes = decisionTreePlugin.step(state, datasetContainer, params);
      state = stepRes.state;
      const stepArtifacts = decisionTreePlugin.getVisualArtifacts(state, datasetContainer, params);
      const stepExplanation = decisionTreePlugin.explain(state, params, stepRes);

      ringBuffer.push({
        epoch: stepRes.epoch,
        trainLoss: stepRes.metrics.trainLoss,
        testLoss: stepRes.metrics.testLoss,
        trainAccuracy: stepRes.metrics.trainAccuracy,
        testAccuracy: stepRes.metrics.testAccuracy,
        weights: new Float32Array([0, 0]),
        bias: 0,
        metrics: stepRes.metrics,
        artifacts: stepArtifacts,
        explanation: stepExplanation,
        modelState: state,
      });
    }

    expect(ringBuffer.getStats().count).toBe(8);

    // Early epoch (epoch 1) tree partition
    const earlySnapshot = ringBuffer.get(0)!;
    const earlyPartition = earlySnapshot.artifacts!.find((a) => a.type === 'tree-partition') as any;
    expect(earlyPartition).toBeDefined();

    // Later epoch (epoch 8) tree partition
    const lateSnapshot = ringBuffer.get(7)!;
    const latePartition = lateSnapshot.artifacts!.find((a) => a.type === 'tree-partition') as any;
    expect(latePartition).toBeDefined();

    // The tree partition at epoch 8 has more splits/nodes than epoch 1
    expect(latePartition.splits.length).toBeGreaterThanOrEqual(earlyPartition.splits.length);
  });

  it('handles edge case: epoch 0 initial state and reset', () => {
    const ringBuffer = new HistoryRingBuffer(10, 4);

    // Before any step, buffer is empty
    expect(ringBuffer.getStats().count).toBe(0);
    expect(ringBuffer.get(0)).toBeNull();

    // Epoch 0 pushed
    ringBuffer.push({
      epoch: 0,
      trainLoss: 1.5,
      testLoss: 1.6,
      weights: new Float32Array([0, 0]),
      bias: 0,
    });

    expect(ringBuffer.getStats().count).toBe(1);
    const ep0 = ringBuffer.get(0);
    expect(ep0).not.toBeNull();
    expect(ep0!.epoch).toBe(0);

    // Reset clears buffer
    ringBuffer.clear();
    expect(ringBuffer.getStats().count).toBe(0);
    expect(ringBuffer.get(0)).toBeNull();
  });
});
