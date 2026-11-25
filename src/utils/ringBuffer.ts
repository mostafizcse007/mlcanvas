import { HistorySnapshot, RingBufferStats } from '../types/history';

export class HistoryRingBuffer {
  private readonly capacity: number;
  private maxFeatures: number;

  // Contiguous Float32Arrays for zero-allocation performance on 4GB devices
  private epochs: Float32Array;
  private trainLosses: Float32Array;
  private testLosses: Float32Array;
  private trainAccuracies: Float32Array;
  private testAccuracies: Float32Array;
  private biases: Float32Array;
  private gradientNorms: Float32Array;
  private margins: Float32Array;
  private nonZeroCounts: Float32Array;
  private weightsBuffer: Float32Array;

  private head: number = 0;
  private count: number = 0;
  private snapshots: (HistorySnapshot | null)[];

  // Full-history series for continuous learning trajectory across thousands of epochs
  private fullEpochs: number[] = [];
  private fullTrainLosses: number[] = [];
  private fullTestLosses: number[] = [];
  private fullEpisodeReturns: number[] = [];
  private fullMovingAvgReturns: number[] = [];

  constructor(capacity: number = 300, maxFeatures: number = 16) {
    this.capacity = capacity;
    this.maxFeatures = maxFeatures;

    this.epochs = new Float32Array(capacity);
    this.trainLosses = new Float32Array(capacity);
    this.testLosses = new Float32Array(capacity);
    this.trainAccuracies = new Float32Array(capacity);
    this.testAccuracies = new Float32Array(capacity);
    this.biases = new Float32Array(capacity);
    this.gradientNorms = new Float32Array(capacity);
    this.margins = new Float32Array(capacity);
    this.nonZeroCounts = new Float32Array(capacity);

    this.weightsBuffer = new Float32Array(capacity * maxFeatures);
    this.snapshots = new Array(capacity).fill(null);
  }

  public push(snapshot: HistorySnapshot): void {
    const slot = (this.head + this.count) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    } else {
      this.head = (this.head + 1) % this.capacity;
    }

    this.epochs[slot] = snapshot.epoch;
    this.trainLosses[slot] = snapshot.trainLoss;
    this.testLosses[slot] = snapshot.testLoss;
    this.trainAccuracies[slot] = snapshot.trainAccuracy ?? 0;
    this.testAccuracies[slot] = snapshot.testAccuracy ?? 0;
    this.biases[slot] = snapshot.bias;
    this.gradientNorms[slot] = snapshot.gradientNorm ?? 0;
    this.margins[slot] = snapshot.margin ?? 0;
    this.nonZeroCounts[slot] = snapshot.nonZeroCount ?? 0;

    // Full-history tracking
    this.fullEpochs.push(snapshot.epoch);
    this.fullTrainLosses.push(snapshot.trainLoss);
    this.fullTestLosses.push(snapshot.testLoss);
    this.fullEpisodeReturns.push(Number(snapshot.metrics?.custom?.episodeReturn ?? 0));
    this.fullMovingAvgReturns.push(Number(snapshot.metrics?.custom?.movingAvgReturn ?? 0));

    const wLen = Math.min(snapshot.weights.length, this.maxFeatures);
    const offset = slot * this.maxFeatures;
    for (let i = 0; i < wLen; i++) {
      this.weightsBuffer[offset + i] = snapshot.weights[i];
    }

    // Capture immutable per-epoch snapshot
    this.snapshots[slot] = {
      epoch: snapshot.epoch,
      trainLoss: snapshot.trainLoss,
      testLoss: snapshot.testLoss,
      trainAccuracy: snapshot.trainAccuracy,
      testAccuracy: snapshot.testAccuracy,
      weights: new Float32Array(snapshot.weights),
      bias: snapshot.bias,
      gradientNorm: snapshot.gradientNorm,
      margin: snapshot.margin,
      nonZeroCount: snapshot.nonZeroCount,
      metrics: snapshot.metrics
        ? {
            ...snapshot.metrics,
            custom: snapshot.metrics.custom ? { ...snapshot.metrics.custom } : undefined,
          }
        : undefined,
      artifacts: snapshot.artifacts ? snapshot.artifacts.map((a) => ({ ...a })) : undefined,
      explanation: snapshot.explanation ? { ...snapshot.explanation } : undefined,
      gridValues: snapshot.gridValues ? new Float32Array(snapshot.gridValues) : undefined,
      modelState: snapshot.modelState
        ? (typeof snapshot.modelState === 'object' ? { ...snapshot.modelState } : snapshot.modelState)
        : undefined,
    };
  }

  public get(logicalIndex: number): HistorySnapshot | null {
    if (logicalIndex < 0 || logicalIndex >= this.count) {
      return null;
    }
    const physicalIndex = (this.head + logicalIndex) % this.capacity;
    return this.readPhysical(physicalIndex);
  }

  public getByEpoch(targetEpoch: number): HistorySnapshot | null {
    for (let i = 0; i < this.count; i++) {
      const snap = this.get(i);
      if (snap && snap.epoch === targetEpoch) {
        return snap;
      }
    }
    return null;
  }

  public getSnapshotIndexForEpoch(targetEpoch: number): number | null {
    for (let i = 0; i < this.count; i++) {
      const snap = this.get(i);
      if (snap && snap.epoch === targetEpoch) {
        return i;
      }
    }
    return null;
  }

  public getLatest(): HistorySnapshot | null {
    if (this.count === 0) return null;
    return this.get(this.count - 1);
  }

  public getStats(): RingBufferStats {
    const earliest = this.count > 0 ? this.epochs[this.head] : 0;
    const latestSlot = this.count > 0 ? (this.head + this.count - 1) % this.capacity : 0;
    const latest = this.count > 0 ? this.epochs[latestSlot] : 0;

    return {
      capacity: this.capacity,
      count: this.count,
      head: this.head,
      earliestEpoch: earliest,
      latestEpoch: latest,
    };
  }

  public getLossSeries(maxDisplayPoints: number = 600): { epochs: number[]; trainLosses: number[]; testLosses: number[] } {
    const total = this.fullEpochs.length;
    if (total === 0) {
      return { epochs: [], trainLosses: [], testLosses: [] };
    }

    if (total <= maxDisplayPoints) {
      return {
        epochs: this.fullEpochs.slice(),
        trainLosses: this.fullTrainLosses.slice(),
        testLosses: this.fullTestLosses.slice(),
      };
    }

    // Dynamic downsampling preserving first, last, and uniform intermediate strides
    const step = (total - 1) / (maxDisplayPoints - 1);
    const epochs: number[] = new Array(maxDisplayPoints);
    const trainLosses: number[] = new Array(maxDisplayPoints);
    const testLosses: number[] = new Array(maxDisplayPoints);

    for (let i = 0; i < maxDisplayPoints; i++) {
      const idx = Math.min(total - 1, Math.round(i * step));
      epochs[i] = this.fullEpochs[idx];
      trainLosses[i] = this.fullTrainLosses[idx];
      testLosses[i] = this.fullTestLosses[idx];
    }

    return { epochs, trainLosses, testLosses };
  }

  public getRLReturnSeries(maxDisplayPoints: number = 600): {
    epochs: number[];
    episodeReturns: number[];
    movingAvgReturns: number[];
  } {
    const total = this.fullEpochs.length;
    if (total === 0) {
      return { epochs: [], episodeReturns: [], movingAvgReturns: [] };
    }

    if (total <= maxDisplayPoints) {
      return {
        epochs: this.fullEpochs.slice(),
        episodeReturns: this.fullEpisodeReturns.slice(),
        movingAvgReturns: this.fullMovingAvgReturns.slice(),
      };
    }

    const step = (total - 1) / (maxDisplayPoints - 1);
    const epochs: number[] = new Array(maxDisplayPoints);
    const episodeReturns: number[] = new Array(maxDisplayPoints);
    const movingAvgReturns: number[] = new Array(maxDisplayPoints);

    for (let i = 0; i < maxDisplayPoints; i++) {
      const idx = Math.min(total - 1, Math.round(i * step));
      epochs[i] = this.fullEpochs[idx];
      episodeReturns[i] = this.fullEpisodeReturns[idx];
      movingAvgReturns[i] = this.fullMovingAvgReturns[idx];
    }

    return { epochs, episodeReturns, movingAvgReturns };
  }

  public getRecentLossSeries(): { epochs: number[]; trainLosses: number[]; testLosses: number[] } {
    const epochs: number[] = new Array(this.count);
    const trainLosses: number[] = new Array(this.count);
    const testLosses: number[] = new Array(this.count);

    for (let i = 0; i < this.count; i++) {
      const idx = (this.head + i) % this.capacity;
      epochs[i] = this.epochs[idx];
      trainLosses[i] = this.trainLosses[idx];
      testLosses[i] = this.testLosses[idx];
    }

    return { epochs, trainLosses, testLosses };
  }

  public clear(): void {
    this.head = 0;
    this.count = 0;
    this.epochs.fill(0);
    this.trainLosses.fill(0);
    this.testLosses.fill(0);
    this.weightsBuffer.fill(0);
    this.snapshots.fill(null);
    this.fullEpochs = [];
    this.fullTrainLosses = [];
    this.fullTestLosses = [];
    this.fullEpisodeReturns = [];
    this.fullMovingAvgReturns = [];
  }

  public dispose(): void {
    this.clear();
  }

  private readPhysical(slot: number): HistorySnapshot {
    const cached = this.snapshots[slot];
    if (cached) {
      return cached;
    }

    const weights = new Float32Array(this.maxFeatures);
    const offset = slot * this.maxFeatures;
    for (let i = 0; i < this.maxFeatures; i++) {
      weights[i] = this.weightsBuffer[offset + i];
    }

    return {
      epoch: this.epochs[slot],
      trainLoss: this.trainLosses[slot],
      testLoss: this.testLosses[slot],
      trainAccuracy: this.trainAccuracies[slot],
      testAccuracy: this.testAccuracies[slot],
      weights,
      bias: this.biases[slot],
      gradientNorm: this.gradientNorms[slot],
      margin: this.margins[slot],
      nonZeroCount: this.nonZeroCounts[slot],
    };
  }
}

