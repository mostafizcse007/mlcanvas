import { VisualArtifact, TelemetryMetrics, ExplanationPayload } from './algorithm';

export interface HistorySnapshot {
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAccuracy?: number;
  testAccuracy?: number;
  weights: Float32Array;
  bias: number;
  gradientNorm?: number;
  margin?: number;
  nonZeroCount?: number;

  // Lightweight per-epoch snapshot payloads for timeline scrubbing
  metrics?: TelemetryMetrics;
  artifacts?: VisualArtifact[];
  explanation?: ExplanationPayload;
  gridValues?: Float32Array;
  modelState?: any;
}

export interface RingBufferStats {
  capacity: number;
  count: number;
  head: number;
  earliestEpoch: number;
  latestEpoch: number;
}

