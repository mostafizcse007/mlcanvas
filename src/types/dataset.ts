export type DatasetType =
  | 'separable-gaussians'
  | 'circles'
  | 'moons'
  | 'xor'
  | 'spiral'
  | 'linear-regression';

export interface DataPoint {
  id: number;
  x1: number;
  x2: number;
  y: number; // +1 or -1 for classification, or target real value for regression
  isTest?: boolean;
}

export interface DatasetConfig {
  type: DatasetType;
  sampleCount: number; // e.g. 100 to 500
  noise: number;       // 0.0 to 0.5
  testRatio: number;   // e.g. 0.2 (20% test, 80% train)
}

export interface FeatureSelection {
  x1: boolean;
  x2: boolean;
  x1Squared: boolean;
  x2Squared: boolean;
  x1x2: boolean;
  sinX1: boolean;
  sinX2: boolean;
}

