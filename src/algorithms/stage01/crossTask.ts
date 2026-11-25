import { DatasetContainer } from '../../types/algorithm';
import { DataPoint2D } from '../../datasets/tabular2d';

/**
 * Projects any target value onto the binary classification label set {-1, +1}
 * around threshold zero (y >= 0 -> +1, y < 0 -> -1).
 * Preserves standard +1 / -1 labels and maps class +2 and exact y = 0 to +1.
 */
export function binarizeTarget(y: number): 1 | -1 {
  return y >= 0 ? 1 : -1;
}

/**
 * Projects any target value onto the Bernoulli label set {0, 1}
 * around threshold zero (y >= 0 -> 1, y < 0 -> 0).
 */
export function binarizeTarget01(y: number): 1 | 0 {
  return y >= 0 ? 1 : 0;
}

/**
 * Detects whether a dataset contains continuous regression targets
 * (any sample whose target y is not in the discrete class label set {-1, 1, 2}).
 */
export function isContinuousRegressionTargets(dataset: DatasetContainer): boolean {
  const hasContinuous = (pts?: DataPoint2D[]) =>
    Boolean(pts && pts.some((p) => p.y !== 1 && p.y !== -1 && p.y !== 2));
  return hasContinuous(dataset.train) || hasContinuous(dataset.test);
}

/**
 * Returns true only for discrete 3-class classification datasets (y in {-1, 1, 2})
 * so continuous regression values that happen to hit y = 2 never trigger 3-class mode.
 */
export function isMultiClassClassificationDataset(dataset: DatasetContainer): boolean {
  if (isContinuousRegressionTargets(dataset)) return false;
  return Boolean(
    dataset.train?.some((p) => p.y === 2) ||
      dataset.test?.some((p) => p.y === 2)
  );
}

export const LOGREG_MIN_THRESHOLD = 0.01;
export const LOGREG_MAX_THRESHOLD = 0.99;

/**
 * Clamps the Logistic Regression probability cutoff τ to [0.01, 0.99]
 * so logit ln(τ / (1 - τ)) remains finite and never produces ±Infinity or NaN.
 */
export function clampThreshold(raw: unknown): number {
  const val = Number(raw);
  if (!Number.isFinite(val)) return 0.5;
  return Math.min(LOGREG_MAX_THRESHOLD, Math.max(LOGREG_MIN_THRESHOLD, val));
}
