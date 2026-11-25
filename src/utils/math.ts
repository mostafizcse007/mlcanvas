import { FeatureSelection, DataPoint } from '../types/dataset';

/**
 * Expands a 2D coordinate point into selected active feature vector
 */
export function extractFeatures(
  x1: number,
  x2: number,
  selection: FeatureSelection
): { vector: number[]; names: string[] } {
  const vector: number[] = [];
  const names: string[] = [];

  if (selection.x1) {
    vector.push(x1);
    names.push('X₁');
  }
  if (selection.x2) {
    vector.push(x2);
    names.push('X₂');
  }
  if (selection.x1Squared) {
    vector.push(x1 * x1);
    names.push('X₁²');
  }
  if (selection.x2Squared) {
    vector.push(x2 * x2);
    names.push('X₂²');
  }
  if (selection.x1x2) {
    vector.push(x1 * x2);
    names.push('X₁X₂');
  }
  if (selection.sinX1) {
    vector.push(Math.sin(x1));
    names.push('sin(X₁)');
  }
  if (selection.sinX2) {
    vector.push(Math.sin(x2));
    names.push('sin(X₂)');
  }

  // Fallback: at least x1, x2 if all deselected
  if (vector.length === 0) {
    vector.push(x1, x2);
    names.push('X₁', 'X₂');
  }

  return { vector, names };
}

export function dot(a: number[] | Float32Array, b: number[] | Float32Array): number {
  let sum = 0;
  const len = a.length;
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function l2Norm(vec: number[] | Float32Array): number {
  return Math.sqrt(dot(vec, vec));
}

export function l1Norm(vec: number[] | Float32Array): number {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) {
    sum += Math.abs(vec[i]);
  }
  return sum;
}

/**
 * Soft-Thresholding Operator S_λ(z) = sign(z) * max(0, |z| - λ)
 * Fundamental building block for Lasso coordinate descent & proximal methods
 */
export function softThreshold(z: number, lambda: number): number {
  if (z > lambda) return z - lambda;
  if (z < -lambda) return z + lambda;
  return 0;
}

/**
 * Sign function returning +1 or -1 (or 0)
 */
export function sign(x: number): number {
  return x > 0 ? 1 : x < 0 ? -1 : 0;
}

/**
 * Computes classification accuracy for binary or Cross-Task binarized targets (y >= 0 ? +1 : -1)
 */
export function computeAccuracy(
  dataset: DataPoint[],
  predictFn: (p: { x1: number; x2: number }) => number
): number {
  if (dataset.length === 0) return 0;
  let correct = 0;
  for (let i = 0; i < dataset.length; i++) {
    const pt = dataset[i];
    const pred = predictFn(pt);
    const predClass = pred >= 0 ? 1 : -1;
    const actualClass = pt.y >= 0 ? 1 : -1;
    if (predClass === actualClass) {
      correct++;
    }
  }
  return (correct / dataset.length) * 100;
}

/**
 * Computes Mean Squared Error for regression
 */
export function computeMSE(
  dataset: DataPoint[],
  predictFn: (p: { x1: number; x2: number }) => number
): number {
  if (dataset.length === 0) return 0;
  let sse = 0;
  for (let i = 0; i < dataset.length; i++) {
    const pt = dataset[i];
    const diff = pt.y - predictFn(pt);
    sse += diff * diff;
  }
  return sse / dataset.length;
}

/**
 * Numerically stable sigmoid function
 */
export function sigmoid(z: number): number {
  if (z >= 15) return 1.0;
  if (z <= -15) return 0.0;
  return 1.0 / (1.0 + Math.exp(-z));
}

/**
 * Numerically stable Softmax for K classes
 */
export function softmax(logits: number[]): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    if (logits[i] > max) max = logits[i];
  }
  let sum = 0;
  const exp = new Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    exp[i] = Math.exp(logits[i] - max);
    sum += exp[i];
  }
  for (let i = 0; i < exp.length; i++) {
    exp[i] = sum > 0 ? exp[i] / sum : 1 / logits.length;
  }
  return exp;
}

/**
 * Huber loss and derivative for robust regression
 */
export function huberLoss(residual: number, delta = 1.0): number {
  const absR = Math.abs(residual);
  if (absR <= delta) {
    return 0.5 * residual * residual;
  }
  return delta * (absR - 0.5 * delta);
}

export function huberDerivative(residual: number, delta = 1.0): number {
  if (Math.abs(residual) <= delta) {
    return residual;
  }
  return delta * Math.sign(residual);
}

/**
 * Distance metrics for 2D vectors
 */
export function euclideanDist(x1: number, y1: number, x2: number, y2: number): number {
  return Math.hypot(x1 - x2, y1 - y2);
}

export function manhattanDist(x1: number, y1: number, x2: number, y2: number): number {
  return Math.abs(x1 - x2) + Math.abs(y1 - y2);
}

/**
 * Kernels for SVM and Gaussian Processes
 */
export function rbfKernel(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  gamma = 0.5
): number {
  const dx = x1 - x2;
  const dy = y1 - y2;
  return Math.exp(-gamma * (dx * dx + dy * dy));
}

export function polyKernel(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  degree = 3,
  c = 1.0
): number {
  return Math.pow(x1 * x2 + y1 * y2 + c, degree);
}

/**
 * 1D Gaussian probability density
 */
export function gaussianPdf(x: number, mean: number, variance: number): number {
  const v = Math.max(1e-6, variance);
  const diff = x - mean;
  return (1.0 / Math.sqrt(2 * Math.PI * v)) * Math.exp(-(diff * diff) / (2 * v));
}

/**
 * Gauss-Jordan matrix inversion with partial pivoting
 */
export function invertMatrix(A: number[][]): number[][] | null {
  const n = A.length;
  // Create augmented matrix [A | I]
  const M: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    M[i] = new Array(2 * n).fill(0);
    for (let j = 0; j < n; j++) {
      M[i][j] = A[i][j];
    }
    M[i][n + i] = 1.0;
  }

  for (let i = 0; i < n; i++) {
    // Find pivot
    let maxRow = i;
    let maxVal = Math.abs(M[i][i]);
    for (let r = i + 1; r < n; r++) {
      if (Math.abs(M[r][i]) > maxVal) {
        maxVal = Math.abs(M[r][i]);
        maxRow = r;
      }
    }
    if (maxVal < 1e-12) return null; // Singular

    // Swap rows
    if (maxRow !== i) {
      const temp = M[i];
      M[i] = M[maxRow];
      M[maxRow] = temp;
    }

    // Scale pivot row
    const pivot = M[i][i];
    for (let c = 0; c < 2 * n; c++) {
      M[i][c] /= pivot;
    }

    // Eliminate other rows
    for (let r = 0; r < n; r++) {
      if (r !== i) {
        const factor = M[r][i];
        if (Math.abs(factor) > 1e-15) {
          for (let c = 0; c < 2 * n; c++) {
            M[r][c] -= factor * M[i][c];
          }
        }
      }
    }
  }

  // Extract inverse
  const inv: number[][] = new Array(n);
  for (let i = 0; i < n; i++) {
    inv[i] = new Array(n);
    for (let j = 0; j < n; j++) {
      inv[i][j] = M[i][n + j];
    }
  }
  return inv;
}

/**
 * Solves normal equations (X^T X + λ I) w = X^T y
 */
export function solveNormalEquations(
  X: number[][],
  y: number[],
  l2Reg = 1e-5
): number[] | null {
  const N = X.length;
  if (N === 0) return null;
  const D = X[0].length;

  // Compute X^T X (D x D)
  const XtX: number[][] = Array.from({ length: D }, () => new Array(D).fill(0));
  const Xty: number[] = new Array(D).fill(0);

  for (let i = 0; i < N; i++) {
    const xi = X[i];
    const yi = y[i];
    for (let j = 0; j < D; j++) {
      Xty[j] += xi[j] * yi;
      for (let k = j; k < D; k++) {
        XtX[j][k] += xi[j] * xi[k];
      }
    }
  }

  // Symmetrize and add L2 regularization
  for (let j = 0; j < D; j++) {
    XtX[j][j] += l2Reg;
    for (let k = j + 1; k < D; k++) {
      XtX[k][j] = XtX[j][k];
    }
  }

  const inv = invertMatrix(XtX);
  if (!inv) return null;

  // w = inv * Xty
  const w = new Array(D).fill(0);
  for (let j = 0; j < D; j++) {
    let sum = 0;
    for (let k = 0; k < D; k++) {
      sum += inv[j][k] * Xty[k];
    }
    w[j] = sum;
  }
  return w;
}

