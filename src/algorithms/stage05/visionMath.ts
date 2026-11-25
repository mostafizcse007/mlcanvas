import { generateSyntheticVisionClass } from '../../datasets/visionGrid';

/**
 * Core mathematical routines and canonical filter banks for Stage 05 (Vision Grid & Patch Architectures).
 */

// ==========================================
// 1. Spatial Dimension & Receptive Field
// ==========================================

export function computeOutputDim(
  inputDim: number,
  kernelSize: number,
  stride: number,
  padding: number
): number {
  return Math.floor((inputDim - kernelSize + 2 * padding) / stride) + 1;
}

export interface ReceptiveFieldInfo {
  rfSize: number; // in pixels
  cumulStride: number;
  startOffset: number;
  cumulPadding?: number;
}

/**
 * Calculates effective receptive field growth:
 * R_l = R_{l-1} + (K_l - 1) * CumulStride_{l-1}
 * cumulPadding_l = cumulPadding_{l-1} + P_l * CumulStride_{l-1}
 */
export function nextReceptiveField(
  prev: ReceptiveFieldInfo,
  kernelSize: number,
  stride: number,
  padding: number = 0
): ReceptiveFieldInfo {
  const rfSize = prev.rfSize + (kernelSize - 1) * prev.cumulStride;
  const cumulStride = prev.cumulStride * stride;
  const cumulPadding = (prev.cumulPadding ?? 0) + padding * prev.cumulStride;
  return {
    rfSize,
    cumulStride,
    startOffset: prev.startOffset + Math.floor(kernelSize / 2) * prev.cumulStride,
    cumulPadding,
  };
}

/**
 * Maps an output pixel (row, col) at layer L back to its bounding box on input (16x16),
 * properly offsetting for cumulative spatial padding across earlier layers.
 */
export function mapOutputToInputBoundingBox(
  outRow: number,
  outCol: number,
  rfSizeOrInfo: number | ReceptiveFieldInfo,
  cumulStride: number = 1,
  cumulPaddingOrDim: number = 0,
  inputDim: number = 16
): [number, number, number, number] {
  let rfSize = 1;
  let stride = cumulStride;
  let padding = cumulPaddingOrDim;
  let dim = inputDim;

  if (typeof rfSizeOrInfo === 'object' && rfSizeOrInfo !== null) {
    rfSize = rfSizeOrInfo.rfSize;
    stride = rfSizeOrInfo.cumulStride;
    padding = rfSizeOrInfo.cumulPadding ?? 0;
    dim = typeof cumulStride === 'number' && cumulStride > 1 ? cumulStride : 16;
  } else {
    rfSize = rfSizeOrInfo;
    if (cumulPaddingOrDim === 16 && inputDim === 16) {
      padding = 0;
      dim = 16;
    }
  }

  const startR = outRow * stride - padding;
  const startC = outCol * stride - padding;
  const endR = startR + rfSize - 1;
  const endC = startC + rfSize - 1;

  const minR = Math.max(0, Math.min(dim - 1, startR));
  const maxR = Math.max(0, Math.min(dim - 1, endR));
  const minC = Math.max(0, Math.min(dim - 1, startC));
  const maxC = Math.max(0, Math.min(dim - 1, endC));

  return [minR, minC, maxR, maxC];
}

// ==========================================
// 2. Canonical Filter Banks (Float32Array)
// ==========================================

export const KERNEL_SOBEL_X_3X3 = new Float32Array([
  -1, 0, 1,
  -2, 0, 2,
  -1, 0, 1,
]);

export const KERNEL_SOBEL_Y_3X3 = new Float32Array([
  -1, -2, -1,
   0,  0,  0,
   1,  2,  1,
]);

export const KERNEL_LAPLACIAN_3X3 = new Float32Array([
   0,  1,  0,
   1, -4,  1,
   0,  1,  0,
]);

export const KERNEL_RIDGE_3X3 = new Float32Array([
  -1, -1, -1,
  -1,  8, -1,
  -1, -1, -1,
]);

export const KERNEL_CORNER_3X3 = new Float32Array([
   1,  1, -1,
   1, -1, -1,
  -1, -1, -1,
]);

export const KERNEL_BOX_BLUR_3X3 = new Float32Array([
  1 / 9, 1 / 9, 1 / 9,
  1 / 9, 1 / 9, 1 / 9,
  1 / 9, 1 / 9, 1 / 9,
]);

export const KERNEL_GABOR_HORIZ_5X5 = new Float32Array([
  -0.1, -0.2, -0.3, -0.2, -0.1,
   0.2,  0.5,  0.8,  0.5,  0.2,
   0.4,  0.9,  1.0,  0.9,  0.4,
   0.2,  0.5,  0.8,  0.5,  0.2,
  -0.1, -0.2, -0.3, -0.2, -0.1,
]);

export const KERNEL_GABOR_VERT_5X5 = new Float32Array([
  -0.1,  0.2,  0.4,  0.2, -0.1,
  -0.2,  0.5,  0.9,  0.5, -0.2,
  -0.3,  0.8,  1.0,  0.8, -0.3,
  -0.2,  0.5,  0.9,  0.5, -0.2,
  -0.1,  0.2,  0.4,  0.2, -0.1,
]);

export const KERNEL_ALEXNET_7X7_EDGE = new Float32Array([
  -0.1, -0.2, -0.3, 0.0, 0.3, 0.2, 0.1,
  -0.2, -0.4, -0.6, 0.0, 0.6, 0.4, 0.2,
  -0.3, -0.6, -0.9, 0.0, 0.9, 0.6, 0.3,
  -0.4, -0.8, -1.0, 0.0, 1.0, 0.8, 0.4,
  -0.3, -0.6, -0.9, 0.0, 0.9, 0.6, 0.3,
  -0.2, -0.4, -0.6, 0.0, 0.6, 0.4, 0.2,
  -0.1, -0.2, -0.3, 0.0, 0.3, 0.2, 0.1,
]);

export const KERNEL_ALEXNET_7X7_BLOB = new Float32Array([
  -0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1,
  -0.2,  0.1,  0.3,  0.4,  0.3,  0.1, -0.2,
  -0.2,  0.3,  0.7,  0.9,  0.7,  0.3, -0.2,
  -0.2,  0.4,  0.9,  1.0,  0.9,  0.4, -0.2,
  -0.2,  0.3,  0.7,  0.9,  0.7,  0.3, -0.2,
  -0.2,  0.1,  0.3,  0.4,  0.3,  0.1, -0.2,
  -0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.1,
]);

// ==========================================
// 3. Fast 2D Convolution & Pooling
// ==========================================

export function conv2dSingleChannel(
  input: Float32Array,
  inH: number,
  inW: number,
  kernel: Float32Array,
  kSize: number,
  stride: number = 1,
  padding: number = 0,
  bias: number = 0
): { output: Float32Array; outH: number; outW: number } {
  const outH = computeOutputDim(inH, kSize, stride, padding);
  const outW = computeOutputDim(inW, kSize, stride, padding);
  const output = new Float32Array(outH * outW);

  for (let r = 0; r < outH; r++) {
    const inRowOrigin = r * stride - padding;
    for (let c = 0; c < outW; c++) {
      const inColOrigin = c * stride - padding;
      let sum = bias;

      for (let kr = 0; kr < kSize; kr++) {
        const inR = inRowOrigin + kr;
        if (inR < 0 || inR >= inH) continue; // zero padding
        for (let kc = 0; kc < kSize; kc++) {
          const inC = inColOrigin + kc;
          if (inC < 0 || inC >= inW) continue; // zero padding

          const inVal = input[inR * inW + inC];
          const kVal = kernel[kr * kSize + kc];
          sum += inVal * kVal;
        }
      }

      output[r * outW + c] = sum;
    }
  }

  return { output, outH, outW };
}

export function maxPool2dSingleChannel(
  input: Float32Array,
  inH: number,
  inW: number,
  poolSize: number = 2,
  stride: number = 2
): { output: Float32Array; outH: number; outW: number } {
  const outH = Math.floor((inH - poolSize) / stride) + 1;
  const outW = Math.floor((inW - poolSize) / stride) + 1;
  const output = new Float32Array(outH * outW);

  for (let r = 0; r < outH; r++) {
    const inR = r * stride;
    for (let c = 0; c < outW; c++) {
      const inC = c * stride;
      let maxVal = -Infinity;

      for (let pr = 0; pr < poolSize; pr++) {
        for (let pc = 0; pc < poolSize; pc++) {
          const row = inR + pr;
          const col = inC + pc;
          if (row < inH && col < inW) {
            const v = input[row * inW + col];
            if (v > maxVal) maxVal = v;
          }
        }
      }

      output[r * outW + c] = maxVal === -Infinity ? 0 : maxVal;
    }
  }

  return { output, outH, outW };
}

export function avgPool2dSingleChannel(
  input: Float32Array,
  inH: number,
  inW: number,
  poolSize: number = 2,
  stride: number = 2
): { output: Float32Array; outH: number; outW: number } {
  const outH = Math.floor((inH - poolSize) / stride) + 1;
  const outW = Math.floor((inW - poolSize) / stride) + 1;
  const output = new Float32Array(outH * outW);
  const count = poolSize * poolSize;

  for (let r = 0; r < outH; r++) {
    const inR = r * stride;
    for (let c = 0; c < outW; c++) {
      const inC = c * stride;
      let sum = 0;

      for (let pr = 0; pr < poolSize; pr++) {
        for (let pc = 0; pc < poolSize; pc++) {
          const row = inR + pr;
          const col = inC + pc;
          if (row < inH && col < inW) {
            sum += input[row * inW + col];
          }
        }
      }

      output[r * outW + c] = sum / count;
    }
  }

  return { output, outH, outW };
}

// ==========================================
// 4. Activations
// ==========================================

export function applyActivation(
  data: Float32Array,
  act: 'relu' | 'tanh' | 'sigmoid' | 'none'
): Float32Array {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (act === 'relu') {
      out[i] = Math.max(0, v);
    } else if (act === 'tanh') {
      out[i] = Math.tanh(v);
    } else if (act === 'sigmoid') {
      out[i] = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, v))));
    } else {
      out[i] = v;
    }
  }
  return out;
}

export function activationDerivative(
  v: number,
  act: 'relu' | 'tanh' | 'sigmoid' | 'none'
): number {
  if (act === 'relu') return v > 0 ? 1 : 0;
  if (act === 'tanh') {
    const t = Math.tanh(v);
    return 1 - t * t;
  }
  if (act === 'sigmoid') {
    const s = 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, v))));
    return s * (1 - s);
  }
  return 1;
}

// ==========================================
// 5. Softmax & Class Logits
// ==========================================

export function softmax(logits: number[]): number[] {
  let maxL = -Infinity;
  for (const l of logits) if (l > maxL) maxL = l;

  let sum = 0;
  const exps = logits.map((l) => {
    const e = Math.exp(Math.max(-50, Math.min(50, l - maxL)));
    sum += e;
    return e;
  });

  return exps.map((e) => e / (sum || 1));
}

/**
 * Canonical +1px shift delta (Total Variation distance between unshifted and
 * +1px-shifted class probability distributions): 0.5 * sum_c |p_shift(c) - p(c)|.
 */
export function computeShiftDelta(
  baseProbs: ArrayLike<number>,
  shiftedProbs: ArrayLike<number>
): number {
  let sum = 0;
  const len = Math.min(baseProbs.length, shiftedProbs.length);
  for (let i = 0; i < len; i++) {
    sum += Math.abs((baseProbs[i] ?? 0) - (shiftedProbs[i] ?? 0));
  }
  return 0.5 * sum;
}

// ==========================================
// 6. ViT Patch Tokenization & Self-Attention
// ==========================================

export interface ViTPatchInfo {
  id: number;
  row: number; // grid row 0..gridDim-1
  col: number; // grid col 0..gridDim-1
  pixels: Float32Array; // P x P
  tokenEmbedding: Float32Array; // Projected vector
}

export function extractPatches(
  inputGrid: Float32Array,
  inputDim: number = 16,
  patchSize: number = 4
): ViTPatchInfo[] {
  const gridDim = Math.floor(inputDim / patchSize);
  const patches: ViTPatchInfo[] = [];
  const pArea = patchSize * patchSize;

  for (let gr = 0; gr < gridDim; gr++) {
    for (let gc = 0; gc < gridDim; gc++) {
      const id = gr * gridDim + gc;
      const pixels = new Float32Array(pArea);
      let pIdx = 0;

      for (let pr = 0; pr < patchSize; pr++) {
        const inR = gr * patchSize + pr;
        for (let pc = 0; pc < patchSize; pc++) {
          const inC = gc * patchSize + pc;
          pixels[pIdx++] = inputGrid[inR * inputDim + inC];
        }
      }

      patches.push({
        id,
        row: gr,
        col: gc,
        pixels,
        tokenEmbedding: new Float32Array(8), // D = 8
      });
    }
  }

  return patches;
}

/**
 * Generates linear patch projection matrix E: R^{P^2} -> R^D
 * Scales with any patch size: 2x2 (4 px), 4x4 (16 px), 8x8 (64 px).
 */
export function getDynamicPatchProjectionMatrix(pArea: number = 16, embedDim: number = 8): Float32Array {
  const E = new Float32Array(pArea * embedDim);
  for (let p = 0; p < pArea; p++) {
    for (let d = 0; d < embedDim; d++) {
      E[p * embedDim + d] = Math.sin((p + 1) * (d + 1) * 0.7);
    }
  }
  return E;
}

/**
 * Computes Scaled Dot-Product Self-Attention matrix:
 * A_{ij} = Softmax((Q_i * K_j^T) / (sqrt(d_k) * tau))
 * For N patches + 1 [CLS] token = (N+1) x (N+1)
 */
export function computeViTSelfAttention(
  patches: ViTPatchInfo[],
  patchEmbeddings: Float32Array[], // length N, each D=8
  clsToken: Float32Array, // D=8
  temperature: number = 1.0,
  heads: number = 2
): {
  attentionWeights: Float32Array; // (N+1) x (N+1)
  attendedTokens: Float32Array[]; // length N+1, each D=8
} {
  const N = patches.length;
  const totalTokens = N + 1; // [CLS] at index 0, patches at 1..N
  const D = 8;
  const dkSqrt = Math.sqrt(D / heads);
  const tau = Math.max(0.1, temperature);

  // Combine tokens: index 0 = clsToken, 1..N = patchEmbeddings
  const allTokens: Float32Array[] = [clsToken, ...patchEmbeddings];
  const attentionWeights = new Float32Array(totalTokens * totalTokens);
  const attendedTokens: Float32Array[] = [];

  for (let i = 0; i < totalTokens; i++) {
    const q = allTokens[i];
    const rawScores: number[] = new Array(totalTokens);

    for (let j = 0; j < totalTokens; j++) {
      const k = allTokens[j];
      let dot = 0;
      for (let d = 0; d < D; d++) {
        dot += q[d] * k[d];
      }
      rawScores[j] = dot / (dkSqrt * tau);
    }

    const probs = softmax(rawScores);
    for (let j = 0; j < totalTokens; j++) {
      attentionWeights[i * totalTokens + j] = probs[j];
    }

    // Weighted sum of V (here V = allTokens for transparent visualization)
    const outToken = new Float32Array(D);
    for (let j = 0; j < totalTokens; j++) {
      const weight = probs[j];
      const v = allTokens[j];
      for (let d = 0; d < D; d++) {
        outToken[d] += weight * v[d];
      }
    }
    attendedTokens.push(outToken);
  }

  return { attentionWeights, attendedTokens };
}

// ==========================================
// 7. Micro-Training SGD Classifier
// ==========================================

export interface MicroTrainState {
  convW1: Float32Array; // 4 filters, 3x3 = 36 weights
  convB1: Float32Array; // 4 biases
  fcW: Float32Array;    // 4 classes x numFeatures
  fcB: Float32Array;    // 4 biases
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainAcc: number;
  testAcc: number;
}

export function initMicroTrainState(seed: number = 42): MicroTrainState {
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return (s / 4294967296 - 0.5) * 0.4;
  };
  const convW1 = new Float32Array(4 * 9);
  for (let i = 0; i < convW1.length; i++) convW1[i] = rand();

  const convB1 = new Float32Array(4);

  // Flattened features after 3x3 conv with valid padding + 2x2 max pool:
  // 16x16 -> 14x14 -> 7x7 = 49 * 4 channels = 196
  const numFeatures = 4 * 7 * 7;
  const fcW = new Float32Array(4 * numFeatures);
  for (let i = 0; i < fcW.length; i++) fcW[i] = rand() * 0.1;

  const fcB = new Float32Array(4);

  return {
    convW1,
    convB1,
    fcW,
    fcB,
    epoch: 0,
    trainLoss: 1.38,
    testLoss: 1.39,
    trainAcc: 25.0,
    testAcc: 25.0,
  };
}

export function stepMicroTrainEpoch(
  state: MicroTrainState,
  trainSamples: { pixels: Float32Array; label: number }[],
  testSamples: { pixels: Float32Array; label: number }[],
  lr: number = 0.05
): MicroTrainState {
  const next = {
    convW1: new Float32Array(state.convW1),
    convB1: new Float32Array(state.convB1),
    fcW: new Float32Array(state.fcW),
    fcB: new Float32Array(state.fcB),
    epoch: state.epoch + 1,
    trainLoss: 0,
    testLoss: 0,
    trainAcc: 0,
    testAcc: 0,
  };

  // Generate fallback synthetic samples if empty
  let train = trainSamples;
  let test = testSamples;
  if (!train || train.length === 0) {
    train = [];
    test = [];
    for (let c = 0; c < 4; c++) {
      for (let s = 0; s < 8; s++) {
        train.push({ pixels: generateSyntheticVisionClass(c, s, 42, 0.03), label: c });
      }
      for (let s = 0; s < 4; s++) {
        test.push({ pixels: generateSyntheticVisionClass(c, 8 + s, 42, 0.03), label: c });
      }
    }
  }

  let totalLoss = 0;
  let correctCount = 0;
  const numFeatures = 4 * 7 * 7;

  // SGD on train samples
  for (const sample of train) {
    // 1. Forward Pass: Conv1 (4 filters 3x3)
    const pooledFeatures = new Float32Array(numFeatures);
    for (let f = 0; f < 4; f++) {
      const k = next.convW1.subarray(f * 9, f * 9 + 9);
      const b = next.convB1[f];
      const { output: cOut } = conv2dSingleChannel(sample.pixels, 16, 16, k, 3, 1, 0, b);
      const relu = applyActivation(cOut, 'relu');
      const { output: pOut } = maxPool2dSingleChannel(relu, 14, 14, 2, 2);
      pooledFeatures.set(pOut, f * 49);
    }

    // 2. FC Dense Layer
    const logits = [next.fcB[0], next.fcB[1], next.fcB[2], next.fcB[3]];
    for (let c = 0; c < 4; c++) {
      const rowOffset = c * numFeatures;
      for (let i = 0; i < numFeatures; i++) {
        logits[c] += next.fcW[rowOffset + i] * pooledFeatures[i];
      }
    }

    const probs = softmax(logits);
    const y = sample.label;
    totalLoss += -Math.log(Math.max(1e-7, probs[y]));

    let predCls = 0;
    let maxP = probs[0];
    for (let c = 1; c < 4; c++) {
      if (probs[c] > maxP) {
        maxP = probs[c];
        predCls = c;
      }
    }
    if (predCls === y) correctCount++;

    // 3. Backward Pass (Cross-entropy gradient = prob - one_hot)
    const gradLogits = probs.map((p, c) => p - (c === y ? 1 : 0));

    // Update FC weights & biases
    for (let c = 0; c < 4; c++) {
      const g = gradLogits[c];
      next.fcB[c] -= lr * g;
      const rowOffset = c * numFeatures;
      for (let i = 0; i < numFeatures; i++) {
        next.fcW[rowOffset + i] -= lr * g * pooledFeatures[i];
      }
    }

    // Backprop into Conv1 weights
    for (let f = 0; f < 4; f++) {
      const rowOffset = y * numFeatures + f * 49;
      let gradSum = 0;
      for (let p = 0; p < 49; p++) gradSum += next.fcW[rowOffset + p];
      const gradFilter = gradLogits[y] * (gradSum / 49);
      next.convB1[f] -= lr * gradFilter * 0.1;
      const kOff = f * 9;
      for (let w = 0; w < 9; w++) {
        next.convW1[kOff + w] -= lr * gradFilter * 0.05;
      }
    }
  }

  next.trainLoss = totalLoss / (train.length || 1);
  next.trainAcc = (correctCount / (train.length || 1)) * 100;

  // Evaluation on test samples
  let testTotalLoss = 0;
  let testCorrect = 0;
  for (const sample of test) {
    const pooledFeatures = new Float32Array(numFeatures);
    for (let f = 0; f < 4; f++) {
      const k = next.convW1.subarray(f * 9, f * 9 + 9);
      const b = next.convB1[f];
      const { output: cOut } = conv2dSingleChannel(sample.pixels, 16, 16, k, 3, 1, 0, b);
      const relu = applyActivation(cOut, 'relu');
      const { output: pOut } = maxPool2dSingleChannel(relu, 14, 14, 2, 2);
      pooledFeatures.set(pOut, f * 49);
    }

    const logits = [next.fcB[0], next.fcB[1], next.fcB[2], next.fcB[3]];
    for (let c = 0; c < 4; c++) {
      const rowOffset = c * numFeatures;
      for (let i = 0; i < numFeatures; i++) {
        logits[c] += next.fcW[rowOffset + i] * pooledFeatures[i];
      }
    }

    const probs = softmax(logits);
    const y = sample.label;
    testTotalLoss += -Math.log(Math.max(1e-7, probs[y]));

    let predCls = 0;
    let maxP = probs[0];
    for (let c = 1; c < 4; c++) {
      if (probs[c] > maxP) {
        maxP = probs[c];
        predCls = c;
      }
    }
    if (predCls === y) testCorrect++;
  }

  next.testLoss = testTotalLoss / (test.length || 1);
  next.testAcc = (testCorrect / (test.length || 1)) * 100;

  return next;
}

/**
 * Executes a full forward inference pass through the SGD-trained micro-network:
 * Conv1 (4 filters 3x3) -> ReLU -> MaxPool (2x2) -> FC Dense -> Softmax.
 */
export function predictMicroTrain(
  inputGrid: Float32Array,
  state: MicroTrainState
): Float32Array {
  const numFeatures = 4 * 7 * 7;
  const pooledFeatures = new Float32Array(numFeatures);
  for (let f = 0; f < 4; f++) {
    const k = state.convW1.subarray(f * 9, f * 9 + 9);
    const b = state.convB1[f];
    const { output: cOut } = conv2dSingleChannel(inputGrid, 16, 16, k, 3, 1, 0, b);
    const relu = applyActivation(cOut, 'relu');
    const { output: pOut } = maxPool2dSingleChannel(relu, 14, 14, 2, 2);
    pooledFeatures.set(pOut, f * 49);
  }

  const logits = [state.fcB[0], state.fcB[1], state.fcB[2], state.fcB[3]];
  for (let c = 0; c < 4; c++) {
    const rowOffset = c * numFeatures;
    for (let i = 0; i < numFeatures; i++) {
      logits[c] += state.fcW[rowOffset + i] * pooledFeatures[i];
    }
  }

  return new Float32Array(softmax(logits));
}

/**
 * Blends visual feature priors with live neural network predictions.
 * At epoch 0, visual priors respond instantly to drawings and presets.
 * As SGD trains, learned network weights seamlessly take over.
 */
export function predictHybrid(
  inputGrid: Float32Array,
  state: MicroTrainState,
  epoch: number,
  featureMap?: Float32Array
): Float32Array {
  const microProbs = predictMicroTrain(inputGrid, state);
  const visualLogits = computeVisualClassLogits(featureMap || inputGrid, 16, 16, inputGrid);
  const visualProbs = softmax(visualLogits);

  const alpha = Math.min(1.0, epoch / 12);
  const out = new Float32Array(4);
  for (let c = 0; c < 4; c++) {
    out[c] = (1 - alpha) * visualProbs[c] + alpha * microProbs[c];
  }
  return out;
}

/**
 * Evaluates canonical visual feature energy for 4-class classification:
 * Class 0: Vertical Bars
 * Class 1: Horizontal Bars
 * Class 2: Diagonal Crosses
 * Class 3: Hollow Rings
 *
 * This provides responsive, real-time feedback in Architecture Inspector mode
 * when the user draws on the 16x16 canvas or picks presets.
 */
export function computeVisualClassLogits(
  featureMap: Float32Array,
  _h: number,
  _w: number,
  inputGrid: Float32Array
): number[] {
  // Column and Row variance from 16x16 input grid
  const colSums = new Float32Array(16);
  const rowSums = new Float32Array(16);
  let totalEnergy = 0;

  for (let r = 0; r < 16; r++) {
    for (let c = 0; c < 16; c++) {
      const v = inputGrid[r * 16 + c];
      colSums[c] += v;
      rowSums[r] += v;
      totalEnergy += v;
    }
  }

  if (totalEnergy < 0.05) {
    // Blank canvas: uniform distribution
    return [0, 0, 0, 0];
  }

  // Column variance (distinctive of vertical stripes / vertical bars)
  const meanCol = totalEnergy / 16;
  let varCol = 0;
  for (let c = 0; c < 16; c++) varCol += (colSums[c] - meanCol) ** 2;
  varCol /= 16;

  // Row variance (distinctive of horizontal stripes / horizontal bars)
  const meanRow = totalEnergy / 16;
  let varRow = 0;
  for (let r = 0; r < 16; r++) varRow += (rowSums[r] - meanRow) ** 2;
  varRow /= 16;

  // Diagonal / Cross energy: pixels along the main diagonals
  let diagSum = 0;
  for (let r = 0; r < 16; r++) {
    for (let c = 0; c < 16; c++) {
      const isDiag1 = Math.abs(r - c) <= 1;
      const isDiag2 = Math.abs(r + c - 15) <= 1;
      if (isDiag1 || isDiag2) {
        diagSum += inputGrid[r * 16 + c];
      }
    }
  }
  const diagRatio = diagSum / (totalEnergy || 1);

  // Hollow Ring energy: illuminated ring band with a hollow center void
  let innerCore = 0, innerCount = 0;
  let ringBand = 0, ringCount = 0;

  for (let r = 0; r < 16; r++) {
    for (let c = 0; c < 16; c++) {
      const dist = Math.sqrt((r - 7.5) ** 2 + (c - 7.5) ** 2);
      const v = inputGrid[r * 16 + c];
      if (dist < 2.8) {
        innerCore += v;
        innerCount++;
      } else if (dist >= 3.2 && dist <= 6.8) {
        ringBand += v;
        ringCount++;
      }
    }
  }
  const meanInner = innerCore / (innerCount || 1);
  const meanRing = ringBand / (ringCount || 1);
  const ringScore = Math.max(0, meanRing - meanInner * 1.5);

  // Feature map energy modulation
  let fmTotal = 0;
  for (let i = 0; i < featureMap.length; i++) fmTotal += Math.abs(featureMap[i]);
  const scale = Math.max(0.6, Math.min(2.0, fmTotal / (featureMap.length * 0.15 || 1)));

  // Discriminate based on spatial symmetry and void structures
  const isStripeVert = varCol > 2.2 * varRow && varRow < 2.5;
  const isStripeHoriz = varRow > 2.2 * varCol && varCol < 2.5;
  const isHollowRing = ringScore > 0.2 && meanInner < 0.18;

  let scoreVert = 0.05;
  let scoreHoriz = 0.05;
  let scoreCross = 0.05;
  let scoreRing = 0.05;

  if (isHollowRing) {
    scoreRing = Math.max(0.05, ringScore * 12.0 + 8.0);
    scoreCross = 0.1;
  } else if (isStripeVert) {
    scoreVert = Math.max(0.05, varCol * 3.5);
  } else if (isStripeHoriz) {
    scoreHoriz = Math.max(0.05, varRow * 3.5);
  } else {
    // Cross or generic pattern
    scoreCross = Math.max(0.05, Math.max((diagRatio - 0.2) * 8.0, 5.0));
    scoreRing = Math.max(0.05, ringScore * 2.0);
  }

  return [
    scoreVert * scale,
    scoreHoriz * scale,
    scoreCross * scale,
    scoreRing * scale,
  ];
}
