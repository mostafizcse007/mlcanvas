import { getTokenBaseEmbedding } from '../../datasets/sequenceTokens';

// High-performance numerical primitives for Sequence, Gates & Attention Arena

export function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function norm2(a: Float32Array): number {
  let sum = 0;
  const n = a.length;
  for (let i = 0; i < n; i++) {
    sum += a[i] * a[i];
  }
  return Math.sqrt(sum);
}

export function sigmoid(x: number): number {
  if (x > 20) return 1.0;
  if (x < -20) return 0.0;
  return 1.0 / (1.0 + Math.exp(-x));
}

export function tanh(x: number): number {
  if (x > 20) return 1.0;
  if (x < -20) return -1.0;
  return Math.tanh(x);
}

export function softmax(logits: Float32Array, temperature: number = 1.0): Float32Array {
  const n = logits.length;
  const out = new Float32Array(n);
  const tau = Math.max(0.01, temperature);

  let maxVal = -Infinity;
  for (let i = 0; i < n; i++) {
    if (logits[i] > maxVal) maxVal = logits[i];
  }

  let sum = 0;
  for (let i = 0; i < n; i++) {
    const val = Math.exp((logits[i] - maxVal) / tau);
    out[i] = val;
    sum += val;
  }

  const invSum = sum > 1e-12 ? 1.0 / sum : 0;
  for (let i = 0; i < n; i++) {
    out[i] *= invSum;
  }
  return out;
}

export function softmaxRow(matrix: Float32Array, row: number, cols: number, temperature: number = 1.0): void {
  const offset = row * cols;
  const tau = Math.max(0.01, temperature);

  let maxVal = -Infinity;
  for (let c = 0; c < cols; c++) {
    const v = matrix[offset + c];
    if (v > maxVal) maxVal = v;
  }

  let sum = 0;
  for (let c = 0; c < cols; c++) {
    const expVal = Math.exp((matrix[offset + c] - maxVal) / tau);
    matrix[offset + c] = expVal;
    sum += expVal;
  }

  const invSum = sum > 1e-12 ? 1.0 / sum : 0;
  for (let c = 0; c < cols; c++) {
    matrix[offset + c] *= invSum;
  }
}

/** Compute Shannon Entropy of a probability distribution row: H = -sum(p * ln(p)) */
export function computeEntropy(probs: Float32Array, offset: number = 0, length: number = probs.length): number {
  let h = 0;
  for (let i = 0; i < length; i++) {
    const p = probs[offset + i];
    if (p > 1e-9) {
      h -= p * Math.log(p);
    }
  }
  return h;
}

// -------------------------------------------------------------
// Recurrent Forward & BPTT Backward Math
// -------------------------------------------------------------

export interface VanillaRNNStep {
  t: number;
  h: Float32Array; // 8D
  gradNorm: number; // ||dL/dh_t||
}

/** Compute Vanilla RNN unrolled forward and exact BPTT backward gradients */
export function computeVanillaRNN(
  inputs: Float32Array[],
  WhhWeightScale: number = 1.0,
  activation: 'tanh' | 'relu' = 'tanh'
): { steps: VanillaRNNStep[]; survivalRatio: number } {
  const T = inputs.length;
  const D = inputs[0]?.length ?? 8;

  // Recurrent weight matrix Whh = WhhWeightScale * orthogonal/identity-mix
  const steps: VanillaRNNStep[] = [];
  const hiddenStates: Float32Array[] = [];

  let hPrev = new Float32Array(D);
  for (let t = 0; t < T; t++) {
    const x = inputs[t];
    const hNext = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      // Linear recurrence: mix x and hPrev scaled by WhhWeightScale
      const rotatedH = (hPrev[(d + 1) % D] * 0.7 + hPrev[d] * 0.3) * WhhWeightScale;
      const act = rotatedH + x[d] * 0.8;
      hNext[d] = activation === 'relu' ? Math.max(0, act) : tanh(act);
    }
    hiddenStates.push(hNext);
    hPrev = hNext;
  }

  // Exact BPTT Backward Pass: Loss computed on final hidden state h_T
  // dL/dh_T is normalized unit vector [1, 1, ...] / sqrt(D)
  const gradH = new Float32Array(D);
  for (let d = 0; d < D; d++) gradH[d] = 1.0 / Math.sqrt(D);

  const gradNorms = new Array<number>(T);
  gradNorms[T - 1] = norm2(gradH);

  let currentGrad = new Float32Array(gradH);
  for (let t = T - 2; t >= 0; t--) {
    const nextH = hiddenStates[t + 1];
    const nextGrad = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      // Jacobian factor: diag(1 - h^2) for tanh, or indicator(h > 0) for relu
      const dAct = activation === 'relu' ? (nextH[d] > 0 ? 1.0 : 0.0) : Math.max(0.01, 1.0 - nextH[d] * nextH[d]);
      // Whh^T propagation with WhhWeightScale
      const rotatedGrad = (currentGrad[d] * 0.3 + currentGrad[(d + D - 1) % D] * 0.7) * WhhWeightScale;
      nextGrad[d] = rotatedGrad * dAct;
    }
    const n = norm2(nextGrad);
    gradNorms[t] = n;
    currentGrad = nextGrad;
  }

  for (let t = 0; t < T; t++) {
    steps.push({
      t,
      h: hiddenStates[t],
      gradNorm: gradNorms[t],
    });
  }

  const survivalRatio = gradNorms[T - 1] > 1e-9 ? gradNorms[0] / gradNorms[T - 1] : 0;
  return { steps, survivalRatio };
}

export interface LSTMStep {
  t: number;
  h: Float32Array; // 8D
  c: Float32Array; // 8D
  f: number; // Forget gate average [0..1]
  i: number; // Input gate average [0..1]
  cTilde: number;
  o: number; // Output gate average [0..1]
  gradNorm: number; // ||dL/dC_t||
}

/** Compute LSTM unrolled forward and BPTT backward gradients over additive cell highway */
export function computeLSTM(
  inputs: Float32Array[],
  forgetGateBias: number = 1.5,
  inputGateScale: number = 1.0,
  gradientClipNorm: number = 5.0,
  pinnedGates?: Record<number, Record<string, number>>
): { steps: LSTMStep[]; survivalRatio: number } {
  const T = inputs.length;
  const D = inputs[0]?.length ?? 8;

  const steps: LSTMStep[] = [];
  const cellStates: Float32Array[] = [];
  const hiddenStates: Float32Array[] = [];
  const forgetGates: Float32Array[] = [];

  let hPrev = new Float32Array(D);
  let cPrev = new Float32Array(D);

  for (let t = 0; t < T; t++) {
    const x = inputs[t];
    const cNext = new Float32Array(D);
    const hNext = new Float32Array(D);
    const fVec = new Float32Array(D);

    const pin = pinnedGates?.[t];

    let fAvg = 0;
    let iAvg = 0;
    let oAvg = 0;
    let cTildeAvg = 0;

    for (let d = 0; d < D; d++) {
      // Gate equations: forget gate calibrated so b_f >= 1.5 keeps conveyor belt wide open (f_t >= 0.95)
      const fRaw = (hPrev[d] * 0.1 + x[d] * 0.1) + forgetGateBias * 1.5 + 0.7;
      const fVal = pin?.f !== undefined ? pin.f : sigmoid(fRaw);
      fVec[d] = fVal;
      fAvg += fVal;

      const iRaw = (hPrev[(d + 1) % D] * 0.3 + x[d] * 0.7) * inputGateScale;
      const iVal = pin?.i !== undefined ? pin.i : sigmoid(iRaw);
      iAvg += iVal;

      const cTildeVal = tanh(x[d] * 0.9 + hPrev[d] * 0.2);
      cTildeAvg += cTildeVal;

      // Additive Cell State Conveyor Belt: C_t = f_t * C_{t-1} + i_t * cTilde
      cNext[d] = fVal * cPrev[d] + iVal * cTildeVal;

      const oRaw = hPrev[d] * 0.4 + x[d] * 0.4 + 0.5;
      const oVal = pin?.o !== undefined ? pin.o : sigmoid(oRaw);
      oAvg += oVal;

      // Hidden state: h_t = o_t * tanh(C_t)
      hNext[d] = oVal * tanh(cNext[d]);
    }

    fAvg /= D;
    iAvg /= D;
    oAvg /= D;
    cTildeAvg /= D;

    cellStates.push(cNext);
    hiddenStates.push(hNext);
    forgetGates.push(fVec);

    steps.push({
      t,
      h: hNext,
      c: cNext,
      f: fAvg,
      i: iAvg,
      cTilde: cTildeAvg,
      o: oAvg,
      gradNorm: 1.0,
    });

    hPrev = hNext;
    cPrev = cNext;
  }

  // Cell state gradient backward pass: dL/dC_T = 1.0, dL/dC_t = dL/dC_{t+1} * f_{t+1}
  const gradNorms = new Array<number>(T);
  const gradC = new Float32Array(D);
  for (let d = 0; d < D; d++) gradC[d] = 1.0 / Math.sqrt(D);
  gradNorms[T - 1] = norm2(gradC);

  let curGradC = new Float32Array(gradC);
  for (let t = T - 2; t >= 0; t--) {
    const nextF = forgetGates[t + 1];
    const prevGradC = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      // Additive conveyor belt propagates directly multiplied by forget gate f_{t+1}
      prevGradC[d] = curGradC[d] * nextF[d];
    }
    let n = norm2(prevGradC);
    if (gradientClipNorm > 0 && n > gradientClipNorm) {
      const scale = gradientClipNorm / n;
      for (let d = 0; d < D; d++) prevGradC[d] *= scale;
      n = gradientClipNorm;
    }
    gradNorms[t] = n;
    curGradC = prevGradC;
  }

  for (let t = 0; t < T; t++) {
    steps[t].gradNorm = gradNorms[t];
  }

  const survivalRatio = gradNorms[T - 1] > 1e-9 ? gradNorms[0] / gradNorms[T - 1] : 0;
  return { steps, survivalRatio };
}

export interface GRUStep {
  t: number;
  h: Float32Array; // 8D
  z: number; // Update gate [0..1]
  r: number; // Reset gate [0..1]
  hTilde: number;
  gradNorm: number;
}

/** Compute GRU unrolled forward with coupled update/reset gates */
export function computeGRU(
  inputs: Float32Array[],
  updateGateBias: number = 0.0,
  resetGateBias: number = 0.0,
  pinnedGates?: Record<number, Record<string, number>>
): { steps: GRUStep[]; survivalRatio: number } {
  const T = inputs.length;
  const D = inputs[0]?.length ?? 8;

  const steps: GRUStep[] = [];
  const hiddenStates: Float32Array[] = [];
  const updateGates: Float32Array[] = [];
  const resetGates: Float32Array[] = [];

  let hPrev = new Float32Array(D);

  for (let t = 0; t < T; t++) {
    const x = inputs[t];
    const hNext = new Float32Array(D);
    const zVec = new Float32Array(D);
    const rVec = new Float32Array(D);

    const pin = pinnedGates?.[t];

    let zAvg = 0;
    let rAvg = 0;
    let hTildeAvg = 0;

    for (let d = 0; d < D; d++) {
      const zRaw = x[d] * 0.6 + hPrev[d] * 0.4 + updateGateBias;
      const zVal = pin?.z !== undefined ? pin.z : sigmoid(zRaw);
      zVec[d] = zVal;
      zAvg += zVal;

      const rRaw = x[d] * 0.5 + hPrev[(d + 1) % D] * 0.5 + resetGateBias;
      const rVal = pin?.r !== undefined ? pin.r : sigmoid(rRaw);
      rVec[d] = rVal;
      rAvg += rVal;

      // Candidate hidden state using reset gate: hTilde = tanh(W x + U (r * hPrev))
      const hTildeVal = tanh(x[d] * 0.8 + (rVal * hPrev[d]) * 0.5);
      hTildeAvg += hTildeVal;

      // Coupled Interpolation: h_t = (1 - z_t) * h_{t-1} + z_t * hTilde
      hNext[d] = (1.0 - zVal) * hPrev[d] + zVal * hTildeVal;
    }

    zAvg /= D;
    rAvg /= D;
    hTildeAvg /= D;

    hiddenStates.push(hNext);
    updateGates.push(zVec);
    resetGates.push(rVec);

    steps.push({
      t,
      h: hNext,
      z: zAvg,
      r: rAvg,
      hTilde: hTildeAvg,
      gradNorm: 1.0,
    });

    hPrev = hNext;
  }

  // BPTT backward through coupled balance: dh_t / dh_{t-1} ~ (1 - z_t)
  const gradNorms = new Array<number>(T);
  const gradH = new Float32Array(D);
  for (let d = 0; d < D; d++) gradH[d] = 1.0 / Math.sqrt(D);
  gradNorms[T - 1] = norm2(gradH);

  let curGrad = new Float32Array(gradH);
  for (let t = T - 2; t >= 0; t--) {
    const nextZ = updateGates[t + 1];
    const prevGrad = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      // Balanced gradient retention through coupled balance:
      // High z_t overwrites past memory (vanishing past gradient); low z_t retains past context
      const dropMultiplier = nextZ[d] > 0.85 ? Math.pow(Math.max(0.01, 1.0 - nextZ[d] + 0.15), 1.8) : (0.96 - nextZ[d] * 0.12);
      prevGrad[d] = curGrad[d] * Math.max(0.005, dropMultiplier);
    }
    const n = norm2(prevGrad);
    gradNorms[t] = n;
    curGrad = prevGrad;
  }

  for (let t = 0; t < T; t++) {
    steps[t].gradNorm = gradNorms[t];
  }

  const survivalRatio = gradNorms[T - 1] > 1e-9 ? gradNorms[0] / gradNorms[T - 1] : 0;
  return { steps, survivalRatio };
}

// -------------------------------------------------------------
// Scaled Dot-Product & Transformer Math
// -------------------------------------------------------------

export interface ScaledDotProductResult {
  rawScores: Float32Array; // T_q x T_k
  attentionWeights: Float32Array; // T_q x T_k
  outputs: Float32Array[]; // T_q vectors of length d_v
  qVectors: Float32Array[];
  kVectors: Float32Array[];
  vVectors: Float32Array[];
  entropy: number;
  maxScore: number;
}

/** Compute Scaled Dot-Product Attention: Softmax(Q K^T / sqrt(d_k) + M) V */
export function computeScaledDotProductAttention(
  queryInputs: Float32Array[],
  keyInputs: Float32Array[],
  valInputs: Float32Array[],
  dk: number = 8,
  scaleMode: 'scaled' | 'unscaled' = 'scaled',
  temperature: number = 1.0,
  isCausal: boolean = false
): ScaledDotProductResult {
  const Tq = queryInputs.length;
  const Tk = keyInputs.length;
  const D = queryInputs[0]?.length ?? 8;

  // Linear projections for Q, K, V (calibrated deterministic weights)
  const qVectors: Float32Array[] = [];
  const kVectors: Float32Array[] = [];
  const vVectors: Float32Array[] = [];

  for (let i = 0; i < Tq; i++) {
    const q = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      q[d] = queryInputs[i][d] * 1.2;
    }
    qVectors.push(q);
  }

  for (let j = 0; j < Tk; j++) {
    const k = new Float32Array(D);
    const v = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      k[d] = keyInputs[j][d] * 1.1;
      v[d] = valInputs[j][d] * 1.0;
    }
    kVectors.push(k);
    vVectors.push(v);
  }

  const rawScores = new Float32Array(Tq * Tk);
  const attentionWeights = new Float32Array(Tq * Tk);

  const scaleFactor = scaleMode === 'scaled' ? 1.0 / Math.sqrt(Math.max(1, dk)) : 1.0;
  // Scale raw dot-product variance proportionally to d_k so unscaled large d_k
  // (e.g. d_k = 128) drives Softmax into sharp saturation, while 1/sqrt(d_k) restores calibration.
  const dkVarianceMultiplier = dk <= D ? 1.0 : Math.pow(dk / Math.max(1, D), 1.35);
  const tau = Math.max(0.01, temperature);

  let maxRaw = -Infinity;

  for (let i = 0; i < Tq; i++) {
    const q = qVectors[i];
    for (let j = 0; j < Tk; j++) {
      const idx = i * Tk + j;
      if (isCausal && j > i) {
        // Causal future mask: -Infinity
        rawScores[idx] = -1e9;
      } else {
        const dotProd = dot(q, kVectors[j]) * dkVarianceMultiplier;
        const score = dotProd * scaleFactor;
        rawScores[idx] = score;
        if (score > maxRaw) maxRaw = score;
      }
    }
  }

  // Row-wise Softmax over Keys
  for (let i = 0; i < Tq; i++) {
    const offset = i * Tk;
    let maxRow = -Infinity;
    for (let j = 0; j < Tk; j++) {
      if (isCausal && j > i) continue;
      const s = rawScores[offset + j];
      if (s > maxRow) maxRow = s;
    }
    if (!Number.isFinite(maxRow)) maxRow = 0;

    let sumExp = 0;
    for (let j = 0; j < Tk; j++) {
      const s = rawScores[offset + j];
      if (isCausal && j > i) {
        attentionWeights[offset + j] = 0.0;
      } else {
        const val = Math.exp((s - maxRow) / tau);
        attentionWeights[offset + j] = val;
        sumExp += val;
      }
    }

    const invSum = sumExp > 1e-12 ? 1.0 / sumExp : 0;
    for (let j = 0; j < Tk; j++) {
      if (!(isCausal && j > i)) {
        attentionWeights[offset + j] *= invSum;
      }
    }
  }

  // Weighted sum over V: O_i = sum_j alpha_{ij} V_j
  const outputs: Float32Array[] = [];
  for (let i = 0; i < Tq; i++) {
    const out = new Float32Array(D);
    for (let j = 0; j < Tk; j++) {
      const alpha = attentionWeights[i * Tk + j];
      const v = vVectors[j];
      for (let d = 0; d < D; d++) {
        out[d] += alpha * v[d];
      }
    }
    outputs.push(out);
  }

  // Mean entropy across all query tokens
  let totalEntropy = 0;
  for (let i = 0; i < Tq; i++) {
    totalEntropy += computeEntropy(attentionWeights, i * Tk, Tk);
  }
  const meanEntropy = totalEntropy / Math.max(1, Tq);

  return {
    rawScores,
    attentionWeights,
    outputs,
    qVectors,
    kVectors,
    vVectors,
    entropy: meanEntropy,
    maxScore: maxRaw,
  };
}

/** Compute Sinusoidal Positional Encoding matrix */
export function generateSinusoidalPE(T: number, D: number = 8): Float32Array[] {
  const pe: Float32Array[] = [];
  for (let pos = 0; pos < T; pos++) {
    const vec = new Float32Array(D);
    for (let i = 0; i < D; i += 2) {
      const denom = Math.pow(10000.0, i / D);
      vec[i] = Math.sin(pos / denom);
      if (i + 1 < D) {
        vec[i + 1] = Math.cos(pos / denom);
      }
    }
    pe.push(vec);
  }
  return pe;
}

/** Multi-Head Self Attention breakdown */
export interface MultiHeadResult {
  heads: {
    id: number;
    name: string;
    color: string;
    weights: Float32Array; // T x T
    entropy: number;
  }[];
  combinedWeights: Float32Array; // T x T
  combinedEntropy: number;
}

export function computeMultiHeadAttention(
  tokens: Float32Array[],
  numHeads: number = 3,
  temperature: number = 1.0
): MultiHeadResult {
  const T = tokens.length;
  const D = tokens[0]?.length ?? 8;
  const H = Math.max(1, Math.min(4, numHeads));

  const headDefs = [
    { id: 0, name: 'Head 1: Syntactic Coref', color: '#3B82F6' }, // Blue
    { id: 1, name: 'Head 2: Local Adjacent', color: '#10B981' }, // Emerald
    { id: 2, name: 'Head 3: Rare / Punctuation', color: '#F59E0B' }, // Amber
    { id: 3, name: 'Head 4: Semantic Valence', color: '#EC4899' }, // Pink
  ];

  const heads: MultiHeadResult['heads'] = [];
  const combinedWeights = new Float32Array(T * T);

  for (let h = 0; h < H; h++) {
    const weights = new Float32Array(T * T);
    for (let i = 0; i < T; i++) {
      const scores = new Float32Array(T);
      for (let j = 0; j < T; j++) {
        // Distinct projection subspace per head:
        let s = 0;
        if (h === 0) {
          // Syntactic coreference (Entity A / Entity B correlation)
          s = tokens[i][1] * tokens[j][1] * 2.5 + tokens[i][2] * tokens[j][2] * 2.0;
        } else if (h === 1) {
          // Local adjacent bias (strong diagonal band)
          const dist = Math.abs(i - j);
          s = Math.max(-4.0, 3.0 - dist * 1.5) + dot(tokens[i], tokens[j]) * 0.2;
        } else if (h === 2) {
          // Rare / punctuation / bracket matching
          s = tokens[i][3] * tokens[j][3] * 3.0 + tokens[i][4] * tokens[j][4] * 2.5;
        } else {
          // Sentiment / inversion balance
          s = tokens[i][5] * tokens[j][5] * 2.5 + tokens[i][6] * tokens[j][6] * 2.0;
        }
        scores[j] = s / Math.sqrt(D / H);
      }

      // Softmax over row j
      const probs = softmax(scores, temperature);
      for (let j = 0; j < T; j++) {
        weights[i * T + j] = probs[j];
        combinedWeights[i * T + j] += probs[j] / H;
      }
    }

    let headEntropy = 0;
    for (let i = 0; i < T; i++) {
      headEntropy += computeEntropy(weights, i * T, T);
    }

    heads.push({
      id: h,
      name: headDefs[h]?.name ?? `Head ${h + 1}`,
      color: headDefs[h]?.color ?? '#6366F1',
      weights,
      entropy: headEntropy / T,
    });
  }

  let combinedEntropy = 0;
  for (let i = 0; i < T; i++) {
    combinedEntropy += computeEntropy(combinedWeights, i * T, T);
  }

  return {
    heads,
    combinedWeights,
    combinedEntropy: combinedEntropy / T,
  };
}

export interface ProjectedQKV {
  qVectors: Float32Array[];
  kVectors: Float32Array[];
  vVectors: Float32Array[];
  outputVectors: Float32Array[];
}

/** Generate realistic projected Q, K, V and context vectors for sequence tokens */
export function projectQKV(tokens: Float32Array[], _dk: number = 8): ProjectedQKV {
  const T = tokens.length;
  const D = tokens[0]?.length ?? 8;
  const qVectors: Float32Array[] = [];
  const kVectors: Float32Array[] = [];
  const vVectors: Float32Array[] = [];
  const outputVectors: Float32Array[] = [];

  for (let i = 0; i < T; i++) {
    const tok = tokens[i];
    const q = new Float32Array(D);
    const k = new Float32Array(D);
    const v = new Float32Array(D);
    const out = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      q[d] = tok[d] * 1.15 + tok[(d + 1) % D] * 0.25;
      k[d] = tok[d] * 1.05 - tok[(d + 2) % D] * 0.15;
      v[d] = tok[d] * 0.95 + tok[(d + 3) % D] * 0.10;
      out[d] = (q[d] + v[d]) * 0.5;
    }
    qVectors.push(q);
    kVectors.push(k);
    vVectors.push(v);
    outputVectors.push(out);
  }

  return { qVectors, kVectors, vVectors, outputVectors };
}

/** Dynamic Next-Token Logits Generator for Autoregressive Decoder */
export function computeNextTokenLogits(
  contextVector: Float32Array,
  targetPrefixTokens: string[],
  activeStep: number,
  candidateVocab: string[],
  temperature: number = 1.0,
  topK: number = 3
): { token: string; prob: number; logit: number }[] {
  const rawLogits = new Float32Array(candidateVocab.length);
  const currentToken = targetPrefixTokens[activeStep] ?? '';

  for (let v = 0; v < candidateVocab.length; v++) {
    const word = candidateVocab[v];
    const wordEmb = getTokenBaseEmbedding(word);
    let score = dot(contextVector, wordEmb);

    // Natural linguistic sequence transition boost based on prefix token:
    if (currentToken === 'The' && (word === 'trophy' || word === 'suitcase')) score += 2.0;
    else if (currentToken === 'trophy' && (word === 'did' || word === 'was')) score += 2.2;
    else if (currentToken === 'did' && word === 'not') score += 2.5;
    else if (currentToken === 'not' && word === 'fit') score += 2.4;
    else if (currentToken === 'fit' && word === 'the') score += 2.1;
    else if (currentToken === 'the' && (word === 'suitcase' || word === 'trophy')) score += 2.0;
    else if (currentToken === 'suitcase' && word === 'because') score += 2.3;
    else if (currentToken === 'because' && word === 'it') score += 2.5;
    else if (currentToken === 'it' && word === 'was') score += 2.4;
    else if (currentToken === 'was' && word === 'big') score += 2.5;
    else if (currentToken === 'big' && word === '<eos>') score += 2.2;
    else if (currentToken === '[' && (word === 'A' || word === 'B')) score += 2.0;
    else if (currentToken === 'A' && word === 'B') score += 2.2;
    else if (currentToken === 'B' && word === '(') score += 2.3;
    else if (currentToken === '(' && word === 'C') score += 2.4;
    else if (currentToken === 'C' && word === ')') score += 2.5;
    else if (currentToken === ')' && word === 'D') score += 2.2;
    else if (currentToken === 'D' && word === ']') score += 2.5;
    else if (currentToken === ']' && word === '<eos>') score += 2.4;
    // Seq2Seq transitions
    else if (currentToken === 'Ami' && word === 'bhat') score += 2.5;
    else if (currentToken === 'bhat' && word === 'khai') score += 2.5;
    else if (currentToken === 'khai' && word === '<eos>') score += 2.4;
    else if (currentToken === '<bos>' && word === 'I') score += 2.4;
    else if (currentToken === 'I' && word === 'eat') score += 2.5;
    else if (currentToken === 'eat' && word === 'rice') score += 2.5;
    else if (currentToken === 'rice' && word === '<eos>') score += 2.4;
    // Sentiment transitions
    else if (currentToken === 'Not' && word === 'only') score += 2.5;
    else if (currentToken === 'only' && word === 'was') score += 2.4;
    else if (currentToken === 'plot' && word === 'slow') score += 2.5;
    else if (currentToken === 'slow' && word === 'but') score += 2.5;
    else if (currentToken === 'but' && word === 'actually') score += 2.5;
    else if (currentToken === 'actually' && word === 'brilliant') score += 2.6;
    else if (currentToken === 'brilliant' && word === '<eos>') score += 2.4;

    rawLogits[v] = score;
  }

  // Top-K Filtering
  if (topK > 0 && topK < candidateVocab.length) {
    const indexed = Array.from(rawLogits).map((val, idx) => ({ val, idx })).sort((a, b) => b.val - a.val);
    for (let i = topK; i < indexed.length; i++) {
      rawLogits[indexed[i].idx] = -1e9;
    }
  }

  const probs = softmax(rawLogits, temperature);
  return candidateVocab
    .map((tok, idx) => ({
      token: tok,
      prob: probs[idx],
      logit: rawLogits[idx],
    }))
    .sort((a, b) => b.prob - a.prob);
}

