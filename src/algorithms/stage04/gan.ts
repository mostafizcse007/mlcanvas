import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import {
  initializeWeights,
} from './neuralUtils';
import { mulberry32, gaussianRandom } from '../../utils/prng';

export interface GANParams {
  dSteps?: number; // k in [1..5]
  lrG?: number;
  lrD?: number;
  lossMode?: 'non-saturating' | 'minimax-saturating';
  noiseVariance?: number;
  hiddenDim?: number;
}

export interface GANState {
  hiddenDim: number;
  // Generator: G_W1 (2 -> H), G_b1 (H), G_W2 (H -> H), G_b2 (H), G_W3 (H -> 2), G_b3 (2)
  GW1: Float32Array; Gb1: Float32Array;
  GW2: Float32Array; Gb2: Float32Array;
  GW3: Float32Array; Gb3: Float32Array;

  // Discriminator: D_W1 (2 -> H), D_b1 (H), D_W2 (H -> H), D_b2 (H), D_W3 (H -> 1), D_b3 (1)
  DW1: Float32Array; Db1: Float32Array;
  DW2: Float32Array; Db2: Float32Array;
  DW3: Float32Array; Db3: Float32Array;

  // Adam Moments Generator
  mGW1: Float32Array; vGW1: Float32Array; mGb1: Float32Array; vGb1: Float32Array;
  mGW2: Float32Array; vGW2: Float32Array; mGb2: Float32Array; vGb2: Float32Array;
  mGW3: Float32Array; vGW3: Float32Array; mGb3: Float32Array; vGb3: Float32Array;

  // Adam Moments Discriminator
  mDW1: Float32Array; vDW1: Float32Array; mDb1: Float32Array; vDb1: Float32Array;
  mDW2: Float32Array; vDW2: Float32Array; mDb2: Float32Array; vDb2: Float32Array;
  mDW3: Float32Array; vDW3: Float32Array; mDb3: Float32Array; vDb3: Float32Array;

  stepCountG: number;
  stepCountD: number;
  epoch: number;
  dLoss: number;
  gLoss: number;
  meanDReal: number;
  meanDFake: number;

  // Persistent tracked fake points with previous coordinates for velocity trails
  fakePoints: {
    id: number;
    z: [number, number];
    x1: number;
    x2: number;
    prevX1: number;
    prevX2: number;
  }[];

  rngSeed: number;
}

export function forwardGenerator(
  z1: number,
  z2: number,
  state: GANState
): {
  x1: number;
  x2: number;
  h1: Float32Array;
  h2: Float32Array;
  zh1: Float32Array;
  zh2: Float32Array;
} {
  const H = state.hiddenDim;

  // Layer 1: 2 -> H (Tanh)
  const zh1 = new Float32Array(H);
  const h1 = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    const sum = state.Gb1[j] + z1 * state.GW1[j * 2] + z2 * state.GW1[j * 2 + 1];
    zh1[j] = sum;
    h1[j] = Math.tanh(sum);
  }

  // Layer 2: H -> H (Tanh)
  const zh2 = new Float32Array(H);
  const h2 = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    let sum = state.Gb2[j];
    const off = j * H;
    for (let i = 0; i < H; i++) sum += h1[i] * state.GW2[off + i];
    zh2[j] = sum;
    h2[j] = Math.tanh(sum);
  }

  // Layer 3: H -> 2 (Linear)
  let sum1 = state.Gb3[0];
  let sum2 = state.Gb3[1];
  for (let i = 0; i < H; i++) {
    sum1 += h2[i] * state.GW3[i];
    sum2 += h2[i] * state.GW3[H + i];
  }

  return { x1: sum1, x2: sum2, h1, h2, zh1, zh2 };
}

export function forwardDiscriminator(
  x1: number,
  x2: number,
  state: GANState
): {
  prob: number; // Sigmoid in [0, 1]
  logit: number;
  h1: Float32Array;
  h2: Float32Array;
  zh1: Float32Array;
  zh2: Float32Array;
} {
  const H = state.hiddenDim;

  // Layer 1: 2 -> H (Leaky ReLU)
  const zh1 = new Float32Array(H);
  const h1 = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    const sum = state.Db1[j] + x1 * state.DW1[j * 2] + x2 * state.DW1[j * 2 + 1];
    zh1[j] = sum;
    h1[j] = sum > 0 ? sum : 0.2 * sum;
  }

  // Layer 2: H -> H (Leaky ReLU)
  const zh2 = new Float32Array(H);
  const h2 = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    let sum = state.Db2[j];
    const off = j * H;
    for (let i = 0; i < H; i++) sum += h1[i] * state.DW2[off + i];
    zh2[j] = sum;
    h2[j] = sum > 0 ? sum : 0.2 * sum;
  }

  // Layer 3: H -> 1 (Logit & Sigmoid)
  let logit = state.Db3[0];
  for (let i = 0; i < H; i++) {
    logit += h2[i] * state.DW3[i];
  }

  const clampedLogit = Math.max(-20, Math.min(20, logit));
  const prob = 1.0 / (1.0 + Math.exp(-clampedLogit));

  return { prob, logit, h1, h2, zh1, zh2 };
}

export function computeDiscriminatorSpatialGradient(
  x1: number,
  x2: number,
  state: GANState
): [number, number] {
  const H = state.hiddenDim;
  const fwd = forwardDiscriminator(x1, x2, state);

  // Gradient of logit w.r.t input coordinates [x1, x2]
  // dLogit/dh2 = DW3
  const delta2 = new Float32Array(H);
  for (let i = 0; i < H; i++) {
    const deriv = fwd.zh2[i] > 0 ? 1 : 0.2;
    delta2[i] = state.DW3[i] * deriv;
  }

  // dLogit/dh1 = DW2^T * delta2
  const delta1 = new Float32Array(H);
  for (let i = 0; i < H; i++) {
    let sum = 0;
    for (let j = 0; j < H; j++) {
      sum += state.DW2[j * H + i] * delta2[j];
    }
    const deriv = fwd.zh1[i] > 0 ? 1 : 0.2;
    delta1[i] = sum * deriv;
  }

  // dLogit/dx = DW1^T * delta1
  let gradX1 = 0;
  let gradX2 = 0;
  for (let j = 0; j < H; j++) {
    gradX1 += state.DW1[j * 2] * delta1[j];
    gradX2 += state.DW1[j * 2 + 1] * delta1[j];
  }

  // Non-saturating pull: grad_x log D(x) = (1 - D(x)) * grad_x logit
  const pullFactor = 1.0 - fwd.prob;
  return [pullFactor * gradX1, pullFactor * gradX2];
}

/**
 * Unaveraged Goodfellow minimax discriminator loss:
 * L_D = -1/N * sum[ log D(x) + log(1 - D(G(z))) ]
 * At Nash Equilibrium D(real) = 0.5 and D(fake) = 0.5, L_D* = 2 * ln(2) ≈ 1.3863.
 */
export function computeDiscriminatorLoss(realProbs: number[], fakeProbs: number[]): number {
  const n = Math.min(realProbs.length, fakeProbs.length);
  if (n === 0) return 0;
  let sumLoss = 0;
  for (let i = 0; i < n; i++) {
    const pReal = Math.max(1e-7, Math.min(1 - 1e-7, realProbs[i]));
    const pFake = Math.max(1e-7, Math.min(1 - 1e-7, fakeProbs[i]));
    sumLoss += -Math.log(pReal) - Math.log(1.0 - pFake);
  }
  return sumLoss / n;
}

export const ganPlugin: AlgorithmPlugin<GANState, GANParams> = {
  id: 'gan',
  name: 'Generative Adversarial Network (Minimax Arena)',
  shortCode: 'GAN',
  stage: 'neural-graph',
  taskType: 'density-estimation',
  executionMode: 'epoch-loop',
  description: 'Two-player zero-sum minimax game between Generator G(z) and Discriminator D(x) with live vector field dynamics.',
  paperReference: 'Goodfellow et al., Generative Adversarial Networks, NeurIPS 2014',

  primaryParams: [
    {
      id: 'dSteps',
      name: 'D Steps per G Step (k)',
      type: 'slider',
      min: 1,
      max: 5,
      step: 1,
      default: 1,
      description: 'Number of discriminator updates per single generator update.',
    },
    {
      id: 'lrG',
      name: 'Generator LR (η_G)',
      type: 'select',
      options: [
        { label: '0.005 (Fine)', value: 0.005 },
        { label: '0.01', value: 0.01 },
        { label: '0.03 (Standard)', value: 0.03 },
        { label: '0.08 (Fast)', value: 0.08 },
      ],
      default: 0.03,
      description: 'Learning rate for generator updates.',
    },
    {
      id: 'lrD',
      name: 'Discriminator LR (η_D)',
      type: 'select',
      options: [
        { label: '0.005 (Fine)', value: 0.005 },
        { label: '0.01', value: 0.01 },
        { label: '0.03 (Standard)', value: 0.03 },
        { label: '0.08 (Fast)', value: 0.08 },
      ],
      default: 0.03,
      description: 'Learning rate for discriminator updates.',
    },
  ],

  advancedParams: [
    {
      id: 'lossMode',
      name: 'Generator Objective',
      type: 'select',
      options: [
        { label: 'Non-Saturating (-log D(G(z)))', value: 'non-saturating' },
        { label: 'Minimax Saturating (log(1 - D))', value: 'minimax-saturating' },
      ],
      default: 'non-saturating',
      description: 'Choice between original minimax and heuristic non-saturating loss.',
    },
    {
      id: 'noiseVariance',
      name: 'Latent Noise Variance (σ²)',
      type: 'select',
      options: [
        { label: '0.5 (Tight)', value: 0.5 },
        { label: '1.0 (Unit Normal)', value: 1.0 },
        { label: '1.5 (Broad)', value: 1.5 },
      ],
      default: 1.0,
      description: 'Standard deviation of Gaussian latent distribution z ~ N(0, σ²I).',
    },
    {
      id: 'hiddenDim',
      name: 'Network Width',
      type: 'slider',
      min: 8,
      max: 32,
      step: 8,
      default: 16,
      description: 'Hidden layer capacity for Generator and Discriminator.',
    },
  ],

  init(dataset: DatasetContainer, params: GANParams, seed: number): GANState {
    const rand = mulberry32(seed);
    const H = Math.max(8, Math.min(32, Number(params.hiddenDim ?? 16)));

    // Generator weights (2 -> H -> H -> 2)
    const GW1 = initializeWeights(2, H, 'xavier', rand);
    const Gb1 = new Float32Array(H);
    const GW2 = initializeWeights(H, H, 'xavier', rand);
    const Gb2 = new Float32Array(H);
    const GW3 = initializeWeights(H, 2, 'xavier', rand);
    const Gb3 = new Float32Array(2);

    // Discriminator weights (2 -> H -> H -> 1)
    const DW1 = initializeWeights(2, H, 'xavier', rand);
    const Db1 = new Float32Array(H);
    const DW2 = initializeWeights(H, H, 'xavier', rand);
    const Db2 = new Float32Array(H);
    const DW3 = initializeWeights(H, 1, 'xavier', rand);
    const Db3 = new Float32Array(1);

    const N = Math.max(64, Math.min(160, dataset.train.length || 100));
    const noiseStd = Math.sqrt(Number(params.noiseVariance ?? 1.0));

    const fakePoints: GANState['fakePoints'] = [];
    for (let i = 0; i < N; i++) {
      const z1 = gaussianRandom(rand, 0, noiseStd);
      const z2 = gaussianRandom(rand, 0, noiseStd);
      fakePoints.push({
        id: i + 1,
        z: [z1, z2],
        x1: 0,
        x2: 0,
        prevX1: 0,
        prevX2: 0,
      });
    }

    const state: GANState = {
      hiddenDim: H,
      GW1, Gb1, GW2, Gb2, GW3, Gb3,
      DW1, Db1, DW2, Db2, DW3, Db3,

      mGW1: new Float32Array(GW1.length), vGW1: new Float32Array(GW1.length),
      mGb1: new Float32Array(Gb1.length), vGb1: new Float32Array(Gb1.length),
      mGW2: new Float32Array(GW2.length), vGW2: new Float32Array(GW2.length),
      mGb2: new Float32Array(Gb2.length), vGb2: new Float32Array(Gb2.length),
      mGW3: new Float32Array(GW3.length), vGW3: new Float32Array(GW3.length),
      mGb3: new Float32Array(Gb3.length), vGb3: new Float32Array(Gb3.length),

      mDW1: new Float32Array(DW1.length), vDW1: new Float32Array(DW1.length),
      mDb1: new Float32Array(Db1.length), vDb1: new Float32Array(Db1.length),
      mDW2: new Float32Array(DW2.length), vDW2: new Float32Array(DW2.length),
      mDb2: new Float32Array(Db2.length), vDb2: new Float32Array(Db2.length),
      mDW3: new Float32Array(DW3.length), vDW3: new Float32Array(DW3.length),
      mDb3: new Float32Array(Db3.length), vDb3: new Float32Array(Db3.length),

      stepCountG: 0,
      stepCountD: 0,
      epoch: 0,
      dLoss: 2 * Math.LN2, // 2 * ln(2) ≈ 1.3863
      gLoss: Math.LN2, // ln(2) ≈ 0.6931
      meanDReal: 0.5,
      meanDFake: 0.5,
      fakePoints,
      rngSeed: seed || 42,
    };

    // Initialize initial fake positions
    for (const fp of state.fakePoints) {
      const fwd = forwardGenerator(fp.z[0], fp.z[1], state);
      fp.x1 = fwd.x1;
      fp.x2 = fwd.x2;
      fp.prevX1 = fwd.x1;
      fp.prevX2 = fwd.x2;
    }

    return state;
  },

  step(state: GANState, dataset: DatasetContainer, params: GANParams): StepResult<GANState> {
    const realData = dataset.train;
    const N = realData.length;
    if (N === 0) {
      return {
        state,
        epoch: state.epoch,
        metrics: { trainLoss: 0, testLoss: 0 },
      };
    }

    const kSteps = Math.max(1, Math.min(5, Number(params.dSteps ?? 1)));
    const lrD = Number(params.lrD ?? 0.03);
    const lrG = Number(params.lrG ?? 0.03);
    const lossMode = params.lossMode ?? 'non-saturating';
    const noiseStd = Math.sqrt(Number(params.noiseVariance ?? 1.0));
    const H = state.hiddenDim;

    let latestDLoss = 0;
    let latestDReal = 0;
    let latestDFake = 0;

    // Adam hyperparams
    const b1 = 0.5; // DCGAN-style beta1
    const b2 = 0.999;
    const eps = 1e-8;

    const initialSeed = (state.rngSeed ?? (state.epoch * 1000 + 42)) >>> 0;
    const rand = mulberry32(initialSeed);
    state.rngSeed = ((initialSeed + 0x6d2b79f5) | 0) >>> 0;

    // 1. Train Discriminator for k steps
    for (let k = 0; k < kSteps; k++) {
      const gDW1 = new Float32Array(state.DW1.length);
      const gDb1 = new Float32Array(state.Db1.length);
      const gDW2 = new Float32Array(state.DW2.length);
      const gDb2 = new Float32Array(state.Db2.length);
      const gDW3 = new Float32Array(state.DW3.length);
      const gDb3 = new Float32Array(state.Db3.length);

      let sumLoss = 0;
      let sumReal = 0;
      let sumFake = 0;

      for (let i = 0; i < N; i++) {
        const real = realData[i];

        // --- Real Point pass ---
        // Target = 1.0. Loss = -log(D(x)) -> dLoss/dLogit = D(x) - 1.0
        const fwdReal = forwardDiscriminator(real.x1, real.x2, state);
        const pReal = Math.max(1e-7, Math.min(1 - 1e-7, fwdReal.prob));
        sumReal += pReal;
        sumLoss += -Math.log(pReal);

        const deltaRealLogit = pReal - 1.0;
        gDb3[0] += deltaRealLogit;
        for (let j = 0; j < H; j++) {
          gDW3[j] += deltaRealLogit * fwdReal.h2[j];
        }

        // Layer 2
        const deltaRealH2 = new Float32Array(H);
        for (let j = 0; j < H; j++) {
          const deriv = fwdReal.zh2[j] > 0 ? 1 : 0.2;
          deltaRealH2[j] = deltaRealLogit * state.DW3[j] * deriv;
          gDb2[j] += deltaRealH2[j];
          for (let p = 0; p < H; p++) {
            gDW2[j * H + p] += deltaRealH2[j] * fwdReal.h1[p];
          }
        }

        // Layer 1
        for (let j = 0; j < H; j++) {
          let sum = 0;
          for (let q = 0; q < H; q++) sum += state.DW2[q * H + j] * deltaRealH2[q];
          const deriv = fwdReal.zh1[j] > 0 ? 1 : 0.2;
          const deltaRealH1 = sum * deriv;
          gDb1[j] += deltaRealH1;
          gDW1[j * 2] += deltaRealH1 * real.x1;
          gDW1[j * 2 + 1] += deltaRealH1 * real.x2;
        }

        // --- Fake Point pass ---
        // Sample z and evaluate G(z)
        const z1 = gaussianRandom(rand, 0, noiseStd);
        const z2 = gaussianRandom(rand, 0, noiseStd);
        const gen = forwardGenerator(z1, z2, state);

        // Target = 0.0. Loss = -log(1 - D(G(z))) -> dLoss/dLogit = D(G(z)) - 0.0
        const fwdFake = forwardDiscriminator(gen.x1, gen.x2, state);
        const pFake = Math.max(1e-7, Math.min(1 - 1e-7, fwdFake.prob));
        sumFake += pFake;
        sumLoss += -Math.log(1.0 - pFake);

        const deltaFakeLogit = pFake;
        gDb3[0] += deltaFakeLogit;
        for (let j = 0; j < H; j++) {
          gDW3[j] += deltaFakeLogit * fwdFake.h2[j];
        }

        const deltaFakeH2 = new Float32Array(H);
        for (let j = 0; j < H; j++) {
          const deriv = fwdFake.zh2[j] > 0 ? 1 : 0.2;
          deltaFakeH2[j] = deltaFakeLogit * state.DW3[j] * deriv;
          gDb2[j] += deltaFakeH2[j];
          for (let p = 0; p < H; p++) {
            gDW2[j * H + p] += deltaFakeH2[j] * fwdFake.h1[p];
          }
        }

        for (let j = 0; j < H; j++) {
          let sum = 0;
          for (let q = 0; q < H; q++) sum += state.DW2[q * H + j] * deltaFakeH2[q];
          const deriv = fwdFake.zh1[j] > 0 ? 1 : 0.2;
          const deltaFakeH1 = sum * deriv;
          gDb1[j] += deltaFakeH1;
          gDW1[j * 2] += deltaFakeH1 * gen.x1;
          gDW1[j * 2 + 1] += deltaFakeH1 * gen.x2;
        }
      }

      // Discriminator Adam Step
      state.stepCountD++;
      const tD = state.stepCountD;
      const scaleD = 1.0 / (N * 2);

      const updateAdamD = (w: Float32Array, gw: Float32Array, m: Float32Array, v: Float32Array) => {
        for (let idx = 0; idx < w.length; idx++) {
          const grad = gw[idx] * scaleD;
          m[idx] = b1 * m[idx] + (1 - b1) * grad;
          v[idx] = b2 * v[idx] + (1 - b2) * grad * grad;
          const mHat = m[idx] / (1 - Math.pow(b1, tD));
          const vHat = v[idx] / (1 - Math.pow(b2, tD));
          w[idx] -= (lrD * mHat) / (Math.sqrt(vHat) + eps);
        }
      };

      updateAdamD(state.DW1, gDW1, state.mDW1, state.vDW1);
      updateAdamD(state.Db1, gDb1, state.mDb1, state.vDb1);
      updateAdamD(state.DW2, gDW2, state.mDW2, state.vDW2);
      updateAdamD(state.Db2, gDb2, state.mDb2, state.vDb2);
      updateAdamD(state.DW3, gDW3, state.mDW3, state.vDW3);
      updateAdamD(state.Db3, gDb3, state.mDb3, state.vDb3);

      latestDLoss = sumLoss / N;
      latestDReal = sumReal / N;
      latestDFake = sumFake / N;
    }

    // 2. Train Generator for 1 step
    const gGW1 = new Float32Array(state.GW1.length);
    const gGb1 = new Float32Array(state.Gb1.length);
    const gGW2 = new Float32Array(state.GW2.length);
    const gGb2 = new Float32Array(state.Gb2.length);
    const gGW3 = new Float32Array(state.GW3.length);
    const gGb3 = new Float32Array(state.Gb3.length);

    let latestGLoss = 0;

    for (let i = 0; i < N; i++) {
      const z1 = gaussianRandom(rand, 0, noiseStd);
      const z2 = gaussianRandom(rand, 0, noiseStd);
      const gen = forwardGenerator(z1, z2, state);

      const fwdD = forwardDiscriminator(gen.x1, gen.x2, state);
      const pD = Math.max(1e-7, Math.min(1 - 1e-7, fwdD.prob));

      // Loss for Generator
      let deltaLogit = 0;
      if (lossMode === 'non-saturating') {
        // L_G = -log(D(G(z))) -> dL/dLogit = pD - 1.0 (strong gradient when pD -> 0)
        latestGLoss += -Math.log(pD);
        deltaLogit = pD - 1.0;
      } else {
        // L_G = log(1 - D(G(z))) -> dL/dLogit = pD (vanishes when pD -> 0)
        latestGLoss += Math.log(1.0 - pD);
        deltaLogit = pD;
      }

      // Backprop delta through Discriminator down to [gen.x1, gen.x2]
      const deltaDH2 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const deriv = fwdD.zh2[j] > 0 ? 1 : 0.2;
        deltaDH2[j] = deltaLogit * state.DW3[j] * deriv;
      }

      const deltaDH1 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        let sum = 0;
        for (let q = 0; q < H; q++) sum += state.DW2[q * H + j] * deltaDH2[q];
        const deriv = fwdD.zh1[j] > 0 ? 1 : 0.2;
        deltaDH1[j] = sum * deriv;
      }

      let dLoss_dx1 = 0;
      let dLoss_dx2 = 0;
      for (let j = 0; j < H; j++) {
        dLoss_dx1 += state.DW1[j * 2] * deltaDH1[j];
        dLoss_dx2 += state.DW1[j * 2 + 1] * deltaDH1[j];
      }

      // Now backprop dLoss_dx into Generator Layer 3
      gGb3[0] += dLoss_dx1;
      gGb3[1] += dLoss_dx2;
      for (let j = 0; j < H; j++) {
        gGW3[j] += dLoss_dx1 * gen.h2[j];
        gGW3[H + j] += dLoss_dx2 * gen.h2[j];
      }

      // Generator Layer 2
      const deltaGH2 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const sum = state.GW3[j] * dLoss_dx1 + state.GW3[H + j] * dLoss_dx2;
        const deriv = 1 - gen.h2[j] * gen.h2[j]; // tanh'
        deltaGH2[j] = sum * deriv;
        gGb2[j] += deltaGH2[j];
        for (let p = 0; p < H; p++) {
          gGW2[j * H + p] += deltaGH2[j] * gen.h1[p];
        }
      }

      // Generator Layer 1
      for (let j = 0; j < H; j++) {
        let sum = 0;
        for (let q = 0; q < H; q++) sum += state.GW2[q * H + j] * deltaGH2[q];
        const deriv = 1 - gen.h1[j] * gen.h1[j]; // tanh'
        const deltaGH1 = sum * deriv;
        gGb1[j] += deltaGH1;
        gGW1[j * 2] += deltaGH1 * z1;
        gGW1[j * 2 + 1] += deltaGH1 * z2;
      }
    }

    state.stepCountG++;
    const tG = state.stepCountG;
    const scaleG = 1.0 / N;

    const updateAdamG = (w: Float32Array, gw: Float32Array, m: Float32Array, v: Float32Array) => {
      for (let idx = 0; idx < w.length; idx++) {
        const grad = gw[idx] * scaleG;
        m[idx] = b1 * m[idx] + (1 - b1) * grad;
        v[idx] = b2 * v[idx] + (1 - b2) * grad * grad;
        const mHat = m[idx] / (1 - Math.pow(b1, tG));
        const vHat = v[idx] / (1 - Math.pow(b2, tG));
        w[idx] -= (lrG * mHat) / (Math.sqrt(vHat) + eps);
      }
    };

    updateAdamG(state.GW1, gGW1, state.mGW1, state.vGW1);
    updateAdamG(state.Gb1, gGb1, state.mGb1, state.vGb1);
    updateAdamG(state.GW2, gGW2, state.mGW2, state.vGW2);
    updateAdamG(state.Gb2, gGb2, state.mGb2, state.vGb2);
    updateAdamG(state.GW3, gGW3, state.mGW3, state.vGW3);
    updateAdamG(state.Gb3, gGb3, state.mGb3, state.vGb3);

    state.epoch++;
    state.dLoss = latestDLoss;
    state.gLoss = latestGLoss / N;
    state.meanDReal = latestDReal;
    state.meanDFake = latestDFake;

    // Update persistent tracked fake points
    for (const fp of state.fakePoints) {
      fp.prevX1 = fp.x1;
      fp.prevX2 = fp.x2;
      const fwd = forwardGenerator(fp.z[0], fp.z[1], state);
      fp.x1 = fwd.x1;
      fp.x2 = fwd.x2;
    }

    const metrics = this.loss(state, dataset, params);

    return {
      state,
      epoch: state.epoch,
      metrics,
      converged: Math.abs(state.meanDReal - 0.5) < 0.05 && Math.abs(state.meanDFake - 0.5) < 0.05,
    };
  },

  predict(input: Float32Array, state: GANState, _params: GANParams): number {
    const fwd = forwardDiscriminator(input[0], input[1], state);
    // Map D(x) in [0, 1] to centered score [-1, +1] for standard decision heatmap rendering:
    // D(x) = 0.5 -> 0, D(x) = 1.0 -> +1.5, D(x) = 0.0 -> -1.5
    return (fwd.prob - 0.5) * 3.0;
  },

  loss(state: GANState, _dataset: DatasetContainer, _params: GANParams): TelemetryMetrics {
    return {
      trainLoss: state.dLoss,
      testLoss: state.gLoss,
      custom: {
        dLoss: state.dLoss.toFixed(4),
        gLoss: state.gLoss.toFixed(4),
        meanDReal: `${(state.meanDReal * 100).toFixed(1)}%`,
        meanDFake: `${(state.meanDFake * 100).toFixed(1)}%`,
      },
    };
  },

  getVisualArtifacts(state: GANState, dataset: DatasetContainer, params: GANParams): VisualArtifact[] {
    const realPoints = dataset.train.map((p) => ({
      id: p.id,
      x1: p.x1,
      x2: p.x2,
      label: p.y,
    }));

    const fakePoints = state.fakePoints.map((fp) => {
      const pull = computeDiscriminatorSpatialGradient(fp.x1, fp.x2, state);
      const fwdD = forwardDiscriminator(fp.x1, fp.x2, state);
      return {
        id: fp.id,
        x1: fp.x1,
        x2: fp.x2,
        prevX1: fp.prevX1,
        prevX2: fp.prevX2,
        pullVector: pull,
        dScore: fwdD.prob,
      };
    });

    // Generate 9x9 warped latent mesh
    const gridRes = 9;
    const latentGridWarp: { originalZ: [number, number]; projectedX: [number, number] }[] = [];
    const span = 2.5;
    for (let r = 0; r < gridRes; r++) {
      const z2 = span - (r / (gridRes - 1)) * 2 * span;
      for (let c = 0; c < gridRes; c++) {
        const z1 = -span + (c / (gridRes - 1)) * 2 * span;
        const gen = forwardGenerator(z1, z2, state);
        latentGridWarp.push({
          originalZ: [z1, z2],
          projectedX: [gen.x1, gen.x2],
        });
      }
    }

    return [
      {
        type: 'gan-minimax-overlay',
        realPoints,
        fakePoints,
        latentGridWarp,
        gridResolution: gridRes,
        dLoss: state.dLoss,
        gLoss: state.gLoss,
        meanDReal: state.meanDReal,
        meanDFake: state.meanDFake,
        lossMode: params.lossMode ?? 'non-saturating',
      },
    ];
  },

  serializeSnapshot(state: GANState): Float32Array {
    let totalLen = 6;
    for (const arr of [
      state.GW1, state.Gb1, state.GW2, state.Gb2, state.GW3, state.Gb3,
      state.DW1, state.Db1, state.DW2, state.Db2, state.DW3, state.Db3,
    ]) {
      totalLen += arr.length;
    }

    const buf = new Float32Array(totalLen);
    buf[0] = state.epoch;
    buf[1] = state.dLoss;
    buf[2] = state.gLoss;
    buf[3] = state.meanDReal;
    buf[4] = state.meanDFake;
    buf[5] = state.stepCountG;

    let idx = 6;
    const write = (arr: Float32Array) => {
      buf.set(arr, idx);
      idx += arr.length;
    };
    for (const arr of [
      state.GW1, state.Gb1, state.GW2, state.Gb2, state.GW3, state.Gb3,
      state.DW1, state.Db1, state.DW2, state.Db2, state.DW3, state.Db3,
    ]) {
      write(arr);
    }

    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: GANState): GANState {
    state.epoch = Math.floor(buffer[0]);
    state.dLoss = buffer[1];
    state.gLoss = buffer[2];
    state.meanDReal = buffer[3];
    state.meanDFake = buffer[4];
    state.stepCountG = Math.floor(buffer[5]);

    let idx = 6;
    const read = (arr: Float32Array) => {
      for (let i = 0; i < arr.length; i++) arr[i] = buffer[idx++];
    };
    for (const arr of [
      state.GW1, state.Gb1, state.GW2, state.Gb2, state.GW3, state.Gb3,
      state.DW1, state.Db1, state.DW2, state.Db2, state.DW3, state.Db3,
    ]) {
      read(arr);
    }

    return state;
  },

  explain(state: GANState, params: GANParams): ExplanationPayload {
    const k = params.dSteps ?? 1;
    const mode = params.lossMode ?? 'non-saturating';

    return {
      summary: `Adversarial zero-sum game pitting Generator G(z) against Discriminator D(x). G folds latent Gaussian noise to match the real data distribution, while D assigns probability D(x) ∈ [0, 1].`,
      coreAssumption: `At Nash Equilibrium, G exactly recovers the true data generating distribution p_data(x), and D is rendered completely uncertain: D(x) = 0.5 everywhere.`,
      whenItFails: `Mode collapse: Generator collapses onto a single high-reward cluster while ignoring other clusters. Overpowered Discriminator: D reaches 100% accuracy, saturating gradients and freezing fake points in place.`,
      microLog: [
        `Discriminator Phase (${k} step${k > 1 ? 's' : ''}): Updated D on real vs fake batches. Mean D(Real) = ${(state.meanDReal * 100).toFixed(1)}%, Mean D(Fake) = ${(state.meanDFake * 100).toFixed(1)}%.`,
        `Generator Phase (1 step): Updated G using ${mode.toUpperCase()} loss (GLoss = ${state.gLoss.toFixed(4)}).`,
        `Vector Dynamics: Evaluated discriminator pull vectors ∇_x log D(x) directing fake point velocities toward real probability mass.`,
      ],
      mathLatex: {
        objective: `\\min_G \\max_D V(D, G) = \\mathbb{E}_{\\mathbf{x} \\sim p_{\\text{data}}}[\\log D(\\mathbf{x})] + \\mathbb{E}_{\\mathbf{z} \\sim p_\\mathbf{z}}[\\log(1 - D(G(\\mathbf{z})))]`,
        lossOrUpdate: mode === 'non-saturating' ? `\\mathcal{L}_G = -\\mathbb{E}_{\\mathbf{z}}[\\log D(G(\\mathbf{z}))], \\quad \\nabla_\\mathbf{x} \\log D(\\mathbf{x}) = (1 - D(\\mathbf{x})) \\nabla_\\mathbf{x} \\operatorname{logit}(\\mathbf{x})` : `\\mathcal{L}_G = \\mathbb{E}_{\\mathbf{z}}[\\log(1 - D(G(\\mathbf{z})))]`,
        dualOrConstraint: `\\text{Nash Equilibrium: } p_g = p_{\\text{data}} \\implies D^*(\\mathbf{x}) = \\frac{1}{2}, \\quad L_D^* = 2 \\ln 2 \\approx 1.386`,
      },
      codeSnippet: {
        sklearn: `# GAN is an adversarial game across coupled deep neural networks without a direct scikit-learn model.
# PyTorch implementation provided below.`,
        pytorch: `import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, hidden=${state.hiddenDim}):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 2)
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, hidden=${state.hiddenDim}):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)`,
      },
    };
  },
};
