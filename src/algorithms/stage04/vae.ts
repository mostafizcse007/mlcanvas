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

export interface VAEParams {
  beta?: number; // KL weight beta
  learningRate?: number;
  samplingNoise?: 'on' | 'off' | boolean;
  hiddenDim?: number;
  probeZ1?: number;
  probeZ2?: number;
}

export interface VAEState {
  hiddenDim: number;
  latentDim: 2; // Fixed 2D for rich Gaussian ellipse visualization
  // Encoder: W_e1 (2 -> H), b_e1 (H), W_mu (H -> 2), b_mu (2), W_logvar (H -> 2), b_logvar (2)
  We1: Float32Array;
  be1: Float32Array;
  Wmu: Float32Array;
  bmu: Float32Array;
  Wlv: Float32Array;
  blv: Float32Array;

  // Decoder: W_d1 (2 -> H), b_d1 (H), W_d2 (H -> 2), b_d2 (2)
  Wd1: Float32Array;
  bd1: Float32Array;
  Wd2: Float32Array;
  bd2: Float32Array;

  // Adam Moments
  mWe1: Float32Array; vWe1: Float32Array;
  mbe1: Float32Array; vbe1: Float32Array;
  mWmu: Float32Array; vWmu: Float32Array;
  mbmu: Float32Array; vbmu: Float32Array;
  mWlv: Float32Array; vWlv: Float32Array;
  mblv: Float32Array; vblv: Float32Array;
  mWd1: Float32Array; vWd1: Float32Array;
  mbd1: Float32Array; vbd1: Float32Array;
  mWd2: Float32Array; vWd2: Float32Array;
  mbd2: Float32Array; vbd2: Float32Array;

  stepCount: number;
  epoch: number;
  reconLoss: number;
  klLoss: number;
  totalLoss: number;
  probePoint: { z: [number, number]; decoded: [number, number] };
  rngSeed: number;
}

export function encodeVAE(
  x1: number,
  x2: number,
  state: VAEState
): { mu: [number, number]; logvar: [number, number]; sigma: [number, number] } {
  const H = state.hiddenDim;

  // Layer 1: 2 -> H (Tanh)
  const h = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    const sum = state.be1[j] + x1 * state.We1[j * 2] + x2 * state.We1[j * 2 + 1];
    h[j] = Math.tanh(sum);
  }

  // Mu head
  const mu1 = state.bmu[0] + dot(h, state.Wmu, 0, H);
  const mu2 = state.bmu[1] + dot(h, state.Wmu, H, H);

  // LogVar head (clamped to [-8, 4] for numerical stability)
  const lv1 = Math.max(-8, Math.min(4, state.blv[0] + dot(h, state.Wlv, 0, H)));
  const lv2 = Math.max(-8, Math.min(4, state.blv[1] + dot(h, state.Wlv, H, H)));

  const sigma1 = Math.exp(0.5 * lv1);
  const sigma2 = Math.exp(0.5 * lv2);

  return {
    mu: [mu1, mu2],
    logvar: [lv1, lv2],
    sigma: [sigma1, sigma2],
  };
}

export function decodeVAE(z1: number, z2: number, state: VAEState): [number, number] {
  const H = state.hiddenDim;

  // Decoder Layer 1: 2 -> H (Tanh)
  const h = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    const sum = state.bd1[j] + z1 * state.Wd1[j * 2] + z2 * state.Wd1[j * 2 + 1];
    h[j] = Math.tanh(sum);
  }

  // Decoder Layer 2: H -> 2 (Linear)
  const xHat1 = state.bd2[0] + dot(h, state.Wd2, 0, H);
  const xHat2 = state.bd2[1] + dot(h, state.Wd2, H, H);

  return [xHat1, xHat2];
}

function dot(vec: Float32Array, matrix: Float32Array, rowOffset: number, len: number): number {
  let s = 0;
  for (let i = 0; i < len; i++) {
    s += vec[i] * matrix[rowOffset + i];
  }
  return s;
}

export const vaePlugin: AlgorithmPlugin<VAEState, VAEParams> = {
  id: 'vae',
  name: 'Variational Autoencoder (β-VAE)',
  shortCode: 'VAE',
  stage: 'neural-graph',
  taskType: 'dimensionality-reduction',
  executionMode: 'epoch-loop',
  description: 'Probabilistic latent generative model optimizing the Evidence Lower Bound (ELBO) with Gaussian uncertainty ellipses.',
  paperReference: 'Kingma & Welling, Auto-Encoding Variational Bayes, ICLR 2014; Higgins et al., beta-VAE, ICLR 2017',

  primaryParams: [
    {
      id: 'beta',
      name: 'KL Weight (β-VAE)',
      type: 'slider',
      min: 0.0,
      max: 5.0,
      step: 0.1,
      default: 0.5,
      description: 'Trade-off between reconstruction fidelity and latent Gaussian prior regularization.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'select',
      options: [
        { label: '0.005 (Fine)', value: 0.005 },
        { label: '0.01', value: 0.01 },
        { label: '0.03 (Standard)', value: 0.03 },
        { label: '0.08 (Fast)', value: 0.08 },
      ],
      default: 0.03,
      description: 'Step size for joint encoder-decoder ELBO optimization.',
    },
    {
      id: 'samplingNoise',
      name: 'Sampling Noise (ε)',
      type: 'select',
      options: [
        { label: 'On (Reparameterized z = μ + σ·ε)', value: 'on' },
        { label: 'Off (Deterministic Mode z = μ)', value: 'off' },
      ],
      default: 'on',
      description: 'Stochastic noise injection during encoder latent sampling.',
    },
  ],

  advancedParams: [
    {
      id: 'hiddenDim',
      name: 'Hidden Layer Width',
      type: 'slider',
      min: 4,
      max: 16,
      step: 2,
      default: 8,
      description: 'Number of hidden neurons in encoder and decoder networks.',
    },
  ],

  init(dataset: DatasetContainer, params: VAEParams, seed: number): VAEState {
    const rand = mulberry32(seed);
    const H = Math.max(4, Math.min(16, Number(params.hiddenDim ?? 8)));

    const We1 = initializeWeights(2, H, 'xavier', rand);
    const be1 = new Float32Array(H);
    const Wmu = initializeWeights(H, 2, 'xavier', rand);
    const bmu = new Float32Array(2);
    const Wlv = initializeWeights(H, 2, 'xavier', rand);
    const blv = new Float32Array(2);

    const Wd1 = initializeWeights(2, H, 'xavier', rand);
    const bd1 = new Float32Array(H);
    const Wd2 = initializeWeights(H, 2, 'xavier', rand);
    const bd2 = new Float32Array(2);

    const probeZ1 = Number(params.probeZ1 ?? 0);
    const probeZ2 = Number(params.probeZ2 ?? 0);

    const state: VAEState = {
      hiddenDim: H,
      latentDim: 2,
      We1, be1, Wmu, bmu, Wlv, blv,
      Wd1, bd1, Wd2, bd2,
      mWe1: new Float32Array(We1.length), vWe1: new Float32Array(We1.length),
      mbe1: new Float32Array(be1.length), vbe1: new Float32Array(be1.length),
      mWmu: new Float32Array(Wmu.length), vWmu: new Float32Array(Wmu.length),
      mbmu: new Float32Array(bmu.length), vbmu: new Float32Array(bmu.length),
      mWlv: new Float32Array(Wlv.length), vWlv: new Float32Array(Wlv.length),
      mblv: new Float32Array(blv.length), vblv: new Float32Array(blv.length),
      mWd1: new Float32Array(Wd1.length), vWd1: new Float32Array(Wd1.length),
      mbd1: new Float32Array(bd1.length), vbd1: new Float32Array(bd1.length),
      mWd2: new Float32Array(Wd2.length), vWd2: new Float32Array(Wd2.length),
      mbd2: new Float32Array(bd2.length), vbd2: new Float32Array(bd2.length),
      stepCount: 0,
      epoch: 0,
      reconLoss: 1.0,
      klLoss: 0.5,
      totalLoss: 1.5,
      probePoint: {
        z: [probeZ1, probeZ2],
        decoded: [0, 0],
      },
      rngSeed: seed || 42,
    };

    state.probePoint.decoded = decodeVAE(probeZ1, probeZ2, state);

    // Compute real Epoch 0 Recon MSE, KL Divergence, and beta-ELBO from initial weights
    const beta = Number(params.beta ?? 0.5);
    const useNoise = params.samplingNoise !== 'off' && params.samplingNoise !== false;
    if (dataset.train.length > 0) {
      const evalRand = mulberry32((seed || 42) >>> 0);
      let totalRecon = 0;
      let totalKL = 0;
      for (let i = 0; i < dataset.train.length; i++) {
        const pt = dataset.train[i];
        const enc = encodeVAE(pt.x1, pt.x2, state);
        const [mu1, mu2] = enc.mu;
        const [lv1, lv2] = enc.logvar;
        const [sig1, sig2] = enc.sigma;

        const eps1 = useNoise ? gaussianRandom(evalRand, 0, 1) : 0;
        const eps2 = useNoise ? gaussianRandom(evalRand, 0, 1) : 0;
        const z1 = mu1 + sig1 * eps1;
        const z2 = mu2 + sig2 * eps2;

        const [xHat1, xHat2] = decodeVAE(z1, z2, state);
        const e1 = xHat1 - pt.x1;
        const e2 = xHat2 - pt.x2;
        totalRecon += 0.5 * (e1 * e1 + e2 * e2);
        totalKL +=
          -0.5 * (1 + lv1 - mu1 * mu1 - sig1 * sig1) +
          -0.5 * (1 + lv2 - mu2 * mu2 - sig2 * sig2);
      }
      state.reconLoss = totalRecon / dataset.train.length;
      state.klLoss = totalKL / dataset.train.length;
      state.totalLoss = state.reconLoss + beta * state.klLoss;
    } else {
      state.reconLoss = 0;
      state.klLoss = 0;
      state.totalLoss = 0;
    }

    return state;
  },

  step(state: VAEState, dataset: DatasetContainer, params: VAEParams): StepResult<VAEState> {
    const data = dataset.train;
    const N = data.length;
    if (N === 0) {
      return {
        state,
        epoch: state.epoch,
        metrics: { trainLoss: 0, testLoss: 0 },
      };
    }

    const lr = Number(params.learningRate ?? 0.03);
    const beta = Number(params.beta ?? 0.5);
    const useNoise = params.samplingNoise !== 'off' && params.samplingNoise !== false;
    const H = state.hiddenDim;

    const initialSeed = (state.rngSeed ?? (state.epoch * 1000 + 42)) >>> 0;
    const rand = mulberry32(initialSeed);
    state.rngSeed = ((initialSeed + 0x6d2b79f5) | 0) >>> 0;

    // Gradient accumulators
    const gWe1 = new Float32Array(state.We1.length);
    const gbe1 = new Float32Array(state.be1.length);
    const gWmu = new Float32Array(state.Wmu.length);
    const gbmu = new Float32Array(state.bmu.length);
    const gWlv = new Float32Array(state.Wlv.length);
    const gblv = new Float32Array(state.blv.length);

    const gWd1 = new Float32Array(state.Wd1.length);
    const gbd1 = new Float32Array(state.bd1.length);
    const gWd2 = new Float32Array(state.Wd2.length);
    const gbd2 = new Float32Array(state.bd2.length);

    let totalRecon = 0;
    let totalKL = 0;

    for (let i = 0; i < N; i++) {
      const pt = data[i];
      const x1 = pt.x1;
      const x2 = pt.x2;

      // 1. Encoder Forward
      const z_e1 = new Float32Array(H);
      const a_e1 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const sum = state.be1[j] + x1 * state.We1[j * 2] + x2 * state.We1[j * 2 + 1];
        z_e1[j] = sum;
        a_e1[j] = Math.tanh(sum);
      }

      // Heads
      const mu1 = state.bmu[0] + dot(a_e1, state.Wmu, 0, H);
      const mu2 = state.bmu[1] + dot(a_e1, state.Wmu, H, H);
      const lv1 = Math.max(-8, Math.min(4, state.blv[0] + dot(a_e1, state.Wlv, 0, H)));
      const lv2 = Math.max(-8, Math.min(4, state.blv[1] + dot(a_e1, state.Wlv, H, H)));

      const sig1 = Math.exp(0.5 * lv1);
      const sig2 = Math.exp(0.5 * lv2);

      // Reparameterization trick
      const eps1 = useNoise ? gaussianRandom(rand, 0, 1) : 0;
      const eps2 = useNoise ? gaussianRandom(rand, 0, 1) : 0;
      const z1 = mu1 + sig1 * eps1;
      const z2 = mu2 + sig2 * eps2;

      // Analytical KL Divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
      const kl1 = -0.5 * (1 + lv1 - mu1 * mu1 - sig1 * sig1);
      const kl2 = -0.5 * (1 + lv2 - mu2 * mu2 - sig2 * sig2);
      const kl_i = kl1 + kl2;
      totalKL += kl_i;

      // 2. Decoder Forward
      const z_d1 = new Float32Array(H);
      const a_d1 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const sum = state.bd1[j] + z1 * state.Wd1[j * 2] + z2 * state.Wd1[j * 2 + 1];
        z_d1[j] = sum;
        a_d1[j] = Math.tanh(sum);
      }

      const xHat1 = state.bd2[0] + dot(a_d1, state.Wd2, 0, H);
      const xHat2 = state.bd2[1] + dot(a_d1, state.Wd2, H, H);

      const e1 = xHat1 - x1;
      const e2 = xHat2 - x2;
      const recon_i = 0.5 * (e1 * e1 + e2 * e2);
      totalRecon += recon_i;

      // 3. Backward Pass
      // Output delta
      const delta_out = [e1, e2];
      gbd2[0] += delta_out[0];
      gbd2[1] += delta_out[1];
      for (let j = 0; j < H; j++) {
        gWd2[j] += delta_out[0] * a_d1[j];
        gWd2[H + j] += delta_out[1] * a_d1[j];
      }

      // Decoder Layer 1
      const delta_d1 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const sum = state.Wd2[j] * delta_out[0] + state.Wd2[H + j] * delta_out[1];
        delta_d1[j] = sum * (1 - a_d1[j] * a_d1[j]); // tanh'
        gbd1[j] += delta_d1[j];
        gWd1[j * 2] += delta_d1[j] * z1;
        gWd1[j * 2 + 1] += delta_d1[j] * z2;
      }

      // Delta w.r.t z
      let delta_z1 = 0;
      let delta_z2 = 0;
      for (let j = 0; j < H; j++) {
        delta_z1 += state.Wd1[j * 2] * delta_d1[j];
        delta_z2 += state.Wd1[j * 2 + 1] * delta_d1[j];
      }

      // Gradients for mu and logvar including beta * KL
      // dKL/dmu = mu, dKL/dlogvar = 0.5 * (exp(logvar) - 1)
      const grad_mu1 = delta_z1 + beta * mu1;
      const grad_mu2 = delta_z2 + beta * mu2;
      const grad_lv1 = 0.5 * delta_z1 * sig1 * eps1 + 0.5 * beta * (sig1 * sig1 - 1);
      const grad_lv2 = 0.5 * delta_z2 * sig2 * eps2 + 0.5 * beta * (sig2 * sig2 - 1);

      gbmu[0] += grad_mu1;
      gbmu[1] += grad_mu2;
      gblv[0] += grad_lv1;
      gblv[1] += grad_lv2;

      for (let j = 0; j < H; j++) {
        gWmu[j] += grad_mu1 * a_e1[j];
        gWmu[H + j] += grad_mu2 * a_e1[j];
        gWlv[j] += grad_lv1 * a_e1[j];
        gWlv[H + j] += grad_lv2 * a_e1[j];
      }

      // Encoder Layer 1 backprop
      for (let j = 0; j < H; j++) {
        const sum =
          state.Wmu[j] * grad_mu1 +
          state.Wmu[H + j] * grad_mu2 +
          state.Wlv[j] * grad_lv1 +
          state.Wlv[H + j] * grad_lv2;
        const delta_e1 = sum * (1 - a_e1[j] * a_e1[j]); // tanh'
        gbe1[j] += delta_e1;
        gWe1[j * 2] += delta_e1 * x1;
        gWe1[j * 2 + 1] += delta_e1 * x2;
      }
    }

    // 4. Adam Update
    const scale = 1.0 / N;
    state.stepCount++;
    const t = state.stepCount;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;

    const applyAdam = (w: Float32Array, gw: Float32Array, m: Float32Array, v: Float32Array) => {
      for (let k = 0; k < w.length; k++) {
        const grad = gw[k] * scale;
        m[k] = b1 * m[k] + (1 - b1) * grad;
        v[k] = b2 * v[k] + (1 - b2) * grad * grad;
        const mHat = m[k] / (1 - Math.pow(b1, t));
        const vHat = v[k] / (1 - Math.pow(b2, t));
        w[k] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    };

    applyAdam(state.We1, gWe1, state.mWe1, state.vWe1);
    applyAdam(state.be1, gbe1, state.mbe1, state.vbe1);
    applyAdam(state.Wmu, gWmu, state.mWmu, state.vWmu);
    applyAdam(state.bmu, gbmu, state.mbmu, state.vbmu);
    applyAdam(state.Wlv, gWlv, state.mWlv, state.vWlv);
    applyAdam(state.blv, gblv, state.mblv, state.vblv);
    applyAdam(state.Wd1, gWd1, state.mWd1, state.vWd1);
    applyAdam(state.bd1, gbd1, state.mbd1, state.vbd1);
    applyAdam(state.Wd2, gWd2, state.mWd2, state.vWd2);
    applyAdam(state.bd2, gbd2, state.mbd2, state.vbd2);

    state.epoch++;
    state.reconLoss = totalRecon / N;
    state.klLoss = totalKL / N;
    state.totalLoss = state.reconLoss + beta * state.klLoss;

    // Update probe point
    const pz1 = Number(params.probeZ1 ?? state.probePoint.z[0]);
    const pz2 = Number(params.probeZ2 ?? state.probePoint.z[1]);
    state.probePoint = {
      z: [pz1, pz2],
      decoded: decodeVAE(pz1, pz2, state),
    };

    const metrics = this.loss(state, dataset, params);

    return {
      state,
      epoch: state.epoch,
      metrics,
      converged: state.reconLoss < 0.02 && state.klLoss < 0.1,
    };
  },

  predict(input: Float32Array, state: VAEState, _params: VAEParams): number {
    const enc = encodeVAE(input[0], input[1], state);
    return enc.mu[0];
  },

  loss(state: VAEState, _dataset: DatasetContainer, params: VAEParams): TelemetryMetrics {
    const beta = Number(params.beta ?? 0.5);
    const recon = state.reconLoss;
    const kl = state.klLoss;
    const elbo = recon + beta * kl;

    return {
      trainLoss: elbo,
      testLoss: elbo,
      trainR2: Math.max(0, 1 - recon),
      custom: {
        reconstructionLoss: recon.toFixed(4),
        klDivergence: kl.toFixed(4),
        betaELBO: elbo.toFixed(4),
        probeCoords: `(${state.probePoint.z[0].toFixed(2)}, ${state.probePoint.z[1].toFixed(2)})`,
        decodedCoords: `(${state.probePoint.decoded[0].toFixed(2)}, ${state.probePoint.decoded[1].toFixed(2)})`,
      },
    };
  },

  getVisualArtifacts(state: VAEState, dataset: DatasetContainer, _params: VAEParams): VisualArtifact[] {
    const trainData = dataset.train;

    const encodedPoints: Extract<VisualArtifact, { type: 'latent-bottleneck-overlay' }>['encodedPoints'] = [];
    const reconstructionVectors: Extract<VisualArtifact, { type: 'latent-bottleneck-overlay' }>['reconstructionVectors'] = [];

    for (let i = 0; i < trainData.length; i++) {
      const p = trainData[i];
      const enc = encodeVAE(p.x1, p.x2, state);
      const [xHat1, xHat2] = decodeVAE(enc.mu[0], enc.mu[1], state);

      encodedPoints.push({
        id: p.id,
        z1: enc.mu[0],
        z2: enc.mu[1],
        sigma1: enc.sigma[0],
        sigma2: enc.sigma[1],
        label: p.y,
      });

      const err = Math.hypot(xHat1 - p.x1, xHat2 - p.x2);
      reconstructionVectors.push({
        id: p.id,
        xOriginal: [p.x1, p.x2],
        xReconstructed: [xHat1, xHat2],
        error: err,
        label: p.y,
      });
    }

    return [
      {
        type: 'latent-bottleneck-overlay',
        latentDim: 2,
        isVAE: true,
        encodedPoints,
        reconstructionVectors,
        probePoint: state.probePoint,
        klDivergence: state.klLoss,
        reconstructionMSE: state.reconLoss,
      },
    ];
  },

  serializeSnapshot(state: VAEState): Float32Array {
    const totalLen =
      4 +
      state.We1.length + state.be1.length +
      state.Wmu.length + state.bmu.length +
      state.Wlv.length + state.blv.length +
      state.Wd1.length + state.bd1.length +
      state.Wd2.length + state.bd2.length;

    const buf = new Float32Array(totalLen);
    buf[0] = state.epoch;
    buf[1] = state.reconLoss;
    buf[2] = state.klLoss;
    buf[3] = state.totalLoss;

    let idx = 4;
    const write = (arr: Float32Array) => {
      buf.set(arr, idx);
      idx += arr.length;
    };
    write(state.We1); write(state.be1);
    write(state.Wmu); write(state.bmu);
    write(state.Wlv); write(state.blv);
    write(state.Wd1); write(state.bd1);
    write(state.Wd2); write(state.bd2);
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: VAEState): VAEState {
    state.epoch = Math.floor(buffer[0]);
    state.reconLoss = buffer[1];
    state.klLoss = buffer[2];
    state.totalLoss = buffer[3];

    let idx = 4;
    const read = (arr: Float32Array) => {
      for (let i = 0; i < arr.length; i++) arr[i] = buffer[idx++];
    };
    read(state.We1); read(state.be1);
    read(state.Wmu); read(state.bmu);
    read(state.Wlv); read(state.blv);
    read(state.Wd1); read(state.bd1);
    read(state.Wd2); read(state.bd2);
    return state;
  },

  explain(state: VAEState, params: VAEParams): ExplanationPayload {
    const beta = Number(params.beta ?? 0.5);
    const elbo = state.reconLoss + beta * state.klLoss;

    return {
      summary: `Variational Autoencoder (β-VAE) mapping 2D observations to probabilistic Gaussian posterior distributions q(z|x) = N(μ(x), σ²(x)) regularized against a standard normal prior N(0, I).`,
      coreAssumption: `Observed data is generated by an unobserved continuous latent variable z ~ N(0, I) through a smooth non-linear decoder p(x|z).`,
      whenItFails: `Posterior collapse: When β is too large, the KL penalty forces q(z|x) to match N(0, I) identically for all inputs (μ → 0, σ → 1), causing the decoder to ignore z entirely and predict the blurry global dataset mean.`,
      microLog: [
        `Encoder Pass: Encoded inputs to mean μ and log-variance log(σ²) using reparameterization trick z = μ + σ ⊙ ε.`,
        `ELBO Evaluation: Computed Recon MSE = ${state.reconLoss.toFixed(4)} and KL Divergence = ${state.klLoss.toFixed(4)} (β = ${beta}).`,
        `Update: Optimized Evidence Lower Bound (Total Loss: ${elbo.toFixed(4)}) across encoder and decoder parameters.`,
      ],
      mathLatex: {
        objective: `\\mathcal{L}_{\\beta\\text{-ELBO}} = \\mathbb{E}_{q_\\phi(\\mathbf{z}|\\mathbf{x})}[\\|\\mathbf{x} - \\hat{\\mathbf{x}}\\|^2] + \\beta \\cdot D_{\\text{KL}}(\\mathcal{N}(\\boldsymbol{\\mu}, \\boldsymbol{\\sigma}^2) \\parallel \\mathcal{N}(\\mathbf{0}, \\mathbf{I}))`,
        lossOrUpdate: `D_{\\text{KL}} = -\\frac{1}{2} \\sum_{k=1}^2 (1 + \\log \\sigma_k^2 - \\mu_k^2 - \\sigma_k^2), \\quad \\mathbf{z} = \\boldsymbol{\\mu} + \\boldsymbol{\\sigma} \\odot \\boldsymbol{\\epsilon}`,
        dualOrConstraint: `\\beta = ${beta.toFixed(2)}, \\quad D_{\\text{KL}} = ${state.klLoss.toFixed(4)}, \\quad \\text{Recon MSE} = ${state.reconLoss.toFixed(4)}`,
      },
      codeSnippet: {
        sklearn: `# VAE is natively a probabilistic deep neural architecture without direct sklearn equivalent.
# Implemented via standard PyTorch module below.`,
        pytorch: `import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, hidden_dim=${state.hiddenDim}):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(2, hidden_dim), nn.Tanh())
        self.fc_mu = nn.Linear(hidden_dim, 2)
        self.fc_logvar = nn.Linear(hidden_dim, 2)
        self.dec = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar`,
      },
    };
  },
};
