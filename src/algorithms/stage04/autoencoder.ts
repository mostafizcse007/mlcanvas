import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import {
  ActivationType,
  activate,
  activateDerivative,
  initializeWeights,
} from './neuralUtils';
import { mulberry32 } from '../../utils/prng';

export interface AutoencoderParams {
  latentDim?: 1 | 2;
  activation?: ActivationType;
  learningRate?: number;
  hiddenDim?: number;
  weightDecay?: number;
  probeZ1?: number;
  probeZ2?: number;
}

export interface AutoencoderState {
  latentDim: 1 | 2;
  hiddenDim: number;
  activation: ActivationType;
  // Encoder: W_e1 (2 -> hidden), b_e1 (hidden), W_e2 (hidden -> latentDim), b_e2 (latentDim)
  We1: Float32Array;
  be1: Float32Array;
  We2: Float32Array;
  be2: Float32Array;
  // Decoder: W_d1 (latentDim -> hidden), b_d1 (hidden), W_d2 (hidden -> 2), b_d2 (2)
  Wd1: Float32Array;
  bd1: Float32Array;
  Wd2: Float32Array;
  bd2: Float32Array;
  // Adam moments
  mWe1: Float32Array; vWe1: Float32Array;
  mbe1: Float32Array; vbe1: Float32Array;
  mWe2: Float32Array; vWe2: Float32Array;
  mbe2: Float32Array; vbe2: Float32Array;
  mWd1: Float32Array; vWd1: Float32Array;
  mbd1: Float32Array; vbd1: Float32Array;
  mWd2: Float32Array; vWd2: Float32Array;
  mbd2: Float32Array; vbd2: Float32Array;

  stepCount: number;
  epoch: number;
  reconstructionMSE: number;
  probePoint: { z: [number, number]; decoded: [number, number] };
  pcaAxis1?: { center: [number, number]; direction: [number, number]; label: string };
}

export function encodePoint(
  x1: number,
  x2: number,
  state: AutoencoderState,
  act: ActivationType
): [number, number] {
  const H = state.hiddenDim;
  const dz = state.latentDim;

  // Layer 1: 2 -> H
  const h = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    const sum = state.be1[j] + x1 * state.We1[j * 2] + x2 * state.We1[j * 2 + 1];
    h[j] = activate(sum, act);
  }

  // Layer 2: H -> dz (linear bottleneck)
  const z1 = state.be2[0] + dot(h, state.We2, 0, H);
  const z2 = dz === 2 ? state.be2[1] + dot(h, state.We2, H, H) : 0;

  return [z1, z2];
}

export function decodePoint(
  z1: number,
  z2: number,
  state: AutoencoderState,
  act: ActivationType
): [number, number] {
  const H = state.hiddenDim;
  const dz = state.latentDim;

  // Decoder Layer 1: dz -> H
  const h = new Float32Array(H);
  for (let j = 0; j < H; j++) {
    let sum = state.bd1[j] + z1 * state.Wd1[j * dz];
    if (dz === 2) {
      sum += z2 * state.Wd1[j * dz + 1];
    }
    h[j] = activate(sum, act);
  }

  // Decoder Layer 2: H -> 2 (linear output)
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

function computePCA1(points: { x1: number; x2: number }[]): {
  center: [number, number];
  direction: [number, number];
  label: string;
} {
  const N = points.length;
  if (N === 0) {
    return { center: [0, 0], direction: [1, 0], label: 'PCA PC1' };
  }
  let mx = 0;
  let my = 0;
  for (let i = 0; i < N; i++) {
    mx += points[i].x1;
    my += points[i].x2;
  }
  mx /= N;
  my /= N;

  let sxx = 0;
  let sxy = 0;
  let syy = 0;
  for (let i = 0; i < N; i++) {
    const dx = points[i].x1 - mx;
    const dy = points[i].x2 - my;
    sxx += dx * dx;
    sxy += dx * dy;
    syy += dy * dy;
  }

  // Analytical 2x2 eigendecomposition
  const tr = sxx + syy;
  const det = sxx * syy - sxy * sxy;
  const gap = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const lambda1 = tr / 2 + gap;

  let vx = sxy;
  let vy = lambda1 - sxx;
  const norm = Math.hypot(vx, vy);
  if (norm > 1e-9) {
    vx /= norm;
    vy /= norm;
  } else {
    vx = 1;
    vy = 0;
  }

  return {
    center: [mx, my],
    direction: [vx, vy],
    label: 'PCA PC1 Axis (Linear AE Subspace)',
  };
}

export const autoencoderPlugin: AlgorithmPlugin<AutoencoderState, AutoencoderParams> = {
  id: 'autoencoder',
  name: 'Deterministic Bottleneck Autoencoder',
  shortCode: 'Autoencoder',
  stage: 'neural-graph',
  taskType: 'dimensionality-reduction',
  executionMode: 'epoch-loop',
  description: 'Unsupervised bottleneck compression learning non-linear low-dimensional principal manifold representations.',
  paperReference: 'Kramer, Nonlinear principal component analysis using autoassociative neural networks, AIChE 1991',

  primaryParams: [
    {
      id: 'latentDim',
      name: 'Bottleneck Latent Dim',
      type: 'select',
      options: [
        { label: '1D Bottleneck (Manifold Line)', value: 1 },
        { label: '2D Bottleneck (Latent Plane)', value: 2 },
      ],
      default: 2,
      description: 'Dimensionality of the compressed bottleneck code space.',
    },
    {
      id: 'activation',
      name: 'Encoder/Decoder Activation',
      type: 'select',
      options: [
        { label: 'Tanh (Curved Manifold)', value: 'tanh' },
        { label: 'Linear (Spans PCA Subspace)', value: 'linear' },
        { label: 'ReLU (Piecewise Affine)', value: 'relu' },
      ],
      default: 'tanh',
      description: 'Non-linearity applied in encoder and decoder hidden layers.',
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
      description: 'Gradient step size minimizing reconstruction MSE.',
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
      description: 'Number of neurons in encoder and decoder hidden layers.',
    },
    {
      id: 'weightDecay',
      name: 'Weight Decay (L2)',
      type: 'select',
      options: [
        { label: '0.0 (None)', value: 0.0 },
        { label: '0.0001', value: 0.0001 },
        { label: '0.001', value: 0.001 },
      ],
      default: 0.0001,
      description: 'L2 regularization preventing unconstrained weight blowup.',
    },
  ],

  init(dataset: DatasetContainer, params: AutoencoderParams, seed: number): AutoencoderState {
    const rand = mulberry32(seed);
    const dz = (Number(params.latentDim) === 1 ? 1 : 2) as 1 | 2;
    const H = Math.max(4, Math.min(16, Number(params.hiddenDim ?? 8)));
    const act = params.activation ?? 'tanh';

    // Allocate flat buffers
    const We1 = initializeWeights(2, H, 'xavier', rand);
    const be1 = new Float32Array(H);
    const We2 = initializeWeights(H, dz, 'xavier', rand);
    const be2 = new Float32Array(dz);

    const Wd1 = initializeWeights(dz, H, 'xavier', rand);
    const bd1 = new Float32Array(H);
    const Wd2 = initializeWeights(H, 2, 'xavier', rand);
    const bd2 = new Float32Array(2);

    const probeZ1 = Number(params.probeZ1 ?? 0);
    const probeZ2 = Number(params.probeZ2 ?? 0);

    const state: AutoencoderState = {
      latentDim: dz,
      hiddenDim: H,
      activation: act,
      We1, be1, We2, be2,
      Wd1, bd1, Wd2, bd2,
      mWe1: new Float32Array(We1.length), vWe1: new Float32Array(We1.length),
      mbe1: new Float32Array(be1.length), vbe1: new Float32Array(be1.length),
      mWe2: new Float32Array(We2.length), vWe2: new Float32Array(We2.length),
      mbe2: new Float32Array(be2.length), vbe2: new Float32Array(be2.length),
      mWd1: new Float32Array(Wd1.length), vWd1: new Float32Array(Wd1.length),
      mbd1: new Float32Array(bd1.length), vbd1: new Float32Array(bd1.length),
      mWd2: new Float32Array(Wd2.length), vWd2: new Float32Array(Wd2.length),
      mbd2: new Float32Array(bd2.length), vbd2: new Float32Array(bd2.length),
      stepCount: 0,
      epoch: 0,
      reconstructionMSE: 1.0,
      probePoint: {
        z: [probeZ1, probeZ2],
        decoded: [0, 0],
      },
      pcaAxis1: computePCA1(dataset.train),
    };

    const dec = decodePoint(probeZ1, probeZ2, state, act);
    state.probePoint.decoded = dec;

    // Compute real Epoch 0 reconstruction MSE from initial weights
    if (dataset.train.length > 0) {
      let totalSqErr = 0;
      for (let i = 0; i < dataset.train.length; i++) {
        const pt = dataset.train[i];
        const [z1, z2] = encodePoint(pt.x1, pt.x2, state, act);
        const [xHat1, xHat2] = decodePoint(z1, z2, state, act);
        const e1 = xHat1 - pt.x1;
        const e2 = xHat2 - pt.x2;
        totalSqErr += e1 * e1 + e2 * e2;
      }
      state.reconstructionMSE = totalSqErr / (dataset.train.length * 2);
    } else {
      state.reconstructionMSE = 0;
    }

    return state;
  },

  step(state: AutoencoderState, dataset: DatasetContainer, params: AutoencoderParams): StepResult<AutoencoderState> {
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
    const act = params.activation ?? 'tanh';
    const H = state.hiddenDim;
    const dz = state.latentDim;
    const decay = Number(params.weightDecay ?? 0.0001);

    // Gradient accumulators
    const gWe1 = new Float32Array(state.We1.length);
    const gbe1 = new Float32Array(state.be1.length);
    const gWe2 = new Float32Array(state.We2.length);
    const gbe2 = new Float32Array(state.be2.length);

    const gWd1 = new Float32Array(state.Wd1.length);
    const gbd1 = new Float32Array(state.bd1.length);
    const gWd2 = new Float32Array(state.Wd2.length);
    const gbd2 = new Float32Array(state.bd2.length);

    let totalSqError = 0;

    for (let i = 0; i < N; i++) {
      const pt = data[i];
      const x1 = pt.x1;
      const x2 = pt.x2;

      // 1. Forward Pass
      // Enc Layer 1
      const z_e1 = new Float32Array(H);
      const a_e1 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const sum = state.be1[j] + x1 * state.We1[j * 2] + x2 * state.We1[j * 2 + 1];
        z_e1[j] = sum;
        a_e1[j] = activate(sum, act);
      }

      // Enc Layer 2 (Bottleneck)
      const z_code = new Float32Array(dz);
      for (let k = 0; k < dz; k++) {
        z_code[k] = state.be2[k] + dot(a_e1, state.We2, k * H, H);
      }

      // Dec Layer 1
      const z_d1 = new Float32Array(H);
      const a_d1 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        let sum = state.bd1[j] + z_code[0] * state.Wd1[j * dz];
        if (dz === 2) {
          sum += z_code[1] * state.Wd1[j * dz + 1];
        }
        z_d1[j] = sum;
        a_d1[j] = activate(sum, act);
      }

      // Dec Layer 2 (Reconstruction)
      const xHat1 = state.bd2[0] + dot(a_d1, state.Wd2, 0, H);
      const xHat2 = state.bd2[1] + dot(a_d1, state.Wd2, H, H);

      const e1 = xHat1 - x1;
      const e2 = xHat2 - x2;
      totalSqError += e1 * e1 + e2 * e2;

      // 2. Backward Pass
      // Loss: L = 0.5 * (e1^2 + e2^2) -> dL/dxHat = [e1, e2]
      const delta_out = [e1, e2];
      gbd2[0] += delta_out[0];
      gbd2[1] += delta_out[1];
      for (let j = 0; j < H; j++) {
        gWd2[j] += delta_out[0] * a_d1[j];
        gWd2[H + j] += delta_out[1] * a_d1[j];
      }

      // Backprop through Dec Layer 1
      const delta_d1 = new Float32Array(H);
      for (let j = 0; j < H; j++) {
        const sum = state.Wd2[j] * delta_out[0] + state.Wd2[H + j] * delta_out[1];
        delta_d1[j] = sum * activateDerivative(z_d1[j], a_d1[j], act);
        gbd1[j] += delta_d1[j];
        for (let k = 0; k < dz; k++) {
          gWd1[j * dz + k] += delta_d1[j] * z_code[k];
        }
      }

      // Backprop through Bottleneck z_code
      const delta_z = new Float32Array(dz);
      for (let k = 0; k < dz; k++) {
        let sum = 0;
        for (let j = 0; j < H; j++) {
          sum += state.Wd1[j * dz + k] * delta_d1[j];
        }
        delta_z[k] = sum;
        gbe2[k] += sum;
        for (let j = 0; j < H; j++) {
          gWe2[k * H + j] += sum * a_e1[j];
        }
      }

      // Backprop through Enc Layer 1
      for (let j = 0; j < H; j++) {
        let sum = 0;
        for (let k = 0; k < dz; k++) {
          sum += state.We2[k * H + j] * delta_z[k];
        }
        const delta_e1 = sum * activateDerivative(z_e1[j], a_e1[j], act);
        gbe1[j] += delta_e1;
        gWe1[j * 2] += delta_e1 * x1;
        gWe1[j * 2 + 1] += delta_e1 * x2;
      }
    }

    // 3. Adam Update with L2 weight decay
    const scale = 1.0 / N;
    state.stepCount++;
    const t = state.stepCount;
    const b1 = 0.9;
    const b2 = 0.999;
    const eps = 1e-8;

    const applyAdam = (w: Float32Array, gw: Float32Array, m: Float32Array, v: Float32Array) => {
      for (let k = 0; k < w.length; k++) {
        const grad = gw[k] * scale + decay * w[k];
        m[k] = b1 * m[k] + (1 - b1) * grad;
        v[k] = b2 * v[k] + (1 - b2) * grad * grad;
        const mHat = m[k] / (1 - Math.pow(b1, t));
        const vHat = v[k] / (1 - Math.pow(b2, t));
        w[k] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
      }
    };

    applyAdam(state.We1, gWe1, state.mWe1, state.vWe1);
    applyAdam(state.be1, gbe1, state.mbe1, state.vbe1);
    applyAdam(state.We2, gWe2, state.mWe2, state.vWe2);
    applyAdam(state.be2, gbe2, state.mbe2, state.vbe2);
    applyAdam(state.Wd1, gWd1, state.mWd1, state.vWd1);
    applyAdam(state.bd1, gbd1, state.mbd1, state.vbd1);
    applyAdam(state.Wd2, gWd2, state.mWd2, state.vWd2);
    applyAdam(state.bd2, gbd2, state.mbd2, state.vbd2);

    state.epoch++;
    state.reconstructionMSE = totalSqError / (N * 2);
    state.pcaAxis1 = computePCA1(dataset.train);

    // Update live probe decoding
    const probeZ1 = Number(params.probeZ1 ?? state.probePoint.z[0]);
    const probeZ2 = dz === 1 ? 0 : Number(params.probeZ2 ?? state.probePoint.z[1]);
    state.probePoint = {
      z: [probeZ1, probeZ2],
      decoded: decodePoint(probeZ1, probeZ2, state, act),
    };

    const metrics = this.loss(state, dataset, params);

    return {
      state,
      epoch: state.epoch,
      metrics,
      converged: state.reconstructionMSE < 0.01,
    };
  },

  predict(input: Float32Array, state: AutoencoderState, params: AutoencoderParams): number {
    const act = params.activation ?? 'tanh';
    const [z1] = encodePoint(input[0], input[1], state, act);
    return z1;
  },

  loss(state: AutoencoderState, _dataset: DatasetContainer, _params: AutoencoderParams): TelemetryMetrics {
    const mse = state.reconstructionMSE;
    return {
      trainLoss: mse,
      testLoss: mse,
      trainR2: Math.max(0, 1 - mse),
      custom: {
        reconstructionMSE: mse,
        latentDim: `${state.latentDim}D Bottleneck`,
        probeZ: `(${state.probePoint.z[0].toFixed(2)}, ${state.probePoint.z[1].toFixed(2)})`,
        decodedHat: `(${state.probePoint.decoded[0].toFixed(2)}, ${state.probePoint.decoded[1].toFixed(2)})`,
      },
    };
  },

  getVisualArtifacts(state: AutoencoderState, dataset: DatasetContainer, params: AutoencoderParams): VisualArtifact[] {
    const act = params.activation ?? 'tanh';
    const trainData = dataset.train;

    const encodedPoints: Extract<VisualArtifact, { type: 'latent-bottleneck-overlay' }>['encodedPoints'] = [];
    const reconstructionVectors: Extract<VisualArtifact, { type: 'latent-bottleneck-overlay' }>['reconstructionVectors'] = [];

    for (let i = 0; i < trainData.length; i++) {
      const p = trainData[i];
      const [z1, z2] = encodePoint(p.x1, p.x2, state, act);
      const [xHat1, xHat2] = decodePoint(z1, z2, state, act);

      encodedPoints.push({
        id: p.id,
        z1,
        z2,
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
        latentDim: state.latentDim,
        isVAE: false,
        encodedPoints,
        reconstructionVectors,
        probePoint: state.probePoint,
        pca1Axis: act === 'linear' && state.latentDim === 1 ? state.pcaAxis1 : undefined,
        reconstructionMSE: state.reconstructionMSE,
      },
    ];
  },

  serializeSnapshot(state: AutoencoderState): Float32Array {
    const totalLen =
      2 +
      state.We1.length + state.be1.length +
      state.We2.length + state.be2.length +
      state.Wd1.length + state.bd1.length +
      state.Wd2.length + state.bd2.length;

    const buf = new Float32Array(totalLen);
    buf[0] = state.epoch;
    buf[1] = state.reconstructionMSE;

    let idx = 2;
    const write = (arr: Float32Array) => {
      buf.set(arr, idx);
      idx += arr.length;
    };
    write(state.We1); write(state.be1); write(state.We2); write(state.be2);
    write(state.Wd1); write(state.bd1); write(state.Wd2); write(state.bd2);
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: AutoencoderState): AutoencoderState {
    state.epoch = Math.floor(buffer[0]);
    state.reconstructionMSE = buffer[1];

    let idx = 2;
    const read = (arr: Float32Array) => {
      for (let i = 0; i < arr.length; i++) arr[i] = buffer[idx++];
    };
    read(state.We1); read(state.be1); read(state.We2); read(state.be2);
    read(state.Wd1); read(state.bd1); read(state.Wd2); read(state.bd2);
    return state;
  },

  explain(state: AutoencoderState, params: AutoencoderParams): ExplanationPayload {
    const act = params.activation ?? 'tanh';
    const dz = state.latentDim;

    return {
      summary: `Deterministic bottleneck autoencoder compressing 2D coordinate space into a ${dz}D latent space Z before reconstructing back to X̂.`,
      coreAssumption: `Data points lie on or near a lower-dimensional smooth manifold parameterized by ${dz} continuous latent coordinates.`,
      whenItFails: `In empty dead zones between training clusters, deterministic autoencoders can decode arbitrary spurious coordinates because the latent space is not regularized into a smooth prior.`,
      microLog: [
        `Encoder Pass: Mapped 2D inputs through [2 → ${state.hiddenDim} → ${dz}] into latent code (${state.probePoint.z[0].toFixed(2)}, ${state.probePoint.z[1].toFixed(2)}).`,
        `Decoder Pass: Reconstructed [${dz} → ${state.hiddenDim} → 2] with MSE = ${state.reconstructionMSE.toFixed(4)}.`,
        `Update: Backpropagated reconstruction error vectors to optimize encoder and decoder parameters.`,
      ],
      mathLatex: {
        objective: `\\mathcal{L}_{\\text{AE}} = \\frac{1}{N} \\sum_{i=1}^N \\|\\mathbf{x}_i - \\mathcal{D}(\\mathcal{E}(\\mathbf{x}_i))\\|^2 + \\lambda \\|\\mathbf{W}\\|_2^2`,
        lossOrUpdate: `\\mathbf{z} = \\mathcal{E}(\\mathbf{x}) \\in \\mathbb{R}^{${dz}}, \\quad \\hat{\\mathbf{x}} = \\mathcal{D}(\\mathbf{z}) \\in \\mathbb{R}^2`,
        dualOrConstraint: act === 'linear' && dz === 1 ? `\\text{Linear AE: } \\operatorname{span}(\\mathcal{W}) \\equiv \\operatorname{span}(\\mathbf{v}_{\\text{PCA}})` : `\\text{MSE: } ${state.reconstructionMSE.toFixed(4)}`,
      },
      codeSnippet: {
        sklearn: `from sklearn.neural_network import MLPRegressor

# Bottleneck Autoencoder (2 -> 8 -> ${dz} -> 8 -> 2)
ae = MLPRegressor(
    hidden_layer_sizes=(${state.hiddenDim}, ${dz}, ${state.hiddenDim}),
    activation='${act}',
    solver='adam',
    max_iter=400,
    random_state=42
)
ae.fit(X_train, X_train)`,
        pytorch: `import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, ${state.hiddenDim}),
            nn.${act === 'relu' ? 'ReLU' : act === 'linear' ? 'Identity' : 'Tanh'}(),
            nn.Linear(${state.hiddenDim}, ${dz})
        )
        self.decoder = nn.Sequential(
            nn.Linear(${dz}, ${state.hiddenDim}),
            nn.${act === 'relu' ? 'ReLU' : act === 'linear' ? 'Identity' : 'Tanh'}(),
            nn.Linear(${state.hiddenDim}, 2)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))`,
      },
    };
  },
};
