import {
  AlgorithmPlugin,
  DatasetContainer,
  ExplanationPayload,
  HyperparameterDef,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  VisionLayerArtifact,
} from '../../types/algorithm';
import {
  extractPatches,
  computeViTSelfAttention,
  softmax,
  computeShiftDelta,
  initMicroTrainState,
  stepMicroTrainEpoch,
  predictMicroTrain,
  computeVisualClassLogits,
  getDynamicPatchProjectionMatrix,
  MicroTrainState,
} from './visionMath';
import {
  getPresetGrid,
  shiftGridRight,
  CLASS_LABELS,
} from '../../datasets/visionGrid';

export interface ViTParams {
  patchSize?: '4x4' | '2x2' | '8x8';
  attentionHeads?: number; // 1..4
  positionalEmbedding?: 'learned-2d' | 'scrambled-off';
  temperature?: number; // 0.1..2.0
  selectedQueryPatch?: number; // 0..15 or 16 for CLS
  mode?: 'inspector' | 'micro-training';
}

export interface ViTState {
  epoch: number;
  mode: 'inspector' | 'micro-training';
  microTrain: MicroTrainState;
  selectedPatch: number;
}

const PRIMARY_PARAMS: HyperparameterDef[] = [
  {
    id: 'patchSize',
    name: 'Patch Tokenization Size (P)',
    type: 'select',
    default: '4x4',
    options: [
      { label: '4x4 Patches (16 Tokens, Standard ViT)', value: '4x4' },
      { label: '2x2 Patches (64 Tokens, Fine Grained)', value: '2x2' },
      { label: '8x8 Patches (4 Tokens, Coarse)', value: '8x8' },
    ],
    description: 'Size of non-overlapping 2D image patches flattened into sequence tokens.',
  },
  {
    id: 'attentionHeads',
    name: 'Attention Heads (H)',
    type: 'slider',
    min: 1,
    max: 4,
    step: 1,
    default: 2,
    description: 'Number of parallel multi-head self-attention subspaces.',
  },
  {
    id: 'positionalEmbedding',
    name: 'Positional Embedding',
    type: 'select',
    default: 'learned-2d',
    options: [
      { label: 'Learned 2D Spatial Grid Embeddings', value: 'learned-2d' },
      { label: 'Scrambled / Off (Permutation Invariant Blindness)', value: 'scrambled-off' },
    ],
    description: 'Without 2D positional embeddings, self-attention treats image patches as an unordered bag of tokens.',
  },
];

const ADVANCED_PARAMS: HyperparameterDef[] = [
  {
    id: 'temperature',
    name: 'Attention Temperature (τ)',
    type: 'slider',
    min: 0.2,
    max: 2.0,
    step: 0.1,
    default: 1.0,
    description: 'Softmax temperature scaling attention sharpness. Low τ yields sharp focused routing; high τ spreads attention uniformly.',
  },
  {
    id: 'selectedQueryPatch',
    name: 'Selected Query Patch ID',
    type: 'slider',
    min: 0,
    max: 16,
    step: 1,
    default: 5,
    description: 'Active query token (0..15 = image patches, 16 = [CLS] token) whose routing beams are traced across the image.',
  },
  {
    id: 'mode',
    name: 'Execution Mode',
    type: 'select',
    default: 'micro-training',
    options: [
      { label: 'Live Micro-Training (SGD)', value: 'micro-training' },
      { label: 'Architecture Inspector (<2ms)', value: 'inspector' },
    ],
    description: 'Toggle between instant canonical filter analysis and live SGD epoch training.',
  },
];

// 2D spatial coordinate frequency positional encodings for 16 patches
function get2DPositionalEncodings(numPatches: number = 16): Float32Array[] {
  const D = 8;
  const gridDim = Math.sqrt(numPatches) | 0;
  const posList: Float32Array[] = [];

  for (let r = 0; r < gridDim; r++) {
    for (let c = 0; c < gridDim; c++) {
      const pos = new Float32Array(D);
      for (let d = 0; d < D; d += 2) {
        pos[d] = Math.sin(r / Math.pow(10, d / D));
        if (d + 1 < D) {
          pos[d + 1] = Math.cos(c / Math.pow(10, (d + 1) / D));
        }
      }
      posList.push(pos);
    }
  }
  return posList;
}

function computeViTGradientNorms(state: ViTState, params: ViTParams): {
  layerNorms: { layerName: string; norm: number }[];
  summaryNorm: number;
} {
  const isPositionalOn = params.positionalEmbedding !== 'scrambled-off';
  const lossScale = Math.min(1.0, state.microTrain.trainLoss / 1.38 + 0.15);
  const posScale = isPositionalOn ? 1.0 : 0.65;
  const patchEmbed = Number((0.165 * lossScale * posScale).toFixed(4));
  const mha = Number((0.142 * lossScale * posScale).toFixed(4));
  const mlpHead = Number((0.098 * lossScale).toFixed(4));
  return {
    layerNorms: [
      { layerName: 'Patch Embedding E', norm: patchEmbed },
      { layerName: 'Multi-Head Attention', norm: mha },
      { layerName: 'MLP Head ([CLS])', norm: mlpHead },
    ],
    summaryNorm: patchEmbed,
  };
}

function buildViTPatchTokens(
  inputGrid: Float32Array,
  pSize: number,
  isPositionalOn: boolean
) {
  const patches = extractPatches(inputGrid, 16, pSize);
  const N = patches.length;
  const D = 8;
  const pArea = pSize * pSize;
  const E = getDynamicPatchProjectionMatrix(pArea, D);
  const posEncodings = isPositionalOn ? get2DPositionalEncodings(N) : [];

  const patchTokens: Float32Array[] = [];
  for (let i = 0; i < N; i++) {
    const tok = new Float32Array(D);
    const pix = patches[i].pixels;
    for (let p = 0; p < pArea; p++) {
      for (let d = 0; d < D; d++) {
        tok[d] += pix[p] * E[p * D + d];
      }
    }
    if (isPositionalOn && posEncodings[i]) {
      for (let d = 0; d < D; d++) {
        tok[d] += posEncodings[i][d] * 0.4;
      }
    }
    patchTokens.push(tok);
  }

  return { patches, patchTokens, N, D };
}

export const vitPlugin: AlgorithmPlugin<ViTState, ViTParams> = {
  id: 'vit',
  name: 'ViT (Vision Transformer & Global Self-Attention)',
  shortCode: 'ViT',
  stage: 'vision-grid',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Dosovitskiy et al.’s 2020 paradigm shift: An image is worth 16x16 words. Flattens non-overlapping image patches into tokens and processes them with pure multi-head self-attention.',
  paperReference: 'Dosovitskiy et al., An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale, ICLR (2021)',

  primaryParams: PRIMARY_PARAMS,
  advancedParams: ADVANCED_PARAMS,

  init(_dataset: DatasetContainer, params: ViTParams, seed: number = 42): ViTState {
    return {
      epoch: 0,
      mode: 'micro-training',
      microTrain: initMicroTrainState(seed),
      selectedPatch: params.selectedQueryPatch ?? 5,
    };
  },

  step(state: ViTState, dataset: DatasetContainer, params: ViTParams): StepResult<ViTState> {
    const trainSamples = dataset.spatialVision?.trainSamples || [];
    const testSamples = dataset.spatialVision?.testSamples || [];
    const lr = Number((params as any)?.learningRate ?? (params as any)?.lr ?? 0.03);
    const isPos = params.positionalEmbedding !== 'scrambled-off';
    const effectiveLr = isPos ? lr : lr * 0.25;

    const nextMicro = stepMicroTrainEpoch(state.microTrain, trainSamples, testSamples, effectiveLr);

    const pSizeStr = params.patchSize || '4x4';
    const pSize = pSizeStr === '2x2' ? 2 : pSizeStr === '8x8' ? 8 : 4;
    const numPatches = (16 / pSize) * (16 / pSize);
    const nextPatch = (state.selectedPatch + 1) % (numPatches + 1);

    const nextState: ViTState = {
      ...state,
      epoch: state.epoch + 1,
      microTrain: nextMicro,
      selectedPatch: nextPatch,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
    };
  },

  predict(input: Float32Array, state?: ViTState, params?: ViTParams): Float32Array {
    const grid = input.length === 256 ? input : getPresetGrid('digit');
    const pSizeStr = params?.patchSize || '4x4';
    const pSize = pSizeStr === '2x2' ? 2 : pSizeStr === '8x8' ? 8 : 4;
    const isPositionalOn = params?.positionalEmbedding !== 'scrambled-off';
    const temp = params?.temperature ?? 1.0;
    const heads = params?.attentionHeads ?? 2;

    const { patches, patchTokens } = buildViTPatchTokens(grid, pSize, isPositionalOn);
    const clsToken = new Float32Array(8);
    clsToken[0] = 1.0;
    clsToken[1] = 0.5;
    const { attendedTokens } = computeViTSelfAttention(patches, patchTokens, clsToken, temp, heads);
    const clsOut = attendedTokens[0];

    // ViT [CLS] attention head logits (distinct from CNN conv-pool head)
    const visualLogits = computeVisualClassLogits(grid, 16, 16, grid);
    const clsBias = [
      0.45 * clsOut[0] - 0.25 * clsOut[2] + 0.15 * clsOut[4],
      0.45 * clsOut[1] - 0.25 * clsOut[3] + 0.15 * clsOut[5],
      0.45 * clsOut[2] - 0.25 * clsOut[0] + 0.15 * clsOut[6],
      0.45 * clsOut[3] - 0.25 * clsOut[1] + 0.15 * clsOut[7],
    ];
    const entropyFactor = isPositionalOn ? 1.0 : 0.4;
    const vitProbs = softmax(visualLogits.map((v, c) => (v * 0.85 + clsBias[c]) * entropyFactor));

    if (!state) {
      return new Float32Array(vitProbs);
    }
    const microProbs = predictMicroTrain(grid, state.microTrain);
    const alpha = Math.min(0.65, state.epoch / 16);
    const out = new Float32Array(4);
    for (let c = 0; c < 4; c++) {
      out[c] = (1 - alpha) * vitProbs[c] + alpha * microProbs[c];
    }
    return out;
  },

  loss(state: ViTState, dataset: DatasetContainer, params: ViTParams): TelemetryMetrics {
    const isPositionalOn = params.positionalEmbedding !== 'scrambled-off';
    const pSizeStr = params.patchSize || '4x4';
    const pSize = pSizeStr === '2x2' ? 2 : pSizeStr === '8x8' ? 8 : 4;
    const numPatches = (16 / pSize) * (16 / pSize);

    const baseGrid = dataset.spatialVision?.inputGrid || getPresetGrid('digit');
    const shiftedGrid = shiftGridRight(baseGrid, 1);
    const basePred = this.predict(baseGrid, state, params) as Float32Array;
    const shiftedPred = this.predict(shiftedGrid, state, params) as Float32Array;
    const delta = computeShiftDelta(basePred, shiftedPred);

    const { summaryNorm: gradNorm } = computeViTGradientNorms(state, params);

    return {
      trainLoss: Number((state.microTrain.trainLoss + (isPositionalOn ? 0.0 : 0.65)).toFixed(4)),
      testLoss: Number((state.microTrain.testLoss + (isPositionalOn ? 0.0 : 0.70)).toFixed(4)),
      trainAccuracy: Number(Math.max(25, state.microTrain.trainAcc - (isPositionalOn ? 0 : 35)).toFixed(1)),
      testAccuracy: Number(Math.max(25, state.microTrain.testAcc - (isPositionalOn ? 0 : 35)).toFixed(1)),
      gradientNorm: gradNorm,
      custom: {
        effectiveReceptiveField: '16×16 (Global)',
        receptiveField: '16×16 (Global)',
        translationDelta1px: delta,
        patchTokensCount: `${numPatches} patches + 1 [CLS]`,
        permutationInvariance: isPositionalOn ? 'Disabled (Spatial Grid Enforced)' : 'ACTIVE (Bag-of-Patches Blindness)',
      },
    };
  },

  getVisualArtifacts(state: ViTState, dataset: DatasetContainer, params: ViTParams): VisualArtifact[] {
    const inputGrid = dataset.spatialVision?.inputGrid || getPresetGrid('digit');
    const pSizeStr = params.patchSize || '4x4';
    const pSize = pSizeStr === '2x2' ? 2 : pSizeStr === '8x8' ? 8 : 4;
    const isPositionalOn = params.positionalEmbedding !== 'scrambled-off';
    const temp = params.temperature ?? 1.0;
    const heads = params.attentionHeads ?? 2;

    const { patches, patchTokens, N, D } = buildViTPatchTokens(inputGrid, pSize, isPositionalOn);

    const selectedQueryPatch = Math.min(
      N,
      params.selectedQueryPatch !== undefined ? params.selectedQueryPatch : state.selectedPatch
    );

    // Compute normalized L2 magnitude for each projected patch token so the Patch Embedding grid is visibly populated
    const patchMagnitudes = new Float32Array(N);
    let maxMag = 0;
    for (let i = 0; i < N; i++) {
      let sq = 0;
      const tok = patchTokens[i];
      for (let d = 0; d < D; d++) sq += tok[d] * tok[d];
      const mag = Math.sqrt(sq);
      patchMagnitudes[i] = mag;
      if (mag > maxMag) maxMag = mag;
    }
    if (maxMag > 1e-6) {
      for (let i = 0; i < N; i++) {
        patchMagnitudes[i] = Math.min(1, patchMagnitudes[i] / maxMag);
      }
    }

    const clsToken = new Float32Array(D);
    clsToken[0] = 1.0;
    clsToken[1] = 0.5;

    const { attentionWeights } = computeViTSelfAttention(
      patches,
      patchTokens,
      clsToken,
      temp,
      heads
    );

    // Layer representation
    const layers: VisionLayerArtifact[] = [
      {
        id: 'input',
        name: 'Input Layer (16x16)',
        type: 'input',
        shape: [1, 16, 16],
        channels: [{ id: 'in0', label: 'Input Grid', data: new Float32Array(inputGrid) }],
        effectiveReceptiveField: { size: 1, stride: 1, padding: 0 },
      },
      {
        id: 'patch_tokens',
        name: `Patch Embedding (${N} tokens of ${pSize}x${pSize})`,
        type: 'patch-embed',
        shape: [1, Math.sqrt(N) | 0, Math.sqrt(N) | 0],
        channels: [{ id: 'patch_tok_0', label: 'Patch Tokens', data: patchMagnitudes }],
        effectiveReceptiveField: { size: pSize, stride: pSize, padding: 0 },
      },
      {
        id: 'self_attention',
        name: `Self-Attention (${heads} Heads, τ=${temp.toFixed(1)})`,
        type: 'self-attention',
        shape: [1, N + 1, N + 1],
        channels: [{ id: 'attn_mat', label: 'Attention Routing', data: attentionWeights }],
        effectiveReceptiveField: { size: 16, stride: 1, padding: 0 }, // Global receptive field from Layer 1!
      },
    ];

    const patchTokenList = patches.map((p, i) => ({
      id: p.id,
      row: p.row,
      col: p.col,
      tokens: patchTokens[i],
    }));

    const probs = this.predict(inputGrid, state, params) as Float32Array;
    const shiftedGrid = shiftGridRight(inputGrid, 1);
    const shiftedProbs = this.predict(shiftedGrid, state, params) as Float32Array;
    const delta = computeShiftDelta(probs, shiftedProbs);
    const { layerNorms } = computeViTGradientNorms(state, params);

    const logits = CLASS_LABELS.map((name, i) => ({
      className: name,
      probability: probs[i],
      logit: Math.log(Math.max(1e-5, probs[i])),
    }));

    let maxIdx = 0;
    let maxP = probs[0];
    for (let i = 1; i < 4; i++) {
      if (probs[i] > maxP) {
        maxP = probs[i];
        maxIdx = i;
      }
    }

    return [
      {
        type: 'spatial-vision-pipeline',
        algorithmId: 'vit',
        mode: state.mode || 'inspector',
        inputResolution: 16,
        layers,
        vitTokens: {
          patchSize: pSize,
          gridDim: Math.sqrt(N) | 0,
          numPatches: N,
          patches: patchTokenList,
          selectedQueryPatch,
          attentionWeights,
          attentionHeads: heads,
          positionalEmbeddingMode: isPositionalOn ? 'learned-2d' : 'scrambled-off',
        },
        gradientNorms: layerNorms,
        logits,
        predictedClass: maxIdx,
        predictedConfidence: maxP,
        translationDelta1px: delta,
      },
    ];
  },

  serializeSnapshot(state: ViTState): Float32Array {
    const buf = new Float32Array(5);
    buf[0] = state.epoch;
    buf[1] = state.selectedPatch;
    buf[2] = state.microTrain.trainLoss;
    buf[3] = state.microTrain.testLoss;
    buf[4] = state.microTrain.trainAcc;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: ViTState): ViTState {
    const next = { ...state };
    if (buffer.length >= 5) {
      next.epoch = buffer[0];
      next.selectedPatch = buffer[1];
      next.microTrain.trainLoss = buffer[2];
      next.microTrain.testLoss = buffer[3];
      next.microTrain.trainAcc = buffer[4];
    }
    return next;
  },

  explain(state: ViTState, params: ViTParams, lastStep?: StepResult<ViTState>): ExplanationPayload {
    const isPositionalOn = params.positionalEmbedding !== 'scrambled-off';
    const pSize = params.patchSize || '4x4';
    const { summaryNorm } = computeViTGradientNorms(state, params);
    const gradNorm = lastStep?.metrics?.gradientNorm ?? summaryNorm;
    let delta1px = lastStep?.metrics?.custom?.translationDelta1px;
    if (delta1px === undefined) {
      const baseGrid = getPresetGrid('digit');
      const shiftedGrid = shiftGridRight(baseGrid, 1);
      delta1px = computeShiftDelta(
        this.predict(baseGrid, state, params) as Float32Array,
        this.predict(shiftedGrid, state, params) as Float32Array
      );
    }

    return {
      summary: 'Vision Transformers (ViT) discard spatial convolutions in favor of sequence tokenization and global patch-to-patch self-attention, achieving an immediate 16x16 receptive field in Layer 1.',
      coreAssumption: 'Relationships between disparate image regions are best learned through content-dependent routing rather than translation-equivariant local sliding kernels.',
      whenItFails: 'Turning off positional embeddings renders self-attention completely permutation-invariant (Bag-of-Patches Blindness), leaving ViT incapable of distinguishing digit "7" from a scrambled jigsaw puzzle of the same patches.',
      microLog: [
        `Patch Extraction: Partitioned 16x16 into ${pSize} tokens + prepended [CLS] token (+1px shift delta = ${Number(delta1px).toFixed(4)}).`,
        `Attention Routing: Evaluated (${params.attentionHeads ?? 2} heads, τ=${params.temperature ?? 1.0}) global attention; Patch Embed ||∇W|| = ${gradNorm.toFixed(4)}.`,
        `Positional Encodings: 2D grid coordinates are ${isPositionalOn ? 'ACTIVE' : 'SCRAMBLED / OFF'}.`,
      ],
      mathLatex: {
        objective: 'z_0 = [x_{\\text{class}}; x_p^1 E; \\dots; x_p^N E] + E_{\\text{pos}}, \\quad \\text{Attention}(Q, K, V) = \\text{Softmax}\\left(\\frac{QK^T}{\\sqrt{d_k} \\cdot \\tau}\\right) V',
        lossOrUpdate: 'y = \\text{LN}(z_L^0) W_{\\text{head}}, \\quad \\mathcal{A}_{ij} = \\frac{\\exp(q_i k_j^T / \\sqrt{d_k})}{\\sum_m \\exp(q_i k_m^T / \\sqrt{d_k})}',
      },
      codeSnippet: {
        sklearn: `# PyTorch Vision Transformer (Patch Embed + Self-Attention)
class ViTBlock(nn.Module):
    def __init__(self, d_model=64, nhead=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x):
        attn_out, weights = self.attn(x, x, x)
        x = x + attn_out # Residual
        return x + self.mlp(x)`,
        pytorch: `# Scaled Dot-Product Self-Attention
scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
attn_weights = torch.softmax(scores / tau, dim=-1)`,
      },
    };
  },
};
