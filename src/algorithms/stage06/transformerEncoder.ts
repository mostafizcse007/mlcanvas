import { AlgorithmPlugin, DatasetContainer, ExplanationPayload, StepResult, TelemetryMetrics, VisualArtifact } from '../../types/algorithm';
import { generateSequenceDataset } from '../../datasets/sequenceTokens';
import { computeMultiHeadAttention, computeNextTokenLogits, generateSinusoidalPE, projectQKV } from './sequenceMath';
import { PRESET_SEQUENCES } from '../../datasets/sequenceTokens';

export interface TransformerEncoderState {
  epoch: number;
  selectedStep: number;
  selectedHead: number; // 0..H-1 or -1 for Combined
  tokens: string[];
  peMatrix: Float32Array[];
  heads: {
    id: number;
    name: string;
    color: string;
    weights: Float32Array;
    entropy: number;
  }[];
  combinedWeights: Float32Array;
  combinedEntropy: number;
  qVectors?: Float32Array[];
  kVectors?: Float32Array[];
  vVectors?: Float32Array[];
  outputVectors?: Float32Array[];
  weights: Float32Array;
}

export const transformerEncoderPlugin: AlgorithmPlugin<TransformerEncoderState, Record<string, any>> = {
  id: 'transformer-encoder',
  name: 'Transformer Encoder Layer',
  shortCode: 'Encoder (BERT)',
  stage: 'sequence-attn',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Bidirectional Multi-Head Self-Attention + Residual LayerNorm + Sinusoidal Positional Encoding.',
  paperReference: 'Devlin, J., et al. (2018). BERT: Pre-training of deep bidirectional transformers for language understanding. arXiv:1810.04805.',

  primaryParams: [
    {
      id: 'numHeads',
      name: 'Attention Heads (H)',
      type: 'select',
      options: [
        { label: 'H = 1 (Single Head)', value: 1 },
        { label: 'H = 2 (Two Heads)', value: 2 },
        { label: 'H = 3 (Three Heads)', value: 3 },
        { label: 'H = 4 (Four Heads)', value: 4 },
      ],
      default: 2,
      description: 'Number of parallel attention heads capturing distinct subspace projections.',
    },
    {
      id: 'posEncoding',
      name: 'Positional Encoding',
      type: 'select',
      options: [
        { label: 'Sinusoidal (Order Aware)', value: 'sinusoidal' },
        { label: 'Off / Bag-of-Words (Permutation Invariant)', value: 'off-bag-of-words' },
      ],
      default: 'sinusoidal',
      description: 'Harmonic sine/cosine waves injected into token embeddings to encode sequence order.',
    },
    {
      id: 'residualLayerNorm',
      name: 'Residual + LayerNorm',
      type: 'select',
      options: [
        { label: 'On (Identity Shortcut x + Attn(x))', value: 'on' },
        { label: 'Off (Severed Highway)', value: 'off' },
      ],
      default: 'on',
      description: 'Identity shortcut connection and layer normalization preventing gradient degradation.',
    },
  ],

  advancedParams: [
    {
      id: 'sequenceLength',
      name: 'Sequence Length (T)',
      type: 'slider',
      min: 4,
      max: 10,
      step: 1,
      default: 10,
      description: 'Number of input tokens.',
    },
    {
      id: 'temperature',
      name: 'Temperature (τ)',
      type: 'slider',
      min: 0.1,
      max: 2.0,
      step: 0.1,
      default: 1.0,
      description: 'Attention temperature scaling.',
    },
  ],

  symbolicLinks: [
    {
      id: 'multi-head',
      symbolLatex: '\\text{MultiHead}(Q, K, V) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_H) W^O',
      label: 'Multi-Head Projection',
      description: 'Jointly attends to information from different representation subspaces at different positions.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: Record<string, any>, _seed: number): TransformerEncoderState {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const H = Number(params.numHeads ?? 2);
    const usePE = params.posEncoding !== 'off-bag-of-words';
    const tau = Number(params.temperature ?? 1.0);

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const rawTokens = seqData.sequence.slice(0, T);
    const tokens = rawTokens.map((s) => s.text);

    const peMatrix = generateSinusoidalPE(T, 8);
    const embeddedTokens: Float32Array[] = [];

    for (let i = 0; i < T; i++) {
      const emb = new Float32Array(rawTokens[i].embedding);
      if (usePE) {
        for (let d = 0; d < 8; d++) {
          emb[d] += peMatrix[i][d] * 0.4;
        }
      }
      embeddedTokens.push(emb);
    }

    const { heads, combinedWeights, combinedEntropy } = computeMultiHeadAttention(embeddedTokens, H, tau);
    const { qVectors, kVectors, vVectors, outputVectors } = projectQKV(embeddedTokens, 8);

    return {
      epoch: 0,
      selectedStep: seqData.selectedTokenIndex < T ? seqData.selectedTokenIndex : 0,
      selectedHead: -1, // Combined by default
      tokens,
      peMatrix,
      heads,
      combinedWeights,
      combinedEntropy,
      qVectors,
      kVectors,
      vVectors,
      outputVectors,
      weights: new Float32Array([H, usePE ? 1 : 0]),
    };
  },

  step(state: TransformerEncoderState, dataset: DatasetContainer, params: Record<string, any>): StepResult<TransformerEncoderState> {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const H = Number(params.numHeads ?? 2);
    const usePE = params.posEncoding !== 'off-bag-of-words';
    const tau = Number(params.temperature ?? 1.0);

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const rawTokens = seqData.sequence.slice(0, T);
    const tokens = rawTokens.map((s) => s.text);
    const peMatrix = generateSinusoidalPE(T, 8);

    const embeddedTokens: Float32Array[] = [];
    for (let i = 0; i < T; i++) {
      const emb = new Float32Array(rawTokens[i].embedding);
      if (usePE) {
        for (let d = 0; d < 8; d++) {
          emb[d] += peMatrix[i][d] * 0.4;
        }
      }
      embeddedTokens.push(emb);
    }

    const { heads, combinedWeights, combinedEntropy } = computeMultiHeadAttention(embeddedTokens, H, tau);
    const { qVectors, kVectors, vVectors, outputVectors } = projectQKV(embeddedTokens, 8);

    const nextEpoch = state.epoch + 1;
    const nextStep = (state.selectedStep + 1) % tokens.length;

    const nextState: TransformerEncoderState = {
      ...state,
      epoch: nextEpoch,
      selectedStep: nextStep,
      tokens,
      peMatrix,
      heads,
      combinedWeights,
      combinedEntropy,
      qVectors,
      kVectors,
      vVectors,
      outputVectors,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
    };
  },

  predict(_input: Float32Array, state: TransformerEncoderState, _params: Record<string, any>): Float32Array {
    return new Float32Array(state.combinedWeights);
  },

  loss(state: TransformerEncoderState, _dataset: DatasetContainer, params: Record<string, any>): TelemetryMetrics {
    const usePE = params.posEncoding !== 'off-bag-of-words';
    const ep = state.epoch;

    const decay = 1.0 / (1.0 + ep * 0.04);
    const pePenalty = usePE ? 0 : 0.85;
    const residualPenalty = params.residualLayerNorm === 'off' ? 0.65 : 0;
    const baseLoss = Math.max(0.06, (0.34 + pePenalty + residualPenalty) * decay + (pePenalty + residualPenalty) * 0.3);

    const T = state.tokens.length;
    const selIdx = Math.max(0, Math.min(T - 1, state.selectedStep));
    const activeWeights =
      state.selectedHead >= 0 && state.selectedHead < state.heads.length
        ? state.heads[state.selectedHead].weights
        : state.combinedWeights;
    let maxAttn = 0;
    for (let j = 0; j < T; j++) {
      const a = activeWeights[selIdx * T + j] ?? 0;
      if (a > maxAttn) maxAttn = a;
    }

    return {
      trainLoss: baseLoss,
      testLoss: baseLoss * 1.08,
      custom: {
        numHeads: state.heads.length,
        combinedEntropy: state.combinedEntropy.toFixed(3),
        maxSoftmaxScore: maxAttn.toFixed(4),
        positionalEncoding: usePE ? 'Sinusoidal PE (Active)' : 'Disabled (Bag-of-Words Trap)',
        residualHighway: params.residualLayerNorm === 'off' ? 'Off (Degraded)' : 'On (Identity)',
        selectedToken: state.tokens[state.selectedStep] ?? '—',
      },
    };
  },

  getVisualArtifacts(state: TransformerEncoderState, dataset: DatasetContainer, params: Record<string, any>): VisualArtifact[] {
    const usePE = params.posEncoding !== 'off-bag-of-words';
    const tau = Number(params.temperature ?? 1.0);
    const activeHeadIndex = state.selectedHead;
    const activeWeights = activeHeadIndex >= 0 && activeHeadIndex < state.heads.length
      ? state.heads[activeHeadIndex].weights
      : state.combinedWeights;

    // Compute [MASK] token predictions if any token is masked
    const maskIdx = state.tokens.findIndex((t) => t === '[MASK]');
    let maskedPredictions: { token: string; prob: number; logit: number }[] | undefined;
    if (maskIdx >= 0) {
      const preset = dataset.sequenceTokens?.preset ?? 'coreference';
      const vocab = Array.from(new Set(PRESET_SEQUENCES[preset]?.tokens ?? state.tokens)).filter((t) => t !== '[MASK]');
      const contextVec = (state.outputVectors ?? state.peMatrix)[maskIdx] ?? new Float32Array(8);
      maskedPredictions = computeNextTokenLogits(contextVec, state.tokens, Math.max(0, maskIdx - 1), vocab, tau, 3);
    }

    return [
      {
        type: 'sequence-gate-attn',
        modelFamily: 'attention',
        algorithmId: 'transformer-encoder',
        executionMode: 'epoch-loop',
        tokens: state.tokens,
        selectedStep: state.selectedStep,
        attention: {
          d_k: Math.max(1, Math.round(8 / Math.max(1, state.heads.length))),
          scale: 1.0 / Math.sqrt(8),
          temperature: tau,
          isCausal: false,
          rawScores: activeWeights,
          attentionWeights: activeWeights,
          entropy: state.combinedEntropy,
          maxScore: 1.0,
          qVectors: state.qVectors ?? state.peMatrix,
          kVectors: state.kVectors ?? state.peMatrix,
          vVectors: state.vVectors ?? state.peMatrix,
          outputVectors: state.outputVectors ?? state.peMatrix,
          heads: state.heads,
          selectedHead: state.selectedHead,
          nextTokens: maskedPredictions,
          positionalEncoding: {
            enabled: usePE,
            matrix: state.peMatrix,
          },
        },
      },
    ];
  },

  serializeSnapshot(state: TransformerEncoderState): Float32Array {
    return new Float32Array(state.weights);
  },

  restoreSnapshot(buffer: Float32Array, state: TransformerEncoderState): TransformerEncoderState {
    return {
      ...state,
      weights: new Float32Array(buffer),
    };
  },

  explain(state: TransformerEncoderState, params: Record<string, any>, _lastStep?: StepResult<TransformerEncoderState>): ExplanationPayload {
    const usePE = params.posEncoding !== 'off-bag-of-words';
    const H = state.heads.length;

    const dHeadRaw = 8 / Math.max(1, H);
    const dHeadFormatted = Number.isInteger(dHeadRaw) ? String(dHeadRaw) : dHeadRaw.toFixed(2);
    return {
      summary: `Transformer Encoder applies bidirectional multi-head self-attention (${H} heads) across all tokens simultaneously, allowing each token to aggregate syntactic, semantic, and positional context without sequential bottlenecks.`,
      coreAssumption: 'Different attention heads specialize in distinct linguistic relations (Head 1 = syntactic coreference, Head 2 = local adjacent tokens, Head 3 = punctuation/structural markers).',
      whenItFails: !usePE
        ? 'Bag-of-Words Collapse: Without positional encodings, self-attention is strictly permutation invariant. "Dog bites man" produces the exact same token attention representations as "Man bites dog"!'
        : 'Bidirectional context peeks into future tokens, making it unsuitable for pure autoregressive text generation without causal masking.',
      microLog: [
        `Multi-head decomposition: ${H} heads projecting into ${dHeadFormatted}D per head.`,
        usePE ? 'Sinusoidal positional encodings added to token embeddings.' : 'Positional encodings stripped (Bag-of-Words mode).',
        `Combined attention entropy: ${state.combinedEntropy.toFixed(3)} nats.`,
      ],
      mathLatex: {
        objective: 'PE_{(pos, 2i)} = \\sin(pos / 10000^{2i/d}), \\quad PE_{(pos, 2i+1)} = \\cos(pos / 10000^{2i/d})',
        lossOrUpdate: 'x_{\\text{out}} = \\text{LayerNorm}(x + \\text{MultiHead}(x, x, x))',
      },
      codeSnippet: {
        sklearn: `# HuggingFace / PyTorch Transformer Encoder
from torch.nn import TransformerEncoderLayer

encoder_layer = TransformerEncoderLayer(d_model=8, nhead=2, batch_first=True)
out = encoder_layer(embedded_tokens)`,
        pytorch: `class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=8, n_heads=2):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)`,
      },
    };
  },
};

