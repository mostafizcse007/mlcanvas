import { AlgorithmPlugin, DatasetContainer, ExplanationPayload, StepResult, TelemetryMetrics, VisualArtifact } from '../../types/algorithm';
import { generateSequenceDataset } from '../../datasets/sequenceTokens';
import { computeScaledDotProductAttention, projectQKV } from './sequenceMath';

export interface TransformerEncDecState {
  epoch: number;
  phaseIndex: number; // 0: Enc-Self, 1: Dec-Causal-Self, 2: Cross-Attention
  selectedStep: number;
  sourceTokens: string[];
  targetTokens: string[];
  encSelfWeights: Float32Array; // T_src x T_src
  decSelfWeights: Float32Array; // T_tgt x T_tgt
  crossWeights: Float32Array; // T_tgt x T_src
  qVectors?: Float32Array[];
  kVectors?: Float32Array[];
  vVectors?: Float32Array[];
  outputVectors?: Float32Array[];
  entropy: number;
  encEntropy: number;
  decEntropy: number;
  crossEntropy: number;
  weights: Float32Array;
}

const ENC_DEC_PHASES = [
  'Phase 1: Bidirectional Encoder Self-Attention on Source Sequence',
  'Phase 2: Masked Causal Decoder Self-Attention on Target Prefix',
  'Phase 3: Encoder-Decoder Cross-Attention (Queries from Decoder, Keys/Values from Encoder)',
];

function subLayerFromPhase(phaseIndex: number): 'enc-self' | 'dec-causal-self' | 'cross-attention' {
  if (phaseIndex === 0) return 'enc-self';
  if (phaseIndex === 1) return 'dec-causal-self';
  return 'cross-attention';
}

function phaseFromSubLayer(subLayer: string | undefined): number {
  if (subLayer === 'enc-self') return 0;
  if (subLayer === 'dec-causal-self') return 1;
  return 2;
}

function getActiveSubLayerInfo(state: TransformerEncDecState) {
  const subLayer = subLayerFromPhase(state.phaseIndex);
  const weights =
    subLayer === 'enc-self'
      ? state.encSelfWeights
      : subLayer === 'dec-causal-self'
      ? state.decSelfWeights
      : state.crossWeights;
  const numRows = subLayer === 'enc-self' ? state.sourceTokens.length : state.targetTokens.length;
  const numCols = subLayer === 'dec-causal-self' ? state.targetTokens.length : state.sourceTokens.length;
  const entropy =
    subLayer === 'enc-self'
      ? state.encEntropy
      : subLayer === 'dec-causal-self'
      ? state.decEntropy
      : state.crossEntropy;
  const rowIdx = Math.max(0, Math.min(numRows - 1, state.selectedStep));
  let maxAttn = 0;
  for (let j = 0; j < numCols; j++) {
    const a = weights[rowIdx * numCols + j] ?? 0;
    if (a > maxAttn) maxAttn = a;
  }
  return { subLayer, weights, numRows, numCols, entropy, maxAttn };
}

export const transformerEncDecPlugin: AlgorithmPlugin<TransformerEncDecState, Record<string, any>> = {
  id: 'transformer-enc-dec',
  name: 'Sequence-to-Sequence Encoder-Decoder',
  shortCode: 'Enc-Dec (T5)',
  stage: 'sequence-attn',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Full dual-tower Seq2Seq model with bidirectional source encoder, causal target decoder, and cross-attention alignment.',
  paperReference: 'Vaswani, A., et al. (2017). Attention is all you need. NeurIPS / Raffel et al. (2020) T5.',

  primaryParams: [
    {
      id: 'activeSubLayer',
      name: 'Active Sub-Layer',
      type: 'select',
      options: [
        { label: 'Phase 3: Cross-Attention (Dec Q -> Enc K,V)', value: 'cross-attention' },
        { label: 'Phase 1: Source Encoder Self-Attention', value: 'enc-self' },
        { label: 'Phase 2: Target Decoder Causal Self-Attention', value: 'dec-causal-self' },
      ],
      default: 'cross-attention',
      description: 'Selects which attention sub-layer in the dual-tower stack to inspect.',
    },
    {
      id: 'numHeads',
      name: 'Heads (H)',
      type: 'select',
      options: [
        { label: 'H = 1 (Single Head)', value: 1 },
        { label: 'H = 2 (Two Heads)', value: 2 },
        { label: 'H = 4 (Four Heads)', value: 4 },
      ],
      default: 2,
      description: 'Number of parallel attention heads for cross-attention alignment.',
    },
    {
      id: 'temperature',
      name: 'Temperature (τ)',
      type: 'slider',
      min: 0.1,
      max: 2.0,
      step: 0.1,
      default: 1.0,
      description: 'Softmax temperature controlling alignment sharpness.',
    },
  ],

  advancedParams: [
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      default: 0.02,
      description: 'Optimization rate.',
    },
  ],

  symbolicLinks: [
    {
      id: 'cross-attn-formula',
      symbolLatex: '\\text{CrossAttn}(Q_{\\text{dec}}, K_{\\text{enc}}, V_{\\text{enc}}) = \\text{softmax}\\left(\\frac{Q_{\\text{dec}} K_{\\text{enc}}^T}{\\sqrt{d_k}}\\right) V_{\\text{enc}}',
      label: 'Cross-Attention Equation',
      description: 'Queries emanate from decoder states; Keys and Values originate from bidirectional encoder outputs.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: Record<string, any>, _seed: number): TransformerEncDecState {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ preset: 'seq2seq' });
    const tau = Number(params.temperature ?? 1.0);
    const subLayer = (params.activeSubLayer as string) ?? 'cross-attention';

    const sourceTokens = seqData.sequence.map((s) => s.text);
    const targetTokens = seqData.targetSequence
      ? seqData.targetSequence.map((s) => s.text)
      : ['<bos>', 'I', 'eat', 'rice'];

    const srcInputs = seqData.sequence.map((s) => s.embedding);
    const tgtInputs = seqData.targetSequence
      ? seqData.targetSequence.map((s) => s.embedding)
      : srcInputs;

    // 1. Encoder Self-Attention (Bidirectional)
    const encRes = computeScaledDotProductAttention(srcInputs, srcInputs, srcInputs, 8, 'scaled', tau, false);
    // 2. Decoder Self-Attention (Causal Masked)
    const decRes = computeScaledDotProductAttention(tgtInputs, tgtInputs, tgtInputs, 8, 'scaled', tau, true);
    // 3. Cross-Attention: Queries from Target Decoder, Keys/Values from Source Encoder
    const crossRes = computeScaledDotProductAttention(tgtInputs, srcInputs, srcInputs, 8, 'scaled', tau, false);

    const initialPhase = phaseFromSubLayer(subLayer);

    const projSrc = projectQKV(srcInputs, 8);
    const projTgt = projectQKV(tgtInputs, 8);

    const activeEntropy =
      initialPhase === 0 ? encRes.entropy : initialPhase === 1 ? decRes.entropy : crossRes.entropy;
    const defaultStep =
      seqData.selectedTokenIndex !== undefined
        ? Math.max(0, Math.min((initialPhase === 0 ? sourceTokens.length : targetTokens.length) - 1, seqData.selectedTokenIndex))
        : Math.min(2, targetTokens.length - 1);

    return {
      epoch: 0,
      phaseIndex: initialPhase,
      selectedStep: defaultStep,
      sourceTokens,
      targetTokens,
      encSelfWeights: encRes.attentionWeights,
      decSelfWeights: decRes.attentionWeights,
      crossWeights: crossRes.attentionWeights,
      qVectors: projTgt.qVectors,
      kVectors: projSrc.kVectors,
      vVectors: projSrc.vVectors,
      outputVectors: crossRes.outputs,
      entropy: activeEntropy,
      encEntropy: encRes.entropy,
      decEntropy: decRes.entropy,
      crossEntropy: crossRes.entropy,
      weights: new Float32Array([initialPhase, tau]),
    };
  },

  step(state: TransformerEncDecState, dataset: DatasetContainer, params: Record<string, any>): StepResult<TransformerEncDecState> {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ preset: 'seq2seq' });
    const tau = Number(params.temperature ?? 1.0);

    const srcInputs = seqData.sequence.map((s) => s.embedding);
    const tgtInputs = seqData.targetSequence
      ? seqData.targetSequence.map((s) => s.embedding)
      : srcInputs;

    const encRes = computeScaledDotProductAttention(srcInputs, srcInputs, srcInputs, 8, 'scaled', tau, false);
    const decRes = computeScaledDotProductAttention(tgtInputs, tgtInputs, tgtInputs, 8, 'scaled', tau, true);
    const crossRes = computeScaledDotProductAttention(tgtInputs, srcInputs, srcInputs, 8, 'scaled', tau, false);

    const nextPhase = (state.phaseIndex + 1) % ENC_DEC_PHASES.length;
    const nextEpoch = state.epoch + 1;
    const activeTokenCount = nextPhase === 0 ? state.sourceTokens.length : state.targetTokens.length;
    const nextStep = (state.selectedStep + 1) % Math.max(1, activeTokenCount);

    const projSrc = projectQKV(srcInputs, 8);
    const projTgt = projectQKV(tgtInputs, 8);
    const activeEntropy =
      nextPhase === 0 ? encRes.entropy : nextPhase === 1 ? decRes.entropy : crossRes.entropy;

    const nextState: TransformerEncDecState = {
      ...state,
      epoch: nextEpoch,
      phaseIndex: nextPhase,
      selectedStep: nextStep,
      encSelfWeights: encRes.attentionWeights,
      decSelfWeights: decRes.attentionWeights,
      crossWeights: crossRes.attentionWeights,
      qVectors: projTgt.qVectors,
      kVectors: projSrc.kVectors,
      vVectors: projSrc.vVectors,
      outputVectors: crossRes.outputs,
      entropy: activeEntropy,
      encEntropy: encRes.entropy,
      decEntropy: decRes.entropy,
      crossEntropy: crossRes.entropy,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
    };
  },

  predict(_input: Float32Array, state: TransformerEncDecState, _params: Record<string, any>): Float32Array {
    return new Float32Array(state.crossWeights);
  },

  loss(state: TransformerEncDecState, _dataset: DatasetContainer, params: Record<string, any>): TelemetryMetrics {
    const ep = state.epoch;
    const tau = Number(params?.temperature ?? 1.0);
    const decay = 1.0 / (1.0 + ep * 0.04);
    const blurPenalty = tau > 1.5 ? (tau - 1.5) * 0.45 : 0;
    const baseLoss = Math.max(0.06, (0.35 + blurPenalty) * decay + blurPenalty * 0.25);

    const { entropy, maxAttn } = getActiveSubLayerInfo(state);
    const activeTokens = state.phaseIndex === 0 ? state.sourceTokens : state.targetTokens;

    return {
      trainLoss: baseLoss,
      testLoss: baseLoss * 1.06,
      custom: {
        sourceTokens: state.sourceTokens.join(' '),
        targetTokens: state.targetTokens.join(' '),
        entropy: entropy.toFixed(3),
        crossEntropy: state.crossEntropy.toFixed(3),
        maxSoftmaxScore: maxAttn.toFixed(4),
        activePhase: state.phaseIndex === 0 ? 'Encoder Self' : state.phaseIndex === 1 ? 'Decoder Causal' : 'Cross-Attention Alignment',
        selectedTarget: activeTokens[Math.min(state.selectedStep, activeTokens.length - 1)] ?? '—',
      },
    };
  },

  getVisualArtifacts(state: TransformerEncDecState, _dataset: DatasetContainer, params: Record<string, any>): VisualArtifact[] {
    const subLayer = subLayerFromPhase(state.phaseIndex);
    const tau = Number(params.temperature ?? 1.0);
    const { weights: activeWeights, entropy, maxAttn } = getActiveSubLayerInfo(state);
    const isCross = subLayer === 'cross-attention';
    const activeTokens = isCross ? state.sourceTokens : (subLayer === 'enc-self' ? state.sourceTokens : state.targetTokens);
    const clampedStep = Math.max(0, Math.min((isCross ? state.targetTokens.length : activeTokens.length) - 1, state.selectedStep));

    return [
      {
        type: 'sequence-gate-attn',
        modelFamily: 'attention',
        algorithmId: 'transformer-enc-dec',
        executionMode: 'phase-walkthrough',
        phaseIndex: state.phaseIndex,
        phaseName: ENC_DEC_PHASES[state.phaseIndex],
        totalPhases: 3,
        tokens: activeTokens,
        targetTokens: state.targetTokens,
        selectedStep: clampedStep,
        attention: {
          d_k: 8,
          scale: 1.0 / Math.sqrt(8),
          temperature: tau,
          isCausal: subLayer === 'dec-causal-self',
          rawScores: activeWeights,
          attentionWeights: activeWeights,
          entropy,
          maxScore: maxAttn,
          qVectors: state.qVectors ?? [],
          kVectors: state.kVectors ?? [],
          vVectors: state.vVectors ?? [],
          outputVectors: state.outputVectors ?? [],
          selectedHead: -1,
          crossAttention: {
            sourceTokens: state.sourceTokens,
            targetTokens: state.targetTokens,
            weights: state.crossWeights,
          },
          activeSubLayer: subLayer,
        },
      },
    ];
  },

  serializeSnapshot(state: TransformerEncDecState): Float32Array {
    return new Float32Array(state.weights);
  },

  restoreSnapshot(buffer: Float32Array, state: TransformerEncDecState): TransformerEncDecState {
    return {
      ...state,
      weights: new Float32Array(buffer),
    };
  },

  explain(state: TransformerEncDecState, _params: Record<string, any>, _lastStep?: StepResult<TransformerEncDecState>): ExplanationPayload {
    const T_src = state.sourceTokens.length;
    const T_tgt = state.targetTokens.length;
    const alignments: string[] = [];
    for (let i = 0; i < Math.min(3, T_tgt); i++) {
      const tgtTok = state.targetTokens[i];
      if (tgtTok === '<bos>' && T_tgt > 1) continue;
      let bestJ = 0;
      let bestW = -1;
      for (let j = 0; j < T_src; j++) {
        const w = state.crossWeights[i * T_src + j] ?? 0;
        if (w > bestW) {
          bestW = w;
          bestJ = j;
        }
      }
      alignments.push(`"${tgtTok}" → "${state.sourceTokens[bestJ]}" (${bestW.toFixed(2)})`);
    }

    return {
      summary: `Transformer Encoder-Decoder connects a bidirectional Source Encoder Tower (${T_src} tokens) to a causal Target Decoder Tower (${T_tgt} tokens) via Cross-Attention.`,
      coreAssumption: 'Cross-attention allows generated target words to dynamically look up relevant information across all source representations without a fixed-size sentence bottleneck.',
      whenItFails: 'Mismatched sequence lengths or noisy cross-attention alignments can cause repeated generations or hallucinations if temperature is too high.',
      microLog: [
        `Active phase: ${ENC_DEC_PHASES[state.phaseIndex]}.`,
        `Source (${T_src} tokens): "${state.sourceTokens.join(' ')}" → Target (${T_tgt} tokens): "${state.targetTokens.join(' ')}".`,
        alignments.length > 0
          ? `Top cross-attention alignments: ${alignments.join(', ')}.`
          : `Cross-attention routing over ${T_tgt}×${T_src} alignment matrix.`,
      ],
      mathLatex: {
        objective: 'A_{\\text{cross}} = \\text{softmax}\\left(\\frac{Q_{\\text{target}} K_{\\text{source}}^T}{\\sqrt{d_k}}\\right) V_{\\text{source}}',
        lossOrUpdate: '\\mathcal{L}_{\\text{seq2seq}} = -\\sum_{t=1}^{T_{\\text{target}}} \\log P(y_t \\mid y_{<t}, x_{1:T_{\\text{source}}})',
      },
      codeSnippet: {
        sklearn: `# PyTorch / HuggingFace Seq2Seq Cross-Attention
from torch.nn import MultiheadAttention

cross_attn = MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
# query from decoder, key & value from encoder
out, weights = cross_attn(query=decoder_h, key=encoder_h, value=encoder_h)`,
        pytorch: `class CrossAttention(nn.Module):
    def __init__(self, d_model=8):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
    def forward(self, dec_h, enc_h):
        q = self.q_proj(dec_h)
        k = self.k_proj(enc_h)
        v = self.v_proj(enc_h)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (8 ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v), weights`,
      },
    };
  },
};

