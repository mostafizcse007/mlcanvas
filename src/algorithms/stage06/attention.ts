import { AlgorithmPlugin, DatasetContainer, ExplanationPayload, StepResult, TelemetryMetrics, VisualArtifact } from '../../types/algorithm';
import { generateSequenceDataset } from '../../datasets/sequenceTokens';
import { computeScaledDotProductAttention } from './sequenceMath';

export interface AttentionState {
  epoch: number;
  phaseIndex: number; // 0..4 for the 5-step walkthrough
  selectedStep: number;
  tokens: string[];
  rawScores: Float32Array;
  attentionWeights: Float32Array;
  qVectors: Float32Array[];
  kVectors: Float32Array[];
  vVectors: Float32Array[];
  outputVectors: Float32Array[];
  entropy: number;
  maxScore: number;
  weights: Float32Array;
}

const PHASES = [
  'Step 1: Project Q, K, V Representations',
  'Step 2: Pairwise Dot-Product Similarity (Q · K^T)',
  'Step 3: Dimensional Scaling (1 / √d_k)',
  'Step 4: Softmax Distribution Normalization (α)',
  'Step 5: Value Routing Weighted Sum (Σ α_ij V_j)',
];

export const attentionPlugin: AlgorithmPlugin<AttentionState, Record<string, any>> = {
  id: 'attention',
  name: 'Scaled Dot-Product Attention',
  shortCode: 'SDP Attention',
  stage: 'sequence-attn',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Softmax(Q K^T / √d_k) V token routing mechanism proving direct O(1) path length and 1/√d_k gradient stabilization.',
  paperReference: 'Vaswani, A., et al. (2017). Attention is all you need. NeurIPS, 30.',

  primaryParams: [
    {
      id: 'dk',
      name: 'Key Dimension (d_k)',
      type: 'select',
      options: [
        { label: 'd_k = 4 (Low Dim)', value: 4 },
        { label: 'd_k = 8 (Standard)', value: 8 },
        { label: 'd_k = 16 (Medium)', value: 16 },
        { label: 'd_k = 64 (High Dim)', value: 64 },
        { label: 'd_k = 128 (Large)', value: 128 },
      ],
      default: 8,
      description: 'Dimensionality of Query and Key projection vectors.',
    },
    {
      id: 'scaleFactor',
      name: 'Scale Factor',
      type: 'select',
      options: [
        { label: '1 / √d_k (Stabilized)', value: 'scaled-sqrt-dk' },
        { label: '1.0 (Unscaled Dot Product)', value: 'unscaled-1.0' },
      ],
      default: 'scaled-sqrt-dk',
      description: 'Stabilizes large magnitude dot products so softmax gradients do not saturate and vanish.',
    },
    {
      id: 'temperature',
      name: 'Softmax Temperature (τ)',
      type: 'slider',
      min: 0.1,
      max: 2.0,
      step: 0.1,
      default: 1.0,
      description: 'Controls concentration of attention distribution (τ < 0.5 = sharp peak, τ > 1.5 = uniform blur).',
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
      description: 'Number of interactive sequence tokens.',
    },
  ],

  symbolicLinks: [
    {
      id: 'scaled-dot-product',
      symbolLatex: '\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}}\\right) V',
      label: 'Scaled Dot-Product Equation',
      description: 'Scaling dot products by 1/√d_k maintains unit variance for large dimensions, preventing softmax saturation.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: Record<string, any>, _seed: number): AttentionState {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const dk = Number(params.dk ?? 8);
    const scaleMode = params.scaleFactor === 'unscaled-1.0' ? 'unscaled' : 'scaled';
    const tau = Number(params.temperature ?? 1.0);

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);
    const tokens = seqData.sequence.slice(0, T).map((s) => s.text);

    const res = computeScaledDotProductAttention(inputs, inputs, inputs, dk, scaleMode, tau, false);

    return {
      epoch: 0,
      phaseIndex: 3, // Default to normalized Softmax view
      selectedStep: seqData.selectedTokenIndex < T ? seqData.selectedTokenIndex : 0,
      tokens,
      rawScores: res.rawScores,
      attentionWeights: res.attentionWeights,
      qVectors: res.qVectors,
      kVectors: res.kVectors,
      vVectors: res.vVectors,
      outputVectors: res.outputs,
      entropy: res.entropy,
      maxScore: res.maxScore,
      weights: new Float32Array([dk, tau]),
    };
  },

  step(state: AttentionState, dataset: DatasetContainer, params: Record<string, any>): StepResult<AttentionState> {
    const nextPhase = (state.phaseIndex + 1) % PHASES.length;
    const nextEpoch = state.epoch + 1;
    const nextStep = (state.selectedStep + 1) % state.tokens.length;

    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const dk = Number(params.dk ?? 8);
    const scaleMode = params.scaleFactor === 'unscaled-1.0' ? 'unscaled' : 'scaled';
    const tau = Number(params.temperature ?? 1.0);

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);
    const res = computeScaledDotProductAttention(inputs, inputs, inputs, dk, scaleMode, tau, false);

    const nextState: AttentionState = {
      ...state,
      epoch: nextEpoch,
      phaseIndex: nextPhase,
      selectedStep: nextStep,
      rawScores: res.rawScores,
      attentionWeights: res.attentionWeights,
      qVectors: res.qVectors,
      kVectors: res.kVectors,
      vVectors: res.vVectors,
      outputVectors: res.outputs,
      entropy: res.entropy,
      maxScore: res.maxScore,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
    };
  },

  predict(input: Float32Array, state: AttentionState, _params: Record<string, any>): Float32Array {
    const out = state.outputVectors[state.selectedStep];
    return out ? new Float32Array(out) : new Float32Array(input.length);
  },

  loss(state: AttentionState, _dataset: DatasetContainer, params: Record<string, any>): TelemetryMetrics {
    const isUnscaled = params.scaleFactor === 'unscaled-1.0';
    const dk = Number(params.dk ?? 8);
    const isSaturated = isUnscaled && dk >= 64;
    const ep = state.epoch;

    const decay = 1.0 / (1.0 + ep * 0.04);
    const saturationPenalty = isSaturated ? 2.1 : 0;
    const baseLoss = Math.max(0.06, (0.38 + saturationPenalty) * decay + saturationPenalty * 0.3);

    const T = state.tokens.length;
    const selIdx = state.selectedStep;
    let maxAttn = 0;
    for (let j = 0; j < T; j++) {
      const a = state.attentionWeights[selIdx * T + j];
      if (a > maxAttn) maxAttn = a;
    }

    return {
      trainLoss: baseLoss,
      testLoss: baseLoss * 1.05,
      custom: {
        entropy: state.entropy.toFixed(3),
        maxSoftmaxScore: maxAttn.toFixed(4),
        rawMaxScore: state.maxScore.toFixed(2),
        pathLength: 'O(1) Direct Hop',
        selectedToken: state.tokens[state.selectedStep] ?? '—',
      },
    };
  },

  getVisualArtifacts(state: AttentionState, _dataset: DatasetContainer, params: Record<string, any>): VisualArtifact[] {
    const dk = Number(params.dk ?? 8);
    const tau = Number(params.temperature ?? 1.0);
    const scaleFactor = params.scaleFactor === 'unscaled-1.0' ? 1.0 : 1.0 / Math.sqrt(dk);

    return [
      {
        type: 'sequence-gate-attn',
        modelFamily: 'attention',
        algorithmId: 'attention',
        executionMode: 'phase-walkthrough',
        phaseIndex: state.phaseIndex,
        phaseName: PHASES[state.phaseIndex],
        totalPhases: PHASES.length,
        tokens: state.tokens,
        selectedStep: state.selectedStep,
        attention: {
          d_k: dk,
          scale: scaleFactor,
          temperature: tau,
          isCausal: false,
          rawScores: state.rawScores,
          attentionWeights: state.attentionWeights,
          entropy: state.entropy,
          maxScore: state.maxScore,
          qVectors: state.qVectors,
          kVectors: state.kVectors,
          vVectors: state.vVectors,
          outputVectors: state.outputVectors,
          selectedHead: -1,
        },
      },
    ];
  },

  serializeSnapshot(state: AttentionState): Float32Array {
    return new Float32Array(state.weights);
  },

  restoreSnapshot(buffer: Float32Array, state: AttentionState): AttentionState {
    return {
      ...state,
      weights: new Float32Array(buffer),
    };
  },

  explain(state: AttentionState, params: Record<string, any>, _lastStep?: StepResult<AttentionState>): ExplanationPayload {
    const dk = Number(params.dk ?? 8);
    const isUnscaled = params.scaleFactor === 'unscaled-1.0';
    const isSaturated = isUnscaled && dk >= 64;

    return {
      summary: 'Scaled Dot-Product Attention connects any token to any other token in a single O(1) hop, completely bypassing the sequential recurrent bottleneck of RNNs.',
      coreAssumption: 'Semantic alignment can be measured as the scaled inner product between Query and Key representations, routing Values accordingly.',
      whenItFails: isSaturated
        ? `Dot Product Saturation Alert: At d_k = ${dk} without scaling, raw dot products reach ${state.maxScore.toFixed(1)}, driving Softmax into one-hot saturation (Entropy = ${state.entropy.toFixed(3)}). Gradients α(1 - α) vanish to 0.000!`
        : 'Computational memory scales quadratically O(T^2) with sequence length.',
      microLog: [
        `Walkthrough phase: ${PHASES[state.phaseIndex]}.`,
        `Direct O(1) routing: Token ${state.tokens.length} attends to Token 1 without ${Math.max(0, state.tokens.length - 1)} intermediate recurrent cell hops.`,
        `Mean Softmax entropy: ${state.entropy.toFixed(3)} nats with scaling factor = ${isUnscaled ? '1.0 (Unscaled)' : `1/√${dk} ≈ ${(1 / Math.sqrt(dk)).toFixed(3)}`}.`,
      ],
      mathLatex: {
        objective: '\\alpha_{ij} = \\frac{\\exp\\left(\\frac{q_i \\cdot k_j^T}{\\sqrt{d_k}}\\right)}{\\sum_{m=1}^T \\exp\\left(\\frac{q_i \\cdot k_m^T}{\\sqrt{d_k}}\\right)}, \\quad O_i = \\sum_{j=1}^T \\alpha_{ij} V_j',
        lossOrUpdate: '\\frac{\\partial \\alpha_{ij}}{\\partial S_{ik}} = \\alpha_{ij}(\\delta_{jk} - \\alpha_{ik})',
      },
      codeSnippet: {
        sklearn: `# PyTorch Scaled Dot-Product Attention
import torch
import torch.nn.functional as F

scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
attn_weights = F.softmax(scores, dim=-1)
output = torch.matmul(attn_weights, V)`,
        pytorch: `def scaled_dot_product_attention(Q, K, V, scale=True):
    d_k = Q.size(-1)
    scale_factor = 1.0 / (d_k ** 0.5) if scale else 1.0
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale_factor
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights`,
      },
    };
  },
};

