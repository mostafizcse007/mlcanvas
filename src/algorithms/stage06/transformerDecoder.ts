import { AlgorithmPlugin, DatasetContainer, ExplanationPayload, StepResult, TelemetryMetrics, VisualArtifact } from '../../types/algorithm';
import { generateSequenceDataset } from '../../datasets/sequenceTokens';
import { computeNextTokenLogits, computeScaledDotProductAttention } from './sequenceMath';

export interface TransformerDecoderState {
  epoch: number;
  phaseIndex: number; // autoregressive step 0..T-1
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
  nextTokens: {
    token: string;
    prob: number;
    logit: number;
  }[];
  weights: Float32Array;
}

const SUBSCRIPT_DIGITS = ['₀', '₁', '₂', '₃', '₄', '₅', '₆', '₇', '₈', '₉'];
function toSubscript(n: number): string {
  return String(n)
    .split('')
    .map((ch) => SUBSCRIPT_DIGITS[Number(ch)] ?? ch)
    .join('');
}

function buildCandidateVocab(tokens: string[], preset?: string): string[] {
  const base = preset === 'seq2seq'
    ? ['<bos>', 'I', 'eat', 'rice', 'Ami', 'bhat', 'khai', '<eos>']
    : [...tokens, '<eos>'];
  const unique: string[] = [];
  for (const tok of base) {
    if (tok !== '[MASK]' && !unique.includes(tok)) unique.push(tok);
  }
  if (!unique.includes('<eos>')) unique.push('<eos>');
  return unique;
}

export const transformerDecoderPlugin: AlgorithmPlugin<TransformerDecoderState, Record<string, any>> = {
  id: 'transformer-decoder',
  name: 'Autoregressive Causal Transformer Decoder',
  shortCode: 'Decoder (GPT)',
  stage: 'sequence-attn',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Autoregressive generation with strictly lower-triangular causal self-attention mask preventing future token look-ahead leaks.',
  paperReference: 'Radford, A., et al. (2018). Improving language understanding by generative pre-training (GPT). OpenAI.',

  primaryParams: [
    {
      id: 'causalMask',
      name: 'Look-Ahead Mask',
      type: 'select',
      options: [
        { label: 'Causal Masked (-∞ upper triangle)', value: 'masked-causal' },
        { label: 'Unmasked (Future Leakage Trap)', value: 'unmasked-leak-trap' },
      ],
      default: 'masked-causal',
      description: 'Enforces strictly autoregressive attention: token t cannot peek at token t+1 during training or inference.',
    },
    {
      id: 'temperature',
      name: 'Sampling Temperature (τ)',
      type: 'slider',
      min: 0.1,
      max: 2.0,
      step: 0.1,
      default: 1.0,
      description: 'Sharpen (τ < 0.5) or flatten (τ > 1.5) next-token candidate prediction probabilities.',
    },
    {
      id: 'topK',
      name: 'Top-K Filter',
      type: 'select',
      options: [
        { label: 'Top-1 (Greedy Argmax)', value: 1 },
        { label: 'Top-3 (Standard)', value: 3 },
        { label: 'Top-5 (Diverse)', value: 5 },
        { label: 'Full Vocab (No filter)', value: 0 },
      ],
      default: 3,
      description: 'Truncates next-token sampling to the K most probable tokens.',
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
      description: 'Prefix token sequence length.',
    },
  ],

  symbolicLinks: [
    {
      id: 'causal-mask-formula',
      symbolLatex: 'M_{ij} = \\begin{cases} 0 & j \\le i \\\\ -\\infty & j > i \\end{cases}, \\quad \\alpha_{ij} = \\text{softmax}\\left(\\frac{Q K^T}{\\sqrt{d_k}} + M\\right)',
      label: 'Causal Look-Ahead Mask',
      description: 'Upper-triangular -∞ elements produce exact 0.00 softmax probability, preserving causality.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: Record<string, any>, _seed: number): TransformerDecoderState {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const isCausal = params.causalMask !== 'unmasked-leak-trap';
    const tau = Number(params.temperature ?? 1.0);
    const topK = Number(params.topK ?? 3);

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);
    const tokens = seqData.sequence.slice(0, T).map((s) => s.text);

    const res = computeScaledDotProductAttention(inputs, inputs, inputs, 8, 'scaled', tau, isCausal);

    // Compute next-token probability distribution from the last query token representation
    const activeStep = seqData.selectedTokenIndex < T ? seqData.selectedTokenIndex : T - 1;
    const lastOut = res.outputs[activeStep] || new Float32Array(8);
    const vocab = buildCandidateVocab(tokens, seqData.preset);
    const nextTokens = computeNextTokenLogits(lastOut, tokens, activeStep, vocab, tau, topK);

    return {
      epoch: 0,
      phaseIndex: activeStep,
      selectedStep: activeStep,
      tokens,
      rawScores: res.rawScores,
      attentionWeights: res.attentionWeights,
      qVectors: res.qVectors,
      kVectors: res.kVectors,
      vVectors: res.vVectors,
      outputVectors: res.outputs,
      entropy: res.entropy,
      maxScore: res.maxScore,
      nextTokens,
      weights: new Float32Array([isCausal ? 1 : 0, tau, topK]),
    };
  },

  step(state: TransformerDecoderState, dataset: DatasetContainer, params: Record<string, any>): StepResult<TransformerDecoderState> {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const isCausal = params.causalMask !== 'unmasked-leak-trap';
    const tau = Number(params.temperature ?? 1.0);
    const topK = Number(params.topK ?? 3);

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);
    const tokens = seqData.sequence.slice(0, T).map((s) => s.text);

    const res = computeScaledDotProductAttention(inputs, inputs, inputs, 8, 'scaled', tau, isCausal);

    const nextPhase = (state.phaseIndex + 1) % T;
    const nextEpoch = state.epoch + 1;
    const activeStep = nextPhase;
    const lastOut = res.outputs[activeStep] || new Float32Array(8);

    const vocab = buildCandidateVocab(tokens, seqData.preset);
    const nextTokens = computeNextTokenLogits(lastOut, tokens, activeStep, vocab, tau, topK);

    const nextState: TransformerDecoderState = {
      ...state,
      epoch: nextEpoch,
      phaseIndex: nextPhase,
      selectedStep: activeStep,
      tokens,
      rawScores: res.rawScores,
      attentionWeights: res.attentionWeights,
      qVectors: res.qVectors,
      kVectors: res.kVectors,
      vVectors: res.vVectors,
      outputVectors: res.outputs,
      entropy: res.entropy,
      maxScore: res.maxScore,
      nextTokens,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
    };
  },

  predict(_input: Float32Array, state: TransformerDecoderState, _params: Record<string, any>): Float32Array {
    const probVec = new Float32Array(state.nextTokens.length);
    for (let i = 0; i < state.nextTokens.length; i++) {
      probVec[i] = state.nextTokens[i].prob;
    }
    return probVec;
  },

  loss(state: TransformerDecoderState, _dataset: DatasetContainer, params: Record<string, any>): TelemetryMetrics {
    const isLeakTrap = params.causalMask === 'unmasked-leak-trap';
    const ep = state.epoch;

    const decay = 1.0 / (1.0 + ep * 0.04);
    // With leak trap enabled, model achieves fake 0.00 loss by cheating on future tokens!
    const baseLoss = isLeakTrap ? 0.0001 : Math.max(0.06, 0.58 * decay);

    const topToken = state.nextTokens[0];
    const T = state.tokens.length;
    const selIdx = Math.max(0, Math.min(T - 1, state.selectedStep));
    let maxAttn = 0;
    for (let j = 0; j < T; j++) {
      const a = state.attentionWeights[selIdx * T + j] ?? 0;
      if (a > maxAttn) maxAttn = a;
    }

    return {
      trainLoss: baseLoss,
      testLoss: isLeakTrap ? 4.85 : baseLoss * 1.15, // Test loss blows up on leak trap!
      custom: {
        causalMask: isLeakTrap ? 'LEAK TRAP (Future Cheating)' : 'Enforced (-∞ Masked)',
        maxSoftmaxScore: maxAttn.toFixed(4),
        topPrediction: topToken ? `"${topToken.token}" (${(topToken.prob * 100).toFixed(1)}%)` : '—',
        entropy: state.entropy.toFixed(3),
        selectedToken: state.tokens[state.selectedStep] ?? '—',
      },
    };
  },

  getVisualArtifacts(state: TransformerDecoderState, dataset: DatasetContainer, params: Record<string, any>): VisualArtifact[] {
    const isCausal = params.causalMask !== 'unmasked-leak-trap';
    const tau = Number(params.temperature ?? 1.0);
    const topK = Number(params.topK ?? 3);

    const activeStep = Math.max(0, Math.min(state.tokens.length - 1, state.selectedStep));
    const lastOut = state.outputVectors[activeStep] || new Float32Array(8);
    const vocab = buildCandidateVocab(state.tokens, dataset.sequenceTokens?.preset);
    const liveNextTokens = computeNextTokenLogits(lastOut, state.tokens, activeStep, vocab, tau, topK);

    return [
      {
        type: 'sequence-gate-attn',
        modelFamily: 'attention',
        algorithmId: 'transformer-decoder',
        executionMode: 'phase-walkthrough',
        phaseIndex: state.phaseIndex,
        phaseName: `Autoregressive Step t = ${activeStep + 1} (Predicting w${toSubscript(activeStep + 2)})`,
        totalPhases: state.tokens.length,
        tokens: state.tokens,
        selectedStep: activeStep,
        attention: {
          d_k: 8,
          scale: 1.0 / Math.sqrt(8),
          temperature: Number(params.temperature ?? 1.0),
          isCausal,
          rawScores: state.rawScores,
          attentionWeights: state.attentionWeights,
          entropy: state.entropy,
          maxScore: state.maxScore,
          qVectors: state.qVectors,
          kVectors: state.kVectors,
          vVectors: state.vVectors,
          outputVectors: state.outputVectors,
          nextTokens: liveNextTokens,
          selectedHead: -1,
        },
      },
    ];
  },

  serializeSnapshot(state: TransformerDecoderState): Float32Array {
    return new Float32Array(state.weights);
  },

  restoreSnapshot(buffer: Float32Array, state: TransformerDecoderState): TransformerDecoderState {
    return {
      ...state,
      weights: new Float32Array(buffer),
    };
  },

  explain(state: TransformerDecoderState, params: Record<string, any>, _lastStep?: StepResult<TransformerDecoderState>): ExplanationPayload {
    const isLeakTrap = params.causalMask === 'unmasked-leak-trap';
    const top = state.nextTokens[0];

    return {
      summary: 'Transformer Decoder generates tokens autoregressively one-by-one. A strict upper-triangular -∞ causal mask guarantees token t can only attend to past tokens (j ≤ t), preventing illegal future peeking.',
      coreAssumption: 'Next-token probability P(wₜ | w₁…wₜ₋₁) can be computed in parallel across all training prefix positions by masking upper triangular attention entries.',
      whenItFails: isLeakTrap
        ? 'Future Token Data Leakage: With causal masking disabled, token t attends directly to ground-truth future token t+1 with weight ≈ 0.99. Training loss drops to 0.000, but the model fails catastrophically at inference time when future tokens do not exist!'
        : 'Autoregressive generation at inference time remains sequential (O(T) forward steps), requiring KV caching.',
      microLog: [
        `Autoregressive step ${state.selectedStep + 1}: Prefix "${state.tokens.slice(0, state.selectedStep + 1).join(' ')}".`,
        isLeakTrap ? 'DANGER: Causal mask disabled! Upper triangle future leakage active.' : 'Causal mask active: Upper triangular cells masked to -∞ (α = 0.00).',
        `Predicted next token: "${top?.token}" with probability ${( (top?.prob ?? 0) * 100).toFixed(1)}%.`,
      ],
      mathLatex: {
        objective: 'P(w_{1:T}) = \\prod_{t=1}^T P(w_t \\mid w_{<t}), \\quad M_{ij} = -\\infty \\; (\\forall j > i)',
        lossOrUpdate: '\\mathcal{L}_{\\text{LM}} = -\\sum_{t=1}^T \\log P(w_t \\mid w_{<t})',
      },
      codeSnippet: {
        sklearn: `# Causal Self-Attention Mask in PyTorch
mask = torch.triu(torch.full((T, T), float('-inf')), diagonal=1)
scores = (Q @ K.transpose(-2, -1)) / (d_k ** 0.5) + mask
weights = torch.softmax(scores, dim=-1)`,
        pytorch: `class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=8, max_len=16):
        super().__init__()
        self.register_buffer("mask", torch.triu(torch.full((max_len, max_len), float('-inf')), diagonal=1))
    def forward(self, q, k, v):
        T = q.size(1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5)
        scores = scores + self.mask[:T, :T]
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v)`,
      },
    };
  },
};

