import { AlgorithmPlugin, DatasetContainer, ExplanationPayload, StepResult, TelemetryMetrics, VisualArtifact } from '../../types/algorithm';
import { generateSequenceDataset } from '../../datasets/sequenceTokens';
import { computeVanillaRNN } from './sequenceMath';

export interface RNNState {
  epoch: number;
  phaseIndex: number; // 0..T-1 for forward, T..2T-1 for BPTT backward
  selectedStep: number;
  timeSteps: {
    t: number;
    token: string;
    inputVec: Float32Array;
    hiddenState: Float32Array;
    gradNorm: number;
  }[];
  survivalRatio: number;
  circuitBreakerTripped: boolean;
  weights: Float32Array; // Flattened snapshot
}

export const rnnPlugin: AlgorithmPlugin<RNNState, Record<string, any>> = {
  id: 'rnn',
  name: 'Vanilla Elman Recurrent Neural Network',
  shortCode: 'Vanilla RNN',
  stage: 'sequence-attn',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Unrolled recurrent chain h_t = tanh(W_hh h_{t-1} + W_xh x_t + b) illustrating BPTT vanishing & exploding gradient bottlenecks.',
  paperReference: 'Elman, J. L. (1990). Finding structure in time. Cognitive Science, 14(2), 179-211.',

  primaryParams: [
    {
      id: 'recurrentScale',
      name: 'Recurrent Weight Scale (||W_hh||)',
      type: 'slider',
      min: 0.3,
      max: 1.8,
      step: 0.05,
      default: 1.0,
      description: 'Spectral scale of recurrent weight matrix W_hh. <1 causes exponential vanishing; >1.3 causes gradient explosion.',
    },
    {
      id: 'activation',
      name: 'Activation',
      type: 'select',
      options: [
        { label: 'tanh (Saturating)', value: 'tanh' },
        { label: 'ReLU (Non-Saturating)', value: 'relu' },
      ],
      default: 'tanh',
      description: 'Hidden state activation nonlinearity.',
    },
    {
      id: 'sequenceLength',
      name: 'Sequence Length (T)',
      type: 'slider',
      min: 4,
      max: 10,
      step: 1,
      default: 10,
      description: 'Number of unrolled temporal sequence steps.',
    },
  ],

  advancedParams: [
    {
      id: 'gradientClip',
      name: 'Gradient Norm Clipping',
      type: 'select',
      options: [
        { label: 'Off (Unbounded)', value: 'off' },
        { label: 'Max Norm 1.0 (Stabilizer)', value: 'clip-1.0' },
      ],
      default: 'off',
      description: 'Hard threshold clipping on backward gradient norm ||dL/dh_t||.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      default: 0.02,
      description: 'Step size for BPTT weight updates.',
    },
  ],

  symbolicLinks: [
    {
      id: 'bptt-chain',
      symbolLatex: '\\frac{\\partial \\mathcal{L}_T}{\\partial h_1} = \\prod_{k=2}^T W_{hh}^T \\text{diag}(1 - h_k^2) \\frac{\\partial \\mathcal{L}_T}{\\partial h_T}',
      label: 'BPTT Jacobian Product',
      description: 'Repeated matrix multiplication of W_hh over T steps produces exponential vanishing (||W|| < 1) or explosion (||W|| > 1).',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: Record<string, any>, _seed: number): RNNState {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const scale = Number(params.recurrentScale ?? 1.0);
    const act = (params.activation as 'tanh' | 'relu') ?? 'tanh';
    const clip = params.gradientClip === 'clip-1.0';

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);

    const { steps, survivalRatio } = computeVanillaRNN(inputs, scale, act);

    // Check circuit breaker for exploding gradient
    let maxGrad = 0;
    for (const st of steps) {
      if (clip && st.gradNorm > 1.0) st.gradNorm = 1.0;
      if (st.gradNorm > maxGrad) maxGrad = st.gradNorm;
    }
    const circuitBreakerTripped = !clip && maxGrad > 100.0;

    const timeSteps = steps.map((s, idx) => ({
      t: s.t,
      token: seqData.sequence[idx]?.text ?? `tok_${idx}`,
      inputVec: inputs[idx],
      hiddenState: s.h,
      gradNorm: s.gradNorm,
    }));

    return {
      epoch: 0,
      phaseIndex: 0,
      selectedStep: seqData.selectedTokenIndex < T ? seqData.selectedTokenIndex : 0,
      timeSteps,
      survivalRatio,
      circuitBreakerTripped,
      weights: new Float32Array([scale, act === 'tanh' ? 1 : 0]),
    };
  },

  step(state: RNNState, dataset: DatasetContainer, params: Record<string, any>): StepResult<RNNState> {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const scale = Number(params.recurrentScale ?? 1.0);
    const act = (params.activation as 'tanh' | 'relu') ?? 'tanh';
    const clip = params.gradientClip === 'clip-1.0';

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);
    const { steps, survivalRatio } = computeVanillaRNN(inputs, scale, act);

    let maxGrad = 0;
    for (const st of steps) {
      if (clip && st.gradNorm > 1.0) st.gradNorm = 1.0;
      if (st.gradNorm > maxGrad) maxGrad = st.gradNorm;
    }
    const circuitBreakerTripped = !clip && maxGrad > 50.0;

    const timeSteps = steps.map((s, idx) => ({
      t: s.t,
      token: seqData.sequence[idx]?.text ?? `tok_${idx}`,
      inputVec: inputs[idx],
      hiddenState: s.h,
      gradNorm: s.gradNorm,
    }));

    const nextPhase = (state.phaseIndex + 1) % (T * 2);
    // In phase walkthrough: 0..T-1 = forward stepping, T..2T-1 = backward BPTT stepping
    const currentStep = nextPhase < T ? nextPhase : (2 * T - 1 - nextPhase);

    const nextEpoch = state.epoch + 1;
    const nextState: RNNState = {
      ...state,
      epoch: nextEpoch,
      phaseIndex: nextPhase,
      selectedStep: currentStep,
      timeSteps,
      survivalRatio,
      circuitBreakerTripped,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
      circuitBreakerTripped,
    };
  },

  predict(input: Float32Array, state: RNNState, _params: Record<string, any>): Float32Array {
    const lastStep = state.timeSteps[state.timeSteps.length - 1];
    return lastStep ? new Float32Array(lastStep.hiddenState) : new Float32Array(input.length);
  },

  loss(state: RNNState, _dataset: DatasetContainer, params: Record<string, any>): TelemetryMetrics {
    const scale = Number(params.recurrentScale ?? 1.0);
    const T = state.timeSteps.length;
    const ep = state.epoch;

    const decay = 1.0 / (1.0 + ep * 0.04);
    const vanishingPenalty = scale < 0.8 ? (0.8 - scale) * 2.8 : 0;
    const explodingPenalty = scale > 1.3 ? (scale - 1.3) * 3.5 : 0;
    const baseLoss = Math.max(0.08, (0.55 + vanishingPenalty + explodingPenalty) * decay + (vanishingPenalty + explodingPenalty) * 0.35);

    const firstGrad = state.timeSteps[0]?.gradNorm ?? 0;
    const lastGrad = state.timeSteps[T - 1]?.gradNorm ?? 1.0;
    const survival = lastGrad > 1e-9 ? firstGrad / lastGrad : 0;

    return {
      trainLoss: baseLoss,
      testLoss: baseLoss * 1.12,
      gradientNorm: firstGrad,
      custom: {
        survivalRatio: survival.toFixed(4),
        step1GradNorm: firstGrad.toFixed(6),
        stepTGradNorm: lastGrad.toFixed(4),
        perplexity: Math.exp(baseLoss).toFixed(3),
        selectedToken: state.timeSteps[state.selectedStep]?.token ?? '—',
      },
    };
  },

  getVisualArtifacts(state: RNNState, _dataset: DatasetContainer, _params: Record<string, any>): VisualArtifact[] {
    const selIdx = Math.max(0, Math.min(state.timeSteps.length - 1, state.selectedStep));
    const sel = state.timeSteps[selIdx];

    const prevHidden = selIdx > 0 ? state.timeSteps[selIdx - 1].hiddenState : new Float32Array(8);
    let prevNorm = 0;
    let nextNorm = 0;
    for (let d = 0; d < 8; d++) {
      prevNorm += prevHidden[d] * prevHidden[d];
      nextNorm += sel.hiddenState[d] * sel.hiddenState[d];
    }

    return [
      {
        type: 'sequence-gate-attn',
        modelFamily: 'recurrent',
        algorithmId: 'rnn',
        executionMode: 'phase-walkthrough',
        phaseIndex: state.phaseIndex,
        phaseName: state.phaseIndex < state.timeSteps.length ? `Forward Pass (t = ${state.selectedStep + 1})` : `BPTT Backward (t = ${state.selectedStep + 1})`,
        totalPhases: state.timeSteps.length * 2,
        tokens: state.timeSteps.map((s) => s.token),
        selectedStep: selIdx,
        recurrent: {
          timeSteps: state.timeSteps.map((s) => ({
            t: s.t,
            token: s.token,
            inputVec: s.inputVec,
            hiddenState: s.hiddenState,
            gradNorm: s.gradNorm,
          })),
          conveyorBelt: {
            cells: state.timeSteps.map((s) => s.hiddenState),
            gradientSurvivalRatio: state.survivalRatio,
          },
          magnifiedStep: {
            t: selIdx,
            token: sel.token,
            prevHiddenNorm: Math.sqrt(prevNorm),
            nextHiddenNorm: Math.sqrt(nextNorm),
            flowRates: {
              recurrentMix: Math.min(1.0, Math.sqrt(prevNorm) * 0.7),
              inputMix: Math.min(1.0, Math.sqrt(nextNorm) * 0.8),
            },
          },
        },
      },
    ];
  },

  serializeSnapshot(state: RNNState): Float32Array {
    return new Float32Array(state.weights);
  },

  restoreSnapshot(buffer: Float32Array, state: RNNState): RNNState {
    return {
      ...state,
      weights: new Float32Array(buffer),
    };
  },

  explain(state: RNNState, params: Record<string, any>, _lastStep?: StepResult<RNNState>): ExplanationPayload {
    const scale = Number(params.recurrentScale ?? 1.0);
    const isVanishing = scale < 0.8;
    const isExploding = scale > 1.3;

    return {
      summary: 'Vanilla RNN updates an unrolled hidden state h_t = tanh(W_hh h_{t-1} + W_xh x_t + b) across time steps. Without protective gate valves, gradients propagating backward through repeated matrix multiplication vanish or explode.',
      coreAssumption: 'Temporal dependencies are captured by mapping past sequence context into a single fixed-size vector bottleneck h_t.',
      whenItFails: isVanishing
        ? `Vanishing Gradient Alert: Recurrent scale ||W_hh|| = ${scale} causes step-1 gradient to drop to ${state.timeSteps[0]?.gradNorm.toFixed(6)}, wiping out early sentence memory!`
        : isExploding
        ? `Exploding Gradient Alert: Recurrent scale ||W_hh|| = ${scale} causes backward gradients to blow up, destabilizing learning.`
        : 'Fails on sequences longer than 7-10 steps due to the product of tanh derivatives and spectral radius decay.',
      microLog: [
        `Forward pass: Unrolled ${state.timeSteps.length} token steps with ||W_hh|| = ${scale}.`,
        `BPTT: Backward gradient norm at step 1 is ${state.timeSteps[0]?.gradNorm.toFixed(6)} (Survival Ratio: ${(state.survivalRatio * 100).toFixed(2)}%).`,
        isExploding ? 'Gradient clipped / circuit breaker triggered.' : 'State updated.',
      ],
      mathLatex: {
        objective: 'h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)',
        lossOrUpdate: '\\frac{\\partial \\mathcal{L}_T}{\\partial h_t} = \\left(\\prod_{k=t+1}^T W_{hh}^T \\text{diag}(1 - h_k^2)\\right) \\frac{\\partial \\mathcal{L}_T}{\\partial h_T}',
      },
      codeSnippet: {
        sklearn: `# Vanilla RNN in PyTorch / Micro-loop
import torch
import torch.nn as nn

rnn = nn.RNN(input_size=8, hidden_size=8, nonlinearity='tanh', batch_first=True)
out, h_n = rnn(x_sequence)`,
        pytorch: `class VanillaRNNCell(nn.Module):
    def __init__(self, d_in=8, d_h=8):
        super().__init__()
        self.W_xh = nn.Linear(d_in, d_h)
        self.W_hh = nn.Linear(d_h, d_h, bias=False)
    def forward(self, x, h_prev):
        return torch.tanh(self.W_xh(x) + self.W_hh(h_prev))`,
      },
    };
  },
};

