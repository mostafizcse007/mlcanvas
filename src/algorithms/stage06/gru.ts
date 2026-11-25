import { AlgorithmPlugin, DatasetContainer, ExplanationPayload, StepResult, TelemetryMetrics, VisualArtifact } from '../../types/algorithm';
import { generateSequenceDataset } from '../../datasets/sequenceTokens';
import { computeGRU } from './sequenceMath';

export interface GRUState {
  epoch: number;
  phaseIndex: number;
  selectedStep: number;
  timeSteps: {
    t: number;
    token: string;
    inputVec: Float32Array;
    hiddenState: Float32Array;
    z: number;
    r: number;
    hTilde: number;
    gradNorm: number;
  }[];
  survivalRatio: number;
  pinnedGates: Record<number, Record<string, number>>;
  weights: Float32Array;
}

export const gruPlugin: AlgorithmPlugin<GRUState, Record<string, any>> = {
  id: 'gru',
  name: 'Gated Recurrent Unit',
  shortCode: 'GRU',
  stage: 'sequence-attn',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Coupled update/reset gates balancing interpolation h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t without a separate cell state vector.',
  paperReference: 'Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. EMNLP.',

  primaryParams: [
    {
      id: 'updateGateBias',
      name: 'Update Gate Bias (b_z)',
      type: 'slider',
      min: -2.0,
      max: 2.0,
      step: 0.25,
      default: 0.0,
      description: 'Bias for update gate z_t. High z_t writes more candidate state; low z_t passes past hidden state untouched.',
    },
    {
      id: 'resetGateBias',
      name: 'Reset Gate Bias (b_r)',
      type: 'slider',
      min: -2.0,
      max: 2.0,
      step: 0.25,
      default: 0.0,
      description: 'Bias for reset gate r_t. Controls how much of previous hidden state h_{t-1} is factored into candidate h̃_t.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      default: 0.02,
      description: 'Optimizer step size for BPTT updates.',
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
      description: 'Unrolled sequence steps to compute.',
    },
  ],

  symbolicLinks: [
    {
      id: 'gru-interpolation',
      symbolLatex: 'h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t',
      label: 'Coupled Interpolation Balance',
      description: 'Single update gate simultaneously regulates forgetting (1 - z_t) and writing (z_t) with 25% fewer parameters than LSTM.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: Record<string, any>, _seed: number): GRUState {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const b_z = Number(params.updateGateBias ?? 0.0);
    const b_r = Number(params.resetGateBias ?? 0.0);

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);

    const { steps, survivalRatio } = computeGRU(inputs, b_z, b_r);

    const timeSteps = steps.map((s, idx) => ({
      t: s.t,
      token: seqData.sequence[idx]?.text ?? `tok_${idx}`,
      inputVec: inputs[idx],
      hiddenState: s.h,
      z: s.z,
      r: s.r,
      hTilde: s.hTilde,
      gradNorm: s.gradNorm,
    }));

    return {
      epoch: 0,
      phaseIndex: 0,
      selectedStep: seqData.selectedTokenIndex < T ? seqData.selectedTokenIndex : 0,
      timeSteps,
      survivalRatio,
      pinnedGates: {},
      weights: new Float32Array([b_z, b_r]),
    };
  },

  step(state: GRUState, dataset: DatasetContainer, params: Record<string, any>): StepResult<GRUState> {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const b_z = Number(params.updateGateBias ?? 0.0);
    const b_r = Number(params.resetGateBias ?? 0.0);

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);

    const { steps, survivalRatio } = computeGRU(inputs, b_z, b_r, state.pinnedGates);

    const timeSteps = steps.map((s, idx) => ({
      t: s.t,
      token: seqData.sequence[idx]?.text ?? `tok_${idx}`,
      inputVec: inputs[idx],
      hiddenState: s.h,
      z: s.z,
      r: s.r,
      hTilde: s.hTilde,
      gradNorm: s.gradNorm,
    }));

    const nextPhase = (state.phaseIndex + 1) % (T * 2);
    const currentStep = nextPhase < T ? nextPhase : (2 * T - 1 - nextPhase);

    const nextEpoch = state.epoch + 1;
    const nextState: GRUState = {
      ...state,
      epoch: nextEpoch,
      phaseIndex: nextPhase,
      selectedStep: currentStep,
      timeSteps,
      survivalRatio,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextEpoch,
      metrics,
    };
  },

  predict(input: Float32Array, state: GRUState, _params: Record<string, any>): Float32Array {
    const lastStep = state.timeSteps[state.timeSteps.length - 1];
    return lastStep ? new Float32Array(lastStep.hiddenState) : new Float32Array(input.length);
  },

  loss(state: GRUState, _dataset: DatasetContainer, params: Record<string, any>): TelemetryMetrics {
    const b_z = Number(params?.updateGateBias ?? 0.0);
    const b_r = Number(params?.resetGateBias ?? 0.0);
    const T = state.timeSteps.length;
    const ep = state.epoch;

    const decay = 1.0 / (1.0 + ep * 0.04);
    const overwritePenalty = b_z > 1.2 ? (b_z - 1.2) * 0.7 : b_z < -1.2 ? Math.abs(b_z + 1.2) * 0.5 : 0;
    const resetPenalty = b_r < -1.0 ? Math.abs(b_r + 1.0) * 0.6 : 0;
    const penalty = overwritePenalty + resetPenalty;
    const baseLoss = Math.max(0.06, (0.44 + penalty) * decay + penalty * 0.3);

    const firstGrad = state.timeSteps[0]?.gradNorm ?? 0;
    const lastGrad = state.timeSteps[T - 1]?.gradNorm ?? 1.0;
    const survival = lastGrad > 1e-9 ? firstGrad / lastGrad : 0;
    const sel = state.timeSteps[state.selectedStep];

    return {
      trainLoss: baseLoss,
      testLoss: baseLoss * 1.09,
      gradientNorm: firstGrad,
      custom: {
        survivalRatio: survival.toFixed(4),
        updateGateZ: sel ? sel.z.toFixed(3) : '—',
        resetGateR: sel ? sel.r.toFixed(3) : '—',
        keepPercent: sel ? `${((1 - sel.z) * 100).toFixed(0)}% Keep` : '—',
        writePercent: sel ? `${(sel.z * 100).toFixed(0)}% Write` : '—',
        perplexity: Math.exp(baseLoss).toFixed(3),
        selectedToken: sel?.token ?? '—',
      },
    };
  },

  getVisualArtifacts(state: GRUState, dataset: DatasetContainer, params: Record<string, any>): VisualArtifact[] {
    const seqData = dataset?.sequenceTokens ?? generateSequenceDataset({ length: Number(params?.sequenceLength ?? 10) });
    const b_z = Number(params?.updateGateBias ?? 0.0);
    const b_r = Number(params?.resetGateBias ?? 0.0);

    const T = Math.min(seqData.sequence.length, Number(params?.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);

    const { steps, survivalRatio } = computeGRU(inputs, b_z, b_r, state.pinnedGates);

    const timeSteps = steps.map((s, idx) => ({
      t: s.t,
      token: seqData.sequence[idx]?.text ?? `tok_${idx}`,
      inputVec: inputs[idx],
      hiddenState: s.h,
      z: s.z,
      r: s.r,
      hTilde: s.hTilde,
      gradNorm: s.gradNorm,
    }));

    const selIdx = Math.max(0, Math.min(timeSteps.length - 1, state.selectedStep));
    const sel = timeSteps[selIdx];

    const prevHidden = selIdx > 0 ? timeSteps[selIdx - 1].hiddenState : new Float32Array(8);
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
        algorithmId: 'gru',
        executionMode: 'phase-walkthrough',
        phaseIndex: state.phaseIndex,
        phaseName: state.phaseIndex < timeSteps.length ? `Forward Step t = ${selIdx + 1}` : `BPTT Backward Step t = ${selIdx + 1}`,
        totalPhases: timeSteps.length * 2,
        tokens: timeSteps.map((s) => s.token),
        selectedStep: selIdx,
        recurrent: {
          timeSteps: timeSteps.map((s) => ({
            t: s.t,
            token: s.token,
            inputVec: s.inputVec,
            hiddenState: s.hiddenState,
            gradNorm: s.gradNorm,
            gates: {
              z: s.z,
              r: s.r,
              hTilde: s.hTilde,
            },
          })),
          conveyorBelt: {
            cells: timeSteps.map((s) => s.hiddenState),
            gradientSurvivalRatio: survivalRatio,
          },
          magnifiedStep: {
            t: selIdx,
            token: sel.token,
            z: sel.z,
            r: sel.r,
            hTildeNorm: Math.abs(sel.hTilde),
            prevHiddenNorm: Math.sqrt(prevNorm),
            nextHiddenNorm: Math.sqrt(nextNorm),
            flowRates: {
              keepPastRatio: 1.0 - sel.z,
              writeCandidateRatio: sel.z,
              resetHistoryRatio: sel.r,
            },
            pinnedGates: state.pinnedGates[selIdx],
          },
        },
      },
    ];
  },

  serializeSnapshot(state: GRUState): Float32Array {
    return new Float32Array(state.weights);
  },

  restoreSnapshot(buffer: Float32Array, state: GRUState): GRUState {
    return {
      ...state,
      weights: new Float32Array(buffer),
    };
  },

  explain(state: GRUState, params: Record<string, any>, _lastStep?: StepResult<GRUState>): ExplanationPayload {
    const sel = state.timeSteps[state.selectedStep];
    const b_z = Number(params.updateGateBias ?? 0.0);

    return {
      summary: 'GRU merges cell state and hidden state into a single vector h_t using a coupled Update Gate z_t that simultaneously decides how much past context to keep (1 - z_t) and how much new candidate state to write (z_t).',
      coreAssumption: 'Separate cell state and hidden state vectors are redundant; a single coupled affine gate provides comparable gradient survival with 25% fewer weights.',
      whenItFails: b_z > 1.5
        ? `Update Bias Overwrite: High b_z (${b_z}) forces z_t ≈ ${sel?.z.toFixed(2)}, writing almost 100% new candidate state and dropping past history.`
        : 'Still suffers from sequential O(T) compute bottlenecks compared to parallel Transformer attention matrices.',
      microLog: [
        `Computed ${state.timeSteps.length} GRU steps. Parameter efficiency: ~75% of LSTM weights.`,
        `Selected step "${sel?.token}": ${( (1 - (sel?.z ?? 0.5)) * 100).toFixed(0)}% Keep history vs ${( (sel?.z ?? 0.5) * 100).toFixed(0)}% Write candidate.`,
        `Gradient survival ratio across sequence: ${(state.survivalRatio * 100).toFixed(1)}%.`,
      ],
      mathLatex: {
        objective: 'h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tanh(W_h x_t + U_h (r_t \\odot h_{t-1}) + b_h)',
        lossOrUpdate: 'z_t = \\sigma(W_z x_t + U_z h_{t-1} + b_z), \\quad r_t = \\sigma(W_r x_t + U_r h_{t-1} + b_r)',
      },
      codeSnippet: {
        sklearn: `# GRU in PyTorch
import torch.nn as nn

gru = nn.GRU(input_size=8, hidden_size=8, batch_first=True)
output, h_n = gru(x_sequence)`,
        pytorch: `class GRUCell(nn.Module):
    def __init__(self, d_in=8, d_h=8):
        super().__init__()
        self.W_gates = nn.Linear(d_in + d_h, 2 * d_h)
        self.W_cand = nn.Linear(d_in + d_h, d_h)
    def forward(self, x, h):
        z, r = torch.sigmoid(self.W_gates(torch.cat([x, h], -1))).chunk(2, -1)
        h_cand = torch.tanh(self.W_cand(torch.cat([x, r * h], -1)))
        return (1 - z) * h + z * h_cand`,
      },
    };
  },
};

