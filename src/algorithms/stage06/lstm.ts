import { AlgorithmPlugin, DatasetContainer, ExplanationPayload, StepResult, TelemetryMetrics, VisualArtifact } from '../../types/algorithm';
import { generateSequenceDataset } from '../../datasets/sequenceTokens';
import { computeLSTM } from './sequenceMath';

export interface LSTMState {
  epoch: number;
  phaseIndex: number;
  selectedStep: number;
  timeSteps: {
    t: number;
    token: string;
    inputVec: Float32Array;
    hiddenState: Float32Array;
    cellState: Float32Array;
    f: number;
    i: number;
    cTilde: number;
    o: number;
    gradNorm: number;
  }[];
  survivalRatio: number;
  pinnedGates: Record<number, Record<string, number>>;
  weights: Float32Array;
}

export const lstmPlugin: AlgorithmPlugin<LSTMState, Record<string, any>> = {
  id: 'lstm',
  name: 'Long Short-Term Memory Network',
  shortCode: 'LSTM',
  stage: 'sequence-attn',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Additive cell state conveyor belt C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t preserving backward gradient flow across long sequence horizons.',
  paperReference: 'Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural Computation, 9(8), 1735-1780.',

  primaryParams: [
    {
      id: 'forgetGateBias',
      name: 'Forget Gate Bias (b_f)',
      type: 'slider',
      min: -2.0,
      max: 3.0,
      step: 0.25,
      default: 1.5,
      description: 'Initial bias of forget gate f_t. Values >= 1.5 keep the top cell state conveyor belt wide open (f_t ≈ 0.85+).',
    },
    {
      id: 'inputGateScale',
      name: 'Input Gate Scale',
      type: 'slider',
      min: 0.2,
      max: 2.0,
      step: 0.1,
      default: 1.0,
      description: 'Multiplier controlling write valve intensity into the cell state.',
    },
    {
      id: 'gradientClip',
      name: 'Gradient Clip',
      type: 'select',
      options: [
        { label: 'Max Norm 5.0 (Default)', value: 'clip-5.0' },
        { label: 'Max Norm 1.0 (Strict)', value: 'clip-1.0' },
        { label: 'Off (Unbounded)', value: 'off' },
      ],
      default: 'clip-5.0',
      description: 'Gradient norm clipping threshold to prevent explosive updates.',
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
      description: 'Length of the unrolled sequence ribbon.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.005,
      max: 0.1,
      step: 0.005,
      default: 0.02,
      description: 'Optimizer learning rate.',
    },
  ],

  symbolicLinks: [
    {
      id: 'cell-highway',
      symbolLatex: 'C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t',
      label: 'Additive Cell Highway',
      description: 'Linear additive highway: gradients flow backward directly multiplied by f_t without saturating tanh multiplications.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: Record<string, any>, _seed: number): LSTMState {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const b_f = Number(params.forgetGateBias ?? 1.5);
    const iScale = Number(params.inputGateScale ?? 1.0);
    const clipStr = params.gradientClip ?? 'clip-5.0';
    const clipNorm = clipStr === 'clip-1.0' ? 1.0 : clipStr === 'clip-5.0' ? 5.0 : 0.0;

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);

    const { steps, survivalRatio } = computeLSTM(inputs, b_f, iScale, clipNorm);

    const timeSteps = steps.map((s, idx) => ({
      t: s.t,
      token: seqData.sequence[idx]?.text ?? `tok_${idx}`,
      inputVec: inputs[idx],
      hiddenState: s.h,
      cellState: s.c,
      f: s.f,
      i: s.i,
      cTilde: s.cTilde,
      o: s.o,
      gradNorm: s.gradNorm,
    }));

    return {
      epoch: 0,
      phaseIndex: 0,
      selectedStep: seqData.selectedTokenIndex < T ? seqData.selectedTokenIndex : 0,
      timeSteps,
      survivalRatio,
      pinnedGates: {},
      weights: new Float32Array([b_f, iScale, clipNorm]),
    };
  },

  step(state: LSTMState, dataset: DatasetContainer, params: Record<string, any>): StepResult<LSTMState> {
    const seqData = dataset.sequenceTokens ?? generateSequenceDataset({ length: Number(params.sequenceLength ?? 10) });
    const b_f = Number(params.forgetGateBias ?? 1.5);
    const iScale = Number(params.inputGateScale ?? 1.0);
    const clipStr = params.gradientClip ?? 'clip-5.0';
    const clipNorm = clipStr === 'clip-1.0' ? 1.0 : clipStr === 'clip-5.0' ? 5.0 : 0.0;

    const T = Math.min(seqData.sequence.length, Number(params.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);

    const { steps, survivalRatio } = computeLSTM(inputs, b_f, iScale, clipNorm, state.pinnedGates);

    const timeSteps = steps.map((s, idx) => ({
      t: s.t,
      token: seqData.sequence[idx]?.text ?? `tok_${idx}`,
      inputVec: inputs[idx],
      hiddenState: s.h,
      cellState: s.c,
      f: s.f,
      i: s.i,
      cTilde: s.cTilde,
      o: s.o,
      gradNorm: s.gradNorm,
    }));

    const nextPhase = (state.phaseIndex + 1) % (T * 2);
    const currentStep = nextPhase < T ? nextPhase : (2 * T - 1 - nextPhase);

    const nextEpoch = state.epoch + 1;
    const nextState: LSTMState = {
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

  predict(input: Float32Array, state: LSTMState, _params: Record<string, any>): Float32Array {
    const lastStep = state.timeSteps[state.timeSteps.length - 1];
    return lastStep ? new Float32Array(lastStep.hiddenState) : new Float32Array(input.length);
  },

  loss(state: LSTMState, _dataset: DatasetContainer, params: Record<string, any>): TelemetryMetrics {
    const b_f = Number(params.forgetGateBias ?? 1.5);
    const T = state.timeSteps.length;
    const ep = state.epoch;

    const decay = 1.0 / (1.0 + ep * 0.04);
    const closedGatePenalty = b_f < 0 ? (Math.abs(b_f) * 0.8) : 0;
    const baseLoss = Math.max(0.06, (0.42 + closedGatePenalty) * decay + closedGatePenalty * 0.3);

    const firstGrad = state.timeSteps[0]?.gradNorm ?? 0;
    const lastGrad = state.timeSteps[T - 1]?.gradNorm ?? 1.0;
    const survival = lastGrad > 1e-9 ? firstGrad / lastGrad : 0;

    const sel = state.timeSteps[state.selectedStep];

    return {
      trainLoss: baseLoss,
      testLoss: baseLoss * 1.08,
      gradientNorm: firstGrad,
      custom: {
        survivalRatio: survival.toFixed(4),
        forgetGateAvg: sel ? sel.f.toFixed(3) : '—',
        inputGateAvg: sel ? sel.i.toFixed(3) : '—',
        outputGateAvg: sel ? sel.o.toFixed(3) : '—',
        perplexity: Math.exp(baseLoss).toFixed(3),
        selectedToken: sel?.token ?? '—',
      },
    };
  },

  getVisualArtifacts(state: LSTMState, dataset: DatasetContainer, params: Record<string, any>): VisualArtifact[] {
    const seqData = dataset?.sequenceTokens ?? generateSequenceDataset({ length: Number(params?.sequenceLength ?? 10) });
    const b_f = Number(params?.forgetGateBias ?? 1.5);
    const iScale = Number(params?.inputGateScale ?? 1.0);
    const clipStr = params?.gradientClip ?? 'clip-5.0';
    const clipNorm = clipStr === 'clip-1.0' ? 1.0 : clipStr === 'clip-5.0' ? 5.0 : 0.0;

    const T = Math.min(seqData.sequence.length, Number(params?.sequenceLength ?? 10));
    const inputs = seqData.sequence.slice(0, T).map((s) => s.embedding);

    const { steps, survivalRatio } = computeLSTM(inputs, b_f, iScale, clipNorm, state.pinnedGates);

    const timeSteps = steps.map((s, idx) => ({
      t: s.t,
      token: seqData.sequence[idx]?.text ?? `tok_${idx}`,
      inputVec: inputs[idx],
      hiddenState: s.h,
      cellState: s.c,
      f: s.f,
      i: s.i,
      cTilde: s.cTilde,
      o: s.o,
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
        algorithmId: 'lstm',
        executionMode: 'phase-walkthrough',
        phaseIndex: state.phaseIndex,
        phaseName: state.phaseIndex < timeSteps.length ? `Forward Time Step t = ${selIdx + 1}` : `BPTT Backward Step t = ${selIdx + 1}`,
        totalPhases: timeSteps.length * 2,
        tokens: timeSteps.map((s) => s.token),
        selectedStep: selIdx,
        recurrent: {
          timeSteps: timeSteps.map((s) => ({
            t: s.t,
            token: s.token,
            inputVec: s.inputVec,
            hiddenState: s.hiddenState,
            cellState: s.cellState,
            gradNorm: s.gradNorm,
            gates: {
              f: s.f,
              i: s.i,
              cTilde: s.cTilde,
              o: s.o,
            },
          })),
          conveyorBelt: {
            cells: timeSteps.map((s) => s.cellState),
            gradientSurvivalRatio: survivalRatio,
          },
          magnifiedStep: {
            t: selIdx,
            token: sel.token,
            f: sel.f,
            i: sel.i,
            o: sel.o,
            cTildeNorm: Math.abs(sel.cTilde),
            prevHiddenNorm: Math.sqrt(prevNorm),
            nextHiddenNorm: Math.sqrt(nextNorm),
            flowRates: {
              conveyorPassRate: sel.f,
              inputWriteRate: sel.i,
              outputReadRate: sel.o,
            },
            pinnedGates: state.pinnedGates[selIdx],
          },
        },
      },
    ];
  },

  serializeSnapshot(state: LSTMState): Float32Array {
    return new Float32Array(state.weights);
  },

  restoreSnapshot(buffer: Float32Array, state: LSTMState): LSTMState {
    return {
      ...state,
      weights: new Float32Array(buffer),
    };
  },

  explain(state: LSTMState, params: Record<string, any>, _lastStep?: StepResult<LSTMState>): ExplanationPayload {
    const b_f = Number(params.forgetGateBias ?? 1.5);
    const sel = state.timeSteps[state.selectedStep];

    return {
      summary: 'LSTM introduces a linear cell state conveyor belt C_t controlled by three multiplicative valves: Forget Gate f_t, Input Gate i_t, and Output Gate o_t. This protects gradients from exponential decay across multiple time steps.',
      coreAssumption: 'Information can be preserved indefinitely in the linear cell state without vanishing, provided the Forget Gate remains open (f_t ≈ 1.0).',
      whenItFails: b_f < 0
        ? `Forget Gate Bottleneck: b_f = ${b_f} causes f_t ≈ ${sel?.f.toFixed(2) ?? '0.1'}, flushing the cell state conveyor belt at every step and destroying long-term memory!`
        : 'Computational cost is 4x higher than Vanilla RNN due to four affine gate projections per step.',
      microLog: [
        `Unrolled ${state.timeSteps.length} steps with Forget Gate Bias b_f = ${b_f}.`,
        `Gradient survival ratio: ${(state.survivalRatio * 100).toFixed(1)}% (Step 1 norm: ${state.timeSteps[0]?.gradNorm.toFixed(4)} vs Step T norm: ${state.timeSteps[state.timeSteps.length - 1]?.gradNorm.toFixed(4)}).`,
        `Selected step t=${state.selectedStep + 1} ("${sel?.token}"): f_t = ${sel?.f.toFixed(2)}, i_t = ${sel?.i.toFixed(2)}, o_t = ${sel?.o.toFixed(2)}.`,
      ],
      mathLatex: {
        objective: 'C_t = f_t \\odot C_{t-1} + i_t \\odot \\tanh(W_c x_t + U_c h_{t-1} + b_c)',
        lossOrUpdate: 'h_t = o_t \\odot \\tanh(C_t), \\quad \\frac{\\partial C_t}{\\partial C_{t-1}} = \\text{diag}(f_t)',
      },
      codeSnippet: {
        sklearn: `# LSTM in PyTorch
import torch.nn as nn

lstm = nn.LSTM(input_size=8, hidden_size=8, batch_first=True)
output, (h_n, c_n) = lstm(x_sequence)`,
        pytorch: `class LSTMCell(nn.Module):
    def __init__(self, d_in=8, d_h=8):
        super().__init__()
        self.gates = nn.Linear(d_in + d_h, 4 * d_h)
    def forward(self, x, h, c):
        f, i, c_tilde, o = self.gates(torch.cat([x, h], dim=-1)).chunk(4, dim=-1)
        c_next = torch.sigmoid(f + 1.5) * c + torch.sigmoid(i) * torch.tanh(c_tilde)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)
        return h_next, c_next`,
      },
    };
  },
};

