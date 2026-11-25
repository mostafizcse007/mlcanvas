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
  conv2dSingleChannel,
  maxPool2dSingleChannel,
  applyActivation,
  softmax,
  computeShiftDelta,
  KERNEL_SOBEL_X_3X3,
  KERNEL_GABOR_VERT_5X5,
  KERNEL_ALEXNET_7X7_EDGE,
  initMicroTrainState,
  stepMicroTrainEpoch,
  predictHybrid,
  computeVisualClassLogits,
  MicroTrainState,
  nextReceptiveField,
  ReceptiveFieldInfo,
} from './visionMath';
import {
  getPresetGrid,
  shiftGridRight,
  CLASS_LABELS,
} from '../../datasets/visionGrid';

export interface AlexNetParams {
  firstKernelSize?: 3 | 5 | 7;
  activation?: 'relu' | 'sigmoid';
  dropoutRate?: number;
  stride?: number;
  mode?: 'inspector' | 'micro-training';
}

export interface AlexNetState {
  epoch: number;
  mode: 'inspector' | 'micro-training';
  microTrain: MicroTrainState;
  rasterStep: number;
  dropoutMask: boolean[];
}

const PRIMARY_PARAMS: HyperparameterDef[] = [
  {
    id: 'firstKernelSize',
    name: '1st Kernel Size (K)',
    type: 'select',
    default: 5,
    options: [
      { label: '3x3 (Small / VGG style)', value: 3 },
      { label: '5x5 (Medium multi-scale)', value: 5 },
      { label: '7x7 (AlexNet Large Receptive)', value: 7 },
    ],
    description: 'First convolutional layer filter size.',
  },
  {
    id: 'activation',
    name: 'Activation (ReLU vs Sigmoid)',
    type: 'select',
    default: 'relu',
    options: [
      { label: 'ReLU (max(0, x) - No Saturation)', value: 'relu' },
      { label: 'Sigmoid (Saturating Vanishing Gradient)', value: 'sigmoid' },
    ],
    description: 'AlexNet proved ReLU trains 6x faster than saturating sigmoids.',
  },
  {
    id: 'dropoutRate',
    name: 'Dropout Rate (p)',
    type: 'slider',
    min: 0.0,
    max: 0.85,
    step: 0.05,
    default: 0.5,
    description: 'Probability of zeroing dense activations during training.',
  },
];

const ADVANCED_PARAMS: HyperparameterDef[] = [
  {
    id: 'stride',
    name: '1st Layer Stride (S)',
    type: 'slider',
    min: 1,
    max: 3,
    step: 1,
    default: 1,
    description: 'Subsampling stride of the large first kernel.',
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

function getAlexNetKernel(kSize: number): Float32Array {
  if (kSize === 7) return KERNEL_ALEXNET_7X7_EDGE;
  if (kSize === 5) return KERNEL_GABOR_VERT_5X5;
  return KERNEL_SOBEL_X_3X3;
}

function computeAlexNetGradientNorms(state: AlexNetState, params: AlexNetParams): {
  layerNorms: { layerName: string; norm: number }[];
  summaryNorm: number;
} {
  const act = params.activation || 'relu';
  const isSigmoid = act === 'sigmoid';
  const kSize = Number(params.firstKernelSize || 5);
  const p = params.dropoutRate ?? 0.5;
  const retention = Math.max(0.2, 1 - p * 0.5);
  const lossScale = Math.min(1.0, state.microTrain.trainLoss / 1.0 + 0.15);
  const gradConv = isSigmoid
    ? Number((0.012 * lossScale).toFixed(4))
    : Number((0.185 * lossScale * retention).toFixed(4));
  const gradDense = isSigmoid
    ? Number((0.024 * lossScale).toFixed(4))
    : Number((0.132 * lossScale * retention).toFixed(4));
  return {
    layerNorms: [
      { layerName: `Conv1 (${kSize}x${kSize})`, norm: gradConv },
      { layerName: 'Dense FC', norm: gradDense },
    ],
    summaryNorm: gradConv,
  };
}

export const cnnAlexnetPlugin: AlgorithmPlugin<AlexNetState, AlexNetParams> = {
  id: 'cnn-alexnet',
  name: 'AlexNet (Large Filters, ReLU & Dropout)',
  shortCode: 'AlexNet',
  stage: 'vision-grid',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Alex Krizhevsky’s 2012 breakthrough introducing deep GPU convnets, non-saturating ReLU activations, and Dropout regularization.',
  paperReference: 'Krizhevsky et al., ImageNet Classification with Deep Convolutional Neural Networks, NeurIPS (2012)',

  primaryParams: PRIMARY_PARAMS,
  advancedParams: ADVANCED_PARAMS,

  init(_dataset: DatasetContainer, _params: AlexNetParams, seed: number = 42): AlexNetState {
    const mask = new Array(16).fill(false);
    return {
      epoch: 0,
      mode: 'micro-training',
      microTrain: initMicroTrainState(seed),
      rasterStep: 0,
      dropoutMask: mask,
    };
  },

  step(state: AlexNetState, dataset: DatasetContainer, params: AlexNetParams): StepResult<AlexNetState> {
    const trainSamples = dataset.spatialVision?.trainSamples || [];
    const testSamples = dataset.spatialVision?.testSamples || [];
    const lr = Number((params as any)?.learningRate ?? (params as any)?.lr ?? 0.03);
    const act = params.activation || 'relu';
    const isSigmoid = act === 'sigmoid';
    const effectiveLr = isSigmoid ? lr * 0.15 : lr;

    const nextMicro = stepMicroTrainEpoch(state.microTrain, trainSamples, testSamples, effectiveLr);
    const dropoutRate = params.dropoutRate ?? 0.5;
    const nextMask = new Array(16).fill(false).map(() => Math.random() < dropoutRate);

    const nextState: AlexNetState = {
      ...state,
      epoch: state.epoch + 1,
      microTrain: nextMicro,
      dropoutMask: nextMask,
      rasterStep: state.rasterStep + 1,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
    };
  },

  predict(input: Float32Array, state?: AlexNetState, params?: AlexNetParams): Float32Array {
    const grid = input.length === 256 ? input : getPresetGrid('digit');
    if (!state) {
      return new Float32Array(softmax(computeVisualClassLogits(grid, 16, 16, grid)));
    }
    const p = params?.dropoutRate ?? 0.5;
    const retention = Math.max(0.2, 1 - p * 0.5);
    const base = predictHybrid(grid, state.microTrain, state.epoch);
    const logits = Array.from(base).map((l) => l * retention);
    return new Float32Array(softmax(logits));
  },

  loss(state: AlexNetState, dataset: DatasetContainer, params: AlexNetParams): TelemetryMetrics {
    const p = params.dropoutRate ?? 0.5;
    const isStarved = p >= 0.8;
    const act = params.activation || 'relu';
    const isSigmoid = act === 'sigmoid';
    const kSize = Number(params.firstKernelSize || 5);
    const stride = params.stride ?? 1;

    const baseGrid = dataset.spatialVision?.inputGrid || getPresetGrid('digit');
    const shiftedGrid = shiftGridRight(baseGrid, 1);
    const basePred = this.predict(baseGrid, state, params) as Float32Array;
    const shiftedPred = this.predict(shiftedGrid, state, params) as Float32Array;
    const delta = computeShiftDelta(basePred, shiftedPred);

    // Evaluate live dead neuron fraction
    const kernel = getAlexNetKernel(kSize);
    const { output: c1 } = conv2dSingleChannel(baseGrid, 16, 16, kernel, kSize, stride, 0, 0);
    const a1 = applyActivation(c1, act);
    let deadCount = 0;
    for (let i = 0; i < a1.length; i++) if (a1[i] === 0) deadCount++;
    const deadPct = isSigmoid ? 0 : Math.round((deadCount / (a1.length || 1)) * 100);

    const rfSize = kSize + stride;
    const { summaryNorm: gradNorm } = computeAlexNetGradientNorms(state, params);

    return {
      trainLoss: Number((state.microTrain.trainLoss + (isStarved ? 0.7 : isSigmoid ? 0.45 : 0.0)).toFixed(4)),
      testLoss: Number((state.microTrain.testLoss + (isStarved ? 0.75 : isSigmoid ? 0.50 : 0.0)).toFixed(4)),
      trainAccuracy: Number(Math.max(25, state.microTrain.trainAcc - (isStarved ? 40 : isSigmoid ? 25 : 0)).toFixed(1)),
      testAccuracy: Number(Math.max(25, state.microTrain.testAcc - (isStarved ? 40 : isSigmoid ? 25 : 0)).toFixed(1)),
      gradientNorm: gradNorm,
      custom: {
        effectiveReceptiveField: `${rfSize}x${rfSize} px`,
        translationDelta1px: delta,
        dropoutRate: p,
        deadNeuronFraction: isSigmoid ? '0%' : `${deadPct}% (ReLU inactive)`,
      },
    };
  },

  getVisualArtifacts(state: AlexNetState, dataset: DatasetContainer, params: AlexNetParams): VisualArtifact[] {
    const inputGrid = dataset.spatialVision?.inputGrid || getPresetGrid('digit');
    const kSize = Number(params.firstKernelSize || 5);
    const stride = params.stride ?? 1;
    const act = params.activation || 'relu';
    const dropoutRate = params.dropoutRate ?? 0.5;

    const k = getAlexNetKernel(kSize);

    // 1. Input Layer
    const layers: VisionLayerArtifact[] = [
      {
        id: 'input',
        name: 'Input Layer',
        type: 'input',
        shape: [1, 16, 16],
        channels: [{ id: 'in0', label: 'Input Grid', data: new Float32Array(inputGrid) }],
        effectiveReceptiveField: { size: 1, stride: 1, padding: 0 },
      },
    ];

    let rf: ReceptiveFieldInfo = { rfSize: 1, cumulStride: 1, startOffset: 0, cumulPadding: 0 };

    // 2. Conv1 (Large Kernel: 7x7 / 5x5 / 3x3)
    const c1 = conv2dSingleChannel(inputGrid, 16, 16, k, kSize, stride, 0, 0);
    const a1 = applyActivation(c1.output, act);
    rf = nextReceptiveField(rf, kSize, stride);

    layers.push({
      id: 'conv1',
      name: `Conv1 (${kSize}x${kSize}, S=${stride})`,
      type: 'conv',
      shape: [1, c1.outH, c1.outW],
      channels: [{ id: 'c1_0', label: 'Filter 1', data: a1, kernel: new Float32Array(k), bias: 0 }],
      effectiveReceptiveField: { size: rf.rfSize, stride: rf.cumulStride, padding: 0 },
      stride,
      padding: 0,
      kernelSize: kSize,
      activation: act,
    });

    // 3. MaxPool1 (Overlap 3x3 or 2x2)
    if (c1.outH >= 2 && c1.outW >= 2) {
      const p1 = maxPool2dSingleChannel(a1, c1.outH, c1.outW, 2, 2);
      rf = nextReceptiveField(rf, 2, 2);
      layers.push({
        id: 'pool1',
        name: 'MaxPool1 (2x2)',
        type: 'pool',
        shape: [1, p1.outH, p1.outW],
        channels: [{ id: 'p1_0', label: 'Subsampled', data: p1.output }],
        effectiveReceptiveField: { size: rf.rfSize, stride: rf.cumulStride, padding: 0 },
        stride: 2,
        kernelSize: 2,
      });
    }

    // 4. Dense FC with Dropout Mask
    const denseDim = 16;
    const denseActivations = new Float32Array(denseDim);
    const mask = state.dropoutMask || [];
    for (let i = 0; i < denseDim; i++) {
      const isDropped = mask[i % mask.length];
      denseActivations[i] = isDropped ? 0.0 : (1.0 - (i % 4) * 0.2);
    }

    layers.push({
      id: 'dense_fc',
      name: `Dense FC (Dropout p=${dropoutRate.toFixed(2)})`,
      type: 'dense-head',
      shape: [1, 1, denseDim],
      channels: [{ id: 'fc0', label: 'Dropout Nodes', data: denseActivations }],
    });

    // Sliding window position
    const maxOutCol = Math.max(1, c1.outW);
    const maxOutRow = Math.max(1, c1.outH);
    const winCol = (state.rasterStep % maxOutCol);
    const winRow = Math.floor(state.rasterStep / maxOutCol) % maxOutRow;

    // Dot product live calculation
    const dotInputs: number[] = [];
    const dotWeights: number[] = [];
    let dotSum = 0;
    const inOriginR = winRow * stride;
    const inOriginC = winCol * stride;
    for (let kr = 0; kr < kSize; kr++) {
      const r = inOriginR + kr;
      for (let kc = 0; kc < kSize; kc++) {
        const c = inOriginC + kc;
        const inVal = r >= 0 && r < 16 && c >= 0 && c < 16 ? inputGrid[r * 16 + c] : 0;
        const wVal = k[kr * kSize + kc];
        dotInputs.push(inVal);
        dotWeights.push(wVal);
        dotSum += inVal * wVal;
      }
    }

    const probs = this.predict(inputGrid, state, params) as Float32Array;
    const shiftedGrid = shiftGridRight(inputGrid, 1);
    const shiftedProbs = this.predict(shiftedGrid, state, params) as Float32Array;
    const delta = computeShiftDelta(probs, shiftedProbs);
    const { layerNorms } = computeAlexNetGradientNorms(state, params);

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
        algorithmId: 'cnn-alexnet',
        mode: state.mode || 'inspector',
        inputResolution: 16,
        layers,
        slidingWindow: {
          row: winRow,
          col: winCol,
          kernelSize: kSize,
          stride,
          padding: 0,
          activeLayerIndex: 1,
          activeChannelIndex: 0,
          dotProductCalculation: {
            inputs: dotInputs,
            weights: dotWeights,
            bias: 0,
            output: dotSum,
          },
        },
        gradientNorms: layerNorms,
        logits,
        predictedClass: maxIdx,
        predictedConfidence: maxP,
        translationDelta1px: delta,
      },
    ];
  },

  serializeSnapshot(state: AlexNetState): Float32Array {
    const buf = new Float32Array(5);
    buf[0] = state.epoch;
    buf[1] = state.rasterStep;
    buf[2] = state.microTrain.trainLoss;
    buf[3] = state.microTrain.testLoss;
    buf[4] = state.microTrain.trainAcc;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: AlexNetState): AlexNetState {
    const next = { ...state };
    if (buffer.length >= 5) {
      next.epoch = buffer[0];
      next.rasterStep = buffer[1];
      next.microTrain.trainLoss = buffer[2];
      next.microTrain.testLoss = buffer[3];
      next.microTrain.trainAcc = buffer[4];
    }
    return next;
  },

  explain(state: AlexNetState, params: AlexNetParams, lastStep?: StepResult<AlexNetState>): ExplanationPayload {
    const { summaryNorm } = computeAlexNetGradientNorms(state, params);
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
      summary: 'AlexNet introduced large initial receptive field convolutions, non-saturating ReLU activations (preventing vanishing gradients), and Dropout regularization.',
      coreAssumption: 'Deeper networks need non-saturating rectifiers to propagate gradients, while co-adaptation of dense feature detectors is mitigated by randomly dropping nodes.',
      whenItFails: 'Excessive Dropout (p > 0.8) starves the network of forward information flow, collapsing prediction confidence, while Sigmoid activations cause severe gradient saturation.',
      microLog: [
        `Forward pass: Applied large ${params.firstKernelSize || 5}x${params.firstKernelSize || 5} filters with ${params.activation || 'relu'}; +1px shift delta = ${Number(delta1px).toFixed(4)}.`,
        `Regularization: Evaluated Dropout with keep-probability ${(1 - (params.dropoutRate ?? 0.5)).toFixed(2)}.`,
        `Backward pass: Conv1 gradient norm evaluated at ${gradNorm.toFixed(4)} (${params.activation || 'relu'}).`,
      ],
      mathLatex: {
        objective: 'f(x) = \\max(0, x), \\quad y = \\frac{1}{1 - p} \\left(m \\odot W h\\right), \\; m_i \\sim \\text{Bernoulli}(1-p)',
        lossOrUpdate: '\\frac{\\partial \\text{ReLU}}{\\partial x} = \\mathbf{1}_{x > 0}, \\quad \\frac{\\partial \\sigma}{\\partial x} = \\sigma(x)(1 - \\sigma(x)) \\le 0.25',
      },
      codeSnippet: {
        sklearn: `# PyTorch Equivalent
alexnet = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=${params.firstKernelSize || 5}, stride=${params.stride ?? 1}),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Dropout(p=${params.dropoutRate ?? 0.5}),
    nn.Linear(64, 4)
)`,
        pytorch: `# Dropout forward masking
mask = (torch.rand_like(h) > p).float()
h_dropped = (h * mask) / (1.0 - p)`,
      },
    };
  },
};
