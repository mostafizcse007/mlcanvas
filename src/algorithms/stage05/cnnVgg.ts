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
  applyActivation,
  softmax,
  computeShiftDelta,
  computeOutputDim,
  KERNEL_SOBEL_X_3X3,
  KERNEL_SOBEL_Y_3X3,
  KERNEL_LAPLACIAN_3X3,
  KERNEL_CORNER_3X3,
  KERNEL_BOX_BLUR_3X3,
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

export interface VggParams {
  stackedDepth?: number; // 1..6
  padding?: 'same-1' | 'valid-0';
  activation?: 'relu' | 'sigmoid';
  stride?: number;
  mode?: 'inspector' | 'micro-training';
}

export interface VggState {
  epoch: number;
  mode: 'inspector' | 'micro-training';
  microTrain: MicroTrainState;
  rasterStep: number;
}

const PRIMARY_PARAMS: HyperparameterDef[] = [
  {
    id: 'stackedDepth',
    name: 'Stacked 3x3 Depth (L)',
    type: 'slider',
    min: 1,
    max: 6,
    step: 1,
    default: 3,
    description: 'Number of consecutive 3x3 conv layers. Each +1 layer expands effective receptive field by +2 pixels.',
  },
  {
    id: 'padding',
    name: 'Padding (P)',
    type: 'select',
    default: 'same-1',
    options: [
      { label: 'Same Zero-Padding (P=1, Preserves Size)', value: 'same-1' },
      { label: 'Valid / None (P=0, Spatial Collapse)', value: 'valid-0' },
    ],
    description: 'P=1 preserves 16x16 across all layers; P=0 shrinks width by -2 pixels per layer.',
  },
  {
    id: 'activation',
    name: 'Activation Function',
    type: 'select',
    default: 'relu',
    options: [
      { label: 'ReLU (Non-Saturating)', value: 'relu' },
      { label: 'Sigmoid (Severe Vanishing Gradient)', value: 'sigmoid' },
    ],
    description: 'Nonlinear activation. Stacking 6 layers with Sigmoid causes deep gradient vanishing.',
  },
];

const ADVANCED_PARAMS: HyperparameterDef[] = [
  {
    id: 'stride',
    name: 'Conv Stride (S)',
    type: 'slider',
    min: 1,
    max: 2,
    step: 1,
    default: 1,
    description: 'Convolution stride. Standard VGG uses stride 1 for stacked feature extraction.',
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

const VGG_KERNELS = [
  KERNEL_SOBEL_X_3X3,
  KERNEL_SOBEL_Y_3X3,
  KERNEL_LAPLACIAN_3X3,
  KERNEL_CORNER_3X3,
  KERNEL_BOX_BLUR_3X3,
  KERNEL_SOBEL_X_3X3,
];

function computeVggGradientNorms(state: VggState, params: VggParams): {
  layerNorms: { layerName: string; norm: number }[];
  summaryNorm: number;
} {
  const depth = Math.min(6, Math.max(1, Number(params.stackedDepth || 3)));
  const pad = params.padding === 'valid-0' ? 0 : 1;
  const stride = params.stride ?? 1;
  const isSigmoid = params.activation === 'sigmoid';
  const decayFactor = isSigmoid ? 0.25 : 0.78;
  const lossFactor = Math.min(1.0, state.microTrain.trainLoss / 1.38 + 0.15);

  const layerNorms: { layerName: string; norm: number }[] = [];
  let currDim = 16;
  for (let l = 1; l <= depth; l++) {
    if (currDim < 3) break;
    const outDim = computeOutputDim(currDim, 3, stride, pad);
    const layerDistFromEnd = depth - l;
    const norm = Number((Math.max(0.0005, Math.pow(decayFactor, layerDistFromEnd + 1) * 0.28 * lossFactor)).toFixed(4));
    layerNorms.push({
      layerName: `Layer ${l} (${outDim}x${outDim})`,
      norm,
    });
    currDim = outDim;
  }

  return {
    layerNorms,
    summaryNorm: layerNorms[0]?.norm ?? 0.0005,
  };
}

export const cnnVggPlugin: AlgorithmPlugin<VggState, VggParams> = {
  id: 'cnn-vgg',
  name: 'VGG (Deep Stacked 3x3 Receptive Fields)',
  shortCode: 'VGG',
  stage: 'vision-grid',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Simonyan & Zisserman’s 2014 architecture proving that deep stacks of small 3x3 filters yield large effective receptive fields with dramatically fewer parameters and greater non-linearity.',
  paperReference: 'Simonyan & Zisserman, Very Deep Convolutional Networks for Large-Scale Image Recognition, ICLR (2015)',

  primaryParams: PRIMARY_PARAMS,
  advancedParams: ADVANCED_PARAMS,

  init(_dataset: DatasetContainer, _params: VggParams, seed: number = 42): VggState {
    return {
      epoch: 0,
      mode: 'micro-training',
      microTrain: initMicroTrainState(seed),
      rasterStep: 0,
    };
  },

  step(state: VggState, dataset: DatasetContainer, params: VggParams): StepResult<VggState> {
    const trainSamples = dataset.spatialVision?.trainSamples || [];
    const testSamples = dataset.spatialVision?.testSamples || [];
    const lr = Number((params as any)?.learningRate ?? (params as any)?.lr ?? 0.03);
    const depth = Math.min(6, Math.max(1, Number(params.stackedDepth || 3)));
    const isSigmoid = params.activation === 'sigmoid';
    const decayFactor = isSigmoid ? 0.35 : 0.88;
    const effectiveLr = lr * Math.pow(decayFactor, depth - 1);

    const nextMicro = stepMicroTrainEpoch(state.microTrain, trainSamples, testSamples, effectiveLr);

    const nextState: VggState = {
      ...state,
      epoch: state.epoch + 1,
      microTrain: nextMicro,
      rasterStep: state.rasterStep + 1,
    };

    const metrics = this.loss(nextState, dataset, params);
    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
    };
  },

  predict(input: Float32Array, state?: VggState, _params?: VggParams): Float32Array {
    const grid = input.length === 256 ? input : getPresetGrid('digit');
    if (!state) {
      return new Float32Array(softmax(computeVisualClassLogits(grid, 16, 16, grid)));
    }
    return predictHybrid(grid, state.microTrain, state.epoch);
  },

  loss(state: VggState, dataset: DatasetContainer, params: VggParams): TelemetryMetrics {
    const depth = Math.min(6, Math.max(1, Number(params.stackedDepth || 3)));
    const isSigmoid = params.activation === 'sigmoid';

    const baseGrid = dataset.spatialVision?.inputGrid || getPresetGrid('digit');
    const shiftedGrid = shiftGridRight(baseGrid, 1);
    const basePred = this.predict(baseGrid, state, params) as Float32Array;
    const shiftedPred = this.predict(shiftedGrid, state, params) as Float32Array;
    const delta = computeShiftDelta(basePred, shiftedPred);

    const { summaryNorm: l1Grad } = computeVggGradientNorms(state, params);
    const rfSize = 1 + 2 * depth;
    const weights3x3 = depth * 9;
    const weightsMono = rfSize * rfSize;
    const savingsPct = ((weightsMono - weights3x3) / weightsMono) * 100;

    const isSeverelyDegraded = depth >= 5 && isSigmoid;

    return {
      trainLoss: Number((state.microTrain.trainLoss + (isSeverelyDegraded ? 0.75 : 0.0)).toFixed(4)),
      testLoss: Number((state.microTrain.testLoss + (isSeverelyDegraded ? 0.80 : 0.0)).toFixed(4)),
      trainAccuracy: Number(Math.max(25, state.microTrain.trainAcc - (isSeverelyDegraded ? 40 : 0)).toFixed(1)),
      testAccuracy: Number(Math.max(25, state.microTrain.testAcc - (isSeverelyDegraded ? 40 : 0)).toFixed(1)),
      gradientNorm: l1Grad,
      custom: {
        effectiveReceptiveField: `${rfSize}x${rfSize} px`,
        translationDelta1px: delta,
        parameterCount: `${weights3x3} weights vs ${weightsMono} in ${rfSize}x${rfSize} (+${savingsPct.toFixed(1)}% savings)`,
        l1GradientNorm: l1Grad.toFixed(4),
      },
    };
  },

  getVisualArtifacts(state: VggState, dataset: DatasetContainer, params: VggParams): VisualArtifact[] {
    const inputGrid = dataset.spatialVision?.inputGrid || getPresetGrid('digit');
    const depth = Math.min(6, Math.max(1, Number(params.stackedDepth || 3)));
    const pad = params.padding === 'valid-0' ? 0 : 1;
    const act = params.activation || 'relu';
    const stride = params.stride ?? 1;

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
    let currData = inputGrid;
    let currH = 16;
    let currW = 16;

    const { layerNorms: gradientNorms } = computeVggGradientNorms(state, params);

    for (let l = 1; l <= depth; l++) {
      if (currH < 3 || currW < 3) break;
      const k = VGG_KERNELS[(l - 1) % VGG_KERNELS.length];
      const res = conv2dSingleChannel(currData, currH, currW, k, 3, stride, pad, 0);
      const aOut = applyActivation(res.output, act);

      rf = nextReceptiveField(rf, 3, stride, pad);

      layers.push({
        id: `vgg_conv${l}`,
        name: `Layer ${l}: 3x3 Conv (${res.outH}x${res.outW})`,
        type: 'conv',
        shape: [1, res.outH, res.outW],
        channels: [{ id: `c${l}_0`, label: `Filter ${l}`, data: aOut, kernel: new Float32Array(k), bias: 0 }],
        effectiveReceptiveField: { size: rf.rfSize, stride: rf.cumulStride, padding: pad },
        stride,
        padding: pad,
        kernelSize: 3,
        activation: act,
      });

      currData = aOut;
      currH = res.outH;
      currW = res.outW;
    }

    // Raster step for sliding window in active layer (last conv)
    const lastLayer = layers[layers.length - 1];
    const outH = lastLayer.shape[1];
    const outW = lastLayer.shape[2];
    const maxOutCol = Math.max(1, outW);
    const maxOutRow = Math.max(1, outH);
    const winCol = (state.rasterStep % maxOutCol);
    const winRow = Math.floor(state.rasterStep / maxOutCol) % maxOutRow;

    // Dot product live calculation
    const activeKernel = VGG_KERNELS[(layers.length - 2) % VGG_KERNELS.length];
    const prevLayerData = layers[layers.length - 2]?.channels[0]?.data || inputGrid;
    const prevLayerH = layers[layers.length - 2]?.shape[1] || 16;
    const prevLayerW = layers[layers.length - 2]?.shape[2] || 16;
    const dotInputs: number[] = [];
    const dotWeights: number[] = [];
    let dotSum = 0;
    const inOriginR = winRow * stride - pad;
    const inOriginC = winCol * stride - pad;
    for (let kr = 0; kr < 3; kr++) {
      const r = inOriginR + kr;
      for (let kc = 0; kc < 3; kc++) {
        const c = inOriginC + kc;
        const inVal = r >= 0 && r < prevLayerH && c >= 0 && c < prevLayerW ? prevLayerData[r * prevLayerW + c] : 0;
        const wVal = activeKernel[kr * 3 + kc];
        dotInputs.push(inVal);
        dotWeights.push(wVal);
        dotSum += inVal * wVal;
      }
    }

    const probs = this.predict(inputGrid, state, params) as Float32Array;
    const shiftedGrid = shiftGridRight(inputGrid, 1);
    const shiftedProbs = this.predict(shiftedGrid, state, params) as Float32Array;
    const delta = computeShiftDelta(probs, shiftedProbs);

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
        algorithmId: 'cnn-vgg',
        mode: state.mode || 'inspector',
        inputResolution: 16,
        layers,
        slidingWindow: {
          row: winRow,
          col: winCol,
          kernelSize: 3,
          stride,
          padding: pad,
          activeLayerIndex: layers.length - 1,
          activeChannelIndex: 0,
          dotProductCalculation: {
            inputs: dotInputs,
            weights: dotWeights,
            bias: 0,
            output: dotSum,
          },
        },
        gradientNorms,
        logits,
        predictedClass: maxIdx,
        predictedConfidence: maxP,
        translationDelta1px: delta,
      },
    ];
  },

  serializeSnapshot(state: VggState): Float32Array {
    const buf = new Float32Array(5);
    buf[0] = state.epoch;
    buf[1] = state.rasterStep;
    buf[2] = state.microTrain.trainLoss;
    buf[3] = state.microTrain.testLoss;
    buf[4] = state.microTrain.trainAcc;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: VggState): VggState {
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

  explain(state: VggState, params: VggParams, lastStep?: StepResult<VggState>): ExplanationPayload {
    const depth = Math.min(6, Math.max(1, Number(params.stackedDepth || 3)));
    const rf = 1 + 2 * depth;
    const weights3x3 = depth * 9;
    const weightsMono = rf * rf;
    const savingsPct = ((weightsMono - weights3x3) / weightsMono) * 100;
    const { summaryNorm } = computeVggGradientNorms(state, params);
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
      summary: `VGG demonstrates that stacking ${depth} layers of 3x3 filters yields a ${rf}x${rf} effective receptive field using only ${weights3x3} parameters instead of ${weightsMono} in a single large filter.`,
      coreAssumption: 'Multiple stacked non-linear 3x3 stages are more expressive and parameter-efficient than monolithic wide filters.',
      whenItFails: 'Omitting zero-padding (P=0) causes rapid spatial resolution collapse (16 -> 14 -> 12 -> 10 -> 8 -> 6 -> 4), while stacking beyond 5 plain layers without skip connections causes early layer gradients to vanish.',
      microLog: [
        `Receptive Field: Stacked ${depth} layers yielding ${rf}x${rf} input coverage (+1px shift delta = ${Number(delta1px).toFixed(4)}).`,
        `Parameter Efficiency: Uses ${weights3x3} weights vs ${weightsMono} in ${rf}x${rf} (saving +${savingsPct.toFixed(1)}%).`,
        `Gradient Survival: Layer 1 gradient norm evaluated at ${gradNorm.toFixed(4)}.`,
      ],
      mathLatex: {
        objective: 'R_L = 1 + \\sum_{l=1}^L (K_l - 1) = 1 + 2L, \\quad \\text{Weights: } L \\cdot (3^2 \\cdot C^2) \\ll (2L+1)^2 \\cdot C^2',
        lossOrUpdate: 'O_l = \\left\\lfloor \\frac{W_l - 3 + 2P}{S} \\right\\rfloor + 1, \\quad \\frac{\\partial \\mathcal{L}}{\\partial W^{(1)}} = \\frac{\\partial \\mathcal{L}}{\\partial W^{(L)}} \\prod_{l=1}^{L-1} W^{(l)} \\sigma\'(z^{(l)})',
      },
      codeSnippet: {
        sklearn: `# PyTorch VGG Block (Stacked 3x3)
vgg_block = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.MaxPool2d(kernel_size=2, stride=2)
)`,
        pytorch: `# Receptive Field growth calculation
rf = 1 + 2 * num_layers`,
      },
    };
  },
};
