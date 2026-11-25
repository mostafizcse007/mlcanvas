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
  KERNEL_SOBEL_X_3X3,
  KERNEL_SOBEL_Y_3X3,
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

export interface ResNetParams {
  residualBlocks?: number; // 2..8
  skipConnection?: 'enabled' | 'disabled';
  residualScale?: number; // 0.1..1.0
  activation?: 'relu' | 'tanh';
  mode?: 'inspector' | 'micro-training';
}

export interface ResNetState {
  epoch: number;
  mode: 'inspector' | 'micro-training';
  microTrain: MicroTrainState;
  rasterStep: number;
}

const PRIMARY_PARAMS: HyperparameterDef[] = [
  {
    id: 'residualBlocks',
    name: 'Residual Blocks (N)',
    type: 'slider',
    min: 2,
    max: 8,
    step: 1,
    default: 4,
    description: 'Number of stacked residual blocks. Each block computes H(x) = F(x) + x.',
  },
  {
    id: 'skipConnection',
    name: 'Identity Skip Highway',
    type: 'select',
    default: 'enabled',
    options: [
      { label: 'Enabled: H(x) = F(x) + x (Gradient Highway)', value: 'enabled' },
      { label: 'Disabled: Plain H(x) = F(x) (Vanishing Gradients)', value: 'disabled' },
    ],
    description: 'Toggles the identity addition shortcut bypassing the conv layers.',
  },
  {
    id: 'residualScale',
    name: 'Residual Scale (α)',
    type: 'slider',
    min: 0.1,
    max: 1.0,
    step: 0.1,
    default: 1.0,
    description: 'Scaling factor applied to residual delta F(x): H(x) = α·F(x) + x.',
  },
];

const ADVANCED_PARAMS: HyperparameterDef[] = [
  {
    id: 'activation',
    name: 'Block Activation',
    type: 'select',
    default: 'relu',
    options: [
      { label: 'ReLU Rectifier', value: 'relu' },
      { label: 'Tanh Non-Linearity', value: 'tanh' },
    ],
    description: 'Nonlinear activation function applied within the residual function F(x).',
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

function computeResNetGradientNorms(state: ResNetState, params: ResNetParams): {
  layerNorms: { layerName: string; norm: number }[];
  summaryNorm: number;
} {
  const numBlocks = Math.min(8, Math.max(2, Number(params.residualBlocks || 4)));
  const isSkip = params.skipConnection !== 'disabled';
  const lossScale = Math.min(1.0, state.microTrain.trainLoss / 0.8 + 0.25);

  const layerNorms: { layerName: string; norm: number }[] = [];
  for (let b = 1; b <= numBlocks; b++) {
    const layerDistFromEnd = numBlocks - b;
    const norm = isSkip
      ? Number(((0.92 + 0.08 * (b / numBlocks)) * lossScale).toFixed(4))
      : Math.max(0.0001, Number((Math.pow(0.48, layerDistFromEnd + 1) * 0.5 * lossScale).toFixed(5)));
    layerNorms.push({
      layerName: `Block ${b} Output`,
      norm,
    });
  }

  return {
    layerNorms,
    summaryNorm: layerNorms[0]?.norm ?? 0.0001,
  };
}

export const cnnResnetPlugin: AlgorithmPlugin<ResNetState, ResNetParams> = {
  id: 'cnn-resnet',
  name: 'ResNet (Residual Highway H(x) = F(x) + x)',
  shortCode: 'ResNet',
  stage: 'vision-grid',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'He et al.’s 2015 landmark paper introducing identity shortcut mappings that solved the deep degradation problem, allowing networks to scale to 152+ layers.',
  paperReference: 'He et al., Deep Residual Learning for Image Recognition, CVPR (2016)',

  primaryParams: PRIMARY_PARAMS,
  advancedParams: ADVANCED_PARAMS,

  init(_dataset: DatasetContainer, _params: ResNetParams, seed: number = 42): ResNetState {
    return {
      epoch: 0,
      mode: 'micro-training',
      microTrain: initMicroTrainState(seed),
      rasterStep: 0,
    };
  },

  step(state: ResNetState, dataset: DatasetContainer, params: ResNetParams): StepResult<ResNetState> {
    const trainSamples = dataset.spatialVision?.trainSamples || [];
    const testSamples = dataset.spatialVision?.testSamples || [];
    const lr = Number((params as any)?.learningRate ?? (params as any)?.lr ?? 0.03);
    const numBlocks = Math.min(8, Math.max(2, Number(params.residualBlocks || 4)));
    const isSkip = params.skipConnection !== 'disabled';
    const effectiveLr = isSkip ? lr : lr * Math.pow(0.25, numBlocks / 2);

    const nextMicro = stepMicroTrainEpoch(state.microTrain, trainSamples, testSamples, effectiveLr);

    const nextState: ResNetState = {
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

  predict(input: Float32Array, state?: ResNetState, params?: ResNetParams): Float32Array {
    const grid = input.length === 256 ? input : getPresetGrid('digit');
    if (!state) {
      return new Float32Array(softmax(computeVisualClassLogits(grid, 16, 16, grid)));
    }
    const numBlocks = Math.min(8, Math.max(2, Number(params?.residualBlocks || 4)));
    const isSkipEnabled = params?.skipConnection !== 'disabled';
    const energyMod = !isSkipEnabled && numBlocks >= 6 ? Math.pow(0.55, numBlocks - 4) : 1.0;
    const base = predictHybrid(grid, state.microTrain, state.epoch);
    const logits = Array.from(base).map((l) => l * energyMod);
    return new Float32Array(softmax(logits));
  },

  loss(state: ResNetState, dataset: DatasetContainer, params: ResNetParams): TelemetryMetrics {
    const numBlocks = Math.min(8, Math.max(2, Number(params.residualBlocks || 4)));
    const isSkip = params.skipConnection !== 'disabled';

    const baseGrid = dataset.spatialVision?.inputGrid || getPresetGrid('digit');
    const shiftedGrid = shiftGridRight(baseGrid, 1);
    const basePred = this.predict(baseGrid, state, params) as Float32Array;
    const shiftedPred = this.predict(shiftedGrid, state, params) as Float32Array;
    const delta = computeShiftDelta(basePred, shiftedPred);

    const { summaryNorm: gradNorm } = computeResNetGradientNorms(state, params);

    const rfSize = Math.min(16, 3 + 4 * numBlocks);
    const isSeverelyVanish = !isSkip && numBlocks >= 6;

    return {
      trainLoss: Number((state.microTrain.trainLoss + (isSeverelyVanish ? 0.85 : 0.0)).toFixed(4)),
      testLoss: Number((state.microTrain.testLoss + (isSeverelyVanish ? 0.90 : 0.0)).toFixed(4)),
      trainAccuracy: Number(Math.max(25, state.microTrain.trainAcc - (isSeverelyVanish ? 45 : 0)).toFixed(1)),
      testAccuracy: Number(Math.max(25, state.microTrain.testAcc - (isSeverelyVanish ? 45 : 0)).toFixed(1)),
      gradientNorm: gradNorm,
      custom: {
        effectiveReceptiveField: `${rfSize}x${rfSize} px`,
        translationDelta1px: delta,
        skipConnection: isSkip ? 'Identity Highway Active (+ x)' : 'SEVERED (Plain F(x))',
        earlyLayerGradientNorm: gradNorm.toFixed(5),
        gradientHighwayBenefit: isSkip ? '100% Signal Preserved' : `${(gradNorm * 100).toFixed(2)}% of Final Gradient`,
      },
    };
  },

  getVisualArtifacts(state: ResNetState, dataset: DatasetContainer, params: ResNetParams): VisualArtifact[] {
    const inputGrid = dataset.spatialVision?.inputGrid || getPresetGrid('digit');
    const numBlocks = Math.min(8, Math.max(2, Number(params.residualBlocks || 4)));
    const isSkip = params.skipConnection !== 'disabled';
    const scale = params.residualScale ?? 1.0;
    const act = params.activation || 'relu';

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

    let x: Float32Array = new Float32Array(inputGrid);
    const skipConnections: NonNullable<Extract<VisualArtifact, { type: 'spatial-vision-pipeline' }>['resnetSkipConnections']> = [];
    const { layerNorms: gradientNorms } = computeResNetGradientNorms(state, params);

    // Conv Stem
    const stem = conv2dSingleChannel(x, 16, 16, KERNEL_SOBEL_X_3X3, 3, 1, 1, 0);
    x = applyActivation(stem.output, act);
    layers.push({
      id: 'stem_conv',
      name: 'Conv Stem (3x3)',
      type: 'conv',
      shape: [1, 16, 16],
      channels: [{ id: 'stem0', label: 'Stem Channel', data: x, kernel: new Float32Array(KERNEL_SOBEL_X_3X3), bias: 0 }],
      effectiveReceptiveField: { size: 3, stride: 1, padding: 1 },
      stride: 1,
      padding: 1,
      kernelSize: 3,
    });

    let rf: ReceptiveFieldInfo = { rfSize: 3, cumulStride: 1, startOffset: 0, cumulPadding: 0 };

    for (let b = 1; b <= numBlocks; b++) {
      const sourceLayerIdx = layers.length - 1;
      const xInput = new Float32Array(x);

      // Residual branch F(x)
      const k = b % 2 === 1 ? KERNEL_SOBEL_Y_3X3 : KERNEL_SOBEL_X_3X3;
      const c1 = conv2dSingleChannel(x, 16, 16, k, 3, 1, 1, 0);
      const a1 = applyActivation(c1.output, act);
      const c2 = conv2dSingleChannel(a1, 16, 16, k, 3, 1, 1, 0);
      const fx = c2.output;

      const sum = new Float32Array(256);
      for (let i = 0; i < 256; i++) {
        sum[i] = isSkip ? fx[i] * scale + xInput[i] : fx[i] * scale;
      }
      x = applyActivation(sum, act);

      rf = nextReceptiveField(rf, 3, 1, 1);
      rf = nextReceptiveField(rf, 3, 1, 1);

      layers.push({
        id: `res_block_${b}`,
        name: `Block ${b}: F(x) ${isSkip ? '+ x' : '(No Skip)'}`,
        type: 'skip-add',
        shape: [1, 16, 16],
        channels: [{ id: `block_${b}_out`, label: `Block ${b} Output`, data: x }],
        effectiveReceptiveField: { size: rf.rfSize, stride: 1, padding: 1 },
      });

      const targetLayerIdx = layers.length - 1;
      skipConnections.push({
        sourceLayerIndex: sourceLayerIdx,
        targetLayerIndex: targetLayerIdx,
        enabled: isSkip,
        xMap: xInput,
        fxMap: fx,
        sumMap: sum,
      });
    }

    const winCol = state.rasterStep % 16;
    const winRow = Math.floor(state.rasterStep / 16) % 16;

    const dotInputs: number[] = [];
    const dotWeights: number[] = [];
    let dotSum = 0;
    const inOriginR = winRow - 1;
    const inOriginC = winCol - 1;
    for (let kr = 0; kr < 3; kr++) {
      const r = inOriginR + kr;
      for (let kc = 0; kc < 3; kc++) {
        const c = inOriginC + kc;
        const inVal = r >= 0 && r < 16 && c >= 0 && c < 16 ? inputGrid[r * 16 + c] : 0;
        const wVal = KERNEL_SOBEL_X_3X3[kr * 3 + kc];
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
        algorithmId: 'cnn-resnet',
        mode: state.mode || 'inspector',
        inputResolution: 16,
        layers,
        slidingWindow: {
          row: winRow,
          col: winCol,
          kernelSize: 3,
          stride: 1,
          padding: 1,
          activeLayerIndex: 1,
          activeChannelIndex: 0,
          dotProductCalculation: {
            inputs: dotInputs,
            weights: dotWeights,
            bias: 0,
            output: dotSum,
          },
        },
        resnetSkipConnections: skipConnections,
        gradientNorms,
        logits,
        predictedClass: maxIdx,
        predictedConfidence: maxP,
        translationDelta1px: delta,
      },
    ];
  },

  serializeSnapshot(state: ResNetState): Float32Array {
    const buf = new Float32Array(5);
    buf[0] = state.epoch;
    buf[1] = state.rasterStep;
    buf[2] = state.microTrain.trainLoss;
    buf[3] = state.microTrain.testLoss;
    buf[4] = state.microTrain.trainAcc;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: ResNetState): ResNetState {
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

  explain(state: ResNetState, params: ResNetParams, lastStep?: StepResult<ResNetState>): ExplanationPayload {
    const isSkip = params.skipConnection !== 'disabled';
    const numBlocks = Math.min(8, Math.max(2, Number(params.residualBlocks || 4)));
    const { summaryNorm } = computeResNetGradientNorms(state, params);
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
      summary: `ResNet reformulates deep layers to fit residual mappings H(x) = F(x) + x. The explicit identity highway ensures gradients flow unimpeded directly back to layer 1: ∂L/∂x = ∂L/∂y · (1 + ∂F/∂x).`,
      coreAssumption: 'Optimizing residual delta F(x) -> 0 is much easier for SGD than fitting unconstrained full transformations H(x) from scratch.',
      whenItFails: 'Disabling the skip connection converts ResNet into a plain deep CNN, causing early layer gradients to vanish exponentially with depth.',
      microLog: [
        `Architecture: Stacked ${numBlocks} residual blocks with identity shortcuts ${isSkip ? 'ENABLED' : 'DISABLED'} (+1px shift delta = ${Number(delta1px).toFixed(4)}).`,
        `Gradient Highway: Early layer gradient survival evaluated at ${isSkip ? gradNorm.toFixed(4) : gradNorm.toFixed(5)}.`,
        'Residual summation: F(x) delta added directly to input identity x via element-wise addition node (⊕).',
      ],
      mathLatex: {
        objective: 'H(x) = \\mathcal{F}(x, \\{W_i\\}) + x, \\quad x_{l+1} = x_l + \\mathcal{F}(x_l, W_l)',
        lossOrUpdate: '\\frac{\\partial \\mathcal{E}}{\\partial x_l} = \\frac{\\partial \\mathcal{E}}{\\partial x_L} \\frac{\\partial x_L}{\\partial x_l} = \\frac{\\partial \\mathcal{E}}{\\partial x_L} \\left( 1 + \\frac{\\partial}{\\partial x_l} \\sum_{i=l}^{L-1} \\mathcal{F}(x_i, W_i) \\right)',
      },
      codeSnippet: {
        sklearn: `# PyTorch Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual # Identity Skip Highway!
        return self.relu(out)`,
        pytorch: `# Identity gradient highway preservation
grad_in = grad_out * (1.0 + grad_residual)`,
      },
    };
  },
};
