import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import {
  ActivationType,
  OptimizerType,
  WeightInitType,
  BasisFeature,
  BASIS_FEATURES,
  computeFeature,
  activate,
  activateDerivative,
  initializeWeights,
  MINI_COUNT,
  MINI_COORDS_X1,
  MINI_COORDS_X2,
} from './neuralUtils';
import { mulberry32 } from '../../utils/prng';

export interface MLPParams {
  problemType?: 'classification' | 'regression';
  learningRate?: number;
  activation?: ActivationType;
  regularization?: 'none' | 'l1' | 'l2';
  regRate?: number;
  optimizer?: OptimizerType;
  batchSize?: number | 'full';
  weightInit?: WeightInitType;
  numLayers?: number;
  layer1?: number;
  layer2?: number;
  layer3?: number;
  layer4?: number;
  features?: string; // e.g. 'x1,x2'
  lesioned?: string; // e.g. '0-0->1-0,0-1->1-0'
}

export interface MLPState {
  layerSizes: number[];
  weights: Float32Array[]; // weights[l] between layer l and l+1
  biases: Float32Array[];  // biases[l] for layer l+1
  mW: Float32Array[];      // optimizer moment 1
  vW: Float32Array[];      // optimizer moment 2
  mB: Float32Array[];
  vB: Float32Array[];
  stepCount: number;
  epoch: number;
  lesionedSynapses: Set<string>;
  activeFeatures: BasisFeature[];
  lastGradNorms: number[];
  deadNeuronCount: number;
  // Pre-computed mini-heatmap arrays for each hidden neuron: [layerIdx][neuronIdx] -> Float32Array(576)
  miniHeatmaps: Float32Array[][];
  isRegression?: boolean;
  is1D?: boolean;
}

export function isMLPRegression(params: MLPParams, dataset?: DatasetContainer): boolean {
  if (params.problemType === 'regression') return true;
  if (params.problemType === 'classification') return false;
  if (dataset?.train && dataset.train.length > 0) {
    return dataset.train.some((p) => p.y !== 1 && p.y !== -1 && p.y !== 2);
  }
  return false;
}

export function isMLP1D(isRegression: boolean, points?: { x1: number; x2: number; y: number }[]): boolean {
  if (!isRegression || !points || points.length < 2) return false;
  return points.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);
}

function parseActiveFeatures(featuresStr?: string): BasisFeature[] {
  if (!featuresStr) return ['x1', 'x2'];
  const parts = featuresStr.split(',').map((s) => s.trim().toLowerCase());
  const valid = parts.filter((p): p is BasisFeature => (BASIS_FEATURES as readonly string[]).includes(p));
  return valid.length > 0 ? valid : ['x1', 'x2'];
}

function getLayerSizes(params: MLPParams, inputDim: number): number[] {
  const numHidden = Math.max(1, Math.min(4, Number(params.numLayers ?? 2)));
  const h1 = Math.max(1, Math.min(8, Number(params.layer1 ?? 4)));
  const h2 = Math.max(1, Math.min(8, Number(params.layer2 ?? 2)));
  const h3 = Math.max(1, Math.min(8, Number(params.layer3 ?? 2)));
  const h4 = Math.max(1, Math.min(8, Number(params.layer4 ?? 2)));

  const hidden = [h1, h2, h3, h4].slice(0, numHidden);
  return [inputDim, ...hidden, 1];
}

export function evaluateForward(
  inputFeats: Float32Array,
  state: MLPState,
  activation: ActivationType,
  layerOutputs?: Float32Array[],
  preActivations?: Float32Array[],
  isRegression?: boolean
): number {
  let prev = inputFeats;
  const numLayers = state.weights.length;

  for (let l = 0; l < numLayers; l++) {
    const fanIn = state.layerSizes[l];
    const fanOut = state.layerSizes[l + 1];
    const w = state.weights[l];
    const b = state.biases[l];
    const z = preActivations ? preActivations[l] : new Float32Array(fanOut);
    const a = layerOutputs ? layerOutputs[l] : new Float32Array(fanOut);

    const isOutput = l === numLayers - 1;

    for (let j = 0; j < fanOut; j++) {
      let sum = b[j];
      const offset = j * fanIn;
      for (let i = 0; i < fanIn; i++) {
        const synKey = `${l}-${i}->${l + 1}-${j}`;
        if (!state.lesionedSynapses.has(synKey)) {
          sum += prev[i] * w[offset + i];
        }
      }
      z[j] = sum;
      a[j] = isOutput
        ? isRegression
          ? sum
          : Math.tanh(sum)
        : activate(sum, activation);
    }
    prev = a;
  }

  return prev[0];
}

function updateMiniHeatmaps(state: MLPState, activation: ActivationType): void {
  const numHiddenLayers = state.weights.length - 1;
  const feats = state.activeFeatures;
  const inputDim = feats.length;

  // Temporary buffers for mini evaluation
  const curInputs = new Float32Array(inputDim);
  const layerOutputs: Float32Array[] = [];
  const preActs: Float32Array[] = [];
  for (let l = 0; l < state.weights.length; l++) {
    const fanOut = state.layerSizes[l + 1];
    layerOutputs.push(new Float32Array(fanOut));
    preActs.push(new Float32Array(fanOut));
  }

  // Pre-allocate mini-heatmaps if not present
  if (state.miniHeatmaps.length !== numHiddenLayers) {
    state.miniHeatmaps = [];
    for (let l = 0; l < numHiddenLayers; l++) {
      const neuronCount = state.layerSizes[l + 1];
      const layerMaps: Float32Array[] = [];
      for (let j = 0; j < neuronCount; j++) {
        layerMaps.push(new Float32Array(MINI_COUNT));
      }
      state.miniHeatmaps.push(layerMaps);
    }
  }

  // Evaluate across the 576 coordinates
  for (let ptIdx = 0; ptIdx < MINI_COUNT; ptIdx++) {
    const x1 = MINI_COORDS_X1[ptIdx];
    const x2 = MINI_COORDS_X2[ptIdx];

    for (let f = 0; f < inputDim; f++) {
      curInputs[f] = computeFeature(feats[f], x1, x2);
    }

    evaluateForward(curInputs, state, activation, layerOutputs, preActs, state.isRegression);

    for (let l = 0; l < numHiddenLayers; l++) {
      const neuronCount = state.layerSizes[l + 1];
      for (let j = 0; j < neuronCount; j++) {
        state.miniHeatmaps[l][j][ptIdx] = layerOutputs[l][j];
      }
    }
  }
}

export const mlpPlugin: AlgorithmPlugin<MLPState, MLPParams> = {
  id: 'mlp',
  name: 'Multi-Layer Perceptron (Deep Feedforward)',
  shortCode: 'MLP',
  stage: 'neural-graph',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Deep feedforward network with exact backpropagation, live neuron activation heatmaps, and adaptive optimizers.',
  paperReference: 'Rumelhart, Hinton, & Williams, Learning representations by back-propagating errors, Nature 1986',

  primaryParams: [
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'select',
      options: [
        { label: '0.003 (Fine)', value: 0.003 },
        { label: '0.01', value: 0.01 },
        { label: '0.03 (Standard)', value: 0.03 },
        { label: '0.1 (Fast)', value: 0.1 },
        { label: '0.3', value: 0.3 },
      ],
      default: 0.03,
      description: 'Step size for gradient descent parameter updates.',
    },
    {
      id: 'activation',
      name: 'Activation Function',
      type: 'select',
      options: [
        { label: 'ReLU (Piecewise Linear)', value: 'relu' },
        { label: 'Tanh (Smooth [-1, 1])', value: 'tanh' },
        { label: 'Sigmoid ([0, 1])', value: 'sigmoid' },
        { label: 'Leaky ReLU (α = 0.01)', value: 'leaky-relu' },
      ],
      default: 'tanh',
      description: 'Non-linear activation applied to hidden layer activations.',
    },
    {
      id: 'regularization',
      name: 'Regularization',
      type: 'select',
      options: [
        { label: 'None', value: 'none' },
        { label: 'L2 Ridge (Weight Decay)', value: 'l2' },
        { label: 'L1 Lasso (Sparse Synapses)', value: 'l1' },
      ],
      default: 'none',
      description: 'Weight penalty constraining parameter magnitudes.',
    },
  ],

  advancedParams: [
    {
      id: 'optimizer',
      name: 'Optimizer Algorithm',
      type: 'select',
      options: [
        { label: 'Adam (Adaptive Moments)', value: 'adam' },
        { label: 'Momentum (β = 0.9)', value: 'momentum' },
        { label: 'SGD (Mini-Batch)', value: 'sgd' },
      ],
      default: 'adam',
      description: 'Optimization algorithm updating synaptic weights.',
    },
    {
      id: 'weightInit',
      name: 'Weight Initialization',
      type: 'select',
      options: [
        { label: 'Xavier / Glorot (Uniform)', value: 'xavier' },
        { label: 'He / Kaiming (Normal for ReLU)', value: 'he' },
        { label: 'Zeros Trap (Symmetry Break Test)', value: 'zeros-trap' },
      ],
      default: 'xavier',
      description: 'Initialization distribution for weight matrices.',
    },
    {
      id: 'regRate',
      name: 'Regularization Rate (λ)',
      type: 'select',
      options: [
        { label: '0.0001', value: 0.0001 },
        { label: '0.001', value: 0.001 },
        { label: '0.01', value: 0.01 },
        { label: '0.1', value: 0.1 },
      ],
      default: 0.001,
      description: 'Strength of L1/L2 penalty on weights.',
    },
    {
      id: 'batchSize',
      name: 'Batch Size',
      type: 'select',
      options: [
        { label: '16', value: 16 },
        { label: '32', value: 32 },
        { label: '64', value: 64 },
        { label: 'Full Dataset', value: 'full' },
      ],
      default: 32,
      description: 'Number of training points evaluated per gradient update.',
    },
    {
      id: 'numLayers',
      name: 'Hidden Layers (1..4)',
      type: 'slider',
      min: 1,
      max: 4,
      step: 1,
      default: 2,
      description: 'Number of hidden layers in the deep network.',
    },
    {
      id: 'layer1',
      name: 'Layer 1 Neurons (1..8)',
      type: 'slider',
      min: 1,
      max: 8,
      step: 1,
      default: 4,
      description: 'Neuron count in first hidden layer.',
    },
    {
      id: 'layer2',
      name: 'Layer 2 Neurons (1..8)',
      type: 'slider',
      min: 1,
      max: 8,
      step: 1,
      default: 2,
      description: 'Neuron count in second hidden layer.',
    },
  ],

  init(_dataset: DatasetContainer, params: MLPParams, seed: number): MLPState {
    const rand = mulberry32(seed);
    const initType = params.weightInit ?? 'xavier';
    const isRegression = isMLPRegression(params, _dataset);
    const is1D = isMLP1D(isRegression, _dataset?.train);
    const parsedFeatures = parseActiveFeatures(params.features);
    const x2Dependent = new Set<BasisFeature>(['x2', 'x2_sq', 'x1_x2', 'sin_x2']);
    const filteredFeatures = is1D
      ? parsedFeatures.filter((f) => !x2Dependent.has(f))
      : parsedFeatures;
    const activeFeatures: BasisFeature[] = filteredFeatures.length > 0 ? filteredFeatures : ['x1'];
    const layerSizes = getLayerSizes(params, activeFeatures.length);

    const numTransitions = layerSizes.length - 1;
    const weights: Float32Array[] = [];
    const biases: Float32Array[] = [];
    const mW: Float32Array[] = [];
    const vW: Float32Array[] = [];
    const mB: Float32Array[] = [];
    const vB: Float32Array[] = [];

    for (let l = 0; l < numTransitions; l++) {
      const fanIn = layerSizes[l];
      const fanOut = layerSizes[l + 1];
      weights.push(initializeWeights(fanIn, fanOut, initType, rand));
      const b = new Float32Array(fanOut);
      // Small bias init or zero
      biases.push(b);
      mW.push(new Float32Array(fanIn * fanOut));
      vW.push(new Float32Array(fanIn * fanOut));
      mB.push(new Float32Array(fanOut));
      vB.push(new Float32Array(fanOut));
    }

    const lesionedSynapses = new Set<string>();
    if (params.lesioned) {
      params.lesioned.split(',').forEach((s) => s.trim() && lesionedSynapses.add(s.trim()));
    }

    const lastGradNorms = new Array(numTransitions).fill(0);

    const state: MLPState = {
      layerSizes,
      weights,
      biases,
      mW,
      vW,
      mB,
      vB,
      stepCount: 0,
      epoch: 0,
      lesionedSynapses,
      activeFeatures,
      lastGradNorms,
      deadNeuronCount: 0,
      miniHeatmaps: [],
      isRegression,
      is1D,
    };

    updateMiniHeatmaps(state, params.activation ?? 'tanh');
    return state;
  },

  step(state: MLPState, dataset: DatasetContainer, params: MLPParams): StepResult<MLPState> {
    const trainData = dataset.train;
    const N = trainData.length;
    if (N === 0) {
      return {
        state,
        epoch: state.epoch,
        metrics: { trainLoss: 0, testLoss: 0 },
      };
    }

    const lr = Number(params.learningRate ?? 0.03);
    const activation = params.activation ?? 'tanh';
    const optimizer = params.optimizer ?? 'adam';
    const regType = params.regularization ?? 'none';
    const regRate = Number(params.regRate ?? 0.001);
    const numTransitions = state.weights.length;
    const feats = state.activeFeatures;
    const inputDim = feats.length;

    // Mini-batch selection
    const batchSizeSetting = params.batchSize ?? 32;
    const batchSize =
      batchSizeSetting === 'full' ? N : Math.min(N, Math.max(4, Number(batchSizeSetting)));

    // Allocate gradient accumulators
    const gradW: Float32Array[] = [];
    const gradB: Float32Array[] = [];
    for (let l = 0; l < numTransitions; l++) {
      gradW.push(new Float32Array(state.weights[l].length));
      gradB.push(new Float32Array(state.biases[l].length));
    }

    // Dead neuron tracking for ReLU
    const isReLU = activation === 'relu';
    const deadCounts: number[] = new Array(state.layerSizes.slice(1, -1).reduce((a, b) => a + b, 0)).fill(0);

    // Stochastic mini-batch sampling
    const startIdx = (state.stepCount * batchSize) % N;

    const isRegression = state.isRegression ?? isMLPRegression(params, dataset);
    const is1D = state.is1D ?? isMLP1D(isRegression, trainData);
    state.isRegression = isRegression;
    state.is1D = is1D;

    for (let b = 0; b < batchSize; b++) {
      const ptIdx = (startIdx + b) % N;
      const pt = trainData[ptIdx];

      // Input layer: if 1D regression, x2 is the continuous label, not an independent input coordinate
      const inX2 = is1D ? 0 : pt.x2;
      const curInputs = new Float32Array(inputDim);
      for (let f = 0; f < inputDim; f++) {
        curInputs[f] = computeFeature(feats[f], pt.x1, inX2);
      }

      const pointA: Float32Array[] = [curInputs];
      const pointZ: Float32Array[] = [];

      let prevA = curInputs;
      let flatNeuronIdx = 0;

      for (let l = 0; l < numTransitions; l++) {
        const fanIn = state.layerSizes[l];
        const fanOut = state.layerSizes[l + 1];
        const w = state.weights[l];
        const bias = state.biases[l];
        const z = new Float32Array(fanOut);
        const a = new Float32Array(fanOut);
        const isOutput = l === numTransitions - 1;

        for (let j = 0; j < fanOut; j++) {
          let sum = bias[j];
          const offset = j * fanIn;
          for (let i = 0; i < fanIn; i++) {
            const synKey = `${l}-${i}->${l + 1}-${j}`;
            if (!state.lesionedSynapses.has(synKey)) {
              sum += prevA[i] * w[offset + i];
            }
          }
          z[j] = sum;
          a[j] = isOutput
            ? isRegression
              ? sum
              : Math.tanh(sum)
            : activate(sum, activation);

          if (!isOutput && isReLU) {
            if (sum <= 0) {
              deadCounts[flatNeuronIdx]++;
            }
            flatNeuronIdx++;
          }
        }
        pointZ.push(z);
        pointA.push(a);
        prevA = a;
      }

      // Target y: continuous for regression, {-1, +1} for classification
      const targetY = isRegression ? pt.y : pt.y > 0 ? 1.0 : -1.0;
      const pred = prevA[0];
      const error = pred - targetY; // derivative of 0.5 * (pred - y)^2

      // Backpropagation
      // Output delta
      const outA = pointA[numTransitions][0];
      let curDelta = new Float32Array([error * (isRegression ? 1.0 : 1 - outA * outA)]);

      for (let l = numTransitions - 1; l >= 0; l--) {
        const fanIn = state.layerSizes[l];
        const fanOut = state.layerSizes[l + 1];
        const w = state.weights[l];
        const lGradW = gradW[l];
        const lGradB = gradB[l];
        const lA = pointA[l];

        // Accumulate gradients
        for (let j = 0; j < fanOut; j++) {
          const dj = curDelta[j];
          lGradB[j] += dj;
          const offset = j * fanIn;
          for (let i = 0; i < fanIn; i++) {
            const synKey = `${l}-${i}->${l + 1}-${j}`;
            if (!state.lesionedSynapses.has(synKey)) {
              lGradW[offset + i] += dj * lA[i];
            }
          }
        }

        if (l > 0) {
          // Propagate delta to layer l
          const prevZ = pointZ[l - 1];
          const prevA = pointA[l];
          const prevDelta = new Float32Array(fanIn);

          for (let i = 0; i < fanIn; i++) {
            let sum = 0;
            for (let j = 0; j < fanOut; j++) {
              const synKey = `${l}-${i}->${l + 1}-${j}`;
              if (!state.lesionedSynapses.has(synKey)) {
                sum += w[j * fanIn + i] * curDelta[j];
              }
            }
            const deriv = activateDerivative(prevZ[i], prevA[i], activation);
            prevDelta[i] = sum * deriv;
          }
          curDelta = prevDelta;
        }
      }
    }

    // Average gradients & apply regularization
    const scale = 1.0 / batchSize;
    state.lastGradNorms = [];

    state.stepCount++;
    const t = state.stepCount;
    const beta1 = 0.9;
    const beta2 = 0.999;
    const eps = 1e-8;

    for (let l = 0; l < numTransitions; l++) {
      const gW = gradW[l];
      const gB = gradB[l];
      const w = state.weights[l];
      const b = state.biases[l];
      const mW = state.mW[l];
      const vW = state.vW[l];
      const mB = state.mB[l];
      const vB = state.vB[l];

      let sumSq = 0;

      for (let k = 0; k < gW.length; k++) {
        let grad = gW[k] * scale;

        // Regularization penalty gradient
        if (regType === 'l2') {
          grad += regRate * w[k];
        } else if (regType === 'l1') {
          grad += regRate * Math.sign(w[k]);
        }

        sumSq += grad * grad;

        if (optimizer === 'sgd') {
          w[k] -= lr * grad;
        } else if (optimizer === 'momentum') {
          mW[k] = 0.9 * mW[k] + grad;
          w[k] -= lr * mW[k];
        } else {
          // Adam
          mW[k] = beta1 * mW[k] + (1 - beta1) * grad;
          vW[k] = beta2 * vW[k] + (1 - beta2) * grad * grad;
          const mHat = mW[k] / (1 - Math.pow(beta1, t));
          const vHat = vW[k] / (1 - Math.pow(beta2, t));
          w[k] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
      }

      state.lastGradNorms.push(Math.sqrt(sumSq / Math.max(1, gW.length)));

      for (let j = 0; j < gB.length; j++) {
        const grad = gB[j] * scale;
        if (optimizer === 'sgd') {
          b[j] -= lr * grad;
        } else if (optimizer === 'momentum') {
          mB[j] = 0.9 * mB[j] + grad;
          b[j] -= lr * mB[j];
        } else {
          mB[j] = beta1 * mB[j] + (1 - beta1) * grad;
          vB[j] = beta2 * vB[j] + (1 - beta2) * grad * grad;
          const mHat = mB[j] / (1 - Math.pow(beta1, t));
          const vHat = vB[j] / (1 - Math.pow(beta2, t));
          b[j] -= (lr * mHat) / (Math.sqrt(vHat) + eps);
        }
      }
    }

    state.epoch++;
    // Dead neuron check (neurons where all batch points had activation 0)
    let deadTotal = 0;
    if (isReLU) {
      deadTotal = deadCounts.filter((c) => c === batchSize).length;
    }
    state.deadNeuronCount = deadTotal;

    // Update live 24x24 mini heatmaps for hidden neurons
    updateMiniHeatmaps(state, activation);

    const metrics = this.loss(state, dataset, params);

    return {
      state,
      epoch: state.epoch,
      metrics,
      converged: isRegression
        ? metrics.trainR2 !== undefined && metrics.trainR2 >= 0.98
        : metrics.trainAccuracy !== undefined && metrics.trainAccuracy >= 99.5,
    };
  },

  predict(input: Float32Array, state: MLPState, params: MLPParams): number {
    const isRegression = state.isRegression ?? isMLPRegression(params);
    const is1D = state.is1D ?? false;
    const feats = state.activeFeatures;
    const inputDim = feats.length;
    const curInputs = new Float32Array(inputDim);
    const inX2 = is1D ? 0 : input[1];
    for (let f = 0; f < inputDim; f++) {
      curInputs[f] = computeFeature(feats[f], input[0], inX2);
    }
    return evaluateForward(curInputs, state, params.activation ?? 'tanh', undefined, undefined, isRegression);
  },

  loss(state: MLPState, dataset: DatasetContainer, params: MLPParams): TelemetryMetrics {
    const activation = params.activation ?? 'tanh';
    const isRegression = state.isRegression ?? isMLPRegression(params, dataset);
    const is1D = state.is1D ?? isMLP1D(isRegression, dataset.train);
    const feats = state.activeFeatures;
    const inputDim = feats.length;
    const curInputs = new Float32Array(inputDim);

    const computeSetMetrics = (points: { x1: number; x2: number; y: number }[]) => {
      if (points.length === 0) return { loss: 0, accuracy: 0, r2: 0 };
      let totalLoss = 0;
      let correct = 0;
      let sumY = 0;
      for (const p of points) sumY += p.y;
      const meanY = sumY / points.length;
      let totalVar = 0;
      let totalSqErr = 0;

      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        const inX2 = is1D ? 0 : p.x2;
        for (let f = 0; f < inputDim; f++) {
          curInputs[f] = computeFeature(feats[f], p.x1, inX2);
        }
        const pred = evaluateForward(curInputs, state, activation, undefined, undefined, isRegression);
        const target = isRegression ? p.y : p.y > 0 ? 1.0 : -1.0;
        const err = pred - target;
        totalLoss += 0.5 * err * err;
        totalSqErr += err * err;
        totalVar += (p.y - meanY) * (p.y - meanY);

        if (!isRegression && ((pred >= 0 && target > 0) || (pred < 0 && target < 0))) {
          correct++;
        }
      }

      const r2 = totalVar > 1e-7 ? Math.max(-1, 1 - totalSqErr / totalVar) : 0;

      return {
        loss: totalLoss / points.length,
        accuracy: (correct / points.length) * 100,
        r2,
      };
    };

    const trainMetrics = computeSetMetrics(dataset.train);
    const testMetrics = computeSetMetrics(dataset.test);

    return {
      trainLoss: trainMetrics.loss,
      testLoss: testMetrics.loss,
      trainAccuracy: isRegression ? undefined : trainMetrics.accuracy,
      testAccuracy: isRegression ? undefined : testMetrics.accuracy,
      trainR2: isRegression ? trainMetrics.r2 : undefined,
      testR2: isRegression ? testMetrics.r2 : undefined,
      gradientNorm: state.lastGradNorms[0] ?? 0,
      custom: {
        deadNeurons: state.deadNeuronCount,
        layerGradNorms: state.lastGradNorms.map((g) => g.toFixed(4)).join(', '),
      },
    };
  },

  getVisualArtifacts(state: MLPState, _dataset: DatasetContainer, _params: MLPParams): VisualArtifact[] {
    const numLayers = state.layerSizes.length;
    const layers: Extract<VisualArtifact, { type: 'neural-graph-overlay' }>['layers'] = [];
    const synapses: Extract<VisualArtifact, { type: 'neural-graph-overlay' }>['synapses'] = [];

    // Construct layer & neuron representations
    for (let l = 0; l < numLayers; l++) {
      const neuronCount = state.layerSizes[l];
      const isInput = l === 0;
      const isOutput = l === numLayers - 1;
      const neurons: (typeof layers)[0]['neurons'] = [];

      for (let j = 0; j < neuronCount; j++) {
        const id = `L${l}N${j}`;
        let label = `N${j + 1}`;
        if (isInput) {
          label = state.activeFeatures[j] ?? `X${j + 1}`;
        } else if (isOutput) {
          label = 'Ŷ';
        }

        const bias = l > 0 ? state.biases[l - 1][j] : 0;
        let activationPreview: number[] | undefined;

        // Hidden layer neurons provide 24x24 mini preview
        if (!isInput && !isOutput && state.miniHeatmaps[l - 1] && state.miniHeatmaps[l - 1][j]) {
          activationPreview = Array.from(state.miniHeatmaps[l - 1][j]);
        }

        neurons.push({
          id,
          label,
          bias,
          activationPreview,
        });
      }

      layers.push({
        index: l,
        neuronCount,
        isInput,
        isOutput,
        neurons,
      });
    }

    // Construct synapse representations
    for (let l = 0; l < numLayers - 1; l++) {
      const fanIn = state.layerSizes[l];
      const fanOut = state.layerSizes[l + 1];
      const w = state.weights[l];
      const gradNorm = state.lastGradNorms[l] ?? 0.01;

      for (let j = 0; j < fanOut; j++) {
        const offset = j * fanIn;
        for (let i = 0; i < fanIn; i++) {
          const synKey = `${l}-${i}->${l + 1}-${j}`;
          const isLesioned = state.lesionedSynapses.has(synKey);
          const weight = isLesioned ? 0 : w[offset + i];

          synapses.push({
            id: synKey,
            sourceLayer: l,
            sourceNeuron: i,
            targetLayer: l + 1,
            targetNeuron: j,
            weight,
            gradientMagnitude: gradNorm,
            isLesioned,
          });
        }
      }
    }

    return [
      {
        type: 'neural-graph-overlay',
        layers,
        synapses,
        layerGradientNorms: [...state.lastGradNorms],
        inputFeatures: [...state.activeFeatures],
      },
    ];
  },

  serializeSnapshot(state: MLPState): Float32Array {
    let totalLen = 2; // epoch, stepCount
    for (const w of state.weights) totalLen += w.length;
    for (const b of state.biases) totalLen += b.length;

    const buffer = new Float32Array(totalLen);
    buffer[0] = state.epoch;
    buffer[1] = state.stepCount;

    let idx = 2;
    for (const w of state.weights) {
      buffer.set(w, idx);
      idx += w.length;
    }
    for (const b of state.biases) {
      buffer.set(b, idx);
      idx += b.length;
    }

    return buffer;
  },

  restoreSnapshot(buffer: Float32Array, state: MLPState): MLPState {
    state.epoch = Math.floor(buffer[0]);
    state.stepCount = Math.floor(buffer[1]);

    let idx = 2;
    for (const w of state.weights) {
      for (let i = 0; i < w.length; i++) {
        w[i] = buffer[idx++];
      }
    }
    for (const b of state.biases) {
      for (let i = 0; i < b.length; i++) {
        b[i] = buffer[idx++];
      }
    }

    return state;
  },

  explain(state: MLPState, params: MLPParams, lastStep?: StepResult<MLPState>): ExplanationPayload {
    const act = params.activation ?? 'tanh';
    const opt = params.optimizer ?? 'adam';
    const lr = params.learningRate ?? 0.03;
    const hiddenTopology = state.layerSizes.slice(1, -1).join(' → ');
    const isRegression = state.isRegression ?? isMLPRegression(params);

    return {
      summary: isRegression
        ? `Deep feedforward regressor approximating continuous manifold y = f(x) via non-linear composition: f(x) = W^{(L)} ... a^{(1)}(W^{(1)}x + b^{(1)}) + b^{(L)}.`
        : `Deep feedforward network approximating arbitrary continuous functions via non-linear composition: f(x) = a^{(L)}(W^{(L)} ... a^{(1)}(W^{(1)}x + b^{(1)}) + b^{(L)}).`,
      coreAssumption: isRegression
        ? `Continuous target response y ∈ ℝ lies on a smooth non-linear hypersurface parameterized by features X.`
        : `Target data manifold can be carved out by smooth or piecewise-linear hyperplanes folded together through non-linear activation layers.`,
      whenItFails: `Saturated activations (Sigmoid/Tanh far from 0) cause vanishing gradients σ'(z) → 0. Dead ReLUs (z ≤ 0 everywhere) produce permanent zero gradient flow.`,
      microLog: [
        `Forward Pass: Propagated ${state.activeFeatures.length} features through [${hiddenTopology}] layers using ${act.toUpperCase()} activations.`,
        isRegression
          ? `Error Measurement: Evaluated Mean Squared Error on outputs against continuous targets y ∈ ℝ (Train MSE: ${(lastStep?.metrics.trainLoss ?? 0).toFixed(4)}, R²: ${(lastStep?.metrics.trainR2 ?? 0).toFixed(4)}).`
          : `Error Measurement: Evaluated Mean Squared Error on outputs against target labels y ∈ {-1, +1} (Current Train Loss: ${(lastStep?.metrics.trainLoss ?? 0).toFixed(4)}).`,
        `Backward Pass: Backpropagated error deltas via recursive chain rule; updated weights using ${opt.toUpperCase()} (η = ${lr}).`,
      ],
      mathLatex: {
        objective: `\\mathcal{L}(\\mathbf{W}, \\mathbf{b}) = \\frac{1}{2N} \\sum_{i=1}^N \\|\\hat{y}_i - y_i\\|^2 + \\frac{\\lambda}{2} \\sum_l \\|\\mathbf{W}^{(l)}\\|_F^2`,
        lossOrUpdate: `\\delta^{(l)} = \\left( (\\mathbf{W}^{(l)})^T \\delta^{(l+1)} \\right) \\odot \\sigma'({\\mathbf{z}}^{(l)}), \\quad \\nabla_{\\mathbf{W}^{(l)}} \\mathcal{L} = \\delta^{(l+1)} (\\mathbf{a}^{(l)})^T`,
        dualOrConstraint: isRegression
          ? `\\text{Topology: } [${state.layerSizes.join(' \\to ')}], \\quad R^2 = ${(lastStep?.metrics.trainR2 ?? 0).toFixed(4)}`
          : `\\text{Topology: } [${state.layerSizes.join(' \\to ')}], \\quad \\|\\nabla \\mathbf{W}^{(1)}\\|_2 = ${(state.lastGradNorms[0] ?? 0).toFixed(4)}`,
      },
      codeSnippet: {
        sklearn: isRegression
          ? `from sklearn.neural_network import MLPRegressor

# Feedforward MLP Regressor with ${hiddenTopology} hidden architecture
reg = MLPRegressor(
    hidden_layer_sizes=(${state.layerSizes.slice(1, -1).join(', ')}),
    activation='${act === 'leaky-relu' ? 'relu' : act}',
    solver='${opt === 'momentum' ? 'sgd' : opt}',
    learning_rate_init=${lr},
    max_iter=500,
    random_state=42
)
reg.fit(X_train, y_train)`
          : `from sklearn.neural_network import MLPClassifier

# Feedforward MLP with ${hiddenTopology} hidden architecture
clf = MLPClassifier(
    hidden_layer_sizes=(${state.layerSizes.slice(1, -1).join(', ')}),
    activation='${act === 'leaky-relu' ? 'relu' : act}',
    solver='${opt === 'momentum' ? 'sgd' : opt}',
    learning_rate_init=${lr},
    max_iter=500,
    random_state=42
)
clf.fit(X_train, y_train)`,
        pytorch: `import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(${state.layerSizes[0]}, ${state.layerSizes[1]}),
            nn.${act === 'relu' ? 'ReLU' : act === 'tanh' ? 'Tanh' : 'Sigmoid'}(),
            nn.Linear(${state.layerSizes[1]}, ${state.layerSizes[2] || 1}),
            ${isRegression ? '# Linear regression output' : 'nn.Tanh()'}
        )
    def forward(self, x):
        return self.net(x)`,
      },
    };
  },
};
