import { DataPoint2D } from '../datasets/tabular2d';

export type StageKind =
  | 'continuous-2d'
  | 'tree-partition'
  | 'unsupervised-2d'
  | 'neural-graph'
  | 'vision-grid'
  | 'sequence-attn'
  | 'rl-gridworld';

export interface StageInfo {
  id: StageKind;
  index: number;
  name: string;
  shortName: string;
  description: string;
}

export type ExecutionMode = 'epoch-loop' | 'phase-walkthrough' | 'closed-form';

export type TaskType =
  | 'classification'
  | 'regression'
  | 'clustering'
  | 'dimensionality-reduction'
  | 'density-estimation'
  | 'reinforcement-learning';

export interface SpatialVisionSample {
  id: number;
  pixels: Float32Array; // Flattened 16x16 = 256
  label: number; // 0..3
  labelName: string;
}

export interface SpatialVisionDataset {
  preset: 'digit' | 'edges' | 'corner-cross' | 'checkerboard' | 'custom' | 'vertical' | 'horizontal';
  inputGrid: Float32Array; // Flattened 16x16 = 256 values in [0, 1]
  noise: number;
  contrast: number;
  inverted: boolean;
  trainSamples: SpatialVisionSample[];
  testSamples: SpatialVisionSample[];
}

export interface SequenceToken {
  id: number;
  text: string;
  embedding: Float32Array; // 8D vector
  isMasked?: boolean;
}

export type SequencePresetKind = 'coreference' | 'bracket' | 'sentiment' | 'seq2seq';

export interface SequenceDataset {
  preset: SequencePresetKind;
  sequence: SequenceToken[];
  targetSequence?: SequenceToken[]; // For seq2seq translation
  length: number; // T in [4..10]
  noise: number; // sigma in [0..0.5]
  selectedTokenIndex: number;
}

export interface DatasetContainer {
  train: DataPoint2D[];
  test: DataPoint2D[];
  spatialVision?: SpatialVisionDataset;
  sequenceTokens?: SequenceDataset;
  rlGridworld?: RLGridworldDataset;
}

export type GridworldTileKind = 'empty' | 'wall' | 'lava' | 'goal' | 'bait' | 'start';
export type GridworldPresetKind = 'cliff-walking' | 'frozen-lake' | 'twin-treasure' | 'four-rooms';

export interface RLGridworldDataset {
  preset: GridworldPresetKind;
  gridSize: number; // 6
  tiles: Uint8Array; // 36 tiles (0: empty, 1: wall, 2: lava, 3: goal, 4: bait, 5: start)
  startPos: [number, number]; // [r, c]
  goalPos: [number, number]; // [r, c]
  livingPenalty: number; // default -0.04
  slipProbability: number; // default 0.0 or 0.20
  seed: number;
}

export interface HyperparameterDef {
  id: string;
  name: string;
  symbol?: string;
  type: 'select' | 'slider';
  min?: number;
  max?: number;
  step?: number;
  options?: { label: string; value: number | string }[];
  default: number | string;
  description: string;
}

export interface TelemetryMetrics {
  trainLoss: number;
  testLoss: number;
  trainAccuracy?: number;
  testAccuracy?: number;
  trainR2?: number;
  testR2?: number;
  marginWidth?: number;
  sparsity?: { zeroCount: number; totalCount: number };
  gradientNorm?: number;
  custom?: Record<string, number | string>;
}

export interface StepResult<TState = unknown> {
  state: TState;
  epoch: number;
  metrics: TelemetryMetrics;
  converged?: boolean;
  circuitBreakerTripped?: boolean;
}

// Visual Artifacts produced by algorithms for canvas/SVG rendering
export type VisualArtifact =
  | {
      type: 'hyperplane';
      normal: [number, number]; // [w1, w2]
      bias: number;
      color?: string;
      style?: 'solid' | 'dashed';
      label?: string;
    }
  | {
      type: 'margin-line';
      normal: [number, number]; // [w1, w2]
      bias: number;
      offset: number; // e.g. +1 or -1
      label?: string;
    }
  | {
      type: 'support-vector-rings';
      pointIds: number[];
    }
  | {
      type: 'weight-constraint';
      constraintType: 'l1-diamond' | 'l2-circle' | 'elastic-net';
      radius: number;
      currentWeight: [number, number];
      l1Ratio?: number; // 0 (L2 circle) to 1 (L1 diamond)
    }
  | {
      type: 'residual-bars';
      residuals: { id: number; x1: number; x2: number; y: number; yHat: number; error: number }[];
    }
  | {
      type: 'curve-fit';
      points: [number, number][]; // Dense [x1, yHat] or [x1, x2]
      color?: string;
      style?: 'solid' | 'dashed';
      label?: string;
      rungeWarning?: boolean;
    }
  | {
      type: 'epsilon-tube';
      pointsUpper: [number, number][];
      pointsLower: [number, number][];
      epsilon: number;
    }
  | {
      type: 'uncertainty-band';
      meanCurve: [number, number][];
      upperBand: [number, number][]; // +2 sigma
      lowerBand: [number, number][]; // -2 sigma
      samples?: [number, number][][]; // 3 random function draws from posterior
    }
  | {
      type: 'uncertainty-surface';
      trainingX: [number, number][];
      lengthScale: number;
      signalVar: number;
      noiseVar: number;
    }
  | {
      type: 'decision-contour';
      level?: number;
      showMargins?: boolean;
      label?: string;
    }
  | {
      type: 'knn-probe-rays';
      probePoint: [number, number];
      neighbors: { id: number; point: [number, number]; distance: number; weight: number; label: number }[];
      metric: 'euclidean' | 'manhattan';
      radius: number;
    }
  | {
      type: 'marginal-gaussians';
      classes: {
        label: number;
        mean: [number, number];
        variance: [number, number]; // [var_x1, var_x2]
        prior: number;
      }[];
    }
  | {
      type: 'vector-addition';
      origin: [number, number];
      wVector: [number, number];
      deltaVector: [number, number];
      currentPointId?: number;
    }
  | {
      type: 'iso-contours';
      levels: { threshold: number; normal: [number, number]; bias: number }[];
    }
  | {
      type: 'discrete-grid';
      binsX: number;
      binsY: number;
      cells: { binX: number; binY: number; countClass0: number; countClass1: number }[];
    }
  | {
      type: 'split-box';
      bounds: [number, number, number, number]; // [xMin, xMax, yMin, yMax]
      value: number;
    }
  | {
      type: 'centroid';
      center: [number, number];
      clusterId: number;
    }
  | {
      type: 'tree-partition';
      root?: TreeNodeData;
      allNodes: TreeNodeData[];
      splits: {
        featureIndex: 0 | 1;
        threshold: number;
        range: [number, number];
        depth: number;
        label?: string;
        isPlaid?: boolean;
      }[];
      leaves: {
        id: number;
        bounds: [number, number, number, number];
        value: number;
        samples: number;
        classCounts?: number[];
      }[];
      totalTreesCount?: number;
      selectedTreeIndex?: number;
      treesRoots?: TreeNodeData[];
      candidateSplit?: {
        featureIndex: 0 | 1;
        threshold: number;
        gain: number;
        criterion: string;
      };
    }
  | {
      type: 'adaboost-weights';
      weights: { id: number; weight: number; normalizedWeight: number; isMisclassified: boolean }[];
    }
  | {
      type: 'bagging-samples';
      samples: { id: number; count: number; isOOB: boolean }[];
      oobError?: number;
    }
  | {
      type: 'meta-feature-projection';
      points: { id: number; p1: number; p2: number; y: number }[];
      weights: [number, number];
      bias: number;
      modelLabels: [string, string];
    }
  | {
      type: 'voting-boundaries';
      baseModels: { name: string; color: string; style: 'dashed'; accuracy: number; weight: number }[];
    }
  | {
      type: 'histogram-grid';
      binsX1: number[];
      binsX2: number[];
    }
  | {
      type: 'ensemble-overlay';
      trees: {
        treeIndex: number;
        splits: { featureIndex: 0 | 1; threshold: number; range: [number, number] }[];
      }[];
    }
  | {
      type: 'clustering-overlay';
      centroids?: { id: number; current: [number, number]; previous?: [number, number]; color?: string }[];
      assignments?: number[]; // cluster index or -1 for noise
      spokes?: { pointIndex: number; centroidId: number }[];
      voronoiSegments?: { start: [number, number]; end: [number, number] }[];
      dbscan?: {
        epsilon: number;
        minPts: number;
        corePointIds: number[];
        borderPointIds: number[];
        noisePointIds: number[];
        sweepIndex?: number;
      };
      gmmEllipses?: {
        componentId: number;
        mean: [number, number];
        cov: [[number, number], [number, number]];
        eigenvalues: [number, number];
        eigenvectors: [[number, number], [number, number]];
        angle: number;
        weight: number;
        color?: string;
      }[];
      softResponsibilities?: number[][]; // N x K
      clusterProfiles?: { k: number; inertia?: number; bic?: number; silhouette?: number }[];
      hierarchicalMerges?: { source: [number, number]; target: [number, number]; distance: number }[];
    }
  | {
      type: 'dendrogram-data';
      nodes: {
        id: number;
        left?: number;
        right?: number;
        distance: number;
        count: number;
        leafIndex?: number;
        x: number;
        y: number;
        clusterId?: number;
      }[];
      cutHeight: number;
      maxDistance: number;
      leafOrder: number[];
      activeClustersCount: number;
    }
  | {
      type: 'projection-overlay';
      method: 'pca' | 'lda' | 'ica';
      center: [number, number];
      axes: {
        id: string;
        direction: [number, number];
        length: number;
        varianceRatio?: number;
        label: string;
        color?: string;
        isGhost?: boolean;
      }[];
      dropLines?: {
        from: [number, number];
        to: [number, number];
        pointId: number;
        label: number;
      }[];
      screeData?: {
        component: number;
        eigenvalue: number;
        varianceRatio: number;
        cumulativeVariance: number;
      }[];
      projected1D?: {
        pointId: number;
        value: number;
        label: number;
      }[];
      ldaScatter?: {
        swEllipses?: { center: [number, number]; radiusX: number; radiusY: number; angle: number; classLabel: number }[];
        sbConnector?: { from: [number, number]; to: [number, number] };
        rayleighScore?: number;
      };
      icaKurtosis?: number[];
    }
  | {
      type: 'manifold-overlay';
      method: 'tsne' | 'umap';
      embeddedPoints: { id: number; x: number; y: number; label: number }[];
      springForces?: { pointId: number; targetId: number; type: 'attractive' | 'repulsive'; strength: number }[];
      fuzzyGraphEdges?: { sourceId: number; targetId: number; weight: number }[];
      lossTelemetry?: { iteration: number; loss: number; name: string }[];
    }
  | {
      type: 'neural-graph-overlay';
      layers: {
        index: number;
        neuronCount: number;
        isInput?: boolean;
        isOutput?: boolean;
        neurons: {
          id: string;
          label: string;
          bias: number;
          activationPreview?: number[]; // Flat 24x24 values normalized [-1, 1]
          isDead?: boolean;
        }[];
      }[];
      synapses: {
        id: string;
        sourceLayer: number;
        sourceNeuron: number;
        targetLayer: number;
        targetNeuron: number;
        weight: number;
        gradientMagnitude: number;
        isLesioned?: boolean;
      }[];
      layerGradientNorms: number[];
      inputFeatures: string[]; // ['x1', 'x2', 'x1_sq', 'x2_sq', 'x1_x2', 'sin_x1', 'sin_x2']
      magnifiedNeuron?: { layer: number; neuron: number };
    }
  | {
      type: 'latent-bottleneck-overlay';
      latentDim: 1 | 2;
      isVAE: boolean;
      encodedPoints: {
        id: number;
        z1: number;
        z2: number;
        label: number;
        sigma1?: number; // for VAE uncertainty ellipse
        sigma2?: number; // for VAE uncertainty ellipse
      }[];
      reconstructionVectors: {
        id: number;
        xOriginal: [number, number];
        xReconstructed: [number, number];
        error: number;
        label: number;
      }[];
      probePoint?: {
        z: [number, number];
        decoded: [number, number];
      };
      pca1Axis?: {
        center: [number, number];
        direction: [number, number];
        label: string;
      };
      klDivergence?: number;
      reconstructionMSE?: number;
    }
  | {
      type: 'gan-minimax-overlay';
      realPoints: { id: number; x1: number; x2: number; label: number }[];
      fakePoints: {
        id: number;
        x1: number;
        x2: number;
        prevX1?: number;
        prevX2?: number;
        pullVector: [number, number]; // grad_x log D(x)
        dScore: number; // D(G(z))
      }[];
      latentGridWarp?: {
        originalZ: [number, number];
        projectedX: [number, number];
      }[];
      gridResolution?: number; // e.g. 9 for 9x9 grid
      dLoss: number;
      gLoss: number;
      meanDReal: number;
      meanDFake: number;
      lossMode: 'non-saturating' | 'minimax-saturating';
    }
  | {
      type: 'spatial-vision-pipeline';
      algorithmId: 'cnn-lenet' | 'cnn-alexnet' | 'cnn-vgg' | 'cnn-resnet' | 'vit' | string;
      mode: 'inspector' | 'micro-training';
      inputResolution: number; // 16
      layers: VisionLayerArtifact[];
      slidingWindow?: {
        row: number; // 0..H-1
        col: number; // 0..W-1
        kernelSize: number;
        stride: number;
        padding: number;
        activeLayerIndex: number;
        activeChannelIndex: number;
        dotProductCalculation?: {
          inputs: number[];
          weights: number[];
          bias: number;
          output: number;
        };
      };
      vitTokens?: {
        patchSize: number; // e.g. 4
        gridDim: number; // e.g. 4 (4x4 patches = 16)
        numPatches: number; // 16
        patches: {
          id: number;
          row: number;
          col: number;
          tokens: Float32Array;
          isCls?: boolean;
        }[];
        selectedQueryPatch: number; // 0..15 or 16 for CLS
        attentionWeights: Float32Array; // (numPatches+1) x (numPatches+1)
        attentionHeads?: number;
        positionalEmbeddingMode: 'learned-2d' | 'scrambled-off';
      };
      resnetSkipConnections?: {
        sourceLayerIndex: number;
        targetLayerIndex: number;
        enabled: boolean;
        xMap?: Float32Array;
        fxMap?: Float32Array;
        sumMap?: Float32Array;
      }[];
      gradientNorms: {
        layerName: string;
        norm: number;
      }[];
      logits: {
        className: string;
        probability: number;
        logit: number;
      }[];
      predictedClass: number;
      predictedConfidence: number;
      translationDelta1px?: number; // 0..1 delta after +1px right shift
    }
  | {
      type: 'sequence-gate-attn';
      modelFamily: 'recurrent' | 'attention';
      algorithmId: string;
      executionMode: ExecutionMode;
      phaseIndex?: number;
      phaseName?: string;
      totalPhases?: number;
      tokens: string[];
      targetTokens?: string[];
      selectedStep: number; // t* in [0..T-1]

      // Recurrent Highway (Layout A)
      recurrent?: {
        timeSteps: {
          t: number;
          token: string;
          inputVec: Float32Array;
          hiddenState: Float32Array;
          cellState?: Float32Array; // for LSTM
          gradNorm: number; // ||dL/dh_t|| or ||dL/dC_t||
          gates?: {
            f?: number; // forget gate
            i?: number; // input gate
            cTilde?: number; // candidate cell state
            o?: number; // output gate
            z?: number; // update gate (GRU)
            r?: number; // reset gate (GRU)
            hTilde?: number; // candidate hidden (GRU)
          };
        }[];
        conveyorBelt?: {
          cells: Float32Array[];
          gradientSurvivalRatio: number; // ||grad_1|| / ||grad_T||
        };
        magnifiedStep: {
          t: number;
          token: string;
          f?: number;
          i?: number;
          o?: number;
          z?: number;
          r?: number;
          cTildeNorm?: number;
          hTildeNorm?: number;
          prevHiddenNorm: number;
          nextHiddenNorm: number;
          flowRates: Record<string, number>;
          pinnedGates?: Record<string, number>;
        };
      };

      // Attention & Transformer (Layout B)
      attention?: {
        d_k: number;
        scale: number;
        temperature: number;
        isCausal: boolean;
        rawScores: Float32Array; // T x T
        attentionWeights: Float32Array; // T x T
        entropy: number;
        maxScore: number;
        qVectors: Float32Array[]; // T vectors of length d_k
        kVectors: Float32Array[];
        vVectors: Float32Array[];
        outputVectors: Float32Array[];
        heads?: {
          id: number;
          name: string;
          color: string;
          weights: Float32Array;
          entropy: number;
        }[];
        selectedHead: number; // 0..H-1 or -1 for Combined
        positionalEncoding?: {
          enabled: boolean;
          matrix: Float32Array[]; // T x d
        };
        nextTokens?: {
          token: string;
          prob: number;
          logit: number;
        }[];
        crossAttention?: {
          sourceTokens: string[];
          targetTokens: string[];
          weights: Float32Array; // T_tgt x T_src
        };
        activeSubLayer?: 'enc-self' | 'dec-causal-self' | 'cross-attention';
      };
    }
  | {
      type: 'rl-gridworld-overlay';
      algorithmId: string;
      gridSize: number; // 6
      tiles: Uint8Array; // 36 tiles (0: empty, 1: wall, 2: lava, 3: goal, 4: bait, 5: start)
      qValues: Float32Array; // 36 * 4
      policyProbabilities?: Float32Array; // 36 * 4 for REINFORCE
      policyLogits?: Float32Array; // 36 * 4 raw logits θ for REINFORCE
      greedyActions: Int32Array; // 36
      agentState: {
        row: number;
        col: number;
        lastAction: number; // 0..3 (-1 if none)
        lastReward: number;
        cumulativeReturn: number;
        steps: number;
        isTerminal: boolean;
      };
      lastTransition?: {
        state: number; // 0..35
        action: number; // 0..3
        reward: number;
        nextState: number; // 0..35
        targetMaxAction: number; // 0..3
        tdError: number;
        targetQ: number;
        currentQ: number;
      };
      trajectory?: {
        row: number;
        col: number;
        action: number;
        reward: number;
        gValue: number;
        advantage: number;
      }[];
      replayBufferSample?: {
        s: number;
        a: number;
        r: number;
        nextS: number;
        done: boolean;
        isSampled: boolean;
      }[];
      targetNetworkSync?: {
        stepCounter: number;
        syncInterval: number;
        syncProgress: number; // 0..1
      };
      rolloutSpeed: 'single-step' | 'full-episode';
    };

export type RLGridworldArtifact = Extract<VisualArtifact, { type: 'rl-gridworld-overlay' }>;

export type SequenceGateAttnArtifact = Extract<VisualArtifact, { type: 'sequence-gate-attn' }>;

export type SpatialVisionPipelineArtifact = Extract<VisualArtifact, { type: 'spatial-vision-pipeline' }>;

export interface VisionLayerArtifact {
  id: string;
  name: string;
  type: 'input' | 'conv' | 'act' | 'pool' | 'skip-add' | 'patch-embed' | 'self-attention' | 'dense-head';
  shape: [number, number, number]; // [Channels, Height, Width]
  channels: {
    id: string;
    label: string;
    data: Float32Array; // Flattened H x W
    kernel?: Float32Array; // Flattened K x K weights
    bias?: number;
  }[];
  effectiveReceptiveField?: {
    size: number; // e.g. 3, 5, 7
    stride: number;
    padding: number;
  };
  stride?: number;
  padding?: number;
  kernelSize?: number;
  activation?: 'relu' | 'tanh' | 'sigmoid' | 'none';
}

export interface TreeNodeData {
  id: number;
  depth: number;
  isLeaf: boolean;
  featureIndex?: 0 | 1;
  threshold?: number;
  leftId?: number;
  rightId?: number;
  samples: number;
  impurity: number;
  value: number;
  classCounts?: number[];
  region: [number, number, number, number]; // [xMin, xMax, yMin, yMax]
  gain?: number;
  similarity?: number;
  weight?: number;
}

export interface ExplanationPayload {
  summary: string;
  coreAssumption: string;
  whenItFails: string;
  microLog: [string, string, string]; // [Forward pass, Measure error, Parameter update]
  mathLatex: {
    objective: string;
    lossOrUpdate: string;
    dualOrConstraint?: string;
  };
  codeSnippet: {
    sklearn: string;
    pytorch: string;
  };
}

export interface SymbolicMathLink {
  id: string;
  symbolLatex: string;
  label: string;
  description: string;
  highlightTarget: 'decision-hyperplane' | 'margin-positive' | 'margin-negative' | 'margin-street' | 'normal-vector' | 'support-vectors' | 'weights-inset';
}

/**
 * EXACT CORE AlgorithmPlugin INTERFACE CONTRACT
 * Every algorithm across all 7 stages implements this contract.
 */
export interface AlgorithmPlugin<TState = any, TParams = Record<string, any>> {
  id: string;
  name: string;
  shortCode: string;
  stage: StageKind;
  taskType: TaskType;
  executionMode: ExecutionMode;
  description: string;
  paperReference?: string;

  primaryParams: HyperparameterDef[];
  advancedParams: HyperparameterDef[];
  symbolicLinks?: SymbolicMathLink[];

  init: (dataset: DatasetContainer, params: TParams, seed: number) => TState;
  step: (state: TState, dataset: DatasetContainer, params: TParams) => StepResult<TState>;
  predict: (input: Float32Array, state: TState, params: TParams) => number | Float32Array;
  loss: (state: TState, dataset: DatasetContainer, params: TParams) => TelemetryMetrics;
  getVisualArtifacts: (state: TState, dataset: DatasetContainer, params: TParams) => VisualArtifact[];
  serializeSnapshot: (state: TState) => Float32Array;
  restoreSnapshot: (buffer: Float32Array, state: TState) => TState;
  explain: (state: TState, params: TParams, lastStep?: StepResult<TState>) => ExplanationPayload;
}
