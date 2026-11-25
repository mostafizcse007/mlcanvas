import { DataPoint2D } from '../datasets/tabular2d';
import { getPlugin, loadPlugin } from '../algorithms/registry';

// Internal Worker State
let currentAlgorithmId = 'linear-svm';
let hyperparams: Record<string, number | string> = {
  learningRate: 0.03,
  cPenalty: 1.0,
  lambdaPenalty: 0.1,
  batchPasses: 5,
  cyclesPerStep: 5,
};
let trainData: DataPoint2D[] = [];
let testData: DataPoint2D[] = [];
let spatialVisionData: any = null;
let sequenceTokensData: any = null;
let rlGridworldData: any = null;
let modelState: any = null;
let lastHealthySnapshot: Float32Array | null = null;
let isRunning = false;
let loopTimer: number | null = null;

// Pre-allocated grid buffers
const gridBuffer64 = new Float32Array(64 * 64);
const gridBuffer96 = new Float32Array(96 * 96);
const evalInput = new Float32Array(2);

function getDatasetContainer() {
  return {
    train: trainData,
    test: testData,
    spatialVision: spatialVisionData
      ? {
          ...spatialVisionData,
          inputGrid: new Float32Array(spatialVisionData.inputGrid),
        }
      : undefined,
    sequenceTokens: sequenceTokensData
      ? {
          ...sequenceTokensData,
          sequence: sequenceTokensData.sequence.map((s: any) => ({
            ...s,
            embedding: new Float32Array(s.embedding),
          })),
          targetSequence: sequenceTokensData.targetSequence
            ? sequenceTokensData.targetSequence.map((s: any) => ({
                ...s,
                embedding: new Float32Array(s.embedding),
              }))
            : undefined,
        }
      : undefined,
    rlGridworld: rlGridworldData
      ? {
          ...rlGridworldData,
          tiles: new Uint8Array(rlGridworldData.tiles),
          startPos: [...rlGridworldData.startPos],
          goalPos: [...rlGridworldData.goalPos],
        }
      : undefined,
  };
}

function computeGridBuffer(resolution: number = 96, bounds: [number, number, number, number] = [-6, 6, -6, 6]): Float32Array {
  const plugin = getPlugin(currentAlgorithmId);
  if (
    !plugin ||
    !modelState ||
    plugin.stage === 'vision-grid' ||
    plugin.stage === 'sequence-attn' ||
    plugin.stage === 'rl-gridworld' ||
    plugin.id === 'autoencoder' ||
    plugin.id === 'vae'
  ) {
    return new Float32Array(0);
  }

  const count = resolution * resolution;
  let buffer: Float32Array;
  if (resolution === 64) {
    buffer = gridBuffer64;
  } else if (resolution === 96) {
    buffer = gridBuffer96;
  } else {
    buffer = new Float32Array(count);
  }

  const [minX, maxX, minY, maxY] = bounds;
  const stepX = (maxX - minX) / (resolution - 1);
  const stepY = (maxY - minY) / (resolution - 1);

  let idx = 0;
  for (let r = 0; r < resolution; r++) {
    const x2 = maxY - r * stepY;
    evalInput[1] = x2;
    for (let c = 0; c < resolution; c++) {
      const x1 = minX + c * stepX;
      evalInput[0] = x1;
      const pred = (plugin as any).predictWithDataset
        ? (plugin as any).predictWithDataset(evalInput, modelState, getDatasetContainer())
        : plugin.predict(evalInput, modelState, hyperparams);
      buffer[idx++] = typeof pred === 'number' ? pred : pred[0];
    }
  }

  return new Float32Array(buffer);
}

function executeStep(resolution: number = 96, bounds: [number, number, number, number] = [-6, 6, -6, 6]): void {
  const plugin = getPlugin(currentAlgorithmId);
  if (!plugin || !modelState) return;

  const dataset = getDatasetContainer();
  const stepResult = plugin.step(modelState, dataset, hyperparams);

  // If circuit breaker or NaN occurs, recover gracefully without pausing execution
  if (stepResult.circuitBreakerTripped || isNaN(stepResult.metrics.trainLoss) || !isFinite(stepResult.metrics.trainLoss)) {
    if (lastHealthySnapshot && plugin) {
      modelState = plugin.restoreSnapshot(lastHealthySnapshot, modelState);
    }
    self.postMessage({
      type: 'CIRCUIT_BREAKER_TRIPPED',
      epoch: stepResult.epoch,
    });
  } else {
    modelState = stepResult.state;
    lastHealthySnapshot = plugin.serializeSnapshot(modelState);
  }

  // NOTE: In continuous mode, do NOT pause on convergence. Keep training loop alive indefinitely.

  const artifacts = plugin.getVisualArtifacts(modelState, dataset, hyperparams);
  const explanation = plugin.explain(modelState, hyperparams, stepResult);
  const gridBuffer = computeGridBuffer(resolution, bounds);

  self.postMessage({
    type: 'STEP_RESULT',
    result: stepResult,
    artifacts,
    explanation,
    serializedState: lastHealthySnapshot ? new Float32Array(lastHealthySnapshot) : null,
    gridBuffer,
    gridResolution: resolution,
    gridBounds: bounds,
  });
}

function evaluateGrid(resolution: number = 96, bounds: [number, number, number, number] = [-6, 6, -6, 6]): void {
  const transferCopy = computeGridBuffer(resolution, bounds);
  self.postMessage({
    type: 'GRID_EVAL_RESULT',
    buffer: transferCopy,
    resolution,
    bounds,
  });
}

function clampAndRecover(): void {
  if (lastHealthySnapshot && pluginRef()) {
    const plugin = pluginRef()!;
    modelState = plugin.restoreSnapshot(lastHealthySnapshot, modelState);
  }
  if (hyperparams.learningRate) {
    hyperparams.learningRate = Math.max(0.001, Number(hyperparams.learningRate) * 0.5);
  }

  self.postMessage({
    type: 'STATE_CHANGED',
    isRunning,
  });
}

function pluginRef() {
  return getPlugin(currentAlgorithmId);
}

function startLoop(): void {
  if (isRunning) return;
  isRunning = true;
  const tick = () => {
    if (!isRunning) return;
    executeStep();
    loopTimer = self.setTimeout(tick, 20) as unknown as number;
  };
  tick();
}

function pauseLoop(): void {
  isRunning = false;
  if (loopTimer !== null) {
    clearTimeout(loopTimer);
    loopTimer = null;
  }
}

function emitInferenceUpdate(): void {
  const plugin = getPlugin(currentAlgorithmId);
  if (!plugin || !modelState) return;

  const dataset = getDatasetContainer();
  const metrics = plugin.loss(modelState, dataset, hyperparams);
  const artifacts = plugin.getVisualArtifacts(modelState, dataset, hyperparams);
  const currentEpoch = modelState.epoch ?? 0;
  const result = {
    state: modelState,
    epoch: currentEpoch,
    metrics,
  };
  const explanation = plugin.explain(modelState, hyperparams, result);
  lastHealthySnapshot = plugin.serializeSnapshot(modelState);

  const gridBuffer = computeGridBuffer(96, [-6, 6, -6, 6]);

  self.postMessage({
    type: 'INFERENCE_RESULT',
    result,
    artifacts,
    explanation,
    serializedState: lastHealthySnapshot ? new Float32Array(lastHealthySnapshot) : null,
    gridBuffer,
    gridResolution: 96,
    gridBounds: [-6, 6, -6, 6],
  });
}

function emitInitialOrResetState(epoch: number = 0): void {
  const plugin = getPlugin(currentAlgorithmId);
  if (!plugin || !modelState) return;

  const dataset = getDatasetContainer();
  const metrics = plugin.loss(modelState, dataset, hyperparams);
  const artifacts = plugin.getVisualArtifacts(modelState, dataset, hyperparams);
  const isClosedForm = plugin.executionMode === 'closed-form';
  const result = {
    state: modelState,
    epoch: isClosedForm ? 1 : epoch,
    metrics,
    converged: isClosedForm ? true : false,
  };
  const explanation = plugin.explain(modelState, hyperparams, result);
  lastHealthySnapshot = plugin.serializeSnapshot(modelState);

  const gridBuffer = computeGridBuffer(96, [-6, 6, -6, 6]);

  self.postMessage({
    type: 'STEP_RESULT',
    result,
    artifacts,
    explanation,
    serializedState: lastHealthySnapshot ? new Float32Array(lastHealthySnapshot) : null,
    gridBuffer,
    gridResolution: 96,
    gridBounds: [-6, 6, -6, 6],
  });
}

self.onmessage = async (event: MessageEvent) => {
  const msg = event.data;
  switch (msg.type) {
    case 'INIT_ALGORITHM': {
      pauseLoop();
      currentAlgorithmId = msg.algorithmId;
      hyperparams = { ...msg.hyperparams };
      trainData = msg.trainData ? [...msg.trainData] : [];
      testData = msg.testData ? [...msg.testData] : [];
      spatialVisionData = msg.spatialVision
        ? { ...msg.spatialVision, inputGrid: new Float32Array(msg.spatialVision.inputGrid) }
        : null;
      sequenceTokensData = msg.sequenceTokens
        ? {
            ...msg.sequenceTokens,
            sequence: msg.sequenceTokens.sequence.map((s: any) => ({
              ...s,
              embedding: new Float32Array(s.embedding),
            })),
            targetSequence: msg.sequenceTokens.targetSequence
              ? msg.sequenceTokens.targetSequence.map((s: any) => ({
                  ...s,
                  embedding: new Float32Array(s.embedding),
                }))
              : undefined,
          }
        : null;
      rlGridworldData = msg.rlGridworld
        ? {
            ...msg.rlGridworld,
            tiles: new Uint8Array(msg.rlGridworld.tiles),
            startPos: [...msg.rlGridworld.startPos],
            goalPos: [...msg.rlGridworld.goalPos],
          }
        : null;

      let plugin = getPlugin(currentAlgorithmId);
      if (!plugin) {
        plugin = await loadPlugin(currentAlgorithmId);
      }
      if (plugin) {
        modelState = plugin.init(getDatasetContainer(), hyperparams, msg.seed ?? 42);
        emitInitialOrResetState(0);
      }
      break;
    }

    case 'SET_DATASET': {
      trainData = msg.trainData ? [...msg.trainData] : [];
      testData = msg.testData ? [...msg.testData] : [];
      if (msg.spatialVision) {
        spatialVisionData = {
          ...msg.spatialVision,
          inputGrid: new Float32Array(msg.spatialVision.inputGrid),
        };
      }
      if (msg.sequenceTokens) {
        sequenceTokensData = {
          ...msg.sequenceTokens,
          sequence: msg.sequenceTokens.sequence.map((s: any) => ({
            ...s,
            embedding: new Float32Array(s.embedding),
          })),
          targetSequence: msg.sequenceTokens.targetSequence
            ? msg.sequenceTokens.targetSequence.map((s: any) => ({
                ...s,
                embedding: new Float32Array(s.embedding),
              }))
            : undefined,
        };
      }
      if (msg.rlGridworld) {
        rlGridworldData = {
          ...msg.rlGridworld,
          tiles: new Uint8Array(msg.rlGridworld.tiles),
          startPos: [...msg.rlGridworld.startPos],
          goalPos: [...msg.rlGridworld.goalPos],
        };
      }
      let plugin = getPlugin(currentAlgorithmId);
      if (!plugin) {
        plugin = await loadPlugin(currentAlgorithmId);
      }
      if (plugin) {
        modelState = plugin.init(getDatasetContainer(), hyperparams, msg.seed ?? 42);
        if (msg.liveUpdate) {
          executeStep(96, [-6, 6, -6, 6]);
        } else {
          emitInitialOrResetState(0);
        }
      }
      break;
    }

    case 'SET_SEQUENCE_TOKENS': {
      if (msg.sequenceTokens) {
        sequenceTokensData = {
          ...msg.sequenceTokens,
          sequence: msg.sequenceTokens.sequence.map((s: any) => ({
            ...s,
            embedding: new Float32Array(s.embedding),
          })),
          targetSequence: msg.sequenceTokens.targetSequence
            ? msg.sequenceTokens.targetSequence.map((s: any) => ({
                ...s,
                embedding: new Float32Array(s.embedding),
              }))
            : undefined,
        };
      }
      if (msg.rlGridworld) {
        rlGridworldData = {
          ...msg.rlGridworld,
          tiles: new Uint8Array(msg.rlGridworld.tiles),
          startPos: [...msg.rlGridworld.startPos],
          goalPos: [...msg.rlGridworld.goalPos],
        };
      }
      const plugin = getPlugin(currentAlgorithmId);
      if (plugin) {
        modelState = plugin.init(getDatasetContainer(), hyperparams, msg.seed ?? 42);
        if (plugin.stage === 'sequence-attn') {
          // Init already computes the full forward/backward state pinned to selectedTokenIndex;
          // emit directly without advancing phaseIndex/selectedStep.
          emitInitialOrResetState(modelState?.epoch ?? 0);
        } else if (msg.liveUpdate) {
          executeStep(96, [-6, 6, -6, 6]);
        } else {
          emitInitialOrResetState(modelState?.epoch ?? 0);
        }
      }
      break;
    }

    case 'PIN_GATE_VALVE': {
      if (modelState && modelState.pinnedGates !== undefined) {
        const { stepIndex, gateId, value } = msg;
        if (!modelState.pinnedGates[stepIndex]) {
          modelState.pinnedGates[stepIndex] = {};
        }
        if (value === undefined || value === null) {
          delete modelState.pinnedGates[stepIndex][gateId];
        } else {
          modelState.pinnedGates[stepIndex][gateId] = Number(value);
        }
        const plugin = getPlugin(currentAlgorithmId);
        if (plugin) {
          emitInitialOrResetState(modelState?.epoch ?? 0);
        }
      }
      break;
    }

    case 'SET_SPATIAL_VISION': {
      if (msg.spatialVision) {
        spatialVisionData = {
          ...msg.spatialVision,
          inputGrid: new Float32Array(msg.spatialVision.inputGrid),
        };
      }
      const plugin = getPlugin(currentAlgorithmId);
      if (plugin && modelState) {
        if (msg.liveUpdate) {
          executeStep(96, [-6, 6, -6, 6]);
        } else {
          emitInferenceUpdate();
        }
      }
      break;
    }

    case 'SET_RL_GRIDWORLD': {
      if (msg.rlGridworld) {
        rlGridworldData = {
          ...msg.rlGridworld,
          tiles: new Uint8Array(msg.rlGridworld.tiles),
          startPos: [...msg.rlGridworld.startPos],
          goalPos: [...msg.rlGridworld.goalPos],
        };
      }
      const plugin = getPlugin(currentAlgorithmId);
      if (plugin) {
        if (msg.preserveState && modelState) {
          // In-situ adaptation: update MDP transition topology, reset agent to start, preserve learned weights
          modelState.agentRow = rlGridworldData.startPos[0];
          modelState.agentCol = rlGridworldData.startPos[1];
          modelState.stepsInEpisode = 0;
          modelState.cumulativeReturn = 0;
        } else {
          modelState = plugin.init(getDatasetContainer(), hyperparams, msg.seed ?? 42);
        }
        if (msg.liveUpdate || plugin.stage === 'rl-gridworld') {
          executeStep(96, [-6, 6, -6, 6]);
        } else {
          emitInitialOrResetState(modelState?.epoch ?? 0);
        }
      }
      break;
    }

    case 'UPDATE_HYPERPARAMS': {
      const prevOptimisticInit = hyperparams.optimisticInit;
      const incoming = msg.hyperparams || {};
      const changedKeys = Object.keys(incoming).filter(
        (k) => incoming[k] !== hyperparams[k]
      );
      const isInferenceOnly =
        Boolean(msg.inferenceOnly) ||
        (currentAlgorithmId === 'logistic-regression' &&
          changedKeys.length > 0 &&
          changedKeys.every((k) => k === 'threshold'));

      hyperparams = { ...hyperparams, ...incoming };
      if (currentAlgorithmId === 'logistic-regression' && hyperparams.threshold !== undefined) {
        const parsed = Number(hyperparams.threshold);
        hyperparams.threshold = Number.isFinite(parsed)
          ? Math.min(0.99, Math.max(0.01, parsed))
          : 0.5;
      }

      const plugin = getPlugin(currentAlgorithmId);
      if (!plugin || !modelState) break;

      if (isInferenceOnly) {
        if (
          typeof modelState === 'object' &&
          modelState !== null &&
          'threshold' in modelState &&
          hyperparams.threshold !== undefined
        ) {
          modelState = { ...modelState, threshold: Number(hyperparams.threshold) };
        }
        emitInferenceUpdate();
        break;
      }

      if (!isRunning) {
        if (plugin.stage === 'rl-gridworld') {
          // In RL, preserve learned Q-table / neural weights across runtime hyperparam adjustments (cadence, epsilon, lr, sync)
          if (msg.hyperparams?.optimisticInit !== undefined && msg.hyperparams.optimisticInit !== prevOptimisticInit) {
            modelState = plugin.init(getDatasetContainer(), hyperparams, 42);
            emitInitialOrResetState(modelState.epoch ?? 0);
          } else {
            emitInitialOrResetState(modelState.epoch ?? 0);
          }
        } else {
          modelState = plugin.init(getDatasetContainer(), hyperparams, 42);
          emitInitialOrResetState(modelState.epoch ?? 0);
        }
      }
      break;
    }

    case 'STEP': {
      executeStep(msg.resolution ?? 96, msg.bounds ?? [-6, 6, -6, 6]);
      break;
    }

    case 'START': {
      startLoop();
      break;
    }

    case 'PAUSE': {
      pauseLoop();
      break;
    }

    case 'RESET': {
      pauseLoop();
      const plugin = getPlugin(currentAlgorithmId);
      if (plugin) {
        modelState = plugin.init(getDatasetContainer(), hyperparams, msg.seed ?? 42);
        emitInitialOrResetState(0);
      }
      break;
    }

    case 'EVAL_GRID': {
      evaluateGrid(msg.resolution ?? 96, msg.bounds ?? [-6, 6, -6, 6]);
      break;
    }

    case 'CLAMP_RECOVER': {
      clampAndRecover();
      break;
    }

    case 'SYNC_CENTROIDS': {
      if (modelState && msg.centroids) {
        if (modelState.centroids) {
          modelState.centroids = msg.centroids.map((c: [number, number]) => [...c]);
        }
        if (modelState.components) {
          for (let i = 0; i < Math.min(modelState.components.length, msg.centroids.length); i++) {
            modelState.components[i].mean = [...msg.centroids[i]];
          }
        }
        emitInitialOrResetState(modelState.epoch ?? 0);
      }
      break;
    }

    case 'RESTORE_SNAPSHOT': {
      const plugin = getPlugin(currentAlgorithmId);
      if (plugin && modelState) {
        modelState = plugin.restoreSnapshot(new Float32Array(msg.snapshotBuffer), modelState);
        const dataset = getDatasetContainer();
        const metrics = plugin.loss(modelState, dataset, hyperparams);
        const currentEp = modelState.epoch ?? 0;
        const result = {
          state: modelState,
          epoch: currentEp,
          metrics,
        };
        const artifacts = plugin.getVisualArtifacts(modelState, dataset, hyperparams);
        const explanation = plugin.explain(modelState, hyperparams, result);
        const gridBuffer = currentEp > 0 ? computeGridBuffer(96, [-6, 6, -6, 6]) : new Float32Array(0);
        self.postMessage({
          type: 'STEP_RESULT',
          result,
          artifacts,
          explanation,
          serializedState: new Float32Array(msg.snapshotBuffer),
          gridBuffer,
          gridResolution: 96,
          gridBounds: [-6, 6, -6, 6],
        });
      }
      break;
    }
  }
};
