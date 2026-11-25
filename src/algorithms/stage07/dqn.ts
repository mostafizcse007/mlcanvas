import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import {
  GRID_SIZE,
  NUM_STATES,
  NUM_ACTIONS,
  createRLGridworldDataset,
  stepEnvironment,
  TILE_VALUES,
} from '../../datasets/rlGridworld';
import { mulberry32, gaussianRandom } from '../../utils/prng';

const FEATURE_DIM = 6;
const HIDDEN_DIM = 24;
const REPLAY_CAPACITY = 256;
const BATCH_SIZE = 16;

export interface DQNState {
  epoch: number;
  episode: number;
  stepCount: number;
  targetSyncCounter: number;

  // Online Network Weights (W1: 24x6, b1: 24, W2: 4x24, b2: 4)
  onlineW1: Float32Array;
  onlineB1: Float32Array;
  onlineW2: Float32Array;
  onlineB2: Float32Array;

  // Target Network Weights (Frozen θ-)
  targetW1: Float32Array;
  targetB1: Float32Array;
  targetW2: Float32Array;
  targetB2: Float32Array;

  // Circular Experience Replay Buffer (N=256)
  replayStates: Float32Array; // 256 * 6
  replayActions: Int32Array; // 256
  replayRewards: Float32Array; // 256
  replayNextStates: Float32Array; // 256 * 6
  replayDones: Uint8Array; // 256
  replayHead: number;
  replayCount: number;
  lastSampledIndices: number[];

  // Agent State
  agentRow: number;
  agentCol: number;
  lastAction: number;
  lastReward: number;
  cumulativeReturn: number;
  stepsInEpisode: number;
  lastCompletedEpisodeLength: number;
  lastRolloutSpeed?: 'single-step' | 'full-episode';
  isTerminal: boolean;

  // Cached Q-values across all 36 states for fast visualization
  cachedQTable: Float32Array; // 36 * 4

  trajectory: { row: number; col: number; action: number; reward: number; gValue: number; advantage: number }[];
  lastTransition?: {
    state: number;
    action: number;
    reward: number;
    nextState: number;
    targetMaxAction: number;
    tdError: number;
    targetQ: number;
    currentQ: number;
  };
  recentReturns: number[];
  recentLosses: number[];
  meanLoss: number;
  seed: number;
}

/** Extract 6-dim normalized feature vector for grid cell (r, c) */
export function extractStateFeatures(
  row: number,
  col: number,
  goalPos: [number, number],
  tiles: Uint8Array
): Float32Array {
  const feat = new Float32Array(FEATURE_DIM);
  feat[0] = row / (GRID_SIZE - 1);
  feat[1] = col / (GRID_SIZE - 1);
  feat[2] = (goalPos[0] - row) / (GRID_SIZE - 1);
  feat[3] = (goalPos[1] - col) / (GRID_SIZE - 1);

  // Check 4-neighbor adjacency to wall or lava
  let nearWall = 0;
  let nearLava = 0;
  const deltas = [[-1, 0], [0, 1], [1, 0], [0, -1]];
  for (const [dr, dc] of deltas) {
    const nr = row + dr;
    const nc = col + dc;
    if (nr >= 0 && nr < GRID_SIZE && nc >= 0 && nc < GRID_SIZE) {
      const tile = tiles[nr * GRID_SIZE + nc];
      if (tile === TILE_VALUES.wall) nearWall = 1;
      if (tile === TILE_VALUES.lava) nearLava = 1;
    }
  }
  feat[4] = nearWall;
  feat[5] = nearLava;
  return feat;
}

/** Forward pass through 2-layer MLP */
function forwardPass(
  features: Float32Array,
  w1: Float32Array,
  b1: Float32Array,
  w2: Float32Array,
  b2: Float32Array
): { hidden: Float32Array; qValues: Float32Array } {
  const hidden = new Float32Array(HIDDEN_DIM);
  const qValues = new Float32Array(NUM_ACTIONS);

  // Layer 1: Input (6) -> Hidden (24) with ReLU
  for (let h = 0; h < HIDDEN_DIM; h++) {
    let sum = b1[h];
    const offset = h * FEATURE_DIM;
    for (let f = 0; f < FEATURE_DIM; f++) {
      sum += features[f] * w1[offset + f];
    }
    hidden[h] = Math.max(0, sum); // ReLU
  }

  // Layer 2: Hidden (24) -> Output (4) Linear
  for (let a = 0; a < NUM_ACTIONS; a++) {
    let sum = b2[a];
    const offset = a * HIDDEN_DIM;
    for (let h = 0; h < HIDDEN_DIM; h++) {
      sum += hidden[h] * w2[offset + h];
    }
    qValues[a] = sum;
  }

  return { hidden, qValues };
}

export const dqnPlugin: AlgorithmPlugin<DQNState> = {
  id: 'dqn',
  name: 'Deep Q-Network (Function Approximation & Replay)',
  shortCode: 'DQN',
  stage: 'rl-gridworld',
  taskType: 'reinforcement-learning',
  executionMode: 'epoch-loop',
  description:
    'Deep Q-Network with Neural Function Approximation, Experience Replay Memory, and a Frozen Target Network θ⁻ breaking temporal correlation and preventing moving-target divergence.',
  paperReference:
    'Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533.',

  primaryParams: [
    {
      id: 'replayBufferMode',
      name: 'Replay Buffer',
      type: 'select',
      options: [
        { label: 'Enabled (Random Mini-Batch)', value: 'enabled-random-batch' },
        { label: 'Disabled (Correlated Online Stream)', value: 'disabled-online-stream' },
      ],
      default: 'enabled-random-batch',
      description: 'Samples uncorrelated past transitions from circular memory vs. training on consecutive steps.',
    },
    {
      id: 'targetNetworkSync',
      name: 'Target Sync (C)',
      type: 'slider',
      min: 1,
      max: 50,
      step: 1,
      default: 20,
      description: 'Frequency in steps at which Target Network θ⁻ copies weights from Online Network θ.',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (η)',
      type: 'slider',
      min: 0.001,
      max: 0.05,
      step: 0.002,
      default: 0.01,
      description: 'SGD learning rate updating neural network weights via backpropagation.',
    },
  ],

  advancedParams: [
    {
      id: 'discountFactor',
      name: 'Discount (γ)',
      type: 'slider',
      min: 0.0,
      max: 0.99,
      step: 0.05,
      default: 0.95,
      description: 'Bellman discount factor for target calculation.',
    },
    {
      id: 'explorationEpsilon',
      name: 'Exploration (ε)',
      type: 'slider',
      min: 0.0,
      max: 1.0,
      step: 0.05,
      default: 0.2,
      description: 'Probability of selecting a random action vs. argmax_a Q_θ(s, a).',
    },
    {
      id: 'rolloutSpeed',
      name: 'Rollout Cadence',
      type: 'select',
      options: [
        { label: '1 Step / Frame', value: 'single-step' },
        { label: '1 Episode / Frame', value: 'full-episode' },
      ],
      default: 'single-step',
      description: 'Single environment transition for fine inspection vs. full episode rollouts.',
    },
  ],

  symbolicLinks: [
    {
      id: 'dqn-loss',
      symbolLatex: 'L(\\theta) = \\mathbb{E}_{(s, a, r, s\') \\sim \\mathcal{D}} \\Big[\\Big(r + \\gamma \\max_{a\'} Q_{\\theta^-}(s\', a\') - Q_\\theta(s, a)\\Big)^2\\Big]',
      label: 'DQN Loss Function',
      description: 'Mean squared error between online network prediction and frozen target network Bellman target.',
      highlightTarget: 'weights-inset',
    },
  ],

  init: (dataset: DatasetContainer, _params: Record<string, any>, seed: number): DQNState => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const rng = mulberry32(seed);

    // He initialization for weights
    const onlineW1 = new Float32Array(HIDDEN_DIM * FEATURE_DIM);
    const onlineB1 = new Float32Array(HIDDEN_DIM);
    const onlineW2 = new Float32Array(NUM_ACTIONS * HIDDEN_DIM);
    const onlineB2 = new Float32Array(NUM_ACTIONS);

    const std1 = Math.sqrt(2.0 / FEATURE_DIM);
    for (let i = 0; i < onlineW1.length; i++) {
      onlineW1[i] = gaussianRandom(rng) * std1;
    }
    const std2 = Math.sqrt(2.0 / HIDDEN_DIM);
    for (let i = 0; i < onlineW2.length; i++) {
      onlineW2[i] = gaussianRandom(rng) * std2;
    }

    // Clone into Target Network θ-
    const targetW1 = new Float32Array(onlineW1);
    const targetB1 = new Float32Array(onlineB1);
    const targetW2 = new Float32Array(onlineW2);
    const targetB2 = new Float32Array(onlineB2);

    // Pre-compute cached Q-values across all 36 cells
    const cachedQTable = new Float32Array(NUM_STATES * NUM_ACTIONS);
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const s = r * GRID_SIZE + c;
        const feat = extractStateFeatures(r, c, grid.goalPos, grid.tiles);
        const { qValues } = forwardPass(feat, onlineW1, onlineB1, onlineW2, onlineB2);
        for (let a = 0; a < NUM_ACTIONS; a++) {
          cachedQTable[s * NUM_ACTIONS + a] = qValues[a];
        }
      }
    }

    return {
      epoch: 0,
      episode: 0,
      stepCount: 0,
      targetSyncCounter: 0,
      onlineW1,
      onlineB1,
      onlineW2,
      onlineB2,
      targetW1,
      targetB1,
      targetW2,
      targetB2,
      replayStates: new Float32Array(REPLAY_CAPACITY * FEATURE_DIM),
      replayActions: new Int32Array(REPLAY_CAPACITY),
      replayRewards: new Float32Array(REPLAY_CAPACITY),
      replayNextStates: new Float32Array(REPLAY_CAPACITY * FEATURE_DIM),
      replayDones: new Uint8Array(REPLAY_CAPACITY),
      replayHead: 0,
      replayCount: 0,
      lastSampledIndices: [],
      agentRow: grid.startPos[0],
      agentCol: grid.startPos[1],
      lastAction: -1,
      lastReward: 0,
      cumulativeReturn: 0,
      stepsInEpisode: 0,
      lastCompletedEpisodeLength: 0,
      lastRolloutSpeed: String(_params.rolloutSpeed ?? 'single-step') as 'single-step' | 'full-episode',
      isTerminal: false,
      cachedQTable,
      trajectory: [],
      recentReturns: [],
      recentLosses: [],
      meanLoss: 0,
      seed,
    };
  },

  step: (state: DQNState, dataset: DatasetContainer, params: Record<string, any>): StepResult<DQNState> => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const gamma = Number(params.discountFactor ?? 0.95);
    const epsilon = Number(params.explorationEpsilon ?? 0.2);
    const lr = Number(params.learningRate ?? 0.01);
    const replayMode = String(params.replayBufferMode ?? 'enabled-random-batch');
    const syncInterval = Math.max(1, Number(params.targetNetworkSync ?? 20));
    const speed = String(params.rolloutSpeed ?? 'single-step');

    const rng = mulberry32(state.seed + state.epoch * 1013 + state.stepCount);

    let nextRow = state.agentRow;
    let nextCol = state.agentCol;
    let cumReturn = state.cumulativeReturn;
    let epSteps = state.stepsInEpisode;
    let lastCompletedEpisodeLength = state.lastCompletedEpisodeLength;
    let episode = state.episode;
    let syncCounter = state.targetSyncCounter;
    let lastTransition = state.lastTransition;
    let lastAction = state.lastAction;
    let lastReward = state.lastReward;
    let isTerminal = state.isTerminal;
    let actualStepsTaken = 0;

    let currentTrajectory = [...state.trajectory];
    const recentReturns = [...state.recentReturns];
    const recentLosses = [...state.recentLosses];

    // If previous tick ended in a terminal state (Goal / Lava / Timeout), reset for new episode
    if (isTerminal) {
      nextRow = grid.startPos[0];
      nextCol = grid.startPos[1];
      cumReturn = 0;
      epSteps = 0;
      isTerminal = false;
      currentTrajectory = [];
    }

    let replayHead = state.replayHead;
    let replayCount = state.replayCount;
    let lastSampledIndices: number[] = [];
    let stepLossTotal = 0;
    let stepLossCount = 0;

    const maxStepsToTake = speed === 'full-episode' ? 120 : 1;

    for (let sIdx = 0; sIdx < maxStepsToTake; sIdx++) {
      actualStepsTaken += 1;
      const s = nextRow * GRID_SIZE + nextCol;
      const currentFeatures = extractStateFeatures(nextRow, nextCol, grid.goalPos, grid.tiles);

      // Select action using ε-greedy over Online Q(s, ·)
      const currentPred = forwardPass(
        currentFeatures,
        state.onlineW1,
        state.onlineB1,
        state.onlineW2,
        state.onlineB2
      );

      let action = 0;
      if (rng() < epsilon) {
        action = Math.floor(rng() * NUM_ACTIONS);
      } else {
        let bestA = 0;
        let bestQ = -Infinity;
        for (let a = 0; a < NUM_ACTIONS; a++) {
          if (currentPred.qValues[a] > bestQ) {
            bestQ = currentPred.qValues[a];
            bestA = a;
          }
        }
        action = bestA;
      }

      // Step environment
      const envResult = stepEnvironment(nextRow, nextCol, action, grid, rng);
      const nextFeatures = extractStateFeatures(envResult.nextRow, envResult.nextCol, grid.goalPos, grid.tiles);
      const r = envResult.reward;
      cumReturn += r;
      epSteps += 1;
      syncCounter += 1;

      // Store transition in circular Replay Buffer
      const bufIdx = replayHead;
      state.replayStates.set(currentFeatures, bufIdx * FEATURE_DIM);
      state.replayActions[bufIdx] = action;
      state.replayRewards[bufIdx] = r;
      state.replayNextStates.set(nextFeatures, bufIdx * FEATURE_DIM);
      state.replayDones[bufIdx] = envResult.done ? 1 : 0;

      replayHead = (replayHead + 1) % REPLAY_CAPACITY;
      if (replayCount < REPLAY_CAPACITY) replayCount += 1;

      // Sample transitions for gradient update
      const sampleIndices: number[] = [];
      if (replayMode === 'disabled-online-stream' || replayCount < BATCH_SIZE) {
        // Online single-step stream: only train on the most recent transition
        sampleIndices.push(bufIdx);
      } else {
        // Random mini-batch sampling from replay buffer
        for (let b = 0; b < BATCH_SIZE; b++) {
          const randIdx = Math.floor(rng() * replayCount);
          sampleIndices.push(randIdx);
        }
      }
      lastSampledIndices = sampleIndices;

      // Accumulate gradients over mini-batch
      const dW1 = new Float32Array(HIDDEN_DIM * FEATURE_DIM);
      const dB1 = new Float32Array(HIDDEN_DIM);
      const dW2 = new Float32Array(NUM_ACTIONS * HIDDEN_DIM);
      const dB2 = new Float32Array(NUM_ACTIONS);
      let batchLoss = 0;

      for (const idx of sampleIndices) {
        const sFeat = state.replayStates.subarray(idx * FEATURE_DIM, (idx + 1) * FEATURE_DIM);
        const aVal = state.replayActions[idx];
        const rVal = state.replayRewards[idx];
        const nextFeat = state.replayNextStates.subarray(idx * FEATURE_DIM, (idx + 1) * FEATURE_DIM);
        const dVal = state.replayDones[idx];

        // 1. Compute Bellman Target y = r + gamma * max_a' Q_target(s', a')
        const targetPred = forwardPass(
          nextFeat,
          state.targetW1,
          state.targetB1,
          state.targetW2,
          state.targetB2
        );
        let maxTargetQ = -Infinity;
        for (let aPrime = 0; aPrime < NUM_ACTIONS; aPrime++) {
          if (targetPred.qValues[aPrime] > maxTargetQ) {
            maxTargetQ = targetPred.qValues[aPrime];
          }
        }
        const targetVal = dVal === 1 ? rVal : rVal + gamma * maxTargetQ;

        // 2. Online forward pass
        const onlinePred = forwardPass(
          sFeat,
          state.onlineW1,
          state.onlineB1,
          state.onlineW2,
          state.onlineB2
        );
        const currentQVal = onlinePred.qValues[aVal];
        const tdErr = targetVal - currentQVal;
        const lossVal = tdErr * tdErr;
        batchLoss += lossVal;

        // Gradient of Loss w.r.t Q(s, a): dL/dQ = -2 * (target - Q)
        const dQ = new Float32Array(NUM_ACTIONS);
        dQ[aVal] = -2 * tdErr;

        // Backprop to Layer 2
        const dHidden = new Float32Array(HIDDEN_DIM);
        for (let a = 0; a < NUM_ACTIONS; a++) {
          const g = dQ[a];
          if (g === 0) continue;
          dB2[a] += g;
          const wOffset = a * HIDDEN_DIM;
          for (let h = 0; h < HIDDEN_DIM; h++) {
            dW2[wOffset + h] += g * onlinePred.hidden[h];
            dHidden[h] += g * state.onlineW2[wOffset + h];
          }
        }

        // Backprop through ReLU to Layer 1
        for (let h = 0; h < HIDDEN_DIM; h++) {
          if (onlinePred.hidden[h] > 0) {
            const g = dHidden[h];
            dB1[h] += g;
            const wOffset = h * FEATURE_DIM;
            for (let f = 0; f < FEATURE_DIM; f++) {
              dW1[wOffset + f] += g * sFeat[f];
            }
          }
        }
      }

      const N = sampleIndices.length;
      stepLossTotal += batchLoss / N;
      stepLossCount += 1;

      // Apply SGD weight update with gradient clipping
      const scale = lr / N;
      for (let i = 0; i < state.onlineW1.length; i++) {
        const grad = Math.max(-10, Math.min(10, dW1[i]));
        state.onlineW1[i] -= scale * grad;
      }
      for (let i = 0; i < state.onlineB1.length; i++) {
        const grad = Math.max(-10, Math.min(10, dB1[i]));
        state.onlineB1[i] -= scale * grad;
      }
      for (let i = 0; i < state.onlineW2.length; i++) {
        const grad = Math.max(-10, Math.min(10, dW2[i]));
        state.onlineW2[i] -= scale * grad;
      }
      for (let i = 0; i < state.onlineB2.length; i++) {
        const grad = Math.max(-10, Math.min(10, dB2[i]));
        state.onlineB2[i] -= scale * grad;
      }

      // Check Target Network Sync (θ- <- θ every C steps)
      if (syncCounter >= syncInterval) {
        state.targetW1.set(state.onlineW1);
        state.targetB1.set(state.onlineB1);
        state.targetW2.set(state.onlineW2);
        state.targetB2.set(state.onlineB2);
        syncCounter = 0;
      }

      lastAction = action;
      lastReward = r;
      nextRow = envResult.nextRow;
      nextCol = envResult.nextCol;

      // Compute transition Bellman target using frozen Target Network θ-
      const singleTargetPred = forwardPass(
        nextFeatures,
        state.targetW1,
        state.targetB1,
        state.targetW2,
        state.targetB2
      );
      let maxTargetQVal = -Infinity;
      let targetMaxAction = 0;
      for (let aPrime = 0; aPrime < NUM_ACTIONS; aPrime++) {
        if (singleTargetPred.qValues[aPrime] > maxTargetQVal) {
          maxTargetQVal = singleTargetPred.qValues[aPrime];
          targetMaxAction = aPrime;
        }
      }
      const transitionTargetQ = envResult.done ? r : r + gamma * maxTargetQVal;
      const transitionCurrentQ = currentPred.qValues[action];
      const transitionTdError = transitionTargetQ - transitionCurrentQ;

      lastTransition = {
        state: s,
        action,
        reward: r,
        nextState: envResult.nextState,
        targetMaxAction,
        tdError: transitionTdError,
        targetQ: transitionTargetQ,
        currentQ: transitionCurrentQ,
      };

      currentTrajectory.push({
        row: nextRow,
        col: nextCol,
        action,
        reward: r,
        gValue: 0,
        advantage: 0,
      });

      if (envResult.done || epSteps >= 120) {
        lastCompletedEpisodeLength = epSteps;
        recentReturns.push(cumReturn);
        if (recentReturns.length > 20) recentReturns.shift();

        episode += 1;
        isTerminal = true;

        if (speed === 'single-step') {
          // In single-step mode, preserve the terminal landing state (Goal, Lava) for 1 frame
          break;
        } else {
          cumReturn = 0;
          epSteps = 0;
          nextRow = grid.startPos[0];
          nextCol = grid.startPos[1];
          isTerminal = false;
          currentTrajectory = [];
          break;
        }
      }
    }

    const avgStepLoss = stepLossCount > 0 ? stepLossTotal / stepLossCount : state.meanLoss;
    recentLosses.push(avgStepLoss);
    if (recentLosses.length > 20) recentLosses.shift();

    // Refresh cached Q-table across all 36 states
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const s = r * GRID_SIZE + c;
        const feat = extractStateFeatures(r, c, grid.goalPos, grid.tiles);
        const { qValues } = forwardPass(
          feat,
          state.onlineW1,
          state.onlineB1,
          state.onlineW2,
          state.onlineB2
        );
        for (let a = 0; a < NUM_ACTIONS; a++) {
          state.cachedQTable[s * NUM_ACTIONS + a] = qValues[a];
        }
      }
    }

    const nextState: DQNState = {
      ...state,
      epoch: state.epoch + 1,
      episode,
      stepCount: state.stepCount + actualStepsTaken,
      targetSyncCounter: syncCounter,
      replayHead,
      replayCount,
      lastSampledIndices,
      agentRow: nextRow,
      agentCol: nextCol,
      lastAction,
      lastReward,
      cumulativeReturn: cumReturn,
      stepsInEpisode: epSteps,
      lastCompletedEpisodeLength,
      lastRolloutSpeed: speed as 'single-step' | 'full-episode',
      isTerminal,
      trajectory: currentTrajectory,
      lastTransition,
      recentReturns,
      recentLosses,
      meanLoss: avgStepLoss,
    };

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: dqnPlugin.loss(nextState, dataset, params),
    };
  },

  predict: (input: Float32Array, state: DQNState, _params?: Record<string, any>): Float32Array => {
    let s: number;
    if (input.length === 1) {
      s = Math.max(0, Math.min(NUM_STATES - 1, Math.round(input[0])));
    } else if (input.length === 2) {
      const r = Math.max(0, Math.min(GRID_SIZE - 1, Math.round(input[0])));
      const c = Math.max(0, Math.min(GRID_SIZE - 1, Math.round(input[1])));
      s = r * GRID_SIZE + c;
    } else {
      // Direct feature vector input (e.g. 6D features)
      const { qValues } = forwardPass(input, state.onlineW1, state.onlineB1, state.onlineW2, state.onlineB2);
      return qValues;
    }
    const actions = new Float32Array(NUM_ACTIONS);
    for (let a = 0; a < NUM_ACTIONS; a++) {
      actions[a] = state.cachedQTable[s * NUM_ACTIONS + a];
    }
    return actions;
  },

  loss: (state: DQNState, _dataset?: DatasetContainer, params?: Record<string, any>): TelemetryMetrics => {
    const last10 = state.recentReturns.slice(-10);
    const movingAvg = last10.length > 0 ? last10.reduce((a, b) => a + b, 0) / last10.length : 0;
    const lastEpReturn = state.recentReturns.length > 0 ? state.recentReturns[state.recentReturns.length - 1] : 0;
    const speed = String(params?.rolloutSpeed ?? state.lastRolloutSpeed ?? 'single-step');
    const episodeLength =
      speed === 'full-episode' || state.isTerminal ? state.lastCompletedEpisodeLength : state.stepsInEpisode;

    // Evaluation suboptimality / regret against optimal path return (~9.6 for cliff walking)
    const testLoss = Number(Math.max(0, 9.6 - movingAvg).toFixed(4));
    const trainLoss = Number(state.meanLoss.toFixed(4));

    return {
      trainLoss,
      testLoss,
      custom: {
        episodeReturn: Number(lastEpReturn.toFixed(2)),
        movingAvgReturn: Number(movingAvg.toFixed(2)),
        episodeLength,
        meanTdError: Number(Math.sqrt(Math.max(0, state.meanLoss)).toFixed(4)),
        replayCount: `${state.replayCount} / ${REPLAY_CAPACITY}`,
        targetSyncCounter: `${state.targetSyncCounter} steps`,
        totalEpisodes: state.episode,
      },
    };
  },

  getVisualArtifacts: (
    state: DQNState,
    dataset: DatasetContainer,
    params: Record<string, any>
  ): VisualArtifact[] => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const speed = (params.rolloutSpeed ?? 'single-step') as 'single-step' | 'full-episode';
    const syncInterval = Math.max(1, Number(params.targetNetworkSync ?? 20));

    // Compute greedy policy actions argmax_a Q(s, a)
    const greedyActions = new Int32Array(NUM_STATES);
    for (let s = 0; s < NUM_STATES; s++) {
      let bestQ = -Infinity;
      let bestA = 0;
      for (let a = 0; a < NUM_ACTIONS; a++) {
        const q = state.cachedQTable[s * NUM_ACTIONS + a];
        if (q > bestQ) {
          bestQ = q;
          bestA = a;
        }
      }
      greedyActions[s] = bestA;
    }

    // Build sample transitions preview for Experience Replay Strip
    const sampledSet = new Set(state.lastSampledIndices);
    const replayBufferSample: {
      s: number;
      a: number;
      r: number;
      nextS: number;
      done: boolean;
      isSampled: boolean;
    }[] = [];

    const displayCount = Math.min(24, state.replayCount);
    for (let i = 0; i < displayCount; i++) {
      const idx = (state.replayHead - displayCount + i + REPLAY_CAPACITY) % REPLAY_CAPACITY;
      // Recover approximate state index from features (r/5, c/5)
      const r = Math.round(state.replayStates[idx * FEATURE_DIM] * (GRID_SIZE - 1));
      const c = Math.round(state.replayStates[idx * FEATURE_DIM + 1] * (GRID_SIZE - 1));
      const nextR = Math.round(state.replayNextStates[idx * FEATURE_DIM] * (GRID_SIZE - 1));
      const nextC = Math.round(state.replayNextStates[idx * FEATURE_DIM + 1] * (GRID_SIZE - 1));

      replayBufferSample.push({
        s: r * GRID_SIZE + c,
        a: state.replayActions[idx],
        r: state.replayRewards[idx],
        nextS: nextR * GRID_SIZE + nextC,
        done: state.replayDones[idx] === 1,
        isSampled: sampledSet.has(idx),
      });
    }

    return [
      {
        type: 'rl-gridworld-overlay',
        algorithmId: 'dqn',
        gridSize: GRID_SIZE,
        tiles: new Uint8Array(grid.tiles),
        qValues: new Float32Array(state.cachedQTable),
        greedyActions,
        agentState: {
          row: state.agentRow,
          col: state.agentCol,
          lastAction: state.lastAction,
          lastReward: state.lastReward,
          cumulativeReturn: state.cumulativeReturn,
          steps: state.stepsInEpisode,
          isTerminal: state.isTerminal,
        },
        lastTransition: state.lastTransition,
        trajectory: state.trajectory,
        replayBufferSample,
        targetNetworkSync: {
          stepCounter: state.targetSyncCounter,
          syncInterval,
          syncProgress: Math.min(1.0, state.targetSyncCounter / syncInterval),
        },
        rolloutSpeed: speed,
      },
    ];
  },

  serializeSnapshot: (state: DQNState): Float32Array => {
    // Pack online network weights into flat snapshot
    const totalLen = state.onlineW1.length + state.onlineB1.length + state.onlineW2.length + state.onlineB2.length;
    const snap = new Float32Array(totalLen);
    let offset = 0;
    snap.set(state.onlineW1, offset);
    offset += state.onlineW1.length;
    snap.set(state.onlineB1, offset);
    offset += state.onlineB1.length;
    snap.set(state.onlineW2, offset);
    offset += state.onlineW2.length;
    snap.set(state.onlineB2, offset);
    return snap;
  },

  restoreSnapshot: (buffer: Float32Array, state: DQNState): DQNState => {
    let offset = 0;
    const onlineW1 = new Float32Array(state.onlineW1.length);
    onlineW1.set(buffer.subarray(offset, offset + onlineW1.length));
    offset += onlineW1.length;

    const onlineB1 = new Float32Array(state.onlineB1.length);
    onlineB1.set(buffer.subarray(offset, offset + onlineB1.length));
    offset += onlineB1.length;

    const onlineW2 = new Float32Array(state.onlineW2.length);
    onlineW2.set(buffer.subarray(offset, offset + onlineW2.length));
    offset += onlineW2.length;

    const onlineB2 = new Float32Array(state.onlineB2.length);
    onlineB2.set(buffer.subarray(offset, offset + onlineB2.length));

    return {
      ...state,
      onlineW1,
      onlineB1,
      onlineW2,
      onlineB2,
      targetW1: new Float32Array(onlineW1),
      targetB1: new Float32Array(onlineB1),
      targetW2: new Float32Array(onlineW2),
      targetB2: new Float32Array(onlineB2),
    };
  },

  explain: (state: DQNState, params: Record<string, any>): ExplanationPayload => {
    const syncInterval = Number(params.targetNetworkSync ?? 20);
    const replayMode = String(params.replayBufferMode ?? 'enabled-random-batch');

    return {
      summary:
        'Deep Q-Network (DQN) replaces tabular state-action lookup tables with a neural network function approximator Q_θ(s, a), utilizing an Experience Replay ring buffer to break autocorrelation and a frozen Target Network θ⁻ to stabilize Bellman targets.',
      coreAssumption:
        'States share continuous underlying representations allowing neural generalization; uncorrelated mini-batch sampling from past experience preserves i.i.d. gradient descent guarantees.',
      whenItFails:
        'Disabling Experience Replay and Target Network causes deadly triad instability: consecutive correlated transitions (s_t, s_{t+1}) paired with a constantly moving target lead to runaway divergence or policy collapse.',
      microLog: [
        `Sample ${replayMode === 'enabled-random-batch' ? 'random mini-batch of 16 past transitions from replay memory' : 'consecutive correlated single step'}.`,
        `Compute target y = r + γ * max_a' Q_θ⁻(s', a') with Target Network (sync every ${syncInterval} steps: ${state.targetSyncCounter}/${syncInterval}).`,
        `Backpropagate MSE loss (Q_θ - y)² through 2-layer MLP with SGD step size η = ${params.learningRate ?? 0.01}.`,
      ],
      mathLatex: {
        objective: 'L(\\theta) = \\mathbb{E}_{(s, a, r, s\') \\sim \\mathcal{D}} \\left[ \\left( r + \\gamma \\max_{a\'} Q_{\\theta^-}(s\', a\') - Q_\\theta(s, a) \\right)^2 \\right]',
        lossOrUpdate: '\\theta \\leftarrow \\theta - \\eta \\nabla_\\theta L(\\theta), \\quad \\theta^- \\leftarrow \\theta \\text{ every } C \\text{ steps}',
      },
      codeSnippet: {
        sklearn: `# Conceptual DQN Experience Replay & Target Step
batch = replay_buffer.sample(batch_size=16)
targets = batch.rewards + gamma * np.max(target_net.predict(batch.next_states), axis=1) * (1 - batch.dones)
q_network.fit(batch.states, targets, epochs=1, verbose=0)`,
        pytorch: `# PyTorch DQN Loss Calculation
q_values = online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
with torch.no_grad():
    max_next_q = target_net(next_states).max(1)[0]
    targets = rewards + gamma * max_next_q * (1 - dones)
loss = F.mse_loss(q_values, targets)`,
      },
    };
  },
};
