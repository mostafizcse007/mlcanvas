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
} from '../../datasets/rlGridworld';
import { mulberry32 } from '../../utils/prng';

export interface QLearningState {
  epoch: number;
  episode: number;
  stepCount: number;
  qTable: Float32Array; // 36 * 4
  agentRow: number;
  agentCol: number;
  lastAction: number;
  lastReward: number;
  cumulativeReturn: number;
  stepsInEpisode: number;
  lastCompletedEpisodeLength: number;
  lastRolloutSpeed?: 'single-step' | 'full-episode';
  isTerminal: boolean;
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
  meanTdError: number;
  seed: number;
}

export const qLearningPlugin: AlgorithmPlugin<QLearningState> = {
  id: 'q-learning',
  name: 'Q-Learning (Off-Policy TD Control)',
  shortCode: 'Q-Learning',
  stage: 'rl-gridworld',
  taskType: 'reinforcement-learning',
  executionMode: 'epoch-loop',
  description:
    'Tabular off-policy temporal-difference control learning optimal action-values Q*(s, a) via Bellman optimality backups independent of the exploration policy.',
  paperReference: 'Watkins & Dayan (1992). Q-learning. Machine Learning, 8(3), 279-292.',

  primaryParams: [
    {
      id: 'discountFactor',
      name: 'Discount (γ)',
      type: 'slider',
      min: 0.0,
      max: 0.99,
      step: 0.05,
      default: 0.95,
      description: 'Temporal discount factor weighing future delayed returns against immediate living rewards.',
    },
    {
      id: 'explorationEpsilon',
      name: 'Exploration (ε)',
      type: 'slider',
      min: 0.0,
      max: 1.0,
      step: 0.05,
      default: 0.2,
      description: 'Probability of selecting a random exploratory action vs. greedy action argmax_a Q(s, a).',
    },
    {
      id: 'learningRate',
      name: 'Learning Rate (α)',
      type: 'slider',
      min: 0.01,
      max: 1.0,
      step: 0.05,
      default: 0.2,
      description: 'Step size for the Bellman TD update blending old Q-values with target estimates.',
    },
  ],

  advancedParams: [
    {
      id: 'explorationStrategy',
      name: 'Exploration Strategy',
      type: 'select',
      options: [
        { label: 'ε-Greedy', value: 'epsilon-greedy' },
        { label: 'Boltzmann Softmax', value: 'boltzmann-softmax' },
      ],
      default: 'epsilon-greedy',
      description: 'Mechanism used to select actions during rollout.',
    },
    {
      id: 'optimisticInit',
      name: 'Optimistic Init (Q₀)',
      type: 'slider',
      min: 0.0,
      max: 5.0,
      step: 0.5,
      default: 0.0,
      description: 'Initial value for all Q-table entries; values > 0 encourage initial exploration of untried actions.',
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
      id: 'q-update',
      symbolLatex: 'Q(s, a) \\leftarrow Q(s, a) + \\alpha [r + \\gamma \\max_{a\'} Q(s\', a\') - Q(s, a)]',
      label: 'Bellman Optimality Backup',
      description: 'Off-policy temporal difference equation driving Q-values toward the optimal action-value function.',
      highlightTarget: 'weights-inset',
    },
  ],

  init: (dataset: DatasetContainer, params: Record<string, any>, seed: number): QLearningState => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const q0 = Number(params.optimisticInit ?? 0.0);
    const qTable = new Float32Array(NUM_STATES * NUM_ACTIONS);
    if (q0 !== 0) {
      qTable.fill(q0);
    }

    return {
      epoch: 0,
      episode: 0,
      stepCount: 0,
      qTable,
      agentRow: grid.startPos[0],
      agentCol: grid.startPos[1],
      lastAction: -1,
      lastReward: 0,
      cumulativeReturn: 0,
      stepsInEpisode: 0,
      lastCompletedEpisodeLength: 0,
      lastRolloutSpeed: String(params.rolloutSpeed ?? 'single-step') as 'single-step' | 'full-episode',
      isTerminal: false,
      trajectory: [],
      recentReturns: [],
      meanTdError: 0,
      seed,
    };
  },

  step: (state: QLearningState, dataset: DatasetContainer, params: Record<string, any>): StepResult<QLearningState> => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const gamma = Number(params.discountFactor ?? 0.95);
    const epsilon = Number(params.explorationEpsilon ?? 0.2);
    const alpha = Number(params.learningRate ?? 0.2);
    const strategy = String(params.explorationStrategy ?? 'epsilon-greedy');
    const speed = String(params.rolloutSpeed ?? 'single-step');

    const rng = mulberry32(state.seed + state.epoch * 997 + state.stepCount);

    let nextRow = state.agentRow;
    let nextCol = state.agentCol;
    let cumReturn = state.cumulativeReturn;
    let epSteps = state.stepsInEpisode;
    let lastCompletedEpisodeLength = state.lastCompletedEpisodeLength;
    let episode = state.episode;
    let lastTransition = state.lastTransition;
    let lastAction = state.lastAction;
    let lastReward = state.lastReward;
    let isTerminal = state.isTerminal;
    let tdErrorsSum = 0;
    let tdSteps = 0;
    let actualStepsTaken = 0;

    let currentTrajectory = [...state.trajectory];
    const recentReturns = [...state.recentReturns];

    // If previous tick ended in a terminal state (Goal / Lava / Timeout), reset for new episode
    if (isTerminal) {
      nextRow = grid.startPos[0];
      nextCol = grid.startPos[1];
      cumReturn = 0;
      epSteps = 0;
      isTerminal = false;
      currentTrajectory = [];
    }

    // Determine how many steps to execute this tick
    const maxStepsToTake = speed === 'full-episode' ? 120 : 1;

    for (let sIdx = 0; sIdx < maxStepsToTake; sIdx++) {
      actualStepsTaken += 1;
      const s = nextRow * GRID_SIZE + nextCol;

      // Select action
      let action = 0;
      if (strategy === 'boltzmann-softmax') {
        const tau = Math.max(0.05, epsilon);
        const expQ = new Float32Array(NUM_ACTIONS);
        let maxQ = -Infinity;
        for (let a = 0; a < NUM_ACTIONS; a++) {
          const q = state.qTable[s * NUM_ACTIONS + a];
          if (q > maxQ) maxQ = q;
        }
        let sumExp = 0;
        for (let a = 0; a < NUM_ACTIONS; a++) {
          expQ[a] = Math.exp((state.qTable[s * NUM_ACTIONS + a] - maxQ) / tau);
          sumExp += expQ[a];
        }
        const roll = rng();
        let acc = 0;
        for (let a = 0; a < NUM_ACTIONS; a++) {
          acc += expQ[a] / sumExp;
          if (roll <= acc || a === NUM_ACTIONS - 1) {
            action = a;
            break;
          }
        }
      } else {
        // ε-Greedy
        if (rng() < epsilon) {
          action = Math.floor(rng() * NUM_ACTIONS);
        } else {
          let bestA = 0;
          let bestQ = -Infinity;
          for (let a = 0; a < NUM_ACTIONS; a++) {
            const q = state.qTable[s * NUM_ACTIONS + a];
            if (q > bestQ) {
              bestQ = q;
              bestA = a;
            }
          }
          action = bestA;
        }
      }

      // Execute in environment
      const envResult = stepEnvironment(nextRow, nextCol, action, grid, rng);
      const nextS = envResult.nextState;
      const r = envResult.reward;
      cumReturn += r;
      epSteps += 1;

      // Compute Bellman target: y = r + gamma * max_a' Q(s', a')
      let maxNextQ = -Infinity;
      let targetMaxAction = 0;
      for (let aPrime = 0; aPrime < NUM_ACTIONS; aPrime++) {
        const nextQ = state.qTable[nextS * NUM_ACTIONS + aPrime];
        if (nextQ > maxNextQ) {
          maxNextQ = nextQ;
          targetMaxAction = aPrime;
        }
      }

      const targetQ = envResult.done ? r : r + gamma * maxNextQ;
      const currentQ = state.qTable[s * NUM_ACTIONS + action];
      const tdError = targetQ - currentQ;

      // Update Q-table: Q(s, a) <- Q(s, a) + alpha * tdError
      state.qTable[s * NUM_ACTIONS + action] = currentQ + alpha * tdError;

      tdErrorsSum += Math.abs(tdError);
      tdSteps += 1;

      lastAction = action;
      lastReward = r;
      nextRow = envResult.nextRow;
      nextCol = envResult.nextCol;

      lastTransition = {
        state: s,
        action,
        reward: r,
        nextState: nextS,
        targetMaxAction,
        tdError,
        targetQ,
        currentQ,
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
        // Episode ended
        lastCompletedEpisodeLength = epSteps;
        recentReturns.push(cumReturn);
        if (recentReturns.length > 20) recentReturns.shift();

        episode += 1;
        isTerminal = true;

        if (speed === 'single-step') {
          // In single-step mode, preserve the terminal landing state (Goal, Lava) for 1 frame
          // so the user sees the agent reach the goal! Next step will reset to startPos.
          break;
        } else {
          // Full episode mode: reset immediately for the next episode
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

    const meanTd = tdSteps > 0 ? tdErrorsSum / tdSteps : state.meanTdError;

    const nextState: QLearningState = {
      ...state,
      epoch: state.epoch + 1,
      episode,
      stepCount: state.stepCount + actualStepsTaken,
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
      meanTdError: meanTd,
    };

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: qLearningPlugin.loss(nextState, dataset, params),
    };
  },

  predict: (input: Float32Array, state: QLearningState): Float32Array => {
    // Input can be 1D state index [s] or 2D [row, col]
    let s: number;
    if (input.length === 1) {
      s = Math.max(0, Math.min(NUM_STATES - 1, Math.round(input[0])));
    } else {
      const r = Math.max(0, Math.min(GRID_SIZE - 1, Math.round(input[0])));
      const c = Math.max(0, Math.min(GRID_SIZE - 1, Math.round(input[1])));
      s = r * GRID_SIZE + c;
    }
    const actions = new Float32Array(NUM_ACTIONS);
    for (let a = 0; a < NUM_ACTIONS; a++) {
      actions[a] = state.qTable[s * NUM_ACTIONS + a];
    }
    return actions;
  },

  loss: (state: QLearningState, _dataset?: DatasetContainer, params?: Record<string, any>): TelemetryMetrics => {
    const last10 = state.recentReturns.slice(-10);
    const movingAvg = last10.length > 0 ? last10.reduce((a, b) => a + b, 0) / last10.length : 0;
    const lastEpReturn = state.recentReturns.length > 0 ? state.recentReturns[state.recentReturns.length - 1] : 0;
    const speed = String(params?.rolloutSpeed ?? state.lastRolloutSpeed ?? 'single-step');
    const episodeLength =
      speed === 'full-episode' || state.isTerminal ? state.lastCompletedEpisodeLength : state.stepsInEpisode;

    // Evaluation suboptimality / regret against optimal path return (~9.6 for cliff walking)
    const testLoss = Number(Math.max(0, 9.6 - movingAvg).toFixed(4));
    const trainLoss = Number(state.meanTdError.toFixed(4));

    return {
      trainLoss,
      testLoss,
      custom: {
        episodeReturn: Number(lastEpReturn.toFixed(2)),
        movingAvgReturn: Number(movingAvg.toFixed(2)),
        episodeLength,
        meanTdError: trainLoss,
        totalEpisodes: state.episode,
        activeState: `(${state.agentRow}, ${state.agentCol})`,
      },
    };
  },

  getVisualArtifacts: (
    state: QLearningState,
    dataset: DatasetContainer,
    params: Record<string, any>
  ): VisualArtifact[] => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const speed = (params.rolloutSpeed ?? 'single-step') as 'single-step' | 'full-episode';

    // Compute greedy policy actions argmax_a Q(s, a)
    const greedyActions = new Int32Array(NUM_STATES);
    for (let s = 0; s < NUM_STATES; s++) {
      let bestQ = -Infinity;
      let bestA = 0;
      for (let a = 0; a < NUM_ACTIONS; a++) {
        const q = state.qTable[s * NUM_ACTIONS + a];
        if (q > bestQ) {
          bestQ = q;
          bestA = a;
        }
      }
      greedyActions[s] = bestA;
    }

    return [
      {
        type: 'rl-gridworld-overlay',
        algorithmId: 'q-learning',
        gridSize: GRID_SIZE,
        tiles: new Uint8Array(grid.tiles),
        qValues: new Float32Array(state.qTable),
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
        rolloutSpeed: speed,
      },
    ];
  },

  serializeSnapshot: (state: QLearningState): Float32Array => {
    // Snapshot Q-table
    return new Float32Array(state.qTable);
  },

  restoreSnapshot: (buffer: Float32Array, state: QLearningState): QLearningState => {
    const qTable = new Float32Array(NUM_STATES * NUM_ACTIONS);
    qTable.set(buffer.subarray(0, NUM_STATES * NUM_ACTIONS));
    return {
      ...state,
      qTable,
    };
  },

  explain: (state: QLearningState, params: Record<string, any>): ExplanationPayload => {
    const gamma = Number(params.discountFactor ?? 0.95);
    const alpha = Number(params.learningRate ?? 0.2);

    return {
      summary:
        'Q-Learning is a model-free, off-policy Temporal-Difference control algorithm that learns the optimal state-action value function Q*(s, a) using Bellman optimality backups without knowing transition probabilities.',
      coreAssumption:
        'The environment satisfies the Markov Decision Process (MDP) property, and all state-action pairs are visited infinitely often under an exploratory policy (e.g. ε-greedy).',
      whenItFails:
        'With zero exploration (ε=0), the agent prematurely locks into the first discovered sub-optimal reward; with myopic discount (γ=0), future delayed rewards remain completely invisible.',
      microLog: [
        `Select action a_t at state (${state.agentRow}, ${state.agentCol}) using ε-greedy strategy.`,
        `Observe reward r and next state s'; compute Bellman target y = r + ${gamma} * max_a' Q(s', a').`,
        `Update Q(s, a) with step size α = ${alpha}: TD error δ = ${state.lastTransition ? state.lastTransition.tdError.toFixed(3) : '0.000'}.`,
      ],
      mathLatex: {
        objective: 'Q^*(s, a) = \\mathbb{E}_{s\'} [r + \\gamma \\max_{a\'} Q^*(s\', a\') \\mid s, a]',
        lossOrUpdate: 'Q(s_t, a_t) \\leftarrow Q(s_t, a_t) + \\alpha \\Big[r_{t+1} + \\gamma \\max_{a\'} Q(s_{t+1}, a\') - Q(s_t, a_t)\\Big]',
      },
      codeSnippet: {
        sklearn: `# Python / Gymnasium Tabular Q-Learning Loop
import numpy as np

# Bellman Optimality Update
best_next_q = np.max(Q[next_state])
td_error = reward + gamma * best_next_q - Q[state, action]
Q[state, action] += alpha * td_error`,
        pytorch: `# PyTorch Vectorized Q-Learning Target
import torch

with torch.no_grad():
    target_q = reward + gamma * torch.max(q_values(next_state))
loss = (q_values(state)[action] - target_q).pow(2)`,
      },
    };
  },
};
