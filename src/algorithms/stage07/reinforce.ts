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

export interface ReinforceState {
  epoch: number;
  episode: number;
  stepCount: number;

  // Policy Logits θ ∈ R^(36 x 4)
  theta: Float32Array; // 36 * 4

  // Baseline moving average
  movingBaseline: number;

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
  pendingSteps: {
    row: number;
    col: number;
    stateIdx: number;
    action: number;
    reward: number;
    probs: Float32Array;
  }[];

  // Full Trajectory Ribbon of the last rollout
  trajectory: {
    row: number;
    col: number;
    action: number;
    reward: number;
    gValue: number;
    advantage: number;
  }[];

  recentReturns: number[];
  recentGradNorms: number[];
  meanEntropy: number;
  seed: number;
}

/** Compute softmax action probabilities π_θ(·|s) */
export function computePolicyProbabilities(
  theta: Float32Array,
  stateIdx: number,
  temperature: number = 1.0
): Float32Array {
  const probs = new Float32Array(NUM_ACTIONS);
  const tau = Math.max(0.1, temperature);

  let maxLogit = -Infinity;
  const offset = stateIdx * NUM_ACTIONS;
  for (let a = 0; a < NUM_ACTIONS; a++) {
    const logit = theta[offset + a];
    if (logit > maxLogit) maxLogit = logit;
  }

  let sumExp = 0;
  for (let a = 0; a < NUM_ACTIONS; a++) {
    probs[a] = Math.exp((theta[offset + a] - maxLogit) / tau);
    sumExp += probs[a];
  }

  for (let a = 0; a < NUM_ACTIONS; a++) {
    probs[a] /= sumExp;
  }

  return probs;
}

export const reinforcePlugin: AlgorithmPlugin<ReinforceState> = {
  id: 'policy-gradient',
  name: 'REINFORCE (Monte-Carlo Policy Gradient)',
  shortCode: 'REINFORCE',
  stage: 'rl-gridworld',
  taskType: 'reinforcement-learning',
  executionMode: 'epoch-loop',
  description:
    'Likelihood-ratio policy gradient optimization maximizing expected trajectory return via Monte-Carlo sampling, moving-average baseline subtraction, and entropy regularization.',
  paperReference:
    'Williams (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3), 229-256.',

  primaryParams: [
    {
      id: 'learningRate',
      name: 'Learning Rate (α)',
      type: 'slider',
      min: 0.002,
      max: 0.1,
      step: 0.005,
      default: 0.02,
      description: 'Step size for policy gradient ascent updates: θ ← θ + α ∇_θ J(θ).',
    },
    {
      id: 'baselineMode',
      name: 'Baseline Subtraction',
      type: 'select',
      options: [
        { label: 'Moving-Average Baseline', value: 'mean-return-baseline' },
        { label: 'No Baseline (Raw Returns)', value: 'no-baseline-raw' },
      ],
      default: 'mean-return-baseline',
      description: 'Subtracts state-independent baseline b from return G_t to reduce gradient variance without introducing bias.',
    },
    {
      id: 'entropyBonus',
      name: 'Entropy Bonus (β)',
      type: 'slider',
      min: 0.0,
      max: 0.2,
      step: 0.01,
      default: 0.02,
      description: 'Entropy regularization coefficient preventing premature deterministic policy collapse.',
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
      description: 'Temporal discount factor in Monte-Carlo return calculation G_t = ∑ γᵏ r_{t+k+1}.',
    },
    {
      id: 'temperature',
      name: 'Softmax Temperature (τ)',
      type: 'slider',
      min: 0.1,
      max: 2.0,
      step: 0.1,
      default: 1.0,
      description: 'Temperature scaling policy logits before softmax normalization.',
    },
    {
      id: 'rolloutSpeed',
      name: 'Rollout Cadence',
      type: 'select',
      options: [
        { label: '1 Step / Frame', value: 'single-step' },
        { label: '1 Episode / Frame', value: 'full-episode' },
      ],
      default: 'full-episode',
      description: 'Step through a trajectory 1 transition per frame or run a complete Monte-Carlo episode per frame.',
    },
  ],

  symbolicLinks: [
    {
      id: 'policy-gradient-theorem',
      symbolLatex: '\\nabla_\\theta J(\\theta) = \\mathbb{E}_\\tau \\left[ \\sum_{t=0}^{T-1} \\nabla_\\theta \\ln \\pi_\\theta(a_t|s_t) \\Big(G_t - b\\Big) \\right]',
      label: 'Policy Gradient Theorem',
      description: 'The foundation of policy-based RL: increases probability of actions that yielded above-average returns.',
      highlightTarget: 'weights-inset',
    },
  ],

  init: (dataset: DatasetContainer, _params: Record<string, any>, seed: number): ReinforceState => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const theta = new Float32Array(NUM_STATES * NUM_ACTIONS); // All logits initially 0 -> uniform policy

    return {
      epoch: 0,
      episode: 0,
      stepCount: 0,
      theta,
      movingBaseline: 0,
      agentRow: grid.startPos[0],
      agentCol: grid.startPos[1],
      lastAction: -1,
      lastReward: 0,
      cumulativeReturn: 0,
      stepsInEpisode: 0,
      lastCompletedEpisodeLength: 0,
      lastRolloutSpeed: String(_params.rolloutSpeed ?? 'full-episode') as 'single-step' | 'full-episode',
      isTerminal: false,
      pendingSteps: [],
      trajectory: [],
      recentReturns: [],
      recentGradNorms: [],
      meanEntropy: Math.log(NUM_ACTIONS), // Initially ln(4) ≈ 1.386 nats
      seed,
    };
  },

  step: (state: ReinforceState, dataset: DatasetContainer, params: Record<string, any>): StepResult<ReinforceState> => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const gamma = Number(params.discountFactor ?? 0.95);
    const lr = Number(params.learningRate ?? 0.02);
    const betaEnt = Number(params.entropyBonus ?? 0.02);
    const baselineMode = String(params.baselineMode ?? 'mean-return-baseline');
    const tau = Number(params.temperature ?? 1.0);
    const speed = String(params.rolloutSpeed ?? 'full-episode') as 'single-step' | 'full-episode';

    const rng = mulberry32(state.seed + state.epoch * 1039 + state.stepCount);
    const MAX_STEPS = 100;

    const samplePolicyAction = (probs: Float32Array): number => {
      const roll = rng();
      let acc = 0;
      for (let a = 0; a < NUM_ACTIONS; a++) {
        acc += probs[a];
        if (roll <= acc || a === NUM_ACTIONS - 1) {
          return a;
        }
      }
      return 0;
    };

    const applyMonteCarloUpdate = (
      rawSteps: ReinforceState['pendingSteps'],
      epReturn: number,
      endRow: number,
      endCol: number,
      stepsAdded: number
    ): StepResult<ReinforceState> => {
      const T = rawSteps.length;
      const gValues = new Float32Array(T);
      let runningG = 0;
      for (let t = T - 1; t >= 0; t--) {
        runningG = rawSteps[t].reward + gamma * runningG;
        gValues[t] = runningG;
      }

      let baseline = 0;
      if (baselineMode === 'mean-return-baseline') {
        baseline = state.movingBaseline === 0 ? epReturn : 0.9 * state.movingBaseline + 0.1 * epReturn;
      }

      let totalGradNormSq = 0;
      const trajectoryRibbon: ReinforceState['trajectory'] = [];

      for (let t = 0; t < T; t++) {
        const { stateIdx, action, probs, row, col, reward } = rawSteps[t];
        const G_t = gValues[t];
        const advantage = G_t - baseline;

        trajectoryRibbon.push({ row, col, action, reward, gValue: G_t, advantage });

        const offset = stateIdx * NUM_ACTIONS;
        let entropy = 0;
        for (let a = 0; a < NUM_ACTIONS; a++) {
          if (probs[a] > 1e-7) {
            entropy -= probs[a] * Math.log(probs[a]);
          }
        }

        for (let a = 0; a < NUM_ACTIONS; a++) {
          const gradLikelihood = ((a === action ? 1.0 : 0.0) - probs[a]) / tau;
          const gradEntropy = probs[a] > 1e-7 ? (-probs[a] * (Math.log(probs[a]) + entropy)) / tau : 0;
          const totalGrad = advantage * gradLikelihood + betaEnt * gradEntropy;

          totalGradNormSq += totalGrad * totalGrad;
          state.theta[offset + a] += lr * totalGrad;
        }
      }

      const gradNorm = Math.sqrt(totalGradNormSq / Math.max(1, T * NUM_ACTIONS));
      const recentGradNorms = [...state.recentGradNorms, gradNorm];
      if (recentGradNorms.length > 20) recentGradNorms.shift();

      const recentReturns = [...state.recentReturns, epReturn];
      if (recentReturns.length > 20) recentReturns.shift();

      let totalEntropy = 0;
      for (let s = 0; s < NUM_STATES; s++) {
        const probs = computePolicyProbabilities(state.theta, s, tau);
        for (let a = 0; a < NUM_ACTIONS; a++) {
          if (probs[a] > 1e-7) totalEntropy -= probs[a] * Math.log(probs[a]);
        }
      }
      const meanEntropy = totalEntropy / NUM_STATES;

      const nextState: ReinforceState = {
        ...state,
        epoch: state.epoch + 1,
        episode: state.episode + 1,
        stepCount: state.stepCount + stepsAdded,
        movingBaseline: baseline,
        agentRow: endRow,
        agentCol: endCol,
        lastAction: T > 0 ? rawSteps[T - 1].action : -1,
        lastReward: T > 0 ? rawSteps[T - 1].reward : 0,
        cumulativeReturn: epReturn,
        stepsInEpisode: T,
        lastCompletedEpisodeLength: T,
        lastRolloutSpeed: speed,
        isTerminal: true,
        pendingSteps: [],
        trajectory: trajectoryRibbon,
        recentReturns,
        recentGradNorms,
        meanEntropy,
      };

      return {
        state: nextState,
        epoch: nextState.epoch,
        metrics: reinforcePlugin.loss(nextState, dataset, params),
      };
    };

    if (speed === 'single-step') {
      let curRow = state.isTerminal ? grid.startPos[0] : state.agentRow;
      let curCol = state.isTerminal ? grid.startPos[1] : state.agentCol;
      let epReturn = state.isTerminal ? 0 : state.cumulativeReturn;
      const pendingSteps = state.isTerminal ? [] : [...state.pendingSteps];
      const partialTrajectory = state.isTerminal ? [] : [...state.trajectory];

      const s = curRow * GRID_SIZE + curCol;
      const probs = computePolicyProbabilities(state.theta, s, tau);
      const action = samplePolicyAction(probs);
      const envRes = stepEnvironment(curRow, curCol, action, grid, rng);
      epReturn += envRes.reward;

      pendingSteps.push({
        row: curRow,
        col: curCol,
        stateIdx: s,
        action,
        reward: envRes.reward,
        probs,
      });
      partialTrajectory.push({
        row: curRow,
        col: curCol,
        action,
        reward: envRes.reward,
        gValue: 0,
        advantage: 0,
      });

      curRow = envRes.nextRow;
      curCol = envRes.nextCol;

      if (envRes.done || pendingSteps.length >= MAX_STEPS) {
        return applyMonteCarloUpdate(pendingSteps, epReturn, curRow, curCol, 1);
      }

      const nextState: ReinforceState = {
        ...state,
        epoch: state.epoch + 1,
        stepCount: state.stepCount + 1,
        agentRow: curRow,
        agentCol: curCol,
        lastAction: action,
        lastReward: envRes.reward,
        cumulativeReturn: epReturn,
        stepsInEpisode: pendingSteps.length,
        lastRolloutSpeed: speed,
        isTerminal: false,
        pendingSteps,
        trajectory: partialTrajectory,
      };

      return {
        state: nextState,
        epoch: nextState.epoch,
        metrics: reinforcePlugin.loss(nextState, dataset, params),
      };
    }

    // Full-episode Monte-Carlo rollout
    let curRow = grid.startPos[0];
    let curCol = grid.startPos[1];
    let epReturn = 0;
    const rawSteps: ReinforceState['pendingSteps'] = [];

    for (let t = 0; t < MAX_STEPS; t++) {
      const s = curRow * GRID_SIZE + curCol;
      const probs = computePolicyProbabilities(state.theta, s, tau);
      const action = samplePolicyAction(probs);
      const envRes = stepEnvironment(curRow, curCol, action, grid, rng);
      epReturn += envRes.reward;

      rawSteps.push({
        row: curRow,
        col: curCol,
        stateIdx: s,
        action,
        reward: envRes.reward,
        probs,
      });

      curRow = envRes.nextRow;
      curCol = envRes.nextCol;

      if (envRes.done) {
        break;
      }
    }

    return applyMonteCarloUpdate(rawSteps, epReturn, curRow, curCol, rawSteps.length);
  },

  predict: (input: Float32Array, state: ReinforceState): Float32Array => {
    let s: number;
    if (input.length === 1) {
      s = Math.max(0, Math.min(NUM_STATES - 1, Math.round(input[0])));
    } else {
      const r = Math.max(0, Math.min(GRID_SIZE - 1, Math.round(input[0])));
      const c = Math.max(0, Math.min(GRID_SIZE - 1, Math.round(input[1])));
      s = r * GRID_SIZE + c;
    }
    return computePolicyProbabilities(state.theta, s);
  },

  loss: (state: ReinforceState, _dataset?: DatasetContainer, params?: Record<string, any>): TelemetryMetrics => {
    const last10 = state.recentReturns.slice(-10);
    const movingAvg = last10.length > 0 ? last10.reduce((a, b) => a + b, 0) / last10.length : 0;
    const lastEpReturn = state.recentReturns.length > 0 ? state.recentReturns[state.recentReturns.length - 1] : 0;
    const lastGrad = state.recentGradNorms.length > 0 ? state.recentGradNorms[state.recentGradNorms.length - 1] : 0;
    const speed = String(params?.rolloutSpeed ?? state.lastRolloutSpeed ?? 'full-episode');
    const episodeLength =
      speed === 'full-episode' || state.isTerminal ? state.lastCompletedEpisodeLength : state.stepsInEpisode;

    // Evaluation suboptimality / regret against optimal path return (~9.6 for cliff walking)
    const testLoss = Number(Math.max(0, 9.6 - movingAvg).toFixed(4));
    const trainLoss = Number(lastGrad.toFixed(4));

    return {
      trainLoss,
      testLoss,
      custom: {
        episodeReturn: Number(lastEpReturn.toFixed(2)),
        movingAvgReturn: Number(movingAvg.toFixed(2)),
        episodeLength,
        policyEntropy: `${state.meanEntropy.toFixed(3)} nats`,
        baselineValue: Number(state.movingBaseline.toFixed(2)),
        gradientNorm: trainLoss,
        totalEpisodes: state.episode,
      },
    };
  },

  getVisualArtifacts: (
    state: ReinforceState,
    dataset: DatasetContainer,
    params: Record<string, any>
  ): VisualArtifact[] => {
    const grid = dataset.rlGridworld || createRLGridworldDataset();
    const speed = (params.rolloutSpeed ?? 'full-episode') as 'single-step' | 'full-episode';
    const tau = Number(params.temperature ?? 1.0);

    // Compute all 36 state probabilities & greedy action indicators
    const policyProbabilities = new Float32Array(NUM_STATES * NUM_ACTIONS);
    const greedyActions = new Int32Array(NUM_STATES);
    // Construct fake Q-values scaled to probabilities for unified colormap renderer
    const pseudoQValues = new Float32Array(NUM_STATES * NUM_ACTIONS);

    for (let s = 0; s < NUM_STATES; s++) {
      const probs = computePolicyProbabilities(state.theta, s, tau);
      let bestA = 0;
      let maxP = -1;
      for (let a = 0; a < NUM_ACTIONS; a++) {
        const p = probs[a];
        policyProbabilities[s * NUM_ACTIONS + a] = p;
        pseudoQValues[s * NUM_ACTIONS + a] = p * 10.0; // scale [0, 1] -> [0, 10]
        if (p > maxP) {
          maxP = p;
          bestA = a;
        }
      }
      greedyActions[s] = bestA;
    }

    return [
      {
        type: 'rl-gridworld-overlay',
        algorithmId: 'policy-gradient',
        gridSize: GRID_SIZE,
        tiles: new Uint8Array(grid.tiles),
        qValues: pseudoQValues,
        policyProbabilities,
        policyLogits: new Float32Array(state.theta),
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
        trajectory: state.trajectory,
        rolloutSpeed: speed,
      },
    ];
  },

  serializeSnapshot: (state: ReinforceState): Float32Array => {
    return new Float32Array(state.theta);
  },

  restoreSnapshot: (buffer: Float32Array, state: ReinforceState): ReinforceState => {
    const theta = new Float32Array(NUM_STATES * NUM_ACTIONS);
    theta.set(buffer.subarray(0, NUM_STATES * NUM_ACTIONS));
    return {
      ...state,
      theta,
    };
  },

  explain: (state: ReinforceState, params: Record<string, any>): ExplanationPayload => {
    const baselineMode = String(params.baselineMode ?? 'mean-return-baseline');

    return {
      summary:
        'REINFORCE is a Monte-Carlo policy gradient algorithm directly optimizing parameterized policy probabilities π_θ(a|s) via likelihood-ratio gradient ascent without bootstrap approximations.',
      coreAssumption:
        'Episodes terminate in finite horizon; unrolled sample trajectories provide unbiased estimates of the true expected policy gradient ∇_θ J(θ).',
      whenItFails:
        'Without a baseline (b=0), positive trajectories all receive positive gradient steps, causing high variance and erratic policy oscillation; without entropy bonus, probabilities quickly collapse to 100% on suboptimal paths.',
      microLog: [
        `Sample complete episode trajectory τ = (s₀, a₀, r₁, ... s_T) of ${state.stepsInEpisode} steps with total return ${state.cumulativeReturn.toFixed(2)}.`,
        `Calculate discounted returns G_t = ∑ γᵏ r_{t+k+1} and subtract ${baselineMode === 'mean-return-baseline' ? `moving baseline b = ${state.movingBaseline.toFixed(2)}` : 'zero baseline (raw returns)'}.`,
        `Update logits θ_{s, a} using policy gradient theorem with entropy regularization bonus β = ${params.entropyBonus ?? 0.02}.`,
      ],
      mathLatex: {
        objective: 'J(\\theta) = \\mathbb{E}_{\\tau \\sim \\pi_\\theta} [R(\\tau)]',
        lossOrUpdate: '\\theta \\leftarrow \\theta + \\alpha \\sum_{t=0}^{T-1} \\nabla_\\theta \\ln \\pi_\\theta(a_t|s_t) \\Big(G_t - b\\Big) + \\beta_{\\text{ent}} \\nabla_\\theta H(\\pi_\\theta)',
      },
      codeSnippet: {
        sklearn: `# Monte-Carlo Policy Gradient (REINFORCE) Concept
# 1. Compute discounted returns
returns = []
G = 0
for r in reversed(episode_rewards):
    G = r + gamma * G
    returns.insert(0, G)
# 2. Baseline-centered Advantage
advantages = np.array(returns) - baseline`,
        pytorch: `# PyTorch REINFORCE Step with Baseline
log_probs = torch.stack(saved_log_probs)
returns = torch.tensor(discounted_returns)
advantages = returns - returns.mean()  # Baseline subtraction
policy_loss = -(log_probs * advantages).sum() - entropy_coef * entropy
optimizer.zero_grad()
policy_loss.backward()
optimizer.step()`,
      },
    };
  },
};

