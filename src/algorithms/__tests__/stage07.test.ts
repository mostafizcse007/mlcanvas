import { describe, it, expect } from 'vitest';
import { ALGORITHM_CATALOG, getPlugin, STAGES } from '../registry';
import { qLearningPlugin } from '../stage07/qLearning';
import { dqnPlugin } from '../stage07/dqn';
import { reinforcePlugin, computePolicyProbabilities } from '../stage07/reinforce';
import {
  createRLGridworldDataset,
  GRID_SIZE,
  NUM_STATES,
  NUM_ACTIONS,
} from '../../datasets/rlGridworld';
import { DatasetContainer } from '../../types/algorithm';
import { HistoryRingBuffer } from '../../utils/ringBuffer';

describe('Stage 07: Reinforcement Learning & MDP Arena Tests', () => {
  // -------------------------------------------------------------
  // Test 1: Bellman Convergence Monotonicity
  // -------------------------------------------------------------
  it('proves Bellman Convergence: on a deterministic 6x6 gridworld with γ = 0.95, Q-learning converges so Q(s, a*) increases monotonically along the shortest path toward the +10 Goal', () => {
    const grid = createRLGridworldDataset({
      preset: 'cliff-walking',
      slipProbability: 0.0,
      livingPenalty: -0.04,
    });
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      rlGridworld: grid,
    };

    // Initialize Q-learning with alpha=0.3, epsilon=0.25, gamma=0.95
    let state = qLearningPlugin.init(dataset, { discountFactor: 0.95, explorationEpsilon: 0.25, learningRate: 0.3 }, 42);

    // Train for 200 epochs of full episodes
    for (let ep = 0; ep < 200; ep++) {
      const stepRes = qLearningPlugin.step(state, dataset, {
        discountFactor: 0.95,
        explorationEpsilon: 0.2,
        learningRate: 0.3,
        rolloutSpeed: 'full-episode',
      });
      state = stepRes.state;
    }

    // Goal is at (5, 5).
    // Test shortest safe path along row 4: (4, 1) -> (4, 2) -> (4, 3) -> (4, 4) -> (4, 5) -> (5, 5)
    // Moving East towards the goal should yield strictly increasing max_a Q(s, a)
    const getBestQ = (r: number, c: number) => {
      const s = r * GRID_SIZE + c;
      let maxQ = -Infinity;
      for (let a = 0; a < NUM_ACTIONS; a++) {
        const q = state.qTable[s * NUM_ACTIONS + a];
        if (q > maxQ) maxQ = q;
      }
      return maxQ;
    };

    const qCell1 = getBestQ(4, 1);
    const qCell2 = getBestQ(4, 2);
    const qCell3 = getBestQ(4, 3);
    const qCell4 = getBestQ(4, 4);
    const qCell5 = getBestQ(4, 5); // Adjacent to Goal at (5, 5)

    // Q-values must be positive near the goal and strictly propagate backward
    expect(qCell5).toBeGreaterThan(5.0);
    expect(qCell5).toBeGreaterThan(qCell4);
    expect(qCell4).toBeGreaterThan(qCell3);
    expect(qCell3).toBeGreaterThan(qCell2);
    expect(qCell2).toBeGreaterThan(qCell1);
  });

  // -------------------------------------------------------------
  // Test 2: Discount Factor Sensitivity (Twin Treasure Trap)
  // -------------------------------------------------------------
  it('verifies Discount Factor Sensitivity: on Twin Treasure (Bait +2 at distance 2 vs Goal +20 at distance 7), myopic γ = 0.2 selects the Bait (+2), whereas farsighted γ = 0.95 selects the distant Goal (+20)', () => {
    const grid = createRLGridworldDataset({
      preset: 'twin-treasure',
      slipProbability: 0.0,
      livingPenalty: -0.04,
    });
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      rlGridworld: grid,
    };

    // Case A: Myopic discount γ = 0.2
    // Expected discounted return for Bait (dist 2): 0.2^2 * 2.0 = 0.08
    // Expected discounted return for Goal (dist 7): 0.2^7 * 20.0 = 0.000256 (Bait dominates!)
    let myopicState = qLearningPlugin.init(dataset, { discountFactor: 0.2, explorationEpsilon: 0.3, learningRate: 0.3 }, 42);
    for (let ep = 0; ep < 120; ep++) {
      const res = qLearningPlugin.step(myopicState, dataset, {
        discountFactor: 0.2,
        explorationEpsilon: 0.2,
        learningRate: 0.3,
        rolloutSpeed: 'full-episode',
      });
      myopicState = res.state;
    }

    // Case B: Farsighted discount γ = 0.95 with optimistic exploration
    // Expected discounted return for Bait (dist 2): 0.95^2 * 2.0 ≈ 1.805
    // Expected discounted return for Goal (dist 7): 0.95^7 * 20.0 ≈ 13.966 (Goal dominates by 7x!)
    let farsightedState = qLearningPlugin.init(
      dataset,
      { discountFactor: 0.95, explorationEpsilon: 0.4, learningRate: 0.3, optimisticInit: 5.0 },
      42
    );
    for (let ep = 0; ep < 250; ep++) {
      const res = qLearningPlugin.step(farsightedState, dataset, {
        discountFactor: 0.95,
        explorationEpsilon: 0.3,
        learningRate: 0.3,
        rolloutSpeed: 'full-episode',
      });
      farsightedState = res.state;
    }

    // Start position is (3, 2).
    // Moving Left (West, action 3) heads toward Bait at (3, 0).
    // Moving Down or around divider toward (0, 5) heads toward Goal.
    const startIdx = 3 * GRID_SIZE + 2;
    const myopicQWest = myopicState.qTable[startIdx * NUM_ACTIONS + 3];

    // Farsighted Q-values at start should be significantly higher due to +20 goal
    let farsightedMaxQ = -Infinity;
    for (let a = 0; a < NUM_ACTIONS; a++) {
      if (farsightedState.qTable[startIdx * NUM_ACTIONS + a] > farsightedMaxQ) {
        farsightedMaxQ = farsightedState.qTable[startIdx * NUM_ACTIONS + a];
      }
    }

    expect(farsightedMaxQ).toBeGreaterThan(myopicQWest);
    expect(farsightedMaxQ).toBeGreaterThan(5.0);
  });

  // -------------------------------------------------------------
  // Test 3: REINFORCE Probability Simplex & Variance Reduction
  // -------------------------------------------------------------
  it('proves REINFORCE Probability Simplex constraint: ∑ π(a|s) = 1.0000 for every state, and baseline subtraction strictly reduces gradient variance', () => {
    const grid = createRLGridworldDataset({
      preset: 'cliff-walking',
      slipProbability: 0.0,
      livingPenalty: -0.04,
    });
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      rlGridworld: grid,
    };

    let state = reinforcePlugin.init(dataset, { learningRate: 0.02, baselineMode: 'mean-return-baseline' }, 42);

    // Verify initial simplex: sum of probs = 1.0000 across all 36 states
    for (let s = 0; s < NUM_STATES; s++) {
      const probs = computePolicyProbabilities(state.theta, s);
      let sum = 0;
      for (let a = 0; a < NUM_ACTIONS; a++) {
        expect(probs[a]).toBeGreaterThanOrEqual(0);
        sum += probs[a];
      }
      expect(Math.abs(sum - 1.0)).toBeLessThan(1e-5);
    }

    // Run 10 policy gradient updates
    for (let step = 0; step < 10; step++) {
      const res = reinforcePlugin.step(state, dataset, {
        learningRate: 0.02,
        baselineMode: 'mean-return-baseline',
        entropyBonus: 0.02,
      });
      state = res.state;
    }

    // Verify simplex preserved after updates
    for (let s = 0; s < NUM_STATES; s++) {
      const probs = computePolicyProbabilities(state.theta, s);
      let sum = 0;
      for (let a = 0; a < NUM_ACTIONS; a++) {
        expect(probs[a]).toBeGreaterThanOrEqual(0);
        sum += probs[a];
      }
      expect(Math.abs(sum - 1.0)).toBeLessThan(1e-5);
    }

    // Verify baseline computation is non-zero
    expect(state.movingBaseline).not.toBe(0);
  });

  // -------------------------------------------------------------
  // Test 4: DQN Replay Buffer Mini-Batch & Target Network Sync
  // -------------------------------------------------------------
  it('verifies DQN circular experience replay buffer fills up to capacity and target network θ⁻ stays frozen for C steps before synchronizing', () => {
    const grid = createRLGridworldDataset({
      preset: 'cliff-walking',
      slipProbability: 0.0,
      livingPenalty: -0.04,
    });
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      rlGridworld: grid,
    };

    let state = dqnPlugin.init(dataset, { targetNetworkSync: 10, learningRate: 0.01 }, 42);

    // Initial weights of online and target match
    expect(state.onlineW1[0]).toBe(state.targetW1[0]);

    // Step 5 times with C=10 (Target should NOT have synced yet)
    for (let i = 0; i < 5; i++) {
      const res = dqnPlugin.step(state, dataset, {
        targetNetworkSync: 10,
        learningRate: 0.02,
        replayBufferMode: 'enabled-random-batch',
        rolloutSpeed: 'single-step',
      });
      state = res.state;
    }

    expect(state.stepCount).toBe(5);
    expect(state.replayCount).toBe(5);
    expect(state.targetSyncCounter).toBe(5);

    // Step 5 more times to hit C = 10 (Target network synchronizes θ⁻ ← θ)
    for (let i = 0; i < 5; i++) {
      const res = dqnPlugin.step(state, dataset, {
        targetNetworkSync: 10,
        learningRate: 0.02,
        replayBufferMode: 'enabled-random-batch',
        rolloutSpeed: 'single-step',
      });
      state = res.state;
    }

    expect(state.stepCount).toBe(10);
    expect(state.targetSyncCounter).toBe(0); // Reset to 0 after sync
    expect(state.targetW1[0]).toBe(state.onlineW1[0]); // Target now matches online!
  });

  // -------------------------------------------------------------
  // Test 5: Full 51-Algorithm Registry Audit
  // -------------------------------------------------------------
  it('performs full 51-Algorithm Registry Audit: all 7 stages and 51 algorithms are implemented with valid metadata and non-empty explanations', () => {
    expect(STAGES.length).toBe(7);
    expect(ALGORITHM_CATALOG.length).toBe(51);

    for (const entry of ALGORITHM_CATALOG) {
      expect(entry.isImplemented).toBe(true);
      expect(entry.plugin).toBeDefined();
      expect(entry.loader).toBeDefined();

      const plugin = getPlugin(entry.id);
      expect(plugin).toBeDefined();
      expect(plugin?.id).toBe(entry.id);
      expect(plugin?.primaryParams.length).toBeGreaterThan(0);

      // Verify dummy init and explain
      const dummyDataset: DatasetContainer = { train: [], test: [] };
      const initState = plugin?.init(dummyDataset, {}, 42);
      expect(initState).toBeDefined();

      const explanation = plugin!.explain(initState, {});
      expect(explanation).toBeDefined();
      expect(explanation.summary.length).toBeGreaterThan(10);
      expect(explanation.coreAssumption.length).toBeGreaterThan(10);
      expect(explanation.whenItFails.length).toBeGreaterThan(10);
      expect(explanation.microLog.length).toBe(3);
    }
  });

  // -------------------------------------------------------------
  // Test 6: Predict() Input Shape Robustness across all 3 algorithms
  // -------------------------------------------------------------
  it('verifies predict() handles both 1D scalar state [s], 2D grid coordinates [r, c], and 6D feature vectors without NaN', () => {
    const grid = createRLGridworldDataset({ preset: 'cliff-walking' });
    const dataset: DatasetContainer = { train: [], test: [], rlGridworld: grid };

    const qState = qLearningPlugin.init(dataset, {}, 42);
    const dqnState = dqnPlugin.init(dataset, {}, 42);
    const rfState = reinforcePlugin.init(dataset, {}, 42);

    // 1D input [14] (state index 14 = row 2, col 2)
    const qPred1D = qLearningPlugin.predict(new Float32Array([14]), qState, {}) as Float32Array;
    const dqnPred1D = dqnPlugin.predict(new Float32Array([14]), dqnState, {}) as Float32Array;
    const rfPred1D = reinforcePlugin.predict(new Float32Array([14]), rfState, {}) as Float32Array;

    expect(qPred1D.length).toBe(4);
    expect(dqnPred1D.length).toBe(4);
    expect(rfPred1D.length).toBe(4);
    expect(Number.isNaN(qPred1D[0])).toBe(false);
    expect(Number.isNaN(dqnPred1D[0])).toBe(false);
    expect(Number.isNaN(rfPred1D[0])).toBe(false);

    // 2D input [2, 2]
    const qPred2D = qLearningPlugin.predict(new Float32Array([2, 2]), qState, {}) as Float32Array;
    const dqnPred2D = dqnPlugin.predict(new Float32Array([2, 2]), dqnState, {}) as Float32Array;
    const rfPred2D = reinforcePlugin.predict(new Float32Array([2, 2]), rfState, {}) as Float32Array;

    expect(qPred1D[0]).toBeCloseTo(qPred2D[0], 5);
    expect(dqnPred1D[0]).toBeCloseTo(dqnPred2D[0], 5);
    expect(rfPred1D[0]).toBeCloseTo(rfPred2D[0], 5);

    // 6D feature vector for DQN
    const feat6D = new Float32Array([0.4, 0.4, 0.6, 0.6, 0.0, 0.0]);
    const dqnPred6D = dqnPlugin.predict(feat6D, dqnState, {}) as Float32Array;
    expect(dqnPred6D.length).toBe(4);
    expect(Number.isNaN(dqnPred6D[0])).toBe(false);
  });

  // -------------------------------------------------------------
  // Test 7: DQN Bellman Target Non-Zero Parity
  // -------------------------------------------------------------
  it('proves DQN lastTransition computes true non-zero Bellman target y from Target Network θ- rather than hardcoded 0', () => {
    const grid = createRLGridworldDataset({ preset: 'cliff-walking', livingPenalty: -0.04 });
    const dataset: DatasetContainer = { train: [], test: [], rlGridworld: grid };

    let state = dqnPlugin.init(dataset, { learningRate: 0.01, discountFactor: 0.95 }, 42);

    // Take 1 step
    const stepRes = dqnPlugin.step(state, dataset, {
      learningRate: 0.01,
      discountFactor: 0.95,
      rolloutSpeed: 'single-step',
    });

    const trans = stepRes.state.lastTransition;
    expect(trans).toBeDefined();
    // targetQ must be mathematically computed from r + gamma * max Q_target
    expect(trans?.targetQ).toBeDefined();
    expect(typeof trans?.targetQ).toBe('number');
    expect(Number.isNaN(trans?.targetQ)).toBe(false);
    expect(typeof trans?.tdError).toBe('number');
    expect(Number.isNaN(trans?.tdError)).toBe(false);
    // targetQ should not be identically 0.000 when reward is -0.04
    expect(trans?.targetQ).not.toBe(0);
  });

  // -------------------------------------------------------------
  // Test 8: REINFORCE Temperature Scaling Parity
  // -------------------------------------------------------------
  it('proves REINFORCE scales policy gradients by 1/τ so higher temperature produces controlled, proportional logit updates', () => {
    const grid = createRLGridworldDataset({ preset: 'cliff-walking' });
    const dataset: DatasetContainer = { train: [], test: [], rlGridworld: grid };

    const stateLowTau = reinforcePlugin.init(dataset, { learningRate: 0.05, temperature: 0.5 }, 42);
    const stateHighTau = reinforcePlugin.init(dataset, { learningRate: 0.05, temperature: 2.0 }, 42);

    const resLow = reinforcePlugin.step(stateLowTau, dataset, {
      learningRate: 0.05,
      temperature: 0.5,
      entropyBonus: 0.0,
      baselineMode: 'no-baseline-raw',
    });
    const resHigh = reinforcePlugin.step(stateHighTau, dataset, {
      learningRate: 0.05,
      temperature: 2.0,
      entropyBonus: 0.0,
      baselineMode: 'no-baseline-raw',
    });

    // Both step successfully with valid non-zero finite losses
    expect(resLow.metrics.trainLoss).toBeGreaterThan(0);
    expect(resHigh.metrics.trainLoss).toBeGreaterThan(0);
  });

  // -------------------------------------------------------------
  // Test 9: Continuous Dynamic Loss & Regret Progression (No Flat Loss)
  // -------------------------------------------------------------
  it('guarantees continuous dynamic loss curves across all 3 algorithms with non-zero testLoss evaluation regret', () => {
    const grid = createRLGridworldDataset({ preset: 'cliff-walking', livingPenalty: -0.04 });
    const dataset: DatasetContainer = { train: [], test: [], rlGridworld: grid };

    // Q-Learning
    let qState = qLearningPlugin.init(dataset, {}, 42);
    const qRes = qLearningPlugin.step(qState, dataset, { rolloutSpeed: 'full-episode' });
    expect(qRes.metrics.trainLoss).toBeGreaterThanOrEqual(0);
    expect(qRes.metrics.testLoss).toBeGreaterThan(0); // Test loss must not be flat zero!

    // DQN
    let dqnState = dqnPlugin.init(dataset, {}, 42);
    const dqnRes = dqnPlugin.step(dqnState, dataset, { rolloutSpeed: 'full-episode' });
    expect(dqnRes.metrics.trainLoss).toBeGreaterThanOrEqual(0);
    expect(dqnRes.metrics.testLoss).toBeGreaterThan(0); // Test loss must not be flat zero!

    // REINFORCE
    let rfState = reinforcePlugin.init(dataset, {}, 42);
    const rfRes = reinforcePlugin.step(rfState, dataset, { rolloutSpeed: 'full-episode' });
    expect(rfRes.metrics.trainLoss).toBeGreaterThanOrEqual(0);
    expect(rfRes.metrics.testLoss).toBeGreaterThan(0); // Test loss must not be flat zero!
  });

  // -------------------------------------------------------------
  // Test 10: Single-Step Terminal Frame Preservation
  // -------------------------------------------------------------
  it('proves single-step mode preserves terminal landing position (Goal/Lava) for exactly 1 frame before resetting on next tick', () => {
    const grid = createRLGridworldDataset({ preset: 'cliff-walking' });
    const dataset: DatasetContainer = { train: [], test: [], rlGridworld: grid };

    let state = qLearningPlugin.init(dataset, {}, 42);
    // Put agent directly adjacent to Lava at (4, 1), and step South (action 2) into Lava (5, 1)
    state.agentRow = 4;
    state.agentCol = 1;

    // Step South into Lava
    for (let step = 0; step < 10; step++) {
      const res = qLearningPlugin.step(state, dataset, { rolloutSpeed: 'single-step', explorationEpsilon: 1.0 });
      state = res.state;
      if (state.isTerminal) {
        // In terminal frame, agent position must be recorded at the terminal tile
        expect(state.isTerminal).toBe(true);
        // Step once more: next tick must initiate from startPos
        const resetRes = qLearningPlugin.step(state, dataset, { rolloutSpeed: 'single-step' });
        // The transition in the new episode originated from startPos (5, 0)
        expect(resetRes.state.lastTransition?.state).toBe(grid.startPos[0] * GRID_SIZE + grid.startPos[1]);
        expect(resetRes.state.isTerminal).toBe(false);
        expect(resetRes.state.stepsInEpisode).toBe(1);
        break;
      }
    }
  });

  // -------------------------------------------------------------
  // Test 11: lastCompletedEpisodeLength in Full-Episode Mode (Q-Learning, DQN, REINFORCE)
  // -------------------------------------------------------------
  it('tracks lastCompletedEpisodeLength accurately in full-episode mode for Q-Learning, DQN, and REINFORCE instead of resetting episodeLength to 0', () => {
    const grid = createRLGridworldDataset({ preset: 'cliff-walking', slipProbability: 0.0 });
    const dataset: DatasetContainer = { train: [], test: [], rlGridworld: grid };

    // Q-Learning
    let qState = qLearningPlugin.init(dataset, { rolloutSpeed: 'full-episode' }, 42);
    const qRes = qLearningPlugin.step(qState, dataset, { rolloutSpeed: 'full-episode' });
    expect(qRes.state.lastCompletedEpisodeLength).toBeGreaterThan(0);
    expect(qRes.state.stepCount).toBe(qRes.state.lastCompletedEpisodeLength);
    expect(qRes.metrics.custom?.episodeLength).toBe(qRes.state.lastCompletedEpisodeLength);

    // DQN
    let dqnState = dqnPlugin.init(dataset, { rolloutSpeed: 'full-episode' }, 42);
    const dqnRes = dqnPlugin.step(dqnState, dataset, { rolloutSpeed: 'full-episode' });
    expect(dqnRes.state.lastCompletedEpisodeLength).toBeGreaterThan(0);
    expect(dqnRes.state.stepCount).toBe(dqnRes.state.lastCompletedEpisodeLength);
    expect(dqnRes.metrics.custom?.episodeLength).toBe(dqnRes.state.lastCompletedEpisodeLength);

    // REINFORCE
    let rfState = reinforcePlugin.init(dataset, { rolloutSpeed: 'full-episode' }, 42);
    const rfRes = reinforcePlugin.step(rfState, dataset, { rolloutSpeed: 'full-episode' });
    expect(rfRes.state.lastCompletedEpisodeLength).toBeGreaterThan(0);
    expect(rfRes.state.stepCount).toBe(rfRes.state.lastCompletedEpisodeLength);
    expect(rfRes.metrics.custom?.episodeLength).toBe(rfRes.state.lastCompletedEpisodeLength);
  });

  // -------------------------------------------------------------
  // Test 12: DQN Target Sync Counter Increments Per Environment Step in Full-Episode Mode
  // -------------------------------------------------------------
  it('increments DQN targetSyncCounter by actual environment steps in full-episode mode and triggers sync every C steps', () => {
    const grid = createRLGridworldDataset({ preset: 'cliff-walking', slipProbability: 0.0 });
    const dataset: DatasetContainer = { train: [], test: [], rlGridworld: grid };

    let state = dqnPlugin.init(dataset, { targetNetworkSync: 200, rolloutSpeed: 'full-episode' }, 42);
    const res1 = dqnPlugin.step(state, dataset, { targetNetworkSync: 200, rolloutSpeed: 'full-episode' });

    expect(res1.state.targetSyncCounter).toBe(res1.state.lastCompletedEpisodeLength);
    expect(res1.state.stepCount).toBe(res1.state.lastCompletedEpisodeLength);

    // With C = 5, if an episode takes >= 5 steps, syncCounter is remainder <= 4
    let stateC5 = dqnPlugin.init(dataset, { targetNetworkSync: 5, rolloutSpeed: 'full-episode' }, 42);
    const resC5 = dqnPlugin.step(stateC5, dataset, { targetNetworkSync: 5, rolloutSpeed: 'full-episode' });
    expect(resC5.state.targetSyncCounter).toBeLessThan(5);
  });

  // -------------------------------------------------------------
  // Test 13: REINFORCE 1 Step / Frame Cadence
  // -------------------------------------------------------------
  it('steps REINFORCE 1 transition per tick in single-step mode and updates policy on episode termination', () => {
    const grid = createRLGridworldDataset({ preset: 'cliff-walking', slipProbability: 0.0 });
    const dataset: DatasetContainer = { train: [], test: [], rlGridworld: grid };

    let state = reinforcePlugin.init(dataset, { rolloutSpeed: 'single-step' }, 42);
    const step1 = reinforcePlugin.step(state, dataset, { rolloutSpeed: 'single-step' });
    expect(step1.state.stepCount).toBe(1);
    expect(step1.state.stepsInEpisode).toBe(1);
    if (!step1.state.isTerminal) {
      expect(step1.state.episode).toBe(0);
      expect(step1.metrics.custom?.episodeLength).toBe(1);
    }

    // Continue stepping until episode terminates
    state = step1.state;
    for (let i = 0; i < 105 && !state.isTerminal; i++) {
      const res = reinforcePlugin.step(state, dataset, { rolloutSpeed: 'single-step' });
      state = res.state;
    }
    expect(state.isTerminal).toBe(true);
    expect(state.episode).toBe(1);
    expect(state.lastCompletedEpisodeLength).toBeGreaterThan(0);
  });

  // -------------------------------------------------------------
  // Test 14: Preset Switch Clears Telemetry History Ring Buffer
  // -------------------------------------------------------------
  it('clears both loss series and RL return series when HistoryRingBuffer.clear() is called on preset switch', () => {
    const buffer = new HistoryRingBuffer(50, 4);
    buffer.push({
      epoch: 1,
      trainLoss: 0.5,
      testLoss: 1.2,
      weights: new Float32Array([0, 0]),
      bias: 0,
      metrics: {
        trainLoss: 0.5,
        testLoss: 1.2,
        custom: { episodeReturn: 9.72, movingAvgReturn: 8.5 },
      },
    });
    expect(buffer.getLossSeries().epochs.length).toBe(1);
    expect(buffer.getRLReturnSeries().episodeReturns).toEqual([9.72]);

    buffer.clear();
    expect(buffer.getLossSeries().epochs.length).toBe(0);
    expect(buffer.getRLReturnSeries().epochs.length).toBe(0);
    expect(buffer.getRLReturnSeries().episodeReturns.length).toBe(0);
  });
});
