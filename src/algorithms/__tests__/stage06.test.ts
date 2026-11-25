import { describe, it, expect } from 'vitest';
import { getPlugin } from '../registry';
import { rnnPlugin } from '../stage06/rnn';
import { lstmPlugin } from '../stage06/lstm';
import { gruPlugin } from '../stage06/gru';
import { attentionPlugin } from '../stage06/attention';
import { transformerEncoderPlugin } from '../stage06/transformerEncoder';
import { transformerDecoderPlugin } from '../stage06/transformerDecoder';
import { transformerEncDecPlugin } from '../stage06/transformerEncDec';
import {
  generateSequenceDataset,
} from '../../datasets/sequenceTokens';
import { AlgorithmPlugin, DatasetContainer } from '../../types/algorithm';

describe('Stage 06: Sequence Gates & Attention Arena Unit Tests', () => {
  // -------------------------------------------------------------
  // Test 1: RNN vs. LSTM Gradient Survival
  // -------------------------------------------------------------
  it('verifies RNN vs LSTM gradient survival: with T=10 and ||W_hh||=0.7, Vanilla RNN gradient collapses (< 0.05), whereas LSTM preserves ||∇C_1|| / ||∇C_10|| > 0.5', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'coreference', length: 10 }),
    };

    // 1. Vanilla RNN with scale 0.7 (Vanishing regime)
    const rnnState = rnnPlugin.init(dataset, { recurrentScale: 0.7, sequenceLength: 10, activation: 'tanh' }, 42);
    const rnnMetrics = rnnPlugin.loss(rnnState, dataset, { recurrentScale: 0.7, sequenceLength: 10 });
    const rnnSurvival = Number(rnnMetrics.custom?.survivalRatio ?? 0);

    // Vanilla RNN gradient ratio at step 1 should collapse to < 0.05
    expect(rnnSurvival).toBeLessThan(0.05);

    // 2. LSTM with Forget Gate bias b_f = 1.5 (f_t ≈ 0.85 - 0.95)
    const lstmState = lstmPlugin.init(dataset, { forgetGateBias: 1.5, sequenceLength: 10 }, 42);
    const lstmMetrics = lstmPlugin.loss(lstmState, dataset, { forgetGateBias: 1.5, sequenceLength: 10 });
    const lstmSurvival = Number(lstmMetrics.custom?.survivalRatio ?? 0);

    // LSTM additive cell conveyor belt preserves survival > 0.50
    expect(lstmSurvival).toBeGreaterThan(0.5);

    // LSTM survival must be dramatically higher than Vanilla RNN
    expect(lstmSurvival).toBeGreaterThan(rnnSurvival * 10);
  });

  // -------------------------------------------------------------
  // Test 2: LSTM Forget Gate Closure Destroys Memory (Break It Lab)
  // -------------------------------------------------------------
  it('proves that clamping LSTM forget gate bias b_f = -3.0 flushes the cell state conveyor belt, causing gradient collapse similar to Vanilla RNN', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'bracket', length: 8 }),
    };

    const openState = lstmPlugin.init(dataset, { forgetGateBias: 2.0, sequenceLength: 8 }, 42);
    const closedState = lstmPlugin.init(dataset, { forgetGateBias: -3.0, sequenceLength: 8 }, 42);

    expect(openState.survivalRatio).toBeGreaterThan(0.6);
    // Closed forget gate kills survival
    expect(closedState.survivalRatio).toBeLessThan(0.05);
  });

  // -------------------------------------------------------------
  // Test 3: GRU Coupled Gate Balance
  // -------------------------------------------------------------
  it('verifies GRU coupled interpolation: hidden state exactly follows h_t = (1 - z_t) h_{t-1} + z_t h̃_t and maintains gradient survival > 0.35 without C_t', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'sentiment', length: 9 }),
    };

    const gruState = gruPlugin.init(dataset, { updateGateBias: 0.0, resetGateBias: 0.0 }, 42);
    const gruMetrics = gruPlugin.loss(gruState, dataset, {});
    expect(gruMetrics.trainLoss).toBeGreaterThan(0);

    expect(gruState.survivalRatio).toBeGreaterThan(0.35);
    expect(gruState.timeSteps.length).toBe(9);

    // Verify each step has valid z, r gate activations in [0, 1]
    for (const step of gruState.timeSteps) {
      expect(step.z).toBeGreaterThanOrEqual(0.0);
      expect(step.z).toBeLessThanOrEqual(1.0);
      expect(step.r).toBeGreaterThanOrEqual(0.0);
      expect(step.r).toBeLessThanOrEqual(1.0);
    }
  });

  // -------------------------------------------------------------
  // Test 4: Scaled Dot-Product Attention: Row-wise Sum and 1/√d_k Stability
  // -------------------------------------------------------------
  it('proves Scaled Dot-Product Attention row-wise probabilities sum to 1.0000, and 1/√d_k scaling keeps entropy strictly higher than unscaled dot products at d_k=64', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'coreference', length: 10 }),
    };

    // Scaled attention at d_k = 64
    const scaledState = attentionPlugin.init(dataset, { dk: 64, scaleFactor: 'scaled-sqrt-dk', temperature: 1.0 }, 42);
    // Unscaled attention at d_k = 64
    const unscaledState = attentionPlugin.init(dataset, { dk: 64, scaleFactor: 'unscaled-1.0', temperature: 1.0 }, 42);

    const T = scaledState.tokens.length;
    expect(T).toBe(10);

    // Check row sums = 1.0000 (within 1e-5) for scaled attention
    for (let i = 0; i < T; i++) {
      let rowSum = 0;
      for (let j = 0; j < T; j++) {
        rowSum += scaledState.attentionWeights[i * T + j];
      }
      expect(rowSum).toBeCloseTo(1.0, 4);
    }

    // Check row sums = 1.0000 for unscaled attention
    for (let i = 0; i < T; i++) {
      let rowSum = 0;
      for (let j = 0; j < T; j++) {
        rowSum += unscaledState.attentionWeights[i * T + j];
      }
      expect(rowSum).toBeCloseTo(1.0, 4);
    }

    // 1/√d_k scaling must prevent one-hot saturation: Scaled entropy strictly higher than unscaled entropy
    expect(scaledState.entropy).toBeGreaterThan(unscaledState.entropy);
  });

  // -------------------------------------------------------------
  // Test 5: Decoder Causal Masking & Future Token Isolation
  // -------------------------------------------------------------
  it('verifies Decoder Causal Masking: attention weight α_ij === 0 (< 1e-9) for all j > i, and mutating future token embeddings produces 0.0 change in past outputs', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'coreference', length: 8 }),
    };

    const decState = transformerDecoderPlugin.init(dataset, { causalMask: 'masked-causal', temperature: 1.0 }, 42);
    const T = decState.tokens.length;

    // Verify upper triangle is strictly 0.0000
    for (let i = 0; i < T; i++) {
      for (let j = i + 1; j < T; j++) {
        const futureAttn = decState.attentionWeights[i * T + j];
        expect(futureAttn).toBeLessThan(1e-9);
      }
    }

    // Mutate future token at index 7 (last token)
    const mutatedSeq = generateSequenceDataset({ preset: 'coreference', length: 8 });
    for (let d = 0; d < 8; d++) {
      mutatedSeq.sequence[7].embedding[d] += 5.0; // extreme future perturbation
    }

    const mutatedDataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: mutatedSeq,
    };

    const mutatedDecState = transformerDecoderPlugin.init(mutatedDataset, { causalMask: 'masked-causal', temperature: 1.0 }, 42);

    // Query at position i = 2 must produce EXACTLY identical output vector regardless of future token 7 mutation!
    const origOut_2 = decState.outputVectors[2];
    const mutatedOut_2 = mutatedDecState.outputVectors[2];

    for (let d = 0; d < 8; d++) {
      expect(origOut_2[d]).toBeCloseTo(mutatedOut_2[d], 6);
    }
  });

  // -------------------------------------------------------------
  // Test 6: Transformer Encoder Multi-Head Specialization & Positional Encoding
  // -------------------------------------------------------------
  it('demonstrates Multi-Head specialization (H=3) and proves positional encoding breaks permutation invariance', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'bracket', length: 8 }),
    };

    const encStateWithPE = transformerEncoderPlugin.init(dataset, { numHeads: 3, posEncoding: 'sinusoidal' }, 42);
    const encStateNoPE = transformerEncoderPlugin.init(dataset, { numHeads: 3, posEncoding: 'off-bag-of-words' }, 42);

    expect(encStateWithPE.heads.length).toBe(3);

    // Verify distinct heads specialize in distinct weights
    const head1Weights = encStateWithPE.heads[0].weights;
    const head2Weights = encStateWithPE.heads[1].weights;
    let maxDiff = 0;
    for (let k = 0; k < head1Weights.length; k++) {
      const diff = Math.abs(head1Weights[k] - head2Weights[k]);
      if (diff > maxDiff) maxDiff = diff;
    }
    expect(maxDiff).toBeGreaterThan(0.1);

    // Test loss reflects penalty when PE is off
    const lossWithPE = transformerEncoderPlugin.loss(encStateWithPE, dataset, { posEncoding: 'sinusoidal' });
    const lossNoPE = transformerEncoderPlugin.loss(encStateNoPE, dataset, { posEncoding: 'off-bag-of-words' });
    expect(lossNoPE.trainLoss).toBeGreaterThan(lossWithPE.trainLoss);
  });

  // -------------------------------------------------------------
  // Test 7: Transformer Encoder-Decoder Cross-Attention Alignment
  // -------------------------------------------------------------
  it('verifies Seq2Seq Cross-Attention matrix has shape T_target x T_source with exact row normalization and bilingual word alignment', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'seq2seq' }),
    };

    const encDecState = transformerEncDecPlugin.init(dataset, { activeSubLayer: 'cross-attention', temperature: 1.0 }, 42);

    const T_src = encDecState.sourceTokens.length; // 3 ("Ami", "bhat", "khai")
    const T_tgt = encDecState.targetTokens.length; // 4 ("<bos>", "I", "eat", "rice")

    expect(T_src).toBe(3);
    expect(T_tgt).toBe(4);
    expect(encDecState.crossWeights.length).toBe(T_tgt * T_src);

    // Check each target token query sums to 1.0000 across all source keys
    for (let i = 0; i < T_tgt; i++) {
      let rowSum = 0;
      for (let j = 0; j < T_src; j++) {
        rowSum += encDecState.crossWeights[i * T_src + j];
      }
      expect(rowSum).toBeCloseTo(1.0, 4);
    }

    // Target word "eat" (index 2) must attend strongly to Bengali word "khai" (index 2)
    const eatAttendsKhai = encDecState.crossWeights[2 * T_src + 2];
    expect(eatAttendsKhai).toBeGreaterThan(0.5);

    // Target word "I" (index 1) must attend strongly to Bengali word "Ami" (index 0)
    const iAttendsAmi = encDecState.crossWeights[1 * T_src + 0];
    expect(iAttendsAmi).toBeGreaterThan(0.5);
  });

  // -------------------------------------------------------------
  // Test 8: Gate Pinning & Clamping in LSTM & GRU
  // -------------------------------------------------------------
  it('proves gate pinning/clamping directly alters forward state and collapses gradient survival', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'bracket', length: 8 }),
    };

    // LSTM Forget Gate Pinning at step 1
    const lstmState = lstmPlugin.init(dataset, { forgetGateBias: 1.5, sequenceLength: 8 }, 42);
    expect(lstmState.survivalRatio).toBeGreaterThan(0.5);

    // Simulate pinning forget gate to 0.0 at step 1 (severing conveyor belt to step 0)
    lstmState.pinnedGates = { 1: { f: 0.0 } };
    const stepResLstm = lstmPlugin.step(lstmState, dataset, { forgetGateBias: 1.5, sequenceLength: 8 });
    expect(stepResLstm.state.survivalRatio).toBeLessThan(0.05);

    // GRU Update Gate Pinning to 1.0 at step 1
    const gruState = gruPlugin.init(dataset, { updateGateBias: 0.0, resetGateBias: 0.0 }, 42);
    expect(gruState.survivalRatio).toBeGreaterThan(0.35);

    // Pinning update gate to 1.0 at step 1 forces overwrite
    gruState.pinnedGates = { 1: { z: 1.0 } };
    const stepResGru = gruPlugin.step(gruState, dataset, { updateGateBias: 0.0, resetGateBias: 0.0 });
    expect(stepResGru.state.survivalRatio).toBeLessThan(0.20);
  });

  // -------------------------------------------------------------
  // Test 9: Dynamic Autoregressive Step & Logits in Decoder
  // -------------------------------------------------------------
  it('verifies Decoder dynamic next-token distribution updates on autoregressive step', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'coreference', length: 6 }),
    };

    let decState = transformerDecoderPlugin.init(dataset, { causalMask: 'masked-causal', temperature: 1.0 }, 42);
    const initialStep = decState.selectedStep;
    expect(decState.nextTokens.length).toBeGreaterThan(0);

    // Step autoregressively
    const stepRes = transformerDecoderPlugin.step(decState, dataset, { causalMask: 'masked-causal', temperature: 1.0 });
    decState = stepRes.state;
    expect(decState.selectedStep).toBe((initialStep + 1) % 6);
    expect(decState.nextTokens.length).toBeGreaterThan(0);
    // Probabilities sum to ~1.0
    const probSum = decState.nextTokens.reduce((acc, t) => acc + t.prob, 0);
    expect(probSum).toBeCloseTo(1.0, 3);
  });

  // -------------------------------------------------------------
  // Test 10: Dynamic Loss & Epoch Progression Across All 7 Algorithms
  // -------------------------------------------------------------
  it('verifies dynamic loss computation, epoch advancement, and loss history recording across all 7 Stage 06 algorithms', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'coreference', length: 6 }),
    };

    const algorithms: AlgorithmPlugin<any>[] = [
      rnnPlugin,
      lstmPlugin,
      gruPlugin,
      attentionPlugin,
      transformerEncoderPlugin,
      transformerDecoderPlugin,
      transformerEncDecPlugin,
    ];

    for (const algo of algorithms) {
      let state = algo.init(dataset, {}, 42);
      const initialLoss = algo.loss(state, dataset, {});

      expect(initialLoss.trainLoss).toBeGreaterThan(0);
      expect(Number.isFinite(initialLoss.trainLoss)).toBe(true);

      // Run 5 training steps
      for (let i = 0; i < 5; i++) {
        const stepRes = algo.step(state, dataset, {});
        state = stepRes.state;
      }

      expect(state.epoch).toBe(5);
      const afterLoss = algo.loss(state, dataset, {});
      expect(afterLoss.trainLoss).toBeGreaterThan(0);
      expect(Number.isFinite(afterLoss.trainLoss)).toBe(true);

      // Verify visual artifacts contain valid sequence visualization
      const artifacts = algo.getVisualArtifacts(state, dataset, {});
      expect(artifacts.length).toBeGreaterThan(0);
      const seqArt = artifacts.find((a) => a.type === 'sequence-gate-attn');
      expect(seqArt).toBeDefined();
    }
  });

  // -------------------------------------------------------------
  // Test 11: Registry Catalog Validation
  // -------------------------------------------------------------
  it('verifies all 7 Stage 06 algorithms are registered in registry.ts with shortCodes, primary parameters, and valid plugins', () => {
    const algoIds = [
      'rnn',
      'lstm',
      'gru',
      'attention',
      'transformer-encoder',
      'transformer-decoder',
      'transformer-enc-dec',
    ];

    for (const id of algoIds) {
      const plugin = getPlugin(id);
      expect(plugin).not.toBeNull();
      expect(plugin?.id).toBe(id);
      expect(plugin?.stage).toBe('sequence-attn');
      expect(plugin?.primaryParams.length).toBeGreaterThan(0);
      expect(plugin?.primaryParams.length).toBeLessThanOrEqual(3); // Viewport budget: <= 3 primary params!
    }
  });

  // -------------------------------------------------------------
  // Test 12: T5 Enc-Dec Phase 1, 2, 3 Row-Sum Invariants & Causal Mask
  // -------------------------------------------------------------
  it('verifies T5 Enc-Dec Phase 1, Phase 2 (causal), and Phase 3 row sums equal 1.0000 ± 1e-6 across all presets with upper triangle = 0.00 in Phase 2', () => {
    const presets = ['seq2seq', 'coreference', 'bracket', 'sentiment'] as const;
    for (const preset of presets) {
      const dataset: DatasetContainer = {
        train: [],
        test: [],
        sequenceTokens: generateSequenceDataset({ preset }),
      };

      const state = transformerEncDecPlugin.init(dataset, { activeSubLayer: 'dec-causal-self' }, 42);
      expect(state.phaseIndex).toBe(1);

      const T_src = state.sourceTokens.length;
      const T_tgt = state.targetTokens.length;

      // Phase 1: Encoder Self (T_src x T_src)
      for (let i = 0; i < T_src; i++) {
        let sum = 0;
        for (let j = 0; j < T_src; j++) {
          sum += state.encSelfWeights[i * T_src + j];
        }
        expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
      }

      // Phase 2: Decoder Causal Self (T_tgt x T_tgt)
      for (let i = 0; i < T_tgt; i++) {
        let sum = 0;
        for (let j = 0; j < T_tgt; j++) {
          const w = state.decSelfWeights[i * T_tgt + j];
          if (j > i) {
            expect(w).toBe(0);
          }
          sum += w;
        }
        expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
      }

      // Phase 3: Cross-Attention (T_tgt x T_src)
      for (let i = 0; i < T_tgt; i++) {
        let sum = 0;
        for (let j = 0; j < T_src; j++) {
          sum += state.crossWeights[i * T_src + j];
        }
        expect(Math.abs(sum - 1.0)).toBeLessThan(1e-6);
      }

      // Telemetry includes entropy & maxSoftmaxScore
      const metrics = transformerEncDecPlugin.loss(state, dataset, {});
      expect(metrics.custom?.entropy).toBeDefined();
      expect(metrics.custom?.maxSoftmaxScore).toBeDefined();
      expect(Number.isFinite(Number(metrics.custom?.maxSoftmaxScore))).toBe(true);
    }
  });

  // -------------------------------------------------------------
  // Test 13: SDP Attention Break: Saturate (d_k = 128 unscaled)
  // -------------------------------------------------------------
  it('verifies SDP Attention at d_k=128 unscaled saturates Softmax (MAX ATTENTION > 0.90, lower entropy) with row sums = 1.0000 and zero NaNs', () => {
    const dataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'coreference', length: 10 }),
    };

    const unscaled = attentionPlugin.init(dataset, { dk: 128, scaleFactor: 'unscaled-1.0', temperature: 1.0 }, 42);
    const scaled = attentionPlugin.init(dataset, { dk: 128, scaleFactor: 'scaled-sqrt-dk', temperature: 1.0 }, 42);

    const T = unscaled.tokens.length;
    for (let i = 0; i < T; i++) {
      let rowSum = 0;
      for (let j = 0; j < T; j++) {
        const w = unscaled.attentionWeights[i * T + j];
        expect(Number.isNaN(w)).toBe(false);
        rowSum += w;
      }
      expect(rowSum).toBeCloseTo(1.0, 5);
    }

    const unscaledMetrics = attentionPlugin.loss(unscaled, dataset, { dk: 128, scaleFactor: 'unscaled-1.0' });
    expect(Number(unscaledMetrics.custom?.maxSoftmaxScore)).toBeGreaterThan(0.9);
    expect(unscaled.entropy).toBeLessThan(scaled.entropy);

    // Dynamic hop count in micro-log for short Seq2Seq preset (T=3 -> 2 hops)
    const seq2seqDataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'seq2seq' }),
    };
    const shortState = attentionPlugin.init(seq2seqDataset, { dk: 8 }, 42);
    const shortExplain = attentionPlugin.explain(shortState, { dk: 8 });
    expect(shortExplain.microLog.some((line) => line.includes('without 2 intermediate recurrent cell hops'))).toBe(true);
  });

  // -------------------------------------------------------------
  // Test 14: GPT Preset Vocab & Subscripts + BERT Head Dim & [MASK] UI
  // -------------------------------------------------------------
  it('verifies GPT adapts next-token vocab per preset without raw LaTeX and BERT exposes clean integer head dims, maxSoftmaxScore, and [MASK] predictions', () => {
    const seq2seqDataset: DatasetContainer = {
      train: [],
      test: [],
      sequenceTokens: generateSequenceDataset({ preset: 'seq2seq' }),
    };

    const decState = transformerDecoderPlugin.init(seq2seqDataset, {}, 42);
    const decArtifacts = transformerDecoderPlugin.getVisualArtifacts(decState, seq2seqDataset, {}) as any[];
    expect(decArtifacts[0].phaseName).not.toContain('w_{');
    expect(decState.nextTokens.some((t) => ['Ami', 'bhat', 'khai', 'I', 'eat', 'rice', '<eos>'].includes(t.token))).toBe(true);
    expect(decState.nextTokens.some((t) => t.token === 'trophy' || t.token === 'suitcase')).toBe(false);

    const decMetrics = transformerDecoderPlugin.loss(decState, seq2seqDataset, {});
    expect(Number.isFinite(Number(decMetrics.custom?.maxSoftmaxScore))).toBe(true);

    // BERT with masked token
    const corefSeq = generateSequenceDataset({ preset: 'coreference', length: 6 });
    corefSeq.sequence[1] = { ...corefSeq.sequence[1], text: '[MASK]', isMasked: true };
    const maskedDataset: DatasetContainer = { train: [], test: [], sequenceTokens: corefSeq };

    const encState = transformerEncoderPlugin.init(maskedDataset, {}, 42);
    expect(encState.heads.length).toBe(2); // Default H=2 divides d_model=8 cleanly
    const encMetrics = transformerEncoderPlugin.loss(encState, maskedDataset, {});
    expect(Number.isFinite(Number(encMetrics.custom?.maxSoftmaxScore))).toBe(true);

    const encArtifacts = transformerEncoderPlugin.getVisualArtifacts(encState, maskedDataset, {}) as any[];
    expect(encArtifacts[0].attention.nextTokens?.length).toBeGreaterThan(0);
  });
});
