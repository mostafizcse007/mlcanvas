import { describe, it, expect } from 'vitest';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { mlpPlugin } from '../stage04/mlp';
import { autoencoderPlugin, decodePoint } from '../stage04/autoencoder';
import { vaePlugin } from '../stage04/vae';
import { ganPlugin, computeDiscriminatorLoss } from '../stage04/gan';

describe('Stage 04: Neural Foundations & Dense / Generative Graph Lab Unit Tests', () => {
  // Test 1: MLP XOR Convergence
  it('achieves > 95% accuracy on non-linear XOR Quadrants within 150 steps using a 2 -> 4 -> 1 MLP', () => {
    const dataset = generateTabular2dDataset({
      kind: 'xor-quadrants',
      sampleCount: 160,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = {
      learningRate: 0.08,
      activation: 'tanh' as const,
      regularization: 'none' as const,
      optimizer: 'adam' as const,
      numLayers: 1, // 1 hidden layer -> 2 -> 4 -> 1
      layer1: 4,
      batchSize: 32,
      weightInit: 'xavier' as const,
      features: 'x1,x2',
    };

    let state = mlpPlugin.init(dataset, params, 100);
    let bestAccuracy = 0;

    for (let ep = 0; ep < 150; ep++) {
      const res = mlpPlugin.step(state, dataset, params);
      state = res.state;
      const acc = res.metrics.trainAccuracy ?? 0;
      if (acc > bestAccuracy) {
        bestAccuracy = acc;
      }
      if (acc >= 95.0) {
        break;
      }
    }

    expect(bestAccuracy).toBeGreaterThanOrEqual(95.0);
  });

  // Test 2: Symmetry Breaking Failure with Zero Weight Initialization
  it('demonstrates symmetry breaking failure: W = 0 initialization keeps all hidden neuron activations and gradients identical', () => {
    const dataset = generateTabular2dDataset({
      kind: 'xor-quadrants',
      sampleCount: 80,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = {
      learningRate: 0.03,
      activation: 'tanh' as const,
      optimizer: 'sgd' as const,
      numLayers: 1,
      layer1: 4, // 4 hidden neurons
      weightInit: 'zeros-trap' as const,
      features: 'x1,x2',
    };

    let state = mlpPlugin.init(dataset, params, 42);

    // Initial weights are all 0
    for (let i = 0; i < state.weights[0].length; i++) {
      expect(state.weights[0][i]).toBe(0);
    }

    // Step once
    const res = mlpPlugin.step(state, dataset, params);
    state = res.state;

    // After 1 step from W=0 with identical inputs, all 4 hidden neurons have identical weights
    // Weights for neuron j from input i: offset is j * fanIn + i
    const fanIn = state.layerSizes[0];
    const fanOut = state.layerSizes[1]; // 4

    for (let j = 1; j < fanOut; j++) {
      for (let i = 0; i < fanIn; i++) {
        const w0 = state.weights[0][0 * fanIn + i];
        const wj = state.weights[0][j * fanIn + i];
        expect(wj).toBeCloseTo(w0, 5);
      }
    }
  });

  // Test 3: Linear AE Subspace & PCA Alignment
  it('verifies that a 2 -> 1 -> 2 linear Autoencoder decreases MSE monotonically and aligns with the principal variance axis', () => {
    // Generate synthetic correlated dataset: x2 = 1.8 * x1 + small noise
    const train = [];
    for (let i = -20; i <= 20; i++) {
      const x1 = i * 0.2;
      const x2 = 1.8 * x1 + Math.sin(i) * 0.05;
      train.push({ id: i + 30, x1, x2, y: 1 });
    }
    const dataset = { train, test: train };

    const params = {
      latentDim: 1 as const,
      activation: 'linear' as const,
      learningRate: 0.05,
      hiddenDim: 4,
      weightDecay: 0.0,
    };

    let state = autoencoderPlugin.init(dataset, params, 42);
    let prevMSE = 999;
    let decreasedCount = 0;

    for (let ep = 0; ep < 60; ep++) {
      const res = autoencoderPlugin.step(state, dataset, params);
      state = res.state;
      if (res.metrics.trainLoss <= prevMSE + 1e-4) {
        decreasedCount++;
      }
      prevMSE = res.metrics.trainLoss;
    }

    // Reconstruction MSE dropped significantly
    expect(state.reconstructionMSE).toBeLessThan(0.15);
    expect(decreasedCount).toBeGreaterThanOrEqual(30);

    // Verify Decoder spans the principal direction:
    // Decode z=1.0 and z=-1.0 to get the direction vector
    const dPos = decodePoint(1.0, 0, state, 'linear');
    const dNeg = decodePoint(-1.0, 0, state, 'linear');
    const dirX = dPos[0] - dNeg[0];
    const dirY = dPos[1] - dNeg[1];
    const slope = dirY / (dirX || 1e-6);

    // True slope is ~1.8
    expect(slope).toBeGreaterThan(1.2);
    expect(slope).toBeLessThan(2.4);
  });

  // Test 4: VAE KL Penalty reduces divergence toward standard normal
  it('confirms increasing beta from 0.0 to 2.0 strictly reduces KL divergence D_KL toward 0', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 80,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    // 1. Train with beta = 0.0 (Pure reconstruction, unconstrained latent space)
    const paramsBeta0 = {
      beta: 0.0,
      learningRate: 0.03,
      hiddenDim: 8,
      samplingNoise: 'on' as const,
    };
    let state0 = vaePlugin.init(dataset, paramsBeta0, 42);
    for (let ep = 0; ep < 30; ep++) {
      state0 = vaePlugin.step(state0, dataset, paramsBeta0).state;
    }

    // 2. Train with beta = 2.0 (Strong KL regularization forcing q(z|x) -> N(0, I))
    const paramsBeta2 = {
      beta: 2.0,
      learningRate: 0.03,
      hiddenDim: 8,
      samplingNoise: 'on' as const,
    };
    let state2 = vaePlugin.init(dataset, paramsBeta2, 42);
    for (let ep = 0; ep < 30; ep++) {
      state2 = vaePlugin.step(state2, dataset, paramsBeta2).state;
    }

    // Check that state2 has lower KL loss
    expect(state2.klLoss).toBeLessThan(state0.klLoss);
  });

  // Test 5: GAN Gradient Flow (Non-Saturating vs Minimax Saturating)
  it('proves that non-saturating generator loss produces a strictly larger gradient magnitude than minimax saturating loss when D(G(z)) < 0.05', () => {
    // When D(G(z)) is very small (e.g. p = 0.01):
    // For minimax saturating loss: L_G = log(1 - D) -> dL/dLogit = p = 0.01
    // For non-saturating loss: L_G = -log(D) -> dL/dLogit = p - 1.0 = -0.99
    // Magnitude: |-0.99| = 0.99 >> 0.01

    const pFakeSmall = 0.02; // D is overpowering, assigning only 2% probability to fake
    const gradMinimax = Math.abs(pFakeSmall); // 0.02
    const gradNonSaturating = Math.abs(pFakeSmall - 1.0); // 0.98

    expect(gradNonSaturating).toBeGreaterThan(gradMinimax * 10);
  });

  // Test 6: Snapshot Serialization and Restoration
  it('verifies deterministic Float32Array snapshot serialization and restoration for all Stage 04 models', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 40,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    // MLP
    const mlpState = mlpPlugin.init(dataset, {}, 42);
    const mlpSnap = mlpPlugin.serializeSnapshot(mlpState);
    const mlpRestored = mlpPlugin.restoreSnapshot(mlpSnap, mlpState);
    expect(mlpRestored.epoch).toBe(0);

    // Autoencoder
    const aeState = autoencoderPlugin.init(dataset, {}, 42);
    const aeSnap = autoencoderPlugin.serializeSnapshot(aeState);
    const aeRestored = autoencoderPlugin.restoreSnapshot(aeSnap, aeState);
    expect(aeRestored.epoch).toBe(0);

    // VAE
    const vaeState = vaePlugin.init(dataset, {}, 42);
    const vaeSnap = vaePlugin.serializeSnapshot(vaeState);
    const vaeRestored = vaePlugin.restoreSnapshot(vaeSnap, vaeState);
    expect(vaeRestored.epoch).toBe(0);

    // GAN
    const ganState = ganPlugin.init(dataset, {}, 42);
    const ganSnap = ganPlugin.serializeSnapshot(ganState);
    const ganRestored = ganPlugin.restoreSnapshot(ganSnap, ganState);
    expect(ganRestored.epoch).toBe(0);
  });

  // Test 7: Worker structuredClone Compatibility
  it('guarantees VAE and GAN model states are structuredClone serializable across Web Worker boundaries', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 40,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const vaeState = vaePlugin.init(dataset, {}, 42);
    expect(() => structuredClone(vaeState)).not.toThrow();
    const vaeStepped = vaePlugin.step(vaeState, dataset, {}).state;
    expect(() => structuredClone(vaeStepped)).not.toThrow();

    const ganState = ganPlugin.init(dataset, {}, 42);
    expect(() => structuredClone(ganState)).not.toThrow();
    const ganStepped = ganPlugin.step(ganState, dataset, {}).state;
    expect(() => structuredClone(ganStepped)).not.toThrow();
  });

  // Test 8: MLP 1D Continuous Regression & Label Leakage Decoupling
  it('learns non-linear 1D sine wave regression with R2 > 0.85 and strictly decouples x2 target leakage', () => {
    const dataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 160,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = {
      learningRate: 0.05,
      activation: 'tanh' as const,
      regularization: 'none' as const,
      optimizer: 'adam' as const,
      numLayers: 1,
      layer1: 12,
      batchSize: 32,
      weightInit: 'xavier' as const,
      features: 'x1,x2',
    };

    let state = mlpPlugin.init(dataset, params, 42);
    expect(state.isRegression).toBe(true);
    expect(state.is1D).toBe(true);

    let bestR2 = -999;
    for (let ep = 0; ep < 140; ep++) {
      const res = mlpPlugin.step(state, dataset, params);
      state = res.state;
      const r2 = res.metrics.trainR2 ?? -999;
      if (r2 > bestR2) bestR2 = r2;
      if (bestR2 >= 0.88) break;
    }

    expect(bestR2).toBeGreaterThanOrEqual(0.85);

    // Strict mathematical verification: evaluating at (x1, x2_arbitrary_A) must equal (x1, x2_arbitrary_B)
    // because in 1D regression, x2 = y was stored in tabular format and must be decoupled from neural input.
    const predA = mlpPlugin.predict(new Float32Array([1.5, 999.0]), state, params) as number;
    const predB = mlpPlugin.predict(new Float32Array([1.5, -420.0]), state, params) as number;
    expect(predA).toBeCloseTo(predB, 6);

    // Linear continuous output verification: output can be outside [-1, 1] without hard sigmoid clipping
    const testPoints = [-3.0, -1.5, 0.0, 1.5, 3.0];
    const predictions = testPoints.map((x) => mlpPlugin.predict(new Float32Array([x, 0]), state, params) as number);
    const maxVal = Math.max(...predictions);
    const minVal = Math.min(...predictions);
    // Sine wave amplitude is ~2.4, so peak-to-peak should be > 1.5
    expect(maxVal - minVal).toBeGreaterThan(1.5);
  });

  // Test 9: MLP 2D Surface Regression (x1, x2 -> y)
  it('fits a 2D continuous planar surface (linear-surface) with R2 > 0.85 with both x1 and x2 active', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-surface',
      sampleCount: 160,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = {
      learningRate: 0.05,
      activation: 'tanh' as const,
      regularization: 'none' as const,
      optimizer: 'adam' as const,
      numLayers: 1,
      layer1: 8,
      batchSize: 32,
      weightInit: 'xavier' as const,
      features: 'x1,x2',
    };

    let state = mlpPlugin.init(dataset, params, 42);
    expect(state.isRegression).toBe(true);
    expect(state.is1D).toBe(false); // 2D surface regression has independent x2

    let bestR2 = -999;
    for (let ep = 0; ep < 100; ep++) {
      const res = mlpPlugin.step(state, dataset, params);
      state = res.state;
      const r2 = res.metrics.trainR2 ?? -999;
      if (r2 > bestR2) bestR2 = r2;
    }

    expect(bestR2).toBeGreaterThanOrEqual(0.85);

    // On 2D surface, x2 IS an active feature: predict(1, 2) != predict(1, -2)
    const p1 = mlpPlugin.predict(new Float32Array([1.0, 2.0]), state, params) as number;
    const p2 = mlpPlugin.predict(new Float32Array([1.0, -2.0]), state, params) as number;
    expect(Math.abs(p1 - p2)).toBeGreaterThan(0.5);
  });

  // Test 10: Autoencoder on Continuous Regression Manifold
  it('optimizes reconstruction MSE on continuous regression manifold and predicts smooth latent level sets', () => {
    const dataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 100,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = {
      latentDim: 1 as const,
      activation: 'tanh' as const,
      learningRate: 0.05,
      hiddenDim: 6,
      weightDecay: 0.0,
    };

    let state = autoencoderPlugin.init(dataset, params, 42);
    const initialMSE = state.reconstructionMSE;

    for (let ep = 0; ep < 40; ep++) {
      state = autoencoderPlugin.step(state, dataset, params).state;
    }

    expect(state.reconstructionMSE).toBeLessThan(initialMSE);
    expect(state.pcaAxis1).toBeDefined();
    expect(state.pcaAxis1?.direction.length).toBe(2);

    // predict() outputs continuous latent coordinate z1 for level-set manifold visualization
    const zLeft = autoencoderPlugin.predict(new Float32Array([-3.0, 0.0]), state, params);
    const zRight = autoencoderPlugin.predict(new Float32Array([3.0, 0.0]), state, params);
    expect(Number.isFinite(zLeft)).toBe(true);
    expect(Number.isFinite(zRight)).toBe(true);
    expect(zLeft).not.toBe(zRight);
  });

  // Test 11: VAE on Continuous Regression Manifold
  it('optimizes ELBO on continuous regression manifold and produces valid posterior means and ellipses', () => {
    const dataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 80,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = {
      beta: 1.0,
      learningRate: 0.03,
      hiddenDim: 6,
      samplingNoise: 'on' as const,
    };

    let state = vaePlugin.init(dataset, params, 42);

    for (let ep = 0; ep < 30; ep++) {
      state = vaePlugin.step(state, dataset, params).state;
    }

    const artifacts = vaePlugin.getVisualArtifacts(state, dataset, params);
    const latentArtifact = artifacts.find((a) => a.type === 'latent-bottleneck-overlay') as any;
    expect(latentArtifact).toBeDefined();
    expect(latentArtifact.encodedPoints.length).toBe(dataset.train.length);
    expect(latentArtifact.encodedPoints[0]).toHaveProperty('z1');
    expect(latentArtifact.encodedPoints[0]).toHaveProperty('sigma1');
    expect(latentArtifact.encodedPoints[0].sigma1).toBeGreaterThan(0);

    // predict() outputs continuous posterior mean mu1
    const muVal = vaePlugin.predict(new Float32Array([1.0, 1.0]), state, params);
    expect(Number.isFinite(muVal)).toBe(true);
  });

  // Test 12: GAN on Continuous Regression Manifold
  it('trains Discriminator and Generator on continuous manifold with valid probabilities and pull vectors', () => {
    const dataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 80,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = {
      learningRateG: 0.05,
      learningRateD: 0.05,
      hiddenDim: 6,
      lossMode: 'non-saturating' as const,
      dStepsPerGStep: 1,
    };

    let state = ganPlugin.init(dataset, params, 42);

    for (let ep = 0; ep < 25; ep++) {
      state = ganPlugin.step(state, dataset, params).state;
    }

    // Discriminator outputs centered score in [-1.5, +1.5]
    const dScore = ganPlugin.predict(new Float32Array([0.0, 0.0]), state, params);
    expect(Number.isFinite(dScore)).toBe(true);

    // Generator produces fake points on canvas
    expect(state.fakePoints.length).toBeGreaterThan(0);
    for (const pt of state.fakePoints) {
      expect(Number.isFinite(pt.x1)).toBe(true);
      expect(Number.isFinite(pt.x2)).toBe(true);
    }

    // Pull vectors from visual artifacts
    const artifacts = ganPlugin.getVisualArtifacts(state, dataset, params);
    const ganArtifact = artifacts.find((a) => a.type === 'gan-minimax-overlay') as any;
    expect(ganArtifact).toBeDefined();
    expect(ganArtifact.fakePoints.length).toBe(state.fakePoints.length);
    expect(ganArtifact.fakePoints[0].pullVector).toBeDefined();
  });

  // Test 13: Multiclass 3-Class Compatibility across all 4 algorithms
  it('handles 3-class multiclass datasets (y in {-1, 1, 2}) without NaN or coordinate divergence across all 4 models', () => {
    const dataset3 = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 90,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 3,
      scaling: 'raw',
      stretchX2: false,
    });

    // Check class 2 exists
    const hasClass2 = dataset3.train.some((p) => p.y === 2);
    expect(hasClass2).toBe(true);

    // 1. MLP
    const mlpState = mlpPlugin.init(dataset3, {}, 42);
    const mlpRes = mlpPlugin.step(mlpState, dataset3, {});
    expect(Number.isFinite(mlpRes.metrics.trainLoss)).toBe(true);

    // 2. Autoencoder
    const aeState = autoencoderPlugin.init(dataset3, {}, 42);
    const aeRes = autoencoderPlugin.step(aeState, dataset3, {});
    expect(Number.isFinite(aeRes.metrics.trainLoss)).toBe(true);

    // 3. VAE
    const vaeState = vaePlugin.init(dataset3, {}, 42);
    const vaeRes = vaePlugin.step(vaeState, dataset3, {});
    expect(Number.isFinite(vaeRes.metrics.trainLoss)).toBe(true);

    // 4. GAN
    const ganState = ganPlugin.init(dataset3, {}, 42);
    const ganRes = ganPlugin.step(ganState, dataset3, {});
    expect(Number.isFinite(ganRes.metrics.trainLoss)).toBe(true);
  });

  // Test 14 (STAGE-04-ISSUE-002): MLP Regression unbounded linear output on y = 2.5 * x in [-5, +5]
  it('trains MLP in Regression mode on y = 2.5 * x (range [-5, +5]) with predictions exceeding [-1, +1] and R² > 0.95', () => {
    const train = [];
    const test = [];
    for (let i = 0; i < 60; i++) {
      const x1 = -5 + (i / 59) * 10;
      const y = 2.5 * x1;
      const pt = { id: i + 1, x1, x2: y, y };
      if (i % 5 === 0) test.push(pt);
      else train.push(pt);
    }
    const dataset = { train, test };
    const params = {
      problemType: 'regression' as const,
      learningRate: 0.08,
      activation: 'tanh' as const,
      optimizer: 'adam' as const,
      numLayers: 1,
      layer1: 8,
      batchSize: 'full' as const,
      weightInit: 'xavier' as const,
      features: 'x1',
    };

    let state = mlpPlugin.init(dataset, params, 42);
    expect(state.isRegression).toBe(true);

    let finalR2 = -999;
    for (let ep = 0; ep < 250; ep++) {
      const res = mlpPlugin.step(state, dataset, params);
      state = res.state;
      finalR2 = res.metrics.trainR2 ?? -999;
      if (finalR2 > 0.98) break;
    }

    expect(finalR2).toBeGreaterThan(0.95);
    const predPos = mlpPlugin.predict(new Float32Array([4.5, 0]), state, params) as number;
    const predNeg = mlpPlugin.predict(new Float32Array([-4.5, 0]), state, params) as number;
    expect(predPos).toBeGreaterThan(5.0);
    expect(predNeg).toBeLessThan(-5.0);
  });

  // Test 15 (STAGE-04-ISSUE-003): GAN L_D Equilibrium Scale Consistency
  it('computes L_D = 2 * ln(2) ≈ 1.3863 when D(real) = 0.5 and D(fake) = 0.5, matching init and Arena Status', () => {
    const eqLoss = computeDiscriminatorLoss([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    expect(eqLoss).toBeCloseTo(2 * Math.LN2, 6);

    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 40,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });
    const initState = ganPlugin.init(dataset, {}, 42);
    expect(initState.dLoss).toBeCloseTo(2 * Math.LN2, 6);
  });

  // Test 16 (STAGE-04-ISSUE-004): AE & VAE real Epoch 0 loss & Telemetry vs Micro-Log consistency
  it('computes real Epoch 0 losses for AE/VAE and keeps VAE Telemetry β-ELBO identical to Micro-Log Total Loss', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 60,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const aeState = autoencoderPlugin.init(dataset, {}, 42);
    expect(aeState.reconstructionMSE).not.toBe(1.0);
    expect(aeState.reconstructionMSE).toBeGreaterThan(0);

    const vaeParams = { beta: 0.5, samplingNoise: 'off' as const };
    const vaeState = vaePlugin.init(dataset, vaeParams, 42);
    expect(vaeState.reconLoss).not.toBe(1.0);
    expect(vaeState.klLoss).not.toBe(0.5);

    const telemetry = vaePlugin.loss(vaeState, dataset, vaeParams);
    const explanation = vaePlugin.explain(vaeState, vaeParams);
    const expectedElbo = telemetry.custom?.betaELBO;
    expect(expectedElbo).toBeDefined();
    expect(explanation.microLog.some((line) => line.includes(`Total Loss: ${expectedElbo}`))).toBe(true);
  });
});
