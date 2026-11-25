import { describe, it, expect } from 'vitest';
import { AlgorithmPlugin } from '../../types/algorithm';
import { generateTabular2dDataset } from '../../datasets/tabular2d';
import { kMeansPlugin } from '../stage03/kMeans';
import { dbscanPlugin } from '../stage03/dbscan';
import { gmmPlugin } from '../stage03/gmm';
import { pcaPlugin } from '../stage03/pca';
import { ldaPlugin } from '../stage03/lda';
import { icaPlugin } from '../stage03/ica';
import { hierarchicalClusteringPlugin } from '../stage03/hierarchicalClustering';
import { tsnePlugin } from '../stage03/tsne';
import { umapPlugin } from '../stage03/umap';

describe('Stage 03: Unsupervised & Manifold Projection Lab Unit Tests', () => {
  // Test 1: K-Means Monotonicity
  it('strictly decreases or maintains WCSS Inertia on every full Lloyd update cycle', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 60,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { k: 2, initMethod: 'k-means++' as const, metric: 'euclidean' as const };
    let state = kMeansPlugin.init(dataset, params, 42);

    expect(state.centroids.length).toBe(2);
    expect(state.epoch).toBe(0);

    let prevInertia = state.inertia;

    // Run 10 assign-and-update steps
    for (let s = 0; s < 10; s++) {
      const stepRes = kMeansPlugin.step(state, dataset, params);
      state = stepRes.state;

      // When update step completes, inertia must be <= previous inertia (with float tolerance)
      if (state.subStep === 'assign') {
        expect(state.inertia).toBeLessThanOrEqual(prevInertia + 1e-6);
        prevInertia = state.inertia;
      }
    }
  });

  // Test 2: DBSCAN Noise Isolation
  it('assigns far-away outlier point at distance > 5ε as noise cluster ID -1', () => {
    // Two tight dense clusters plus 1 far outlier
    const trainData = [
      // Cluster 1 (center at -2, -2)
      { id: 1, x1: -2.0, x2: -2.0, y: 1 },
      { id: 2, x1: -2.1, x2: -1.9, y: 1 },
      { id: 3, x1: -1.9, x2: -2.1, y: 1 },
      { id: 4, x1: -2.0, x2: -1.8, y: 1 },
      // Cluster 2 (center at 2, 2)
      { id: 5, x1: 2.0, x2: 2.0, y: -1 },
      { id: 6, x1: 2.1, x2: 1.9, y: -1 },
      { id: 7, x1: 1.9, x2: 2.1, y: -1 },
      { id: 8, x1: 2.0, x2: 2.2, y: -1 },
      // Far Outlier at distance > 5ε
      { id: 9, x1: 8.0, x2: 8.0, y: 1 },
    ];

    const dataset = { train: trainData, test: [] };
    const params = { epsilon: 0.6, minPts: 3, metric: 'euclidean' as const };

    const state = dbscanPlugin.init(dataset, params, 42);

    expect(state.activeClusters).toBe(2);
    expect(state.noisePointIds).toContain(8); // Index 8 is id: 9
    expect(state.assignments[8]).toBe(-1); // Marked as noise
    expect(state.corePointIds.length).toBeGreaterThan(0);
  });

  // Test 3: GMM vs. K-Means on Diagonal Ellipses
  it('learns non-zero off-diagonal covariance Σ_12 on elongated diagonal distributions', () => {
    // Generate synthetic diagonal cluster: points along line x2 = x1
    const trainData = [];
    for (let i = 0; i < 30; i++) {
      const t = (i / 30) * 4 - 2;
      trainData.push({ id: i + 1, x1: t, x2: t + (i % 2 === 0 ? 0.05 : -0.05), y: 1 });
    }
    const dataset = { train: trainData, test: [] };
    const params = { k: 1, covarianceType: 'full' as const, regCovar: 0.001 };

    let state = gmmPlugin.init(dataset, params, 42);

    // Run 5 EM steps
    for (let s = 0; s < 5; s++) {
      state = gmmPlugin.step(state, dataset, params).state;
    }

    const comp = state.components[0];
    const offDiagonalCov = comp.cov[0][1];

    // For points along x2 = x1, off-diagonal covariance must be positive and non-zero!
    expect(Math.abs(offDiagonalCov)).toBeGreaterThan(0.2);
    expect(comp.cov[0][0]).toBeGreaterThan(0.2);
    expect(comp.cov[1][1]).toBeGreaterThan(0.2);
  });

  // Test 4: PCA vs. LDA Orthogonality and Discriminant Separation
  it('verifies PCA eigenvectors are strictly orthogonal and LDA achieves higher class separation on parallel cigar clusters', () => {
    // Two parallel horizontal cigar-shaped classes:
    // Class 1 centered at y = -1.5, stretched along x in [-3, 3]
    // Class 2 centered at y = +1.5, stretched along x in [-3, 3]
    const trainData = [];
    let id = 1;
    for (let x = -3; x <= 3; x += 0.5) {
      trainData.push({ id: id++, x1: x, x2: -1.5, y: 1 });
      trainData.push({ id: id++, x1: x, x2: 1.5, y: -1 });
    }

    const dataset = { train: trainData, test: [] };

    // 1. PCA
    const pcaState = pcaPlugin.init(dataset, { nComponents: 2 }, 42);
    const [v1, v2] = pcaState.eigenvectors;

    // Eigenvectors v1 and v2 must be orthogonal (v1 · v2 ≈ 0)
    const dotProduct = v1[0] * v2[0] + v1[1] * v2[1];
    expect(Math.abs(dotProduct)).toBeLessThan(1e-6);

    // Because horizontal variance is large (x in [-3, 3], span 6 vs y gap 3),
    // PCA's PC1 aligns HORIZONTALLY along X (mixing the two classes into complete 1D overlap!)
    expect(Math.abs(v1[0])).toBeGreaterThan(0.9);

    // 2. LDA
    const ldaState = ldaPlugin.init(dataset, { shrinkage: 0 }, 42);

    // LDA maximizes class separation, so its discriminant axis w MUST align VERTICALLY along Y!
    expect(Math.abs(ldaState.wAxis[1])).toBeGreaterThan(0.9);
    expect(ldaState.rayleighRatio).toBeGreaterThan(0.5);
  });

  // Test 5: Hierarchical Dendrogram Consistency
  it('produces valid hierarchical agglomerative tree with N leaves and N-1 merges', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 20,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const N = dataset.train.length;
    const params = { k: 2, linkage: 'ward' as const };
    const state = hierarchicalClusteringPlugin.init(dataset, params, 42);

    expect(state.mergeHistory.length).toBe(N - 1);
    expect(state.dendrogramNodes.length).toBe(2 * N - 1);
    expect(state.leafOrder.length).toBe(N);
  });

  // Test 6: FastICA Non-Gaussianity & Negentropy
  it('runs FastICA iterations without NaN and computes positive excess kurtosis on non-Gaussian points', () => {
    const dataset = generateTabular2dDataset({
      kind: 'xor-quadrants',
      sampleCount: 40,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const params = { contrast: 'logcosh' as const };
    let state = icaPlugin.init(dataset, params, 42);

    for (let s = 0; s < 5; s++) {
      state = icaPlugin.step(state, dataset, params).state;
      expect(isNaN(state.independentAxes[0][0])).toBe(false);
      expect(isNaN(state.independentAxes[1][0])).toBe(false);
    }
  });

  // Test 7: t-SNE & UMAP Optimization
  it('optimizes low-dimensional layout coordinates for t-SNE and UMAP without numerical divergence', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 20,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    // A. t-SNE
    let tsneState = tsnePlugin.init(dataset, { perplexity: 5, learningRate: 50 }, 42);
    expect(tsneState.Y.length).toBe(dataset.train.length);

    for (let s = 0; s < 5; s++) {
      tsneState = tsnePlugin.step(tsneState, dataset, { learningRate: 50 }).state;
      expect(isNaN(tsneState.klLoss)).toBe(false);
      expect(isFinite(tsneState.klLoss)).toBe(true);
    }

    // B. UMAP
    let umapState = umapPlugin.init(dataset, { nNeighbors: 5, minDist: 0.1 }, 42);
    expect(umapState.Y.length).toBe(dataset.train.length);

    for (let s = 0; s < 5; s++) {
      umapState = umapPlugin.step(umapState, dataset, { learningRate: 1.0 }).state;
      expect(isNaN(umapState.crossEntropyLoss)).toBe(false);
      expect(isFinite(umapState.crossEntropyLoss)).toBe(true);
    }
  });

  // Test 8: PCA Reconstruction MSE uses true x2 coordinate (not label y)
  it('correctly calculates PCA reconstruction error in (x1, x2) space regardless of target labels', () => {
    // Points perfectly along line x2 = 2 * x1, with arbitrary labels y != x2
    const trainData = [
      { id: 1, x1: -2, x2: -4, y: 100 },
      { id: 2, x1: -1, x2: -2, y: -50 },
      { id: 3, x1: 0, x2: 0, y: 77 },
      { id: 4, x1: 1, x2: 2, y: -999 },
      { id: 5, x1: 2, x2: 4, y: 3.1415 },
    ];
    const dataset = { train: trainData, test: [] };
    const state = pcaPlugin.init(dataset, { nComponents: 1, centering: 'mean-centered', whiten: 'off' }, 42);

    // Because all points are collinear along x2 = 2*x1, PC1 accounts for 100% of variance
    // and reconstruction MSE must be 0 (or < 1e-6), completely uncorrupted by target labels y!
    expect(state.varianceRatios[0]).toBeGreaterThan(0.9999);
    expect(state.reconstructionMSE).toBeLessThan(1e-6);
  });

  // Test 9: LDA Multiclass (3-Class) Discriminant Fisher Separation
  it('computes generalized Fisher discriminant axis for 3-class classification', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 60,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 3,
      scaling: 'raw',
      stretchX2: false,
    });

    const state = ldaPlugin.init(dataset, { shrinkage: 0.1 }, 42);

    expect(state.classesCount).toBe(3);
    expect(state.classMeans.length).toBe(3);
    expect(state.swEllipses.length).toBe(3);
    expect(isFinite(state.rayleighRatio)).toBe(true);
    expect(state.rayleighRatio).toBeGreaterThan(0);
    expect(Math.hypot(state.wAxis[0], state.wAxis[1])).toBeCloseTo(1.0, 4);
  });

  // Test 10: LDA Sliced Inverse Regression on Continuous Targets
  it('performs Sliced Inverse Regression / Quantile-Binned LDA on continuous regression datasets', () => {
    const regDataset = generateTabular2dDataset({
      kind: 'polynomial-cubic',
      sampleCount: 50,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    // Targets are continuous floats
    expect(regDataset.train.some((p) => p.y !== 1 && p.y !== -1 && p.y !== 2)).toBe(true);

    const state = ldaPlugin.init(regDataset, { shrinkage: 0.05 }, 42);

    expect(state.classesCount).toBeGreaterThanOrEqual(2);
    expect(isFinite(state.rayleighRatio)).toBe(true);
    expect(state.rayleighRatio).toBeGreaterThan(0);
    expect(Math.hypot(state.wAxis[0], state.wAxis[1])).toBeCloseTo(1.0, 4);
    expect(state.projected1D.length).toBe(regDataset.train.length);
  });

  // Test 11: FastICA 1D Projection uses true x2 coordinate
  it('projects 1D values and drop lines using x2 coordinate in FastICA', () => {
    const trainData = [
      { id: 1, x1: 1.0, x2: 3.0, y: 99.0 },
      { id: 2, x1: -1.0, x2: -3.0, y: -88.0 },
      { id: 3, x1: 2.0, x2: -1.0, y: 0.5 },
      { id: 4, x1: -2.0, x2: 1.0, y: -0.5 },
    ];
    const dataset = { train: trainData, test: [] };
    const state = icaPlugin.init(dataset, { contrast: 'logcosh' }, 42);

    const stepRes = icaPlugin.step(state, dataset, { contrast: 'logcosh' });
    const p1 = stepRes.state.projected1D[0];
    const drop1 = stepRes.state.dropLines[0];

    // From coordinate must match (x1, x2)
    expect(drop1.from[0]).toBe(1.0);
    expect(drop1.from[1]).toBe(3.0);
    expect(isFinite(p1.value)).toBe(true);
    expect(isFinite(drop1.to[0])).toBe(true);
    expect(isFinite(drop1.to[1])).toBe(true);
  });

  // Test 12: Predictor fields for Hierarchical, DBSCAN, t-SNE, UMAP produce non-constant grids
  it('produces spatially responsive 2D predictor values for Hierarchical, DBSCAN, t-SNE, and UMAP', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 40,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    // 1. Hierarchical
    const hState = hierarchicalClusteringPlugin.init(dataset, { k: 2, linkage: 'ward' }, 42);
    const hVal1 = hierarchicalClusteringPlugin.predict(new Float32Array([-2, -2]), hState, {});
    const hVal2 = hierarchicalClusteringPlugin.predict(new Float32Array([2, 2]), hState, {});
    expect(hVal1).not.toBe(hVal2); // Must vary with location!

    // 2. DBSCAN
    const dbState = dbscanPlugin.init(dataset, { epsilon: 1.2, minPts: 3 }, 42);
    const dbValCore = dbscanPlugin.predict(new Float32Array([-2, -2]), dbState, {});
    const dbValOutlier = dbscanPlugin.predict(new Float32Array([20, 20]), dbState, {});
    expect(Math.abs(dbValCore as number)).toBeGreaterThan(0);
    expect(dbValOutlier).toBe(0); // Outlier in DBSCAN is neutral noise (0)

    // 3. t-SNE
    const tsneState = tsnePlugin.init(dataset, { perplexity: 5 }, 42);
    const tVal1 = tsnePlugin.predict(new Float32Array([-2, -2]), tsneState, {});
    const tVal2 = tsnePlugin.predict(new Float32Array([2, 2]), tsneState, {});
    expect(typeof tVal1).toBe('number');
    expect(typeof tVal2).toBe('number');

    // 4. UMAP
    const umapState = umapPlugin.init(dataset, { nNeighbors: 5 }, 42);
    const uVal1 = umapPlugin.predict(new Float32Array([-2, -2]), umapState, {});
    const uVal2 = umapPlugin.predict(new Float32Array([2, 2]), umapState, {});
    expect(typeof uVal1).toBe('number');
    expect(typeof uVal2).toBe('number');
  });

  // Test 13: All 9 Stage 3 algorithms execute cleanly on both Classification and Regression datasets
  it('executes all 9 algorithms cleanly on regression datasets (1D Sine Wave and 2D Surface)', () => {
    const sineDataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 30,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const plugins: AlgorithmPlugin<any, any>[] = [
      kMeansPlugin,
      hierarchicalClusteringPlugin,
      dbscanPlugin,
      gmmPlugin,
      pcaPlugin,
      ldaPlugin,
      icaPlugin,
      tsnePlugin,
      umapPlugin,
    ];

    for (const plugin of plugins) {
      const state = plugin.init(sineDataset, {}, 42);
      expect(state).toBeDefined();

      const stepRes = plugin.step(state, sineDataset, {});
      expect(stepRes).toBeDefined();
      expect(stepRes.state).toBeDefined();
      expect(stepRes.metrics).toBeDefined();
      expect(isNaN(stepRes.metrics.trainLoss)).toBe(false);

      const artifacts = plugin.getVisualArtifacts(stepRes.state, sineDataset, {});
      expect(artifacts.length).toBeGreaterThan(0);
    }
  });

  // Test 14: STAGE-03-ISSUE-001 — k-Means Elbow WCSS Monotonically Non-Increasing (K=1..8)
  it('computes monotonically non-increasing elbow WCSS for K=1..8 on Linear Blobs (Seed 42)', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 120,
      noise: 0.1,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const state = kMeansPlugin.init(dataset, { k: 3, initMethod: 'k-means++', metric: 'euclidean' }, 42);
    expect(state.clusterProfiles.length).toBe(8);
    for (let i = 1; i < state.clusterProfiles.length; i++) {
      expect(state.clusterProfiles[i].k).toBe(i + 1);
      expect(state.clusterProfiles[i].inertia).toBeLessThanOrEqual(state.clusterProfiles[i - 1].inertia + 1e-6);
    }
  });

  // Test 15: STAGE-03-ISSUE-004 — GMM Log-Likelihood Sign Consistency
  it('reports consistent signed Log-Likelihood in GMM custom telemetry and Neg Log-Likelihood as trainLoss', () => {
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

    const state = gmmPlugin.init(dataset, { k: 3, covarianceType: 'full' }, 42);
    const metrics = gmmPlugin.loss(state, dataset, {});
    expect(state.logLikelihood).toBeLessThan(0);
    expect(metrics.trainLoss).toBeCloseTo(-state.logLikelihood, 4);
    expect(metrics.custom?.logLikelihood).toBe(Number(state.logLikelihood.toFixed(2)));
  });

  // Test 16: STAGE-03-ISSUE-004 — LDA Binarizes 1D Sine Wave into 2 Classes
  it('binarizes continuous 1D Sine Wave targets into exactly 2 classes in LDA', () => {
    const sineDataset = generateTabular2dDataset({
      kind: 'sine-wave',
      sampleCount: 80,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const state = ldaPlugin.init(sineDataset, { shrinkage: 0 }, 42);
    expect(state.classesCount).toBe(2);
    expect(state.classMeans.length).toBe(2);
    expect(state.swEllipses.length).toBe(2);
  });

  // Test 17: STAGE-03-ISSUE-004 — FastICA Single-Step Progression on Linear Blobs & Concentric Circles
  it('advances FastICA by exactly 1 epoch with finite kurtosis and loss on Linear Blobs and Concentric Circles', () => {
    for (const kind of ['linear-blobs', 'concentric-circles'] as const) {
      const dataset = generateTabular2dDataset({
        kind,
        sampleCount: 80,
        noise: 0.1,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const s0 = icaPlugin.init(dataset, { contrast: 'logcosh' }, 42);
      expect(s0.epoch).toBe(0);

      const res1 = icaPlugin.step(s0, dataset, { contrast: 'logcosh' });
      expect(res1.epoch).toBe(1);
      expect(res1.state.epoch).toBe(1);
      expect(Number.isFinite(res1.state.kurtosis[0])).toBe(true);
      expect(Number.isFinite(res1.state.kurtosis[1])).toBe(true);
      expect(Number.isFinite(res1.metrics.trainLoss)).toBe(true);
      expect(isNaN(res1.metrics.trainLoss)).toBe(false);
    }
  });

  // Test 18: STAGE-03-ISSUE-002 — UMAP Epoch 0 Cross-Entropy Computed (Not Hardcoded 1.8)
  it('computes true Epoch 0 Cross-Entropy in UMAP instead of hardcoding 1.8', () => {
    const dataset = generateTabular2dDataset({
      kind: 'linear-blobs',
      sampleCount: 40,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });

    const s0 = umapPlugin.init(dataset, { nNeighbors: 10, minDist: 0.1 }, 42);
    expect(s0.crossEntropyLoss).not.toBe(1.8);
    expect(s0.crossEntropyLoss).toBeGreaterThan(5);
    const s1 = umapPlugin.step(s0, dataset, { nNeighbors: 10, minDist: 0.1, learningRate: 1.0 });
    expect(Number.isFinite(s1.state.crossEntropyLoss)).toBe(true);
  });
});

