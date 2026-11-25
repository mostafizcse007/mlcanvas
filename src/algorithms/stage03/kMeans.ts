import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import {
  dist2D,
  distManhattan2D,
  computeInertia,
  computeSilhouetteScore,
  computeVoronoiSegments,
  CLUSTER_PALETTE,
} from './clusteringUtils';
import { mulberry32 } from '../../utils/prng';

export interface KMeansState {
  k: number;
  metric: 'euclidean' | 'manhattan';
  initMethod: 'k-means++' | 'random' | 'manual-drag';
  centroids: [number, number][];
  prevCentroids: [number, number][];
  assignments: number[];
  subStep: 'assign' | 'update';
  epoch: number;
  converged: boolean;
  inertia: number;
  silhouette: number;
  clusterProfiles: { k: number; inertia: number; silhouette: number }[];
}

export interface KMeansParams {
  k?: number;
  initMethod?: 'k-means++' | 'random' | 'manual-drag';
  metric?: 'euclidean' | 'manhattan';
  maxIter?: number;
  customCentroids?: [number, number][];
}

function initCentroidsKMeansPP(
  points: [number, number][],
  k: number,
  rand: () => number
): [number, number][] {
  if (points.length === 0) {
    return Array.from({ length: k }, () => [0, 0]);
  }
  const centroids: [number, number][] = [];
  // 1. Choose first centroid uniformly at random
  const firstIdx = Math.floor(rand() * points.length);
  centroids.push([...points[firstIdx]]);

  // 2. Choose remaining k - 1 centroids with probability proportional to D(x)^2
  while (centroids.length < k) {
    const distSq: number[] = new Array(points.length);
    let sumDistSq = 0;

    for (let i = 0; i < points.length; i++) {
      let minDist = Infinity;
      for (let c = 0; c < centroids.length; c++) {
        const d = dist2D(points[i], centroids[c]);
        if (d < minDist) minDist = d;
      }
      distSq[i] = minDist * minDist;
      sumDistSq += distSq[i];
    }

    if (sumDistSq < 1e-9) {
      // Degenerate points, pick random
      centroids.push([...points[Math.floor(rand() * points.length)]]);
      continue;
    }

    const r = rand() * sumDistSq;
    let acc = 0;
    let chosenIdx = points.length - 1;
    for (let i = 0; i < points.length; i++) {
      acc += distSq[i];
      if (acc >= r) {
        chosenIdx = i;
        break;
      }
    }
    centroids.push([...points[chosenIdx]]);
  }

  return centroids;
}

function assignPoints(
  points: [number, number][],
  centroids: [number, number][],
  metric: 'euclidean' | 'manhattan'
): number[] {
  const distFn = metric === 'euclidean' ? dist2D : distManhattan2D;
  const assignments: number[] = new Array(points.length);

  for (let i = 0; i < points.length; i++) {
    let minDist = Infinity;
    let bestC = 0;
    for (let c = 0; c < centroids.length; c++) {
      const d = distFn(points[i], centroids[c]);
      if (d < minDist) {
        minDist = d;
        bestC = c;
      }
    }
    assignments[i] = bestC;
  }
  return assignments;
}

function updateCentroids(
  points: [number, number][],
  assignments: number[],
  oldCentroids: [number, number][]
): { newCentroids: [number, number][]; maxShift: number } {
  const k = oldCentroids.length;
  const sumsX = new Array(k).fill(0);
  const sumsY = new Array(k).fill(0);
  const counts = new Array(k).fill(0);

  for (let i = 0; i < points.length; i++) {
    const c = assignments[i];
    if (c >= 0 && c < k) {
      sumsX[c] += points[i][0];
      sumsY[c] += points[i][1];
      counts[c]++;
    }
  }

  const newCentroids: [number, number][] = [];
  let maxShift = 0;

  for (let c = 0; c < k; c++) {
    if (counts[c] > 0) {
      const nx = sumsX[c] / counts[c];
      const ny = sumsY[c] / counts[c];
      const shift = dist2D([nx, ny], oldCentroids[c]);
      if (shift > maxShift) maxShift = shift;
      newCentroids.push([nx, ny]);
    } else {
      // If empty, keep previous centroid
      newCentroids.push([...oldCentroids[c]]);
    }
  }

  return { newCentroids, maxShift };
}

function computeProfilesAcrossK(
  points: [number, number][],
  maxK = 8
): { k: number; inertia: number; silhouette: number }[] {
  if (points.length === 0) return [];
  const profiles: { k: number; inertia: number; silhouette: number }[] = [];
  const nRestarts = 10;
  const maxProfileIter = 20;
  let prevBestInertia = Infinity;

  for (let curK = 1; curK <= Math.min(maxK, points.length); curK++) {
    let bestInertia = Infinity;
    let bestAssignments: number[] = [];

    for (let restart = 0; restart < nRestarts; restart++) {
      const rand = mulberry32(42 + curK * 101 + restart * 17);
      let cents = initCentroidsKMeansPP(points, curK, rand);
      let asgn = assignPoints(points, cents, 'euclidean');

      for (let it = 0; it < maxProfileIter; it++) {
        const { newCentroids, maxShift } = updateCentroids(points, asgn, cents);
        cents = newCentroids;
        asgn = assignPoints(points, cents, 'euclidean');
        if (maxShift < 1e-4) break;
      }

      const inertia = computeInertia(points, cents, asgn);
      if (inertia < bestInertia) {
        bestInertia = inertia;
        bestAssignments = asgn;
      }
    }

    // True optimal WCSS(K) is monotonically non-increasing in K
    if (bestInertia > prevBestInertia) {
      bestInertia = prevBestInertia;
    }
    prevBestInertia = bestInertia;

    const sil = curK > 1 ? computeSilhouetteScore(points, bestAssignments, curK) : 0;
    profiles.push({ k: curK, inertia: bestInertia, silhouette: sil });
  }

  return profiles;
}

export const kMeansPlugin: AlgorithmPlugin<KMeansState, KMeansParams> = {
  id: 'k-means',
  name: "k-Means (Lloyd's Algorithm & K-Means++)",
  shortCode: 'k-Means',
  stage: 'unsupervised-2d',
  taskType: 'clustering',
  executionMode: 'phase-walkthrough',
  description:
    "Iterative alternating Voronoi assignment and center-of-mass updates minimizing within-cluster sum of squares (WCSS).",

  primaryParams: [
    {
      id: 'k',
      name: 'Clusters (K)',
      type: 'slider',
      min: 1,
      max: 8,
      step: 1,
      default: 3,
      description: 'Number of clusters and Voronoi partition cells.',
    },
    {
      id: 'initMethod',
      name: 'Init Method',
      type: 'select',
      options: [
        { label: 'k-means++ (Smart Spread)', value: 'k-means++' },
        { label: 'Uniform Random', value: 'random' },
        { label: 'Manual Draggable', value: 'manual-drag' },
      ],
      default: 'k-means++',
      description: 'Seeding strategy for initial centroid placement.',
    },
    {
      id: 'metric',
      name: 'Distance',
      type: 'select',
      options: [
        { label: 'Euclidean (L2)', value: 'euclidean' },
        { label: 'Manhattan (L1)', value: 'manhattan' },
      ],
      default: 'euclidean',
      description: 'Metric used to compute distance from points to centroids.',
    },
  ],

  advancedParams: [
    {
      id: 'maxIter',
      name: 'Max Iterations',
      type: 'slider',
      min: 5,
      max: 100,
      step: 5,
      default: 30,
      description: 'Maximum EM iterations before terminating.',
    },
  ],

  init(dataset: DatasetContainer, params: KMeansParams, seed: number = 42): KMeansState {
    const k = Math.max(1, Math.min(8, Math.round(Number(params.k ?? 3))));
    const metric = (params.metric as 'euclidean' | 'manhattan') || 'euclidean';
    const initMethod = params.initMethod || 'k-means++';

    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const rand = mulberry32(seed);

    let centroids: [number, number][];
    if (params.customCentroids && params.customCentroids.length === k) {
      centroids = params.customCentroids.map((c) => [...c]);
    } else if (initMethod === 'k-means++') {
      centroids = initCentroidsKMeansPP(points, k, rand);
    } else {
      // Random pick
      centroids = [];
      for (let i = 0; i < k; i++) {
        if (points.length > 0) {
          const idx = Math.floor(rand() * points.length);
          centroids.push([...points[idx]]);
        } else {
          centroids.push([(rand() - 0.5) * 6, (rand() - 0.5) * 6]);
        }
      }
    }

    const assignments = assignPoints(points, centroids, metric);
    const inertia = computeInertia(points, centroids, assignments);
    const silhouette = k > 1 ? computeSilhouetteScore(points, assignments, k) : 0;
    const clusterProfiles = computeProfilesAcrossK(points, 8);

    return {
      k,
      metric,
      initMethod,
      centroids,
      prevCentroids: centroids.map((c) => [...c]),
      assignments,
      subStep: 'assign',
      epoch: 0,
      converged: false,
      inertia,
      silhouette,
      clusterProfiles,
    };
  },

  step(state: KMeansState, dataset: DatasetContainer, params: KMeansParams): StepResult<KMeansState> {
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const k = state.k;
    const metric = state.metric;

    if (points.length === 0 || k < 1) {
      return {
        state,
        epoch: state.epoch,
        metrics: this.loss(state, dataset, params),
        converged: true,
      };
    }

    let nextCentroids = state.centroids.map((c) => [...c] as [number, number]);
    let prevCentroids = state.prevCentroids.map((c) => [...c] as [number, number]);
    let nextAssignments = [...state.assignments];
    let nextSubStep: 'assign' | 'update' = state.subStep;
    let converged = state.converged;

    if (state.subStep === 'assign') {
      // Sub-Step A: Re-assign points to closest centroid
      nextAssignments = assignPoints(points, state.centroids, metric);
      nextSubStep = 'update';
    } else {
      // Sub-Step B: Move centroids to center of mass
      prevCentroids = state.centroids.map((c) => [...c]);
      const { newCentroids, maxShift } = updateCentroids(points, state.assignments, state.centroids);
      nextCentroids = newCentroids;
      if (maxShift < 1e-4) {
        converged = true;
      }
      nextSubStep = 'assign';
    }

    const inertia = computeInertia(points, nextCentroids, nextAssignments);
    const silhouette = k > 1 ? computeSilhouetteScore(points, nextAssignments, k) : 0;

    const nextState: KMeansState = {
      ...state,
      centroids: nextCentroids,
      prevCentroids,
      assignments: nextAssignments,
      subStep: nextSubStep,
      epoch: state.epoch + 1,
      converged,
      inertia,
      silhouette,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged,
    };
  },

  predict(input: Float32Array, state: KMeansState, params: KMeansParams): number {
    const metric = (params.metric as 'euclidean' | 'manhattan') || state.metric || 'euclidean';
    const distFn = metric === 'euclidean' ? dist2D : distManhattan2D;
    const pt: [number, number] = [input[0], input[1]];

    let minDist = Infinity;
    let secondMinDist = Infinity;
    let bestC = 0;
    for (let c = 0; c < state.centroids.length; c++) {
      const d = distFn(pt, state.centroids[c]);
      if (d < minDist) {
        secondMinDist = minDist;
        minDist = d;
        bestC = c;
      } else if (d < secondMinDist) {
        secondMinDist = d;
      }
    }
    if (state.centroids.length < 2) return bestC;
    const margin = Math.min(3.0, secondMinDist - minDist);
    return bestC % 2 === 0 ? margin : -margin;
  },

  loss(state: KMeansState, _dataset: DatasetContainer, _params: KMeansParams): TelemetryMetrics {
    const activeClusters = new Set(state.assignments).size;
    return {
      trainLoss: state.inertia,
      testLoss: state.inertia,
      trainAccuracy: Math.round(state.silhouette * 1000) / 10, // Silhouette percentage
      custom: {
        inertia: Number(state.inertia.toFixed(2)),
        silhouette: Number(state.silhouette.toFixed(3)),
        activeClusters,
        subStep: state.subStep === 'assign' ? 'Assign Points (Voronoi)' : 'Update Centroids (Mean)',
      },
    };
  },

  getVisualArtifacts(state: KMeansState, dataset: DatasetContainer, _params: KMeansParams): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);

    // 1. Voronoi cell boundaries
    const voronoiSegments = computeVoronoiSegments(state.centroids, [-6, 6, -6, 6]);

    // 2. Centroids with motion trails and color badges
    const centroidArtifacts = state.centroids.map((c, i) => ({
      id: i,
      current: c,
      previous: state.prevCentroids[i],
      color: CLUSTER_PALETTE[i % CLUSTER_PALETTE.length],
    }));

    // 3. Spoke lines connecting points to assigned centroid
    const spokes = points.map((_p, i) => ({
      pointIndex: i,
      centroidId: state.assignments[i] ?? 0,
    }));

    artifacts.push({
      type: 'clustering-overlay',
      centroids: centroidArtifacts,
      assignments: state.assignments,
      spokes,
      voronoiSegments,
      clusterProfiles: state.clusterProfiles,
    });

    return artifacts;
  },

  serializeSnapshot(state: KMeansState): Float32Array {
    // [epoch, k, subStep, converged, inertia, silhouette, ...centroids(2*k)]
    const k = state.centroids.length;
    const buf = new Float32Array(6 + k * 2);
    buf[0] = state.epoch;
    buf[1] = k;
    buf[2] = state.subStep === 'assign' ? 0 : 1;
    buf[3] = state.converged ? 1 : 0;
    buf[4] = state.inertia;
    buf[5] = state.silhouette;
    for (let i = 0; i < k; i++) {
      buf[6 + i * 2] = state.centroids[i][0];
      buf[6 + i * 2 + 1] = state.centroids[i][1];
    }
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: KMeansState): KMeansState {
    const k = Math.round(buffer[1]);
    const centroids: [number, number][] = [];
    for (let i = 0; i < k; i++) {
      centroids.push([buffer[6 + i * 2], buffer[6 + i * 2 + 1]]);
    }
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      k,
      subStep: buffer[2] === 0 ? 'assign' : 'update',
      converged: buffer[3] === 1,
      inertia: buffer[4],
      silhouette: buffer[5],
      centroids,
    };
  },

  explain(state: KMeansState, _params: KMeansParams, _lastStep?: StepResult<KMeansState>): ExplanationPayload {
    return {
      summary: `k-Means partitions space into ${state.k} convex Voronoi polyhedra by iteratively assigning points to the nearest centroid and shifting centroids to cluster means.`,
      coreAssumption:
        'Clusters are spherical, linearly separable (convex), and share similar variance and scale across all dimensions.',
      whenItFails:
        'Concentric circles or entangled manifolds (Voronoi hyperplanes slice straight through non-convex curves) and adversarial co-linear initializations.',
      microLog: [
        `Assign ${state.assignments.length} points to nearest of ${state.k} centroids via Voronoi distance partition`,
        `Compute WCSS Inertia = ${state.inertia.toFixed(2)} (Silhouette = ${state.silhouette.toFixed(3)})`,
        `Update centroid coordinates μ_k to center-of-mass of assigned points`,
      ],
      mathLatex: {
        objective: `\\min_{\\{\\mu_k\\}, \\{r_{ik}\\}} \\sum_{i=1}^N \\sum_{k=1}^K r_{ik} \\|x_i - \\mu_k\\|^2`,
        lossOrUpdate: `\\mu_k = \\frac{\\sum_i r_{ik} x_i}{\\sum_i r_{ik}}, \\quad r_{ik} = \\mathbb{I}(k = \\arg\\min_j \\|x_i - \\mu_j\\|^2)`,
      },
      codeSnippet: {
        sklearn: `from sklearn.cluster import KMeans\nkmeans = KMeans(n_clusters=${state.k}, init='${state.initMethod}', n_init=10)\ny_pred = kmeans.fit_predict(X)`,
        pytorch: `# Custom PyTorch Lloyd K-Means\n# Alternating pairwise distance argmin & scatter_add center-of-mass`,
      },
    };
  },
};
