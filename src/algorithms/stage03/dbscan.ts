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
  computeSilhouetteScore,
  CLUSTER_PALETTE,
} from './clusteringUtils';

export interface DBSCANState {
  epsilon: number;
  minPts: number;
  metric: 'euclidean' | 'manhattan';
  assignments: number[]; // cluster ID >= 0, or -1 for noise
  corePointIds: number[];
  borderPointIds: number[];
  noisePointIds: number[];
  corePoints?: { point: [number, number]; clusterId: number }[];
  sweepIndex: number;
  visited: boolean[];
  epoch: number;
  converged: boolean;
  silhouette: number;
  activeClusters: number;
}

export interface DBSCANParams {
  epsilon?: number;
  minPts?: number;
  metric?: 'euclidean' | 'manhattan';
}

function findNeighbors(
  pointIdx: number,
  points: [number, number][],
  epsilon: number,
  metric: 'euclidean' | 'manhattan'
): number[] {
  const distFn = metric === 'euclidean' ? dist2D : distManhattan2D;
  const neighbors: number[] = [];
  const p = points[pointIdx];

  for (let i = 0; i < points.length; i++) {
    if (distFn(p, points[i]) <= epsilon) {
      neighbors.push(i);
    }
  }
  return neighbors;
}

function runFullDBSCAN(
  points: [number, number][],
  epsilon: number,
  minPts: number,
  metric: 'euclidean' | 'manhattan'
): {
  assignments: number[];
  corePointIds: number[];
  borderPointIds: number[];
  noisePointIds: number[];
  clustersCount: number;
} {
  const n = points.length;
  const assignments = new Array(n).fill(-1); // -1 = unvisited / noise
  const visited = new Array(n).fill(false);
  const coreSet = new Set<number>();
  const borderSet = new Set<number>();
  const noiseSet = new Set<number>();

  let clusterId = 0;

  for (let i = 0; i < n; i++) {
    if (visited[i]) continue;
    visited[i] = true;

    const neighbors = findNeighbors(i, points, epsilon, metric);

    if (neighbors.length < minPts) {
      noiseSet.add(i);
      assignments[i] = -1;
    } else {
      coreSet.add(i);
      noiseSet.delete(i);
      assignments[i] = clusterId;

      // Expand cluster queue
      const queue = [...neighbors];
      let qIdx = 0;

      while (qIdx < queue.length) {
        const neighborIdx = queue[qIdx++];

        if (!visited[neighborIdx]) {
          visited[neighborIdx] = true;
          const neighborNeighbors = findNeighbors(neighborIdx, points, epsilon, metric);
          if (neighborNeighbors.length >= minPts) {
            coreSet.add(neighborIdx);
            noiseSet.delete(neighborIdx);
            for (const nn of neighborNeighbors) {
              if (!queue.includes(nn)) {
                queue.push(nn);
              }
            }
          } else {
            borderSet.add(neighborIdx);
          }
        }

        // If unassigned or previously marked as noise
        if (assignments[neighborIdx] === -1) {
          assignments[neighborIdx] = clusterId;
          noiseSet.delete(neighborIdx);
          if (!coreSet.has(neighborIdx)) {
            borderSet.add(neighborIdx);
          }
        }
      }

      clusterId++;
    }
  }

  return {
    assignments,
    corePointIds: Array.from(coreSet),
    borderPointIds: Array.from(borderSet),
    noisePointIds: Array.from(noiseSet),
    clustersCount: clusterId,
  };
}

export const dbscanPlugin: AlgorithmPlugin<DBSCANState, DBSCANParams> = {
  id: 'dbscan',
  name: 'DBSCAN (Density-Based Spatial Clustering of Applications with Noise)',
  shortCode: 'DBSCAN',
  stage: 'unsupervised-2d',
  taskType: 'clustering',
  executionMode: 'phase-walkthrough',
  description:
    'Density-connected cluster expansion with automatic noise rejection and arbitrary non-convex shape discovery.',

  primaryParams: [
    {
      id: 'epsilon',
      name: 'Epsilon Radius (ε)',
      type: 'slider',
      min: 0.1,
      max: 2.0,
      step: 0.05,
      default: 0.8,
      description: 'Maximum neighborhood distance threshold to consider points density-connected.',
    },
    {
      id: 'minPts',
      name: 'MinPts',
      type: 'slider',
      min: 2,
      max: 15,
      step: 1,
      default: 4,
      description: 'Minimum neighboring points required inside ε-ball to form a Core Point.',
    },
    {
      id: 'metric',
      name: 'Distance Metric',
      type: 'select',
      options: [
        { label: 'Euclidean (L2)', value: 'euclidean' },
        { label: 'Manhattan (L1)', value: 'manhattan' },
      ],
      default: 'euclidean',
      description: 'Metric used to calculate neighborhood ball.',
    },
  ],

  advancedParams: [],

  init(dataset: DatasetContainer, params: DBSCANParams, _seed: number = 42): DBSCANState {
    const epsilon = Math.max(0.05, Number(params.epsilon ?? 0.8));
    const minPts = Math.max(2, Math.round(Number(params.minPts ?? 4)));
    const metric = (params.metric as 'euclidean' | 'manhattan') || 'euclidean';
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);

    const { assignments, corePointIds, borderPointIds, noisePointIds, clustersCount } = runFullDBSCAN(
      points,
      epsilon,
      minPts,
      metric
    );

    const silhouette = clustersCount > 1 ? computeSilhouetteScore(points, assignments, clustersCount) : 0;

    const corePoints = corePointIds.map((idx) => ({
      point: points[idx],
      clusterId: assignments[idx],
    }));

    return {
      epsilon,
      minPts,
      metric,
      assignments,
      corePointIds,
      borderPointIds,
      noisePointIds,
      corePoints,
      sweepIndex: 0,
      visited: new Array(points.length).fill(true),
      epoch: 0,
      converged: true,
      silhouette,
      activeClusters: clustersCount,
    };
  },

  step(state: DBSCANState, dataset: DatasetContainer, params: DBSCANParams): StepResult<DBSCANState> {
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const nextSweep = (state.sweepIndex + 1) % Math.max(1, points.length);

    // Step walks through core points as active sweep radar
    const nextState: DBSCANState = {
      ...state,
      sweepIndex: nextSweep,
      epoch: state.epoch + 1,
      converged: nextSweep === 0,
    };

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: this.loss(nextState, dataset, params),
      converged: nextState.converged,
    };
  },

  predict(input: Float32Array, state: DBSCANState, params: DBSCANParams): number {
    const metric = (params.metric as 'euclidean' | 'manhattan') || state.metric || 'euclidean';
    const distFn = metric === 'euclidean' ? dist2D : distManhattan2D;
    const pt: [number, number] = [input[0], input[1]];
    const eps = Number(params.epsilon ?? state.epsilon);

    if (!state.corePoints || state.corePoints.length === 0) {
      return -1;
    }

    let minDist = Infinity;
    let closestCluster = -1;

    for (let i = 0; i < state.corePoints.length; i++) {
      const cp = state.corePoints[i];
      const d = distFn(pt, cp.point);
      if (d < minDist) {
        minDist = d;
        closestCluster = cp.clusterId;
      }
    }

    if (minDist <= eps && closestCluster >= 0) {
      const reachability = ((eps - minDist) / eps) * 2.5;
      return closestCluster % 2 === 0 ? reachability : -reachability;
    }
    return 0; // Neutral noise background
  },

  loss(state: DBSCANState, dataset: DatasetContainer, _params: DBSCANParams): TelemetryMetrics {
    const noiseFraction = dataset.train.length > 0 ? (state.noisePointIds.length / dataset.train.length) * 100 : 0;
    return {
      trainLoss: noiseFraction,
      testLoss: noiseFraction,
      trainAccuracy: Math.round(state.silhouette * 1000) / 10,
      custom: {
        activeClusters: state.activeClusters,
        noisePoints: state.noisePointIds.length,
        corePoints: state.corePointIds.length,
        borderPoints: state.borderPointIds.length,
        silhouette: Number(state.silhouette.toFixed(3)),
      },
    };
  },

  getVisualArtifacts(state: DBSCANState, dataset: DatasetContainer, _params: DBSCANParams): VisualArtifact[] {
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);

    // Calculate cluster centroids for active clusters
    const clusterCentroids: { id: number; current: [number, number]; color: string }[] = [];
    for (let c = 0; c < state.activeClusters; c++) {
      let sumX = 0;
      let sumY = 0;
      let count = 0;
      for (let i = 0; i < points.length; i++) {
        if (state.assignments[i] === c) {
          sumX += points[i][0];
          sumY += points[i][1];
          count++;
        }
      }
      if (count > 0) {
        clusterCentroids.push({
          id: c,
          current: [sumX / count, sumY / count],
          color: CLUSTER_PALETTE[c % CLUSTER_PALETTE.length],
        });
      }
    }

    return [
      {
        type: 'clustering-overlay',
        centroids: clusterCentroids,
        assignments: state.assignments,
        dbscan: {
          epsilon: state.epsilon,
          minPts: state.minPts,
          corePointIds: state.corePointIds,
          borderPointIds: state.borderPointIds,
          noisePointIds: state.noisePointIds,
          sweepIndex: state.sweepIndex,
        },
      },
    ];
  },

  serializeSnapshot(state: DBSCANState): Float32Array {
    const buf = new Float32Array(6);
    buf[0] = state.epoch;
    buf[1] = state.epsilon;
    buf[2] = state.minPts;
    buf[3] = state.activeClusters;
    buf[4] = state.noisePointIds.length;
    buf[5] = state.silhouette;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: DBSCANState): DBSCANState {
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      epsilon: buffer[1],
      minPts: Math.round(buffer[2]),
      activeClusters: Math.round(buffer[3]),
      silhouette: buffer[5],
    };
  },

  explain(state: DBSCANState, _params: DBSCANParams, _lastStep?: StepResult<DBSCANState>): ExplanationPayload {
    return {
      summary: `DBSCAN partitions dense connected points into clusters of arbitrary geometry while isolating sparse outliers as noise (identified ${state.activeClusters} clusters, ${state.noisePointIds.length} noise points).`,
      coreAssumption:
        'Clusters are high-density regions separated by contiguous low-density areas; density across all clusters is relatively homogeneous.',
      whenItFails:
        'Multi-density clusters: a single global ε cannot simultaneously separate dense clusters without fracturing sparse ones.',
      microLog: [
        `Evaluate ε-neighborhoods (radius ε = ${state.epsilon.toFixed(2)}, MinPts = ${state.minPts})`,
        `Label ${state.corePointIds.length} Core Points, ${state.borderPointIds.length} Border Points`,
        `Reject ${state.noisePointIds.length} isolated points as unassigned Noise (-1)`,
      ],
      mathLatex: {
        objective: `N_\\varepsilon(p) = \\{q \\in D \\mid \\text{dist}(p, q) \\le \\varepsilon\\}, \\quad |N_\\varepsilon(p)| \\ge \\text{MinPts}`,
        lossOrUpdate: `p \\underset{\\text{density}}{\\longleftrightarrow} q \\iff \\exists p_1, \\dots, p_k \\text{ s.t. } p_{i+1} \\in N_\\varepsilon(p_i) \\text{ and } p_i \\text{ is Core}`,
      },
      codeSnippet: {
        sklearn: `from sklearn.cluster import DBSCAN\ndb = DBSCAN(eps=${state.epsilon}, min_samples=${state.minPts}, metric='${state.metric}')\ny_pred = db.fit_predict(X)  # -1 represents noise points`,
        pytorch: `# Custom PyTorch pairwise distance matrix thresholding for DBSCAN`,
      },
    };
  },
};
