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
  computeSilhouetteScore,
  CLUSTER_PALETTE,
} from './clusteringUtils';

export interface DendrogramNode {
  id: number;
  left?: number;
  right?: number;
  distance: number;
  count: number;
  leafIndex?: number;
  x: number;
  y: number;
  clusterId?: number;
}

export interface HierarchicalState {
  k: number;
  linkage: 'single' | 'complete' | 'average' | 'ward';
  cutHeight: number;
  maxDistance: number;
  mergeHistory: {
    step: number;
    clusterA: number;
    clusterB: number;
    distance: number;
    sizeA: number;
    sizeB: number;
    centerA: [number, number];
    centerB: [number, number];
  }[];
  dendrogramNodes: DendrogramNode[];
  leafOrder: number[];
  activeStep: number;
  assignments: number[];
  centroids: [number, number][];
  epoch: number;
  converged: boolean;
  silhouette: number;
}

export interface HierarchicalParams {
  k?: number;
  linkage?: 'single' | 'complete' | 'average' | 'ward';
  cutHeight?: number;
}

interface ClusterGroup {
  id: number;
  pointIndices: number[];
  centroid: [number, number];
}

function computeLinkageDistance(
  groupA: ClusterGroup,
  groupB: ClusterGroup,
  points: [number, number][],
  linkage: 'single' | 'complete' | 'average' | 'ward'
): number {
  const ptsA = groupA.pointIndices;
  const ptsB = groupB.pointIndices;

  switch (linkage) {
    case 'single': {
      let minDist = Infinity;
      for (let i = 0; i < ptsA.length; i++) {
        for (let j = 0; j < ptsB.length; j++) {
          const d = dist2D(points[ptsA[i]], points[ptsB[j]]);
          if (d < minDist) minDist = d;
        }
      }
      return minDist;
    }
    case 'complete': {
      let maxDist = 0;
      for (let i = 0; i < ptsA.length; i++) {
        for (let j = 0; j < ptsB.length; j++) {
          const d = dist2D(points[ptsA[i]], points[ptsB[j]]);
          if (d > maxDist) maxDist = d;
        }
      }
      return maxDist;
    }
    case 'average': {
      let sum = 0;
      for (let i = 0; i < ptsA.length; i++) {
        for (let j = 0; j < ptsB.length; j++) {
          sum += dist2D(points[ptsA[i]], points[ptsB[j]]);
        }
      }
      return sum / (ptsA.length * ptsB.length);
    }
    case 'ward': {
      const nA = ptsA.length;
      const nB = ptsB.length;
      const dCentroid = dist2D(groupA.centroid, groupB.centroid);
      return Math.sqrt((nA * nB) / (nA + nB)) * dCentroid;
    }
  }
}

function runFullAgglomeration(
  points: [number, number][],
  linkage: 'single' | 'complete' | 'average' | 'ward'
): {
  nodes: DendrogramNode[];
  merges: HierarchicalState['mergeHistory'];
  leafOrder: number[];
  maxDistance: number;
} {
  const n = points.length;
  if (n === 0) {
    return { nodes: [], merges: [], leafOrder: [], maxDistance: 1 };
  }

  // Active clusters initialized to 1 point each
  let activeClusters: ClusterGroup[] = points.map((p, i) => ({
    id: i,
    pointIndices: [i],
    centroid: [p[0], p[1]],
  }));

  const nodes: DendrogramNode[] = points.map((_p, i) => ({
    id: i,
    distance: 0,
    count: 1,
    leafIndex: i,
    x: 0, // Assigned later after ordering
    y: 0,
  }));

  const merges: HierarchicalState['mergeHistory'] = [];
  let nextNodeId = n;

  while (activeClusters.length > 1) {
    let minD = Infinity;
    let bestA = 0;
    let bestB = 1;

    for (let i = 0; i < activeClusters.length; i++) {
      for (let j = i + 1; j < activeClusters.length; j++) {
        const d = computeLinkageDistance(activeClusters[i], activeClusters[j], points, linkage);
        if (d < minD) {
          minD = d;
          bestA = i;
          bestB = j;
        }
      }
    }

    const cA = activeClusters[bestA];
    const cB = activeClusters[bestB];

    // Compute merged centroid
    const mergedIndices = [...cA.pointIndices, ...cB.pointIndices];
    let sumX = 0;
    let sumY = 0;
    for (const idx of mergedIndices) {
      sumX += points[idx][0];
      sumY += points[idx][1];
    }
    const mergedCentroid: [number, number] = [
      sumX / mergedIndices.length,
      sumY / mergedIndices.length,
    ];

    merges.push({
      step: merges.length,
      clusterA: cA.id,
      clusterB: cB.id,
      distance: minD,
      sizeA: cA.pointIndices.length,
      sizeB: cB.pointIndices.length,
      centerA: cA.centroid,
      centerB: cB.centroid,
    });

    const parentNode: DendrogramNode = {
      id: nextNodeId++,
      left: cA.id,
      right: cB.id,
      distance: minD,
      count: mergedIndices.length,
      x: 0,
      y: minD,
    };
    nodes.push(parentNode);

    // Replace cA and cB with new cluster
    const newGroup: ClusterGroup = {
      id: parentNode.id,
      pointIndices: mergedIndices,
      centroid: mergedCentroid,
    };

    activeClusters = activeClusters.filter((_, idx) => idx !== bestA && idx !== bestB);
    activeClusters.push(newGroup);
  }

  // Determine leaf ordering by depth-first tree traversal
  const leafOrder: number[] = [];
  const rootId = nodes.length - 1;

  function traverseLeaves(nodeId: number) {
    const node = nodes[nodeId];
    if (!node) return;
    if (node.left === undefined || node.right === undefined) {
      leafOrder.push(node.id);
      return;
    }
    traverseLeaves(node.left);
    traverseLeaves(node.right);
  }

  if (rootId >= 0) {
    traverseLeaves(rootId);
  }

  // Assign x coordinates based on leaf rank
  const leafXMap = new Map<number, number>();
  const totalLeaves = leafOrder.length;
  leafOrder.forEach((leafId, idx) => {
    const xCoord = totalLeaves > 1 ? idx / (totalLeaves - 1) : 0.5;
    leafXMap.set(leafId, xCoord);
    if (nodes[leafId]) {
      nodes[leafId].x = xCoord;
    }
  });

  // Bottom-up assign internal node x coordinates as midpoint of children
  for (let i = n; i < nodes.length; i++) {
    const node = nodes[i];
    if (node.left !== undefined && node.right !== undefined) {
      const xLeft = nodes[node.left]?.x ?? 0;
      const xRight = nodes[node.right]?.x ?? 1;
      node.x = (xLeft + xRight) / 2;
    }
  }

  const maxDistance = merges.length > 0 ? Math.max(...merges.map((m) => m.distance), 0.1) : 1;

  return { nodes, merges, leafOrder, maxDistance };
}

function partitionByCutHeight(
  nodes: DendrogramNode[],
  nLeaves: number,
  cutHeight: number
): number[] {
  const assignments = new Array(nLeaves).fill(0);
  if (nodes.length === 0) return assignments;

  let currentClusterId = 0;
  const visited = new Set<number>();

  function assignSubtree(nodeId: number, clusterId: number) {
    const node = nodes[nodeId];
    if (!node) return;
    if (node.left === undefined && node.right === undefined) {
      if (node.id < nLeaves) {
        assignments[node.id] = clusterId;
      }
      return;
    }
    if (node.left !== undefined) assignSubtree(node.left, clusterId);
    if (node.right !== undefined) assignSubtree(node.right, clusterId);
  }

  const rootId = nodes.length - 1;

  function findBranches(nodeId: number) {
    const node = nodes[nodeId];
    if (!node || visited.has(nodeId)) return;
    visited.add(nodeId);

    // If node distance <= cutHeight or is a leaf, it's an intact cluster
    if (node.distance <= cutHeight || (node.left === undefined && node.right === undefined)) {
      assignSubtree(nodeId, currentClusterId++);
      return;
    }

    // Otherwise split into children
    if (node.left !== undefined) findBranches(node.left);
    if (node.right !== undefined) findBranches(node.right);
  }

  findBranches(rootId);
  return assignments;
}

function computeClusterCentroids(points: [number, number][], assignments: number[]): [number, number][] {
  const distinctClusters = Array.from(new Set(assignments)).sort((a, b) => a - b);
  return distinctClusters.map((cId) => {
    let sumX = 0;
    let sumY = 0;
    let count = 0;
    for (let i = 0; i < points.length; i++) {
      if (assignments[i] === cId) {
        sumX += points[i][0];
        sumY += points[i][1];
        count++;
      }
    }
    return [count > 0 ? sumX / count : 0, count > 0 ? sumY / count : 0] as [number, number];
  });
}

export const hierarchicalClusteringPlugin: AlgorithmPlugin<HierarchicalState, HierarchicalParams> = {
  id: 'hierarchical-clustering',
  name: 'Hierarchical Clustering (Agglomerative Bottom-Up)',
  shortCode: 'Hierarchical',
  stage: 'unsupervised-2d',
  taskType: 'clustering',
  executionMode: 'phase-walkthrough',
  description:
    'Recursive bottom-up cluster merging with interactive dendrogram visualization and dynamic horizontal cut-height slider.',

  primaryParams: [
    {
      id: 'k',
      name: 'Target Clusters (K)',
      type: 'slider',
      min: 1,
      max: 8,
      step: 1,
      default: 3,
      description: 'Number of clusters to extract by cutting the dendrogram tree.',
    },
    {
      id: 'linkage',
      name: 'Linkage Criterion',
      type: 'select',
      options: [
        { label: 'Ward (Min Variance)', value: 'ward' },
        { label: 'Complete (Max Dist)', value: 'complete' },
        { label: 'Average (Mean Dist)', value: 'average' },
        { label: 'Single (Min Dist / Chaining)', value: 'single' },
      ],
      default: 'ward',
      description: 'Metric determining inter-cluster distance when merging.',
    },
  ],

  advancedParams: [
    {
      id: 'cutHeight',
      name: 'Cut Height (h)',
      type: 'slider',
      min: 0.1,
      max: 10.0,
      step: 0.1,
      default: 2.0,
      description: 'Horizontal cut line threshold on the dendrogram tree.',
    },
  ],

  init(dataset: DatasetContainer, params: HierarchicalParams, _seed: number = 42): HierarchicalState {
    const k = Math.max(1, Math.min(8, Math.round(Number(params.k ?? 3))));
    const linkage = params.linkage || 'ward';
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);

    const { nodes, merges, leafOrder, maxDistance } = runFullAgglomeration(points, linkage);

    // Calculate cutHeight that yields target K clusters
    let cutHeight = params.cutHeight ?? (maxDistance * 0.4);
    if (params.cutHeight === undefined && merges.length >= k) {
      // Cut strictly below merge step (N - K) so that exactly K clusters exist
      const mergeIdx = Math.max(0, points.length - k);
      if (merges[mergeIdx]) {
        const prevDist = mergeIdx > 0 ? merges[mergeIdx - 1].distance : 0;
        cutHeight = (merges[mergeIdx].distance + prevDist) / 2;
      }
    }

    const assignments = partitionByCutHeight(nodes, points.length, cutHeight);
    const silhouette = k > 1 ? computeSilhouetteScore(points, assignments, k) : 0;
    const centroids = computeClusterCentroids(points, assignments);

    return {
      k,
      linkage,
      cutHeight,
      maxDistance,
      mergeHistory: merges,
      dendrogramNodes: nodes,
      leafOrder,
      activeStep: 0,
      assignments,
      centroids,
      epoch: 0,
      converged: false,
      silhouette,
    };
  },

  step(state: HierarchicalState, dataset: DatasetContainer, params: HierarchicalParams): StepResult<HierarchicalState> {
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const maxSteps = state.mergeHistory.length;
    const nextStep = Math.min(maxSteps, state.activeStep + 1);
    const converged = nextStep >= maxSteps;

    // Advance cut height or step
    const currentMerge = state.mergeHistory[nextStep - 1];
    const cutHeight = currentMerge ? currentMerge.distance : state.cutHeight;
    const assignments = partitionByCutHeight(state.dendrogramNodes, points.length, cutHeight);

    const distinctClusters = new Set(assignments).size;
    const silhouette = distinctClusters > 1 ? computeSilhouetteScore(points, assignments, distinctClusters) : 0;
    const centroids = computeClusterCentroids(points, assignments);

    const nextState: HierarchicalState = {
      ...state,
      activeStep: nextStep,
      cutHeight,
      assignments,
      centroids,
      epoch: state.epoch + 1,
      converged,
      silhouette,
    };

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: this.loss(nextState, dataset, params),
      converged,
    };
  },

  predict(input: Float32Array, state: HierarchicalState, _params: HierarchicalParams): number {
    const centroids = state.centroids;
    if (!centroids || centroids.length === 0) {
      return state.assignments.length > 0 ? state.assignments[0] : 0;
    }
    const pt: [number, number] = [input[0], input[1]];
    let minDist = Infinity;
    let secondMinDist = Infinity;
    let bestC = 0;
    for (let c = 0; c < centroids.length; c++) {
      const d = dist2D(pt, centroids[c]);
      if (d < minDist) {
        secondMinDist = minDist;
        minDist = d;
        bestC = c;
      } else if (d < secondMinDist) {
        secondMinDist = d;
      }
    }
    if (centroids.length < 2) return bestC;
    const margin = Math.min(3.0, secondMinDist - minDist);
    return bestC % 2 === 0 ? margin : -margin;
  },

  loss(state: HierarchicalState, _dataset: DatasetContainer, _params: HierarchicalParams): TelemetryMetrics {
    const distinctClusters = new Set(state.assignments).size;
    return {
      trainLoss: state.cutHeight,
      testLoss: state.cutHeight,
      trainAccuracy: Math.round(state.silhouette * 1000) / 10,
      custom: {
        cutHeight: Number(state.cutHeight.toFixed(3)),
        activeClusters: distinctClusters,
        silhouette: Number(state.silhouette.toFixed(3)),
        linkage: state.linkage,
      },
    };
  },

  getVisualArtifacts(state: HierarchicalState, dataset: DatasetContainer, _params: HierarchicalParams): VisualArtifact[] {
    const points: [number, number][] = dataset.train.map((p) => [p.x1, p.x2]);
    const distinctClusters = Array.from(new Set(state.assignments)).sort((a, b) => a - b);

    // Compute cluster centroids
    const centroids = distinctClusters.map((cId) => {
      let sumX = 0;
      let sumY = 0;
      let count = 0;
      for (let i = 0; i < points.length; i++) {
        if (state.assignments[i] === cId) {
          sumX += points[i][0];
          sumY += points[i][1];
          count++;
        }
      }
      return {
        id: cId,
        current: [count > 0 ? sumX / count : 0, count > 0 ? sumY / count : 0] as [number, number],
        color: CLUSTER_PALETTE[cId % CLUSTER_PALETTE.length],
      };
    });

    const hierarchicalMerges = state.mergeHistory.slice(0, state.activeStep).map((m) => ({
      source: m.centerA,
      target: m.centerB,
      distance: m.distance,
    }));

    return [
      {
        type: 'clustering-overlay',
        centroids,
        assignments: state.assignments,
        hierarchicalMerges,
      },
      {
        type: 'dendrogram-data',
        nodes: state.dendrogramNodes,
        cutHeight: state.cutHeight,
        maxDistance: state.maxDistance,
        leafOrder: state.leafOrder,
        activeClustersCount: distinctClusters.length,
      },
    ];
  },

  serializeSnapshot(state: HierarchicalState): Float32Array {
    const buf = new Float32Array(5);
    buf[0] = state.epoch;
    buf[1] = state.k;
    buf[2] = state.cutHeight;
    buf[3] = state.silhouette;
    buf[4] = state.activeStep;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: HierarchicalState): HierarchicalState {
    const cutHeight = buffer[2];
    const assignments = partitionByCutHeight(state.dendrogramNodes, state.assignments.length, cutHeight);
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      k: Math.round(buffer[1]),
      cutHeight,
      silhouette: buffer[3],
      activeStep: Math.round(buffer[4]),
      assignments,
    };
  },

  explain(state: HierarchicalState, _params: HierarchicalParams, _lastStep?: StepResult<HierarchicalState>): ExplanationPayload {
    return {
      summary: `Agglomerative Hierarchical Clustering builds a nested hierarchy of clusters by iteratively fusing the closest pair of clusters under the ${state.linkage} linkage criterion.`,
      coreAssumption:
        'Data exhibits hierarchical or multi-scale cluster structure; linkage criterion correctly characterizes inter-cluster proximity.',
      whenItFails:
        "Single linkage suffers from the 'chaining trap' where sparse noise points bridge two separate clusters into a single elongated chain.",
      microLog: [
        `Construct complete agglomerative tree with ${state.dendrogramNodes.length} merge nodes`,
        `Apply horizontal cut-line at height h = ${state.cutHeight.toFixed(2)}`,
        `Extract ${new Set(state.assignments).size} active clusters (Silhouette = ${state.silhouette.toFixed(3)})`,
      ],
      mathLatex: {
        objective: `d(A, B) = \\min_{a \\in A, b \\in B} \\|a - b\\| \\quad (\\text{Single Linkage})`,
        lossOrUpdate: `\\Delta \\text{Var}(A, B) = \\frac{|A| |B|}{|A| + |B|} \\|\\mu_A - \\mu_B\\|^2 \\quad (\\text{Ward's Criterion})`,
      },
      codeSnippet: {
        sklearn: `from sklearn.cluster import AgglomerativeClustering\nmodel = AgglomerativeClustering(n_clusters=${state.k}, linkage='${state.linkage}')\ny_pred = model.fit_predict(X)`,
        pytorch: `# Scipy Dendrogram hierarchy link matrix & fcluster\nfrom scipy.cluster.hierarchy import linkage, dendrogram, fcluster`,
      },
    };
  },
};

