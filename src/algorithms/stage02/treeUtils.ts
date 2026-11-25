import { TreeNodeData, DatasetContainer } from '../../types/algorithm';

export interface TreeNode {
  id: number;
  depth: number;
  isLeaf: boolean;
  featureIndex?: 0 | 1;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  samples: number;
  impurity: number;
  value: number; // class (+1, -1, 2) or continuous regression value
  classCounts?: [number, number, number]; // [class 1, class -1, class 2]
  region: [number, number, number, number]; // [xMin, xMax, yMin, yMax]
  gain?: number;
  similarity?: number;
  weight?: number;
}

export function isRegressionDataset(
  dataset: DatasetContainer,
  params?: object
): boolean {
  const p = params as { task?: string; criterion?: string } | undefined;
  if (p?.task === 'regression' || p?.criterion === 'mse') return true;
  const train = dataset.train;
  if (!train || train.length === 0) return false;
  if (train.some((pt) => pt.y !== 1 && pt.y !== -1 && pt.y !== 2)) return true;
  if (p?.task === 'classification') return false;
  return false;
}

export function is1DRegressionDataset(
  dataset: DatasetContainer,
  isRegression: boolean
): boolean {
  if (!isRegression) return false;
  const train = dataset.train;
  if (!train || train.length < 2) return false;
  return train.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);
}

export function getAllowedFeatures(
  dataset: DatasetContainer,
  isRegression: boolean
): (0 | 1)[] {
  return is1DRegressionDataset(dataset, isRegression) ? [0] : [0, 1];
}

export function computeBoundingBox(points: { x1: number; x2: number }[]): [number, number, number, number] {
  if (points.length === 0) return [-6, 6, -6, 6];
  let minX = -6, maxX = 6, minY = -6, maxY = 6;
  for (let i = 0; i < points.length; i++) {
    if (points[i].x1 - 1 < minX) minX = Math.floor(points[i].x1 - 1);
    if (points[i].x1 + 1 > maxX) maxX = Math.ceil(points[i].x1 + 1);
    if (points[i].x2 - 1 < minY) minY = Math.floor(points[i].x2 - 1);
    if (points[i].x2 + 1 > maxY) maxY = Math.ceil(points[i].x2 + 1);
  }
  return [minX, maxX, minY, maxY];
}

export type ImpurityCriterion = 'gini' | 'entropy' | 'mse';

export function computeImpurity(
  targets: number[],
  criterion: ImpurityCriterion
): { impurity: number; value: number; classCounts?: [number, number, number] } {
  const n = targets.length;
  if (n === 0) return { impurity: 0, value: 0 };

  if (criterion === 'mse') {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += targets[i];
    const mean = sum / n;
    let varSum = 0;
    for (let i = 0; i < n; i++) {
      const d = targets[i] - mean;
      varSum += d * d;
    }
    return { impurity: varSum / n, value: mean };
  }

  // Classification (classes +1, -1, 2)
  let countPos = 0;
  let countNeg = 0;
  let countThird = 0;

  for (let i = 0; i < n; i++) {
    const y = targets[i];
    if (y === 1) countPos++;
    else if (y === 2) countThird++;
    else countNeg++;
  }

  const pPos = countPos / n;
  const pNeg = countNeg / n;
  const pThird = countThird / n;

  // Majority class
  let value = 1;
  let maxCount = countPos;
  if (countNeg > maxCount) {
    maxCount = countNeg;
    value = -1;
  }
  if (countThird > maxCount) {
    value = 2;
  }

  if (criterion === 'gini') {
    const gini = 1 - (pPos * pPos + pNeg * pNeg + pThird * pThird);
    return { impurity: Math.max(0, gini), value, classCounts: [countPos, countNeg, countThird] };
  } else {
    // Entropy (log base 2)
    let ent = 0;
    if (pPos > 0) ent -= pPos * Math.log2(pPos);
    if (pNeg > 0) ent -= pNeg * Math.log2(pNeg);
    if (pThird > 0) ent -= pThird * Math.log2(pThird);
    return { impurity: Math.max(0, ent), value, classCounts: [countPos, countNeg, countThird] };
  }
}

export interface SplitCandidate {
  featureIndex: 0 | 1;
  threshold: number;
  gain: number;
  leftIndices: number[];
  rightIndices: number[];
  leftImpurity: number;
  rightImpurity: number;
  leftValue: number;
  rightValue: number;
  leftCounts?: [number, number, number];
  rightCounts?: [number, number, number];
}

export function findBestSplitForPoints(
  points: { x1: number; x2: number; y: number }[],
  indices: number[],
  criterion: ImpurityCriterion,
  minSamplesSplit: number = 2,
  featureSubset: (0 | 1)[] = [0, 1]
): SplitCandidate | null {
  const n = indices.length;
  if (n < minSamplesSplit) return null;

  const currentTargets = indices.map((idx) => points[idx].y);
  const parent = computeImpurity(currentTargets, criterion);
  if (parent.impurity <= 1e-7) return null; // already pure

  let bestGain = -Infinity;
  let bestSplit: SplitCandidate | null = null;

  for (const featureIndex of featureSubset) {
    // Extract unique feature values sorted
    const pairs: { val: number; idx: number }[] = [];
    for (let i = 0; i < n; i++) {
      const idx = indices[i];
      const val = featureIndex === 0 ? points[idx].x1 : points[idx].x2;
      pairs.push({ val, idx });
    }
    pairs.sort((a, b) => a.val - b.val);

    // Test midpoints
    for (let i = 0; i < n - 1; i++) {
      if (pairs[i].val === pairs[i + 1].val) continue;

      const threshold = (pairs[i].val + pairs[i + 1].val) / 2;
      const leftIdxs: number[] = [];
      const rightIdxs: number[] = [];

      for (let j = 0; j < n; j++) {
        if (pairs[j].val <= threshold) {
          leftIdxs.push(pairs[j].idx);
        } else {
          rightIdxs.push(pairs[j].idx);
        }
      }

      if (leftIdxs.length === 0 || rightIdxs.length === 0) continue;

      const leftTargets = leftIdxs.map((idx) => points[idx].y);
      const rightTargets = rightIdxs.map((idx) => points[idx].y);

      const leftImp = computeImpurity(leftTargets, criterion);
      const rightImp = computeImpurity(rightTargets, criterion);

      const nL = leftIdxs.length;
      const nR = rightIdxs.length;
      const gain = parent.impurity - (nL / n) * leftImp.impurity - (nR / n) * rightImp.impurity;

      if (gain > bestGain && gain > 1e-7) {
        bestGain = gain;
        bestSplit = {
          featureIndex,
          threshold,
          gain,
          leftIndices: leftIdxs,
          rightIndices: rightIdxs,
          leftImpurity: leftImp.impurity,
          rightImpurity: rightImp.impurity,
          leftValue: leftImp.value,
          rightValue: rightImp.value,
          leftCounts: leftImp.classCounts,
          rightCounts: rightImp.classCounts,
        };
      }
    }
  }

  return bestSplit;
}

export function splitNode(
  node: TreeNode,
  split: SplitCandidate,
  nextId: () => number
): { left: TreeNode; right: TreeNode } {
  node.isLeaf = false;
  node.featureIndex = split.featureIndex;
  node.threshold = split.threshold;
  node.gain = split.gain;

  const [xMin, xMax, yMin, yMax] = node.region;
  let leftRegion: [number, number, number, number];
  let rightRegion: [number, number, number, number];

  if (split.featureIndex === 0) {
    leftRegion = [xMin, split.threshold, yMin, yMax];
    rightRegion = [split.threshold, xMax, yMin, yMax];
  } else {
    leftRegion = [xMin, xMax, yMin, split.threshold];
    rightRegion = [xMin, xMax, split.threshold, yMax];
  }

  const leftNode: TreeNode = {
    id: nextId(),
    depth: node.depth + 1,
    isLeaf: true,
    samples: split.leftIndices.length,
    impurity: split.leftImpurity,
    value: split.leftValue,
    classCounts: split.leftCounts,
    region: leftRegion,
  };

  const rightNode: TreeNode = {
    id: nextId(),
    depth: node.depth + 1,
    isLeaf: true,
    samples: split.rightIndices.length,
    impurity: split.rightImpurity,
    value: split.rightValue,
    classCounts: split.rightCounts,
    region: rightRegion,
  };

  node.left = leftNode;
  node.right = rightNode;

  return { left: leftNode, right: rightNode };
}

export function predictTree(node: TreeNode, x1: number, x2: number): number {
  if (node.isLeaf || node.featureIndex === undefined || node.threshold === undefined) {
    return node.value;
  }
  const val = node.featureIndex === 0 ? x1 : x2;
  if (val <= node.threshold) {
    return node.left ? predictTree(node.left, x1, x2) : node.value;
  } else {
    return node.right ? predictTree(node.right, x1, x2) : node.value;
  }
}

export function predictTreeSoft(node: TreeNode, x1: number, x2: number): number {
  if (node.isLeaf || node.featureIndex === undefined || node.threshold === undefined) {
    if (node.classCounts) {
      const pos = node.classCounts[0] || 0;
      const neg = node.classCounts[1] || 0;
      const tot = pos + neg;
      if (tot > 0) return (pos - neg) / tot;
    }
    return node.value;
  }
  const val = node.featureIndex === 0 ? x1 : x2;
  if (val <= node.threshold) {
    return node.left ? predictTreeSoft(node.left, x1, x2) : node.value;
  } else {
    return node.right ? predictTreeSoft(node.right, x1, x2) : node.value;
  }
}

export function predictTreePositiveProb(node: TreeNode, x1: number, x2: number): number {
  if (node.isLeaf || node.featureIndex === undefined || node.threshold === undefined) {
    if (node.classCounts) {
      const pos = node.classCounts[0] || 0;
      const neg = node.classCounts[1] || 0;
      const tot = pos + neg;
      if (tot > 0) return pos / tot;
    }
    return node.value >= 0 ? 1 : 0;
  }
  const val = node.featureIndex === 0 ? x1 : x2;
  if (val <= node.threshold) {
    return node.left ? predictTreePositiveProb(node.left, x1, x2) : (node.value >= 0 ? 1 : 0);
  } else {
    return node.right ? predictTreePositiveProb(node.right, x1, x2) : (node.value >= 0 ? 1 : 0);
  }
}

export function findActiveLeafPath(node: TreeNode, x1: number, x2: number, path: number[] = []): number[] {
  path.push(node.id);
  if (node.isLeaf || node.featureIndex === undefined || node.threshold === undefined) {
    return path;
  }
  const val = node.featureIndex === 0 ? x1 : x2;
  if (val <= node.threshold && node.left) {
    return findActiveLeafPath(node.left, x1, x2, path);
  } else if (node.right) {
    return findActiveLeafPath(node.right, x1, x2, path);
  }
  return path;
}

export function flattenTree(root: TreeNode): TreeNodeData[] {
  const result: TreeNodeData[] = [];

  function traverse(n: TreeNode) {
    result.push({
      id: n.id,
      depth: n.depth,
      isLeaf: n.isLeaf,
      featureIndex: n.featureIndex,
      threshold: n.threshold,
      leftId: n.left?.id,
      rightId: n.right?.id,
      samples: n.samples,
      impurity: n.impurity,
      value: n.value,
      classCounts: n.classCounts,
      region: n.region,
      gain: n.gain,
      similarity: n.similarity,
      weight: n.weight,
    });
    if (n.left) traverse(n.left);
    if (n.right) traverse(n.right);
  }

  traverse(root);
  return result;
}

export function reconstructTree(nodes: TreeNodeData[]): TreeNode {
  if (nodes.length === 0) {
    return {
      id: 0,
      depth: 0,
      isLeaf: true,
      samples: 0,
      impurity: 0,
      value: 0,
      region: [-6, 6, -6, 6],
    };
  }

  const map = new Map<number, TreeNode>();
  for (const nd of nodes) {
    map.set(nd.id, {
      id: nd.id,
      depth: nd.depth,
      isLeaf: nd.isLeaf,
      featureIndex: nd.featureIndex,
      threshold: nd.threshold,
      samples: nd.samples,
      impurity: nd.impurity,
      value: nd.value,
      classCounts: nd.classCounts ? [nd.classCounts[0] || 0, nd.classCounts[1] || 0, nd.classCounts[2] || 0] : undefined,
      region: nd.region,
      gain: nd.gain,
      similarity: nd.similarity,
      weight: nd.weight,
    });
  }

  for (const nd of nodes) {
    const parent = map.get(nd.id);
    if (!parent) continue;
    if (nd.leftId !== undefined && map.has(nd.leftId)) {
      parent.left = map.get(nd.leftId);
    }
    if (nd.rightId !== undefined && map.has(nd.rightId)) {
      parent.right = map.get(nd.rightId);
    }
  }

  return map.get(nodes[0].id) || nodes[0] as unknown as TreeNode;
}

export function extractSplits(root: TreeNode): {
  featureIndex: 0 | 1;
  threshold: number;
  range: [number, number];
  depth: number;
  label?: string;
  isPlaid?: boolean;
}[] {
  const splits: {
    featureIndex: 0 | 1;
    threshold: number;
    range: [number, number];
    depth: number;
    label?: string;
    isPlaid?: boolean;
  }[] = [];

  function traverse(n: TreeNode) {
    if (!n.isLeaf && n.featureIndex !== undefined && n.threshold !== undefined) {
      const [xMin, xMax, yMin, yMax] = n.region;
      splits.push({
        featureIndex: n.featureIndex,
        threshold: n.threshold,
        range: n.featureIndex === 0 ? [yMin, yMax] : [xMin, xMax],
        depth: n.depth,
        label: `${n.featureIndex === 0 ? 'X₁' : 'X₂'} < ${n.threshold.toFixed(2)}`,
      });
      if (n.left) traverse(n.left);
      if (n.right) traverse(n.right);
    }
  }

  traverse(root);
  return splits;
}

export function extractLeaves(root: TreeNode): {
  id: number;
  bounds: [number, number, number, number];
  value: number;
  samples: number;
  classCounts?: number[];
}[] {
  const leaves: {
    id: number;
    bounds: [number, number, number, number];
    value: number;
    samples: number;
    classCounts?: number[];
  }[] = [];

  function traverse(n: TreeNode) {
    if (n.isLeaf) {
      leaves.push({
        id: n.id,
        bounds: n.region,
        value: n.value,
        samples: n.samples,
        classCounts: n.classCounts,
      });
    } else {
      if (n.left) traverse(n.left);
      if (n.right) traverse(n.right);
    }
  }

  traverse(root);
  return leaves;
}

export function pruneCostComplexity(root: TreeNode, ccpAlpha: number): TreeNode {
  if (ccpAlpha <= 0 || root.isLeaf) return root;

  function pruneRecursive(node: TreeNode): void {
    if (node.isLeaf) return;
    if (node.left) pruneRecursive(node.left);
    if (node.right) pruneRecursive(node.right);

    // If both children are leaves, check if we should collapse
    if (node.left && node.left.isLeaf && node.right && node.right.isLeaf) {
      const parentCost = node.impurity;
      const childrenCost =
        (node.left.samples / node.samples) * node.left.impurity +
        (node.right.samples / node.samples) * node.right.impurity;
      const deltaCost = parentCost - childrenCost;

      // In CART pruning: collapse if gain is less than ccpAlpha
      if (deltaCost <= ccpAlpha) {
        node.isLeaf = true;
        node.left = undefined;
        node.right = undefined;
        node.featureIndex = undefined;
        node.threshold = undefined;
      }
    }
  }

  pruneRecursive(root);
  return root;
}

export function serializeTreeToFloat32(root: TreeNode): Float32Array {
  const flat = flattenTree(root);
  // Each node stores 12 numbers:
  // [id, depth, isLeaf(1/0), featureIndex(0/1/-1), threshold, leftId(-1), rightId(-1), samples, impurity, value, xMin, xMax, yMin, yMax] -> 14 numbers
  const stride = 14;
  const buf = new Float32Array(1 + flat.length * stride);
  buf[0] = flat.length;

  for (let i = 0; i < flat.length; i++) {
    const nd = flat[i];
    const off = 1 + i * stride;
    buf[off + 0] = nd.id;
    buf[off + 1] = nd.depth;
    buf[off + 2] = nd.isLeaf ? 1 : 0;
    buf[off + 3] = nd.featureIndex !== undefined ? nd.featureIndex : -1;
    buf[off + 4] = nd.threshold !== undefined ? nd.threshold : 0;
    buf[off + 5] = nd.leftId !== undefined ? nd.leftId : -1;
    buf[off + 6] = nd.rightId !== undefined ? nd.rightId : -1;
    buf[off + 7] = nd.samples;
    buf[off + 8] = nd.impurity;
    buf[off + 9] = nd.value;
    buf[off + 10] = nd.region[0];
    buf[off + 11] = nd.region[1];
    buf[off + 12] = nd.region[2];
    buf[off + 13] = nd.region[3];
  }
  return buf;
}

export function deserializeTreeFromFloat32(buf: Float32Array): TreeNode {
  if (buf.length < 15) {
    return {
      id: 0,
      depth: 0,
      isLeaf: true,
      samples: 0,
      impurity: 0,
      value: 0,
      region: [-6, 6, -6, 6],
    };
  }

  const count = Math.round(buf[0]);
  const stride = 14;
  const nodes: TreeNodeData[] = [];

  for (let i = 0; i < count; i++) {
    const off = 1 + i * stride;
    const fIdx = Math.round(buf[off + 3]);
    const lId = Math.round(buf[off + 5]);
    const rId = Math.round(buf[off + 6]);

    nodes.push({
      id: Math.round(buf[off + 0]),
      depth: Math.round(buf[off + 1]),
      isLeaf: buf[off + 2] === 1,
      featureIndex: fIdx === 0 || fIdx === 1 ? fIdx : undefined,
      threshold: buf[off + 4],
      leftId: lId >= 0 ? lId : undefined,
      rightId: rId >= 0 ? rId : undefined,
      samples: Math.round(buf[off + 7]),
      impurity: buf[off + 8],
      value: buf[off + 9],
      region: [buf[off + 10], buf[off + 11], buf[off + 12], buf[off + 13]],
    });
  }

  return reconstructTree(nodes);
}
