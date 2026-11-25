/**
 * Clustering & Dimensionality Reduction Utilities for Stage 03
 */

export interface EigenResult2D {
  values: [number, number]; // sorted descending lambda1 >= lambda2
  vectors: [[number, number], [number, number]]; // [v1, v2] column or row vectors
  angle: number; // orientation angle of principal eigenvector in radians
}

/**
 * Closed-form eigendecomposition of a 2x2 symmetric matrix:
 * [ a  b ]
 * [ b  d ]
 */
export function eigen2x2Symmetric(a: number, b: number, d: number): EigenResult2D {
  const trace = a + d;
  const det = a * d - b * b;
  const disc = Math.max(0, (trace * trace) / 4 - det);
  const sqrtDisc = Math.sqrt(disc);

  const lambda1 = trace / 2 + sqrtDisc;
  const lambda2 = trace / 2 - sqrtDisc;

  let v1: [number, number];
  if (Math.abs(b) > 1e-9) {
    const x = lambda1 - d;
    const y = b;
    const norm = Math.hypot(x, y) || 1;
    v1 = [x / norm, y / norm];
  } else if (a >= d) {
    v1 = [1, 0];
  } else {
    v1 = [0, 1];
  }

  // v2 is orthogonal to v1
  const v2: [number, number] = [-v1[1], v1[0]];
  const angle = Math.atan2(v1[1], v1[0]);

  return {
    values: [lambda1, lambda2],
    vectors: [v1, v2],
    angle,
  };
}

/**
 * Calculates Euclidean distance between two 2D points
 */
export function dist2D(a: [number, number], b: [number, number]): number {
  const dx = a[0] - b[0];
  const dy = a[1] - b[1];
  return Math.hypot(dx, dy);
}

/**
 * Calculates Manhattan distance between two 2D points
 */
export function distManhattan2D(a: [number, number], b: [number, number]): number {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

/**
 * Computes WCSS (Within-Cluster Sum of Squares) Inertia
 */
export function computeInertia(
  points: [number, number][],
  centroids: [number, number][],
  assignments: number[]
): number {
  let inertia = 0;
  for (let i = 0; i < points.length; i++) {
    const cIdx = assignments[i];
    if (cIdx >= 0 && cIdx < centroids.length) {
      const c = centroids[cIdx];
      const dx = points[i][0] - c[0];
      const dy = points[i][1] - c[1];
      inertia += dx * dx + dy * dy;
    }
  }
  return inertia;
}

/**
 * Computes Silhouette Score s in [-1, +1]
 */
export function computeSilhouetteScore(
  points: [number, number][],
  assignments: number[],
  k: number
): number {
  const n = points.length;
  if (n <= 1 || k <= 1) return 0;

  // Group point indices by cluster
  const clusters: number[][] = Array.from({ length: k }, () => []);
  for (let i = 0; i < n; i++) {
    const c = assignments[i];
    if (c >= 0 && c < k) {
      clusters[c].push(i);
    }
  }

  // Count active non-empty clusters
  const activeClusters = clusters.filter((cl) => cl.length > 0);
  if (activeClusters.length <= 1) return 0;

  let totalSilhouette = 0;
  let validPoints = 0;

  for (let i = 0; i < n; i++) {
    const ci = assignments[i];
    if (ci < 0 || ci >= k) continue; // Noise / unassigned
    const sameCluster = clusters[ci];

    if (sameCluster.length <= 1) {
      // Cluster with single point has silhouette 0
      validPoints++;
      continue;
    }

    // Mean intra-cluster distance a(i)
    let aSum = 0;
    for (let j = 0; j < sameCluster.length; j++) {
      const oIdx = sameCluster[j];
      if (oIdx !== i) {
        aSum += dist2D(points[i], points[oIdx]);
      }
    }
    const a = aSum / (sameCluster.length - 1);

    // Smallest mean inter-cluster distance b(i)
    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ci) continue;
      const otherCluster = clusters[c];
      if (otherCluster.length === 0) continue;

      let bSum = 0;
      for (let j = 0; j < otherCluster.length; j++) {
        bSum += dist2D(points[i], points[otherCluster[j]]);
      }
      const bAvg = bSum / otherCluster.length;
      if (bAvg < b) {
        b = bAvg;
      }
    }

    if (b === Infinity) continue;

    const maxAB = Math.max(a, b);
    const s = maxAB > 1e-9 ? (b - a) / maxAB : 0;
    totalSilhouette += s;
    validPoints++;
  }

  return validPoints > 0 ? totalSilhouette / validPoints : 0;
}

/**
 * Computes exact straight-line Voronoi cell boundaries equidistant between pairs of centroids
 * clipped to bounds [minX, maxX, minY, maxY]
 */
export function computeVoronoiSegments(
  centroids: [number, number][],
  bounds: [number, number, number, number] = [-6, 6, -6, 6]
): { start: [number, number]; end: [number, number] }[] {
  const K = centroids.length;
  if (K < 2) return [];

  const [minX, maxX, minY, maxY] = bounds;
  const segments: { start: [number, number]; end: [number, number] }[] = [];

  // Pairwise bisectors
  for (let i = 0; i < K; i++) {
    for (let j = i + 1; j < K; j++) {
      const c1 = centroids[i];
      const c2 = centroids[j];

      // Midpoint
      const mx = (c1[0] + c2[0]) / 2;
      const my = (c1[1] + c2[1]) / 2;

      // Normal vector from c1 to c2
      const nx = c2[0] - c1[0];
      const ny = c2[1] - c1[1];
      const len = Math.hypot(nx, ny);
      if (len < 1e-6) continue;

      // Direction perpendicular to (nx, ny): (-ny, nx)
      const dx = -ny / len;
      const dy = nx / len;

      // Find intersections of ray (mx + t*dx, my + t*dy) with bounding box
      const tVals: number[] = [];

      // Intersect with x = minX and x = maxX
      if (Math.abs(dx) > 1e-6) {
        const tLeft = (minX - mx) / dx;
        const yLeft = my + tLeft * dy;
        if (yLeft >= minY - 0.1 && yLeft <= maxY + 0.1) tVals.push(tLeft);

        const tRight = (maxX - mx) / dx;
        const yRight = my + tRight * dy;
        if (yRight >= minY - 0.1 && yRight <= maxY + 0.1) tVals.push(tRight);
      }

      // Intersect with y = minY and y = maxY
      if (Math.abs(dy) > 1e-6) {
        const tBottom = (minY - my) / dy;
        const xBottom = mx + tBottom * dx;
        if (xBottom >= minX - 0.1 && xBottom <= maxX + 0.1) tVals.push(tBottom);

        const tTop = (maxY - my) / dy;
        const xTop = mx + tTop * dx;
        if (xTop >= minX - 0.1 && xTop <= maxX + 0.1) tVals.push(tTop);
      }

      if (tVals.length >= 2) {
        tVals.sort((a, b) => a - b);
        const tStart = tVals[0];
        const tEnd = tVals[tVals.length - 1];

        // Sample midpoint to ensure it's strictly closer to c1 and c2 than other centroids
        const pMid: [number, number] = [
          mx + ((tStart + tEnd) / 2) * dx,
          my + ((tStart + tEnd) / 2) * dy,
        ];
        const d12 = dist2D(pMid, c1);
        let valid = true;

        for (let k = 0; k < K; k++) {
          if (k !== i && k !== j) {
            if (dist2D(pMid, centroids[k]) < d12 - 0.2) {
              valid = false;
              break;
            }
          }
        }

        if (valid) {
          segments.push({
            start: [mx + tStart * dx, my + tStart * dy],
            end: [mx + tEnd * dx, my + tEnd * dy],
          });
        }
      }
    }
  }

  return segments;
}

/**
 * Standard palette for clusters
 */
export const CLUSTER_PALETTE = [
  '#2563EB', // Blue
  '#DC2626', // Red
  '#059669', // Emerald
  '#D97706', // Amber
  '#7C3AED', // Purple
  '#DB2777', // Pink
  '#0891B2', // Cyan
  '#4B5563', // Slate
];

