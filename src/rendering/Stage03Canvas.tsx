import React, { useRef, useEffect, useState, useCallback } from 'react';
import { DataPoint2D, PointerToolMode } from '../datasets/tabular2d';
import { AlgorithmPlugin, VisualArtifact, TaskType } from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { DirtyManager } from './dirty-manager';
import { CLUSTER_PALETTE } from '../algorithms/stage03/clusteringUtils';
import { populateHeatmapImageData } from './color-maps';

export type UnsupervisedViewMode = 'neutral' | 'ground-truth' | 'clusters';

export interface Stage03CanvasProps {
  theme: ThemeMode;
  trainData: DataPoint2D[];
  testData: DataPoint2D[];
  artifacts: VisualArtifact[];
  gridValues?: Float32Array | null;
  gridResolution?: number;
  dirtyManager: DirtyManager;
  algorithm: AlgorithmPlugin;
  datasetTaskType?: TaskType;
  hyperparams?: Record<string, number | string>;
  pointerMode: PointerToolMode;
  activeClass: number;
  onAddPoint: (pt: { x1: number; x2: number; y: number }) => void;
  onUpdatePoint: (pt: DataPoint2D) => void;
  onDeletePoint: (id: number) => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
  predictValueAtPoint?: (x1: number, x2: number) => number;
  onUpdateHyperparam?: (key: string, value: number | string) => void;
  onSyncCentroids?: (centroids: [number, number][]) => void;
}

const VIEWPORT = { minX: -6.0, maxX: 6.0, minY: -6.0, maxY: 6.0 };

export const Stage03Canvas: React.FC<Stage03CanvasProps> = ({
  theme,
  trainData,
  testData: _testData,
  artifacts,
  gridValues,
  gridResolution: _gridResolution = 96,
  dirtyManager,
  algorithm,
  datasetTaskType,
  hyperparams = {},
  pointerMode,
  activeClass,
  onAddPoint,
  onUpdatePoint,
  onDeletePoint,
  onDragStart,
  onDragEnd,
  onUpdateHyperparam,
  onSyncCentroids,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);
  const insetCanvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const cachedImageDataRef = useRef<ImageData | null>(null);

  const [discretize, setDiscretize] = useState<boolean>(false);
  const [viewMode, setViewMode] = useState<UnsupervisedViewMode>('clusters');
  const [draggingCentroidIdx, setDraggingCentroidIdx] = useState<number | null>(null);
  const [draggingPointId, setDraggingPointId] = useState<number | null>(null);
  const [draggingCutHeight, setDraggingCutHeight] = useState<boolean>(false);
  const [elbowHoverIdx, setElbowHoverIdx] = useState<number | null>(null);
  const [renderTick, setRenderTick] = useState<number>(0);

  const isCarbon = theme === 'carbon';
  const isClustering = ['k-means', 'hierarchical-clustering', 'dbscan', 'gmm'].includes(algorithm.id);
  const isManifold = ['pca', 'lda', 'ica', 't-sne', 'umap'].includes(algorithm.id);
  const showInset = algorithm.id !== 'gmm' && algorithm.id !== 'dbscan';

  const isRegressionDataset = datasetTaskType !== undefined
    ? datasetTaskType === 'regression'
    : trainData.some((p) => p.y !== 1 && p.y !== -1 && p.y !== 2);

  const getGroundTruthColor = (y: number): string => {
    if (isRegressionDataset) {
      return y >= 0 ? (isCarbon ? '#C26752' : '#D94E34') : (isCarbon ? '#5276B8' : '#1E40AF');
    }
    if (y === 2) return isCarbon ? '#4D8F68' : '#059669';
    if (y < 0) return isCarbon ? '#5276B8' : '#DC2626';
    return isCarbon ? '#C26752' : '#2563EB';
  };

  // Extract Visual Artifacts
  const clusteringArtifact = artifacts.find((a) => a.type === 'clustering-overlay') as
    | Extract<VisualArtifact, { type: 'clustering-overlay' }>
    | undefined;

  const dendrogramArtifact = artifacts.find((a) => a.type === 'dendrogram-data') as
    | Extract<VisualArtifact, { type: 'dendrogram-data' }>
    | undefined;

  const projectionArtifact = artifacts.find((a) => a.type === 'projection-overlay') as
    | Extract<VisualArtifact, { type: 'projection-overlay' }>
    | undefined;

  const manifoldArtifact = artifacts.find((a) => a.type === 'manifold-overlay') as
    | Extract<VisualArtifact, { type: 'manifold-overlay' }>
    | undefined;

  // Local mutable centroids for smooth dragging before worker round-trip
  const [localCentroids, setLocalCentroids] = useState<[number, number][]>([]);

  useEffect(() => {
    if (clusteringArtifact?.centroids) {
      setLocalCentroids(clusteringArtifact.centroids.map((c) => [...c.current]));
    }
  }, [clusteringArtifact?.centroids]);

  // Coordinate conversion helpers
  const toPxX = (x: number, w: number) => ((x - VIEWPORT.minX) / (VIEWPORT.maxX - VIEWPORT.minX)) * w;
  const toPxY = (y: number, h: number) => ((VIEWPORT.maxY - y) / (VIEWPORT.maxY - VIEWPORT.minY)) * h;
  const toCoordX = (px: number, w: number) => VIEWPORT.minX + (px / w) * (VIEWPORT.maxX - VIEWPORT.minX);
  const toCoordY = (py: number, h: number) => VIEWPORT.maxY - (py / h) * (VIEWPORT.maxY - VIEWPORT.minY);

  // Resize handling
  const resizeCanvases = useCallback(() => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    if (mainCanvasRef.current) {
      const c = mainCanvasRef.current;
      const rect = c.getBoundingClientRect();
      const targetW = Math.max(1, Math.floor(rect.width * dpr));
      const targetH = Math.max(1, Math.floor(rect.height * dpr));
      if (c.width !== targetW || c.height !== targetH) {
        c.width = targetW;
        c.height = targetH;
      }
    }

    if (insetCanvasRef.current) {
      const ic = insetCanvasRef.current;
      const rect = ic.getBoundingClientRect();
      const targetW = Math.max(1, Math.floor(rect.width * dpr));
      const targetH = Math.max(1, Math.floor(rect.height * dpr));
      if (ic.width !== targetW || ic.height !== targetH) {
        ic.width = targetW;
        ic.height = targetH;
      }
    }

    dirtyManager.markDirty();
    setRenderTick((t) => t + 1);
  }, [dirtyManager]);

  useEffect(() => {
    resizeCanvases();
    const ro = new ResizeObserver(() => resizeCanvases());
    if (containerRef.current) ro.observe(containerRef.current);
    if (mainCanvasRef.current) ro.observe(mainCanvasRef.current);
    if (insetCanvasRef.current) ro.observe(insetCanvasRef.current);
    return () => ro.disconnect();
  }, [resizeCanvases, showInset]);

  // Update offscreen heatmap buffer
  useEffect(() => {
    const offCanvas = offscreenCanvasRef.current;
    if (!gridValues || gridValues.length === 0) {
      if (offCanvas.width > 0 && offCanvas.height > 0) {
        const offCtx = offCanvas.getContext('2d');
        if (offCtx) offCtx.clearRect(0, 0, offCanvas.width, offCanvas.height);
      }
      dirtyManager.markDirty();
      return;
    }

    const actualRes = Math.max(1, Math.round(Math.sqrt(gridValues.length)));

    if (offCanvas.width !== actualRes || offCanvas.height !== actualRes || !cachedImageDataRef.current) {
      offCanvas.width = actualRes;
      offCanvas.height = actualRes;
      const offCtx = offCanvas.getContext('2d');
      if (offCtx) {
        cachedImageDataRef.current = offCtx.createImageData(actualRes, actualRes);
      }
    }

    if (cachedImageDataRef.current) {
      populateHeatmapImageData(cachedImageDataRef.current, gridValues, theme, discretize);
      const offCtx = offCanvas.getContext('2d');
      if (offCtx) {
        offCtx.putImageData(cachedImageDataRef.current, 0, 0);
      }
    }
    dirtyManager.markDirty();
  }, [gridValues, theme, discretize, dirtyManager]);

  // =========================================================================
  // MAIN CANVAS RENDERER (65% width)
  // =========================================================================
  useEffect(() => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w <= 0 || h <= 0) return;
    const targetW = Math.floor(w * dpr);
    const targetH = Math.floor(h * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background
    ctx.fillStyle = isCarbon ? '#131312' : '#FAF9F5';
    ctx.fillRect(0, 0, w, h);

    // Draw Density / Soft Voronoi Heatmap Buffer
    if (gridValues && gridValues.length > 0 && offscreenCanvasRef.current.width > 0) {
      ctx.save();
      ctx.imageSmoothingEnabled = !discretize;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(offscreenCanvasRef.current, 0, 0, w, h);
      ctx.restore();
    }

    // Subtle grid lines at integer coordinates
    ctx.strokeStyle = isCarbon ? 'rgba(255, 255, 255, 0.04)' : 'rgba(0, 0, 0, 0.04)';
    ctx.lineWidth = 1;
    for (let x = -5; x <= 5; x += 1) {
      ctx.beginPath();
      ctx.moveTo(toPxX(x, w), 0);
      ctx.lineTo(toPxX(x, w), h);
      ctx.stroke();
    }
    for (let y = -5; y <= 5; y += 1) {
      ctx.beginPath();
      ctx.moveTo(0, toPxY(y, h));
      ctx.lineTo(w, toPxY(y, h));
      ctx.stroke();
    }

    // Origin axes
    ctx.strokeStyle = isCarbon ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.12)';
    ctx.beginPath();
    ctx.moveTo(toPxX(0, w), 0);
    ctx.lineTo(toPxX(0, w), h);
    ctx.moveTo(0, toPxY(0, h));
    ctx.lineTo(w, toPxY(0, h));
    ctx.stroke();

    // -----------------------------------------------------------------------
    // A. CLUSTERING OVERLAYS
    // -----------------------------------------------------------------------
    if (isClustering && clusteringArtifact) {
      // 1. Voronoi Segments
      if (clusteringArtifact.voronoiSegments) {
        ctx.save();
        ctx.strokeStyle = isCarbon ? 'rgba(156, 163, 175, 0.4)' : 'rgba(107, 114, 128, 0.4)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([4, 4]);
        for (const seg of clusteringArtifact.voronoiSegments) {
          ctx.beginPath();
          ctx.moveTo(toPxX(seg.start[0], w), toPxY(seg.start[1], h));
          ctx.lineTo(toPxX(seg.end[0], w), toPxY(seg.end[1], h));
          ctx.stroke();
        }
        ctx.restore();
      }

      // 2. Centroid Motion Trails & Spokes
      if (clusteringArtifact.centroids) {
        ctx.save();
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        for (const c of clusteringArtifact.centroids) {
          if (c.previous && (c.previous[0] !== c.current[0] || c.previous[1] !== c.current[1])) {
            ctx.strokeStyle = c.color || '#2563EB';
            ctx.beginPath();
            ctx.moveTo(toPxX(c.previous[0], w), toPxY(c.previous[1], h));
            ctx.lineTo(toPxX(c.current[0], w), toPxY(c.current[1], h));
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      // 3. DBSCAN neighborhood circles & radar sweep
      if (clusteringArtifact.dbscan) {
        const { epsilon, corePointIds, sweepIndex } = clusteringArtifact.dbscan;
        const epsRadiusPx = Math.abs(toPxX(epsilon, w) - toPxX(0, w));

        ctx.save();
        ctx.strokeStyle = isCarbon ? 'rgba(59, 130, 246, 0.15)' : 'rgba(37, 99, 235, 0.15)';
        ctx.fillStyle = isCarbon ? 'rgba(59, 130, 246, 0.03)' : 'rgba(37, 99, 235, 0.03)';
        ctx.lineWidth = 1;

        for (const cId of corePointIds) {
          const pt = trainData[cId];
          if (pt) {
            ctx.beginPath();
            ctx.arc(toPxX(pt.x1, w), toPxY(pt.x2, h), epsRadiusPx, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
          }
        }

        // Active sweep radar
        if (sweepIndex !== undefined && trainData[sweepIndex]) {
          const swPt = trainData[sweepIndex];
          ctx.strokeStyle = '#F59E0B'; // Amber
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(toPxX(swPt.x1, w), toPxY(swPt.x2, h), epsRadiusPx, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.restore();
      }

      // 4. GMM Covariance Ellipses
      if (clusteringArtifact.gmmEllipses) {
        for (const ell of clusteringArtifact.gmmEllipses) {
          const cx = toPxX(ell.mean[0], w);
          const cy = toPxY(ell.mean[1], h);
          const r1 = Math.sqrt(Math.max(1e-4, ell.eigenvalues[0]));
          const r2 = Math.sqrt(Math.max(1e-4, ell.eigenvalues[1]));
          const rxPx = Math.abs(toPxX(r1, w) - toPxX(0, w));
          const ryPx = Math.abs(toPxY(r2, h) - toPxY(0, h));
          const color = ell.color || '#2563EB';

          ctx.save();
          ctx.translate(cx, cy);
          ctx.rotate(-ell.angle); // Invert for canvas screen coordinates

          // 2-sigma outer ellipse (95% confidence)
          ctx.fillStyle = color + (isCarbon ? '12' : '10');
          ctx.strokeStyle = color + (isCarbon ? '40' : '30');
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.ellipse(0, 0, rxPx * 2, ryPx * 2, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();

          // 1-sigma inner ellipse (68% confidence)
          ctx.fillStyle = color + (isCarbon ? '25' : '18');
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.ellipse(0, 0, rxPx, ryPx, 0, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();

          ctx.restore();
        }
      }

      // 5. Hierarchical Linkage Bridges
      if (clusteringArtifact.hierarchicalMerges) {
        ctx.save();
        ctx.strokeStyle = isCarbon ? 'rgba(217, 119, 6, 0.4)' : 'rgba(180, 83, 9, 0.35)';
        ctx.lineWidth = 1.5;
        for (const m of clusteringArtifact.hierarchicalMerges) {
          ctx.beginPath();
          ctx.moveTo(toPxX(m.source[0], w), toPxY(m.source[1], h));
          ctx.lineTo(toPxX(m.target[0], w), toPxY(m.target[1], h));
          ctx.stroke();
        }
        ctx.restore();
      }
    }

    // -----------------------------------------------------------------------
    // B. DIMENSIONALITY REDUCTION & MANIFOLD OVERLAYS
    // -----------------------------------------------------------------------
    if (isManifold) {
      // 1. Orthogonal Projection Drop-Lines (PCA / LDA / ICA)
      if (projectionArtifact?.dropLines) {
        ctx.save();
        ctx.strokeStyle = isCarbon ? 'rgba(156, 163, 175, 0.25)' : 'rgba(107, 114, 128, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        for (const dl of projectionArtifact.dropLines) {
          ctx.beginPath();
          ctx.moveTo(toPxX(dl.from[0], w), toPxY(dl.from[1], h));
          ctx.lineTo(toPxX(dl.to[0], w), toPxY(dl.to[1], h));
          ctx.stroke();
        }
        ctx.restore();
      }

      // 2. LDA Scatter Ellipses & Centroid Connector
      if (projectionArtifact?.ldaScatter) {
        const { swEllipses, sbConnector } = projectionArtifact.ldaScatter;
        if (sbConnector) {
          ctx.save();
          ctx.strokeStyle = '#DC2626';
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 4]);
          ctx.beginPath();
          ctx.moveTo(toPxX(sbConnector.from[0], w), toPxY(sbConnector.from[1], h));
          ctx.lineTo(toPxX(sbConnector.to[0], w), toPxY(sbConnector.to[1], h));
          ctx.stroke();
          ctx.restore();
        }
        if (swEllipses) {
          for (const se of swEllipses) {
            const rxPx = Math.abs(toPxX(se.radiusX, w) - toPxX(0, w));
            const ryPx = Math.abs(toPxY(se.radiusY, h) - toPxY(0, h));
            ctx.save();
            ctx.translate(toPxX(se.center[0], w), toPxY(se.center[1], h));
            ctx.rotate(-se.angle);
            ctx.strokeStyle = se.classLabel === 1 ? '#2563EB' : '#DC2626';
            ctx.lineWidth = 1.2;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.ellipse(0, 0, rxPx, ryPx, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
          }
        }
      }

      // 3. Projection & Basis Axes
      if (projectionArtifact?.axes) {
        const center = projectionArtifact.center || [0, 0];
        const cx = toPxX(center[0], w);
        const cy = toPxY(center[1], h);

        for (const ax of projectionArtifact.axes) {
          const dxPx = (ax.direction[0] * ax.length * w) / (VIEWPORT.maxX - VIEWPORT.minX);
          const dyPx = -(ax.direction[1] * ax.length * h) / (VIEWPORT.maxY - VIEWPORT.minY);

          ctx.save();
          ctx.strokeStyle = ax.color || '#2563EB';
          ctx.lineWidth = ax.isGhost ? 1.5 : 2.5;
          if (ax.isGhost) ctx.setLineDash([4, 4]);

          // Draw full spanning axis
          ctx.beginPath();
          ctx.moveTo(cx - dxPx * 1.5, cy - dyPx * 1.5);
          ctx.lineTo(cx + dxPx * 1.5, cy + dyPx * 1.5);
          ctx.stroke();

          // Arrowhead in positive direction
          const headLen = 9;
          const endX = cx + dxPx;
          const endY = cy + dyPx;
          const angle = Math.atan2(dyPx, dxPx);

          ctx.fillStyle = ax.color || '#2563EB';
          ctx.beginPath();
          ctx.moveTo(endX, endY);
          ctx.lineTo(
            endX - headLen * Math.cos(angle - Math.PI / 6),
            endY - headLen * Math.sin(angle - Math.PI / 6)
          );
          ctx.lineTo(
            endX - headLen * Math.cos(angle + Math.PI / 6),
            endY - headLen * Math.sin(angle + Math.PI / 6)
          );
          ctx.closePath();
          ctx.fill();

          // Axis badge label (clamped inside canvas bounds so it never clips)
          ctx.font = '10px "JetBrains Mono", monospace';
          const labelW = ctx.measureText(ax.label).width;
          const rawLabelX = endX + 8;
          const labelX = Math.max(8, Math.min(w - labelW - 8, rawLabelX));
          const labelY = Math.max(14, Math.min(h - 8, endY + 4));
          ctx.fillText(ax.label, labelX, labelY);
          ctx.restore();
        }
      }

      // 4. UMAP Fuzzy Simplicial Graph Edges
      if (manifoldArtifact?.fuzzyGraphEdges && algorithm.id === 'umap') {
        ctx.save();
        for (const edge of manifoldArtifact.fuzzyGraphEdges) {
          const p1 = trainData[edge.sourceId];
          const p2 = trainData[edge.targetId];
          if (p1 && p2) {
            ctx.strokeStyle = isCarbon
              ? `rgba(99, 102, 241, ${Math.min(0.6, edge.weight * 0.7)})`
              : `rgba(79, 70, 229, ${Math.min(0.5, edge.weight * 0.6)})`;
            ctx.lineWidth = Math.max(0.5, edge.weight * 1.8);
            ctx.beginPath();
            ctx.moveTo(toPxX(p1.x1, w), toPxY(p1.x2, h));
            ctx.lineTo(toPxX(p2.x1, w), toPxY(p2.x2, h));
            ctx.stroke();
          }
        }
        ctx.restore();
      }
    }

    // -----------------------------------------------------------------------
    // C. 2D DATA POINT CLOUD
    // -----------------------------------------------------------------------
    for (let i = 0; i < trainData.length; i++) {
      const pt = trainData[i];
      const px = toPxX(pt.x1, w);
      const py = toPxY(pt.x2, h);

      // Determine point color based on viewMode
      let fillColor = '#94A3B8'; // Neutral Slate

      if (viewMode === 'ground-truth') {
        fillColor = getGroundTruthColor(pt.y);
      } else if (viewMode === 'clusters') {
        if (clusteringArtifact?.assignments) {
          const cId = clusteringArtifact.assignments[i];
          if (cId === -1) {
            // Noise point in DBSCAN
            fillColor = '#64748B';
          } else if (clusteringArtifact.softResponsibilities && clusteringArtifact.softResponsibilities[i]) {
            // GMM soft blend (primary component)
            fillColor = CLUSTER_PALETTE[cId % CLUSTER_PALETTE.length];
          } else {
            fillColor = CLUSTER_PALETTE[cId % CLUSTER_PALETTE.length];
          }
        } else {
          fillColor = getGroundTruthColor(pt.y);
        }
      }

      // Check if DBSCAN noise point
      const isDBSCANNoise = clusteringArtifact?.dbscan?.noisePointIds.includes(i);

      ctx.save();
      if (isDBSCANNoise) {
        // Render noise as cross marker (×)
        ctx.strokeStyle = isCarbon ? '#94A3B8' : '#64748B';
        ctx.lineWidth = 1.5;
        const crossSize = 3.5;
        ctx.beginPath();
        ctx.moveTo(px - crossSize, py - crossSize);
        ctx.lineTo(px + crossSize, py + crossSize);
        ctx.moveTo(px + crossSize, py - crossSize);
        ctx.lineTo(px - crossSize, py + crossSize);
        ctx.stroke();
      } else {
        // Normal point
        ctx.fillStyle = fillColor;
        ctx.strokeStyle = isCarbon ? '#131312' : '#FFFFFF';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(px, py, 4.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    }

    // -----------------------------------------------------------------------
    // D. DRAGGABLE CENTROIDS (K-Means & GMM)
    // -----------------------------------------------------------------------
    if (isClustering && (localCentroids.length > 0 || clusteringArtifact?.centroids)) {
      const centroidsToDraw = localCentroids.length > 0
        ? localCentroids
        : (clusteringArtifact?.centroids?.map((c) => c.current) ?? []);

      centroidsToDraw.forEach((c, idx) => {
        const cx = toPxX(c[0], w);
        const cy = toPxY(c[1], h);
        const color = CLUSTER_PALETTE[idx % CLUSTER_PALETTE.length];
        const isDragging = draggingCentroidIdx === idx;

        ctx.save();
        // Pulsing drag halo if active
        if (isDragging) {
          ctx.fillStyle = color + '33';
          ctx.beginPath();
          ctx.arc(cx, cy, 18, 0, Math.PI * 2);
          ctx.fill();
        }

        // White border ring
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.arc(cx, cy, 8.5, 0, Math.PI * 2);
        ctx.fill();

        // Solid centroid color dot
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(cx, cy, 6.5, 0, Math.PI * 2);
        ctx.fill();

        // Center crosshair / plus
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - 3, cy);
        ctx.lineTo(cx + 3, cy);
        ctx.moveTo(cx, cy - 3);
        ctx.lineTo(cx, cy + 3);
        ctx.stroke();

        // Label μ_k
        ctx.font = 'bold 9px "JetBrains Mono", monospace';
        ctx.fillStyle = isCarbon ? '#E2E8F0' : '#1E293B';
        ctx.fillText(`μ${idx + 1}`, cx + 11, cy + 3);
        ctx.restore();
      });
    }
  }, [
    theme,
    trainData,
    artifacts,
    clusteringArtifact,
    projectionArtifact,
    manifoldArtifact,
    localCentroids,
    viewMode,
    isClustering,
    isManifold,
    isCarbon,
    draggingCentroidIdx,
    gridValues,
    discretize,
    isRegressionDataset,
    renderTick,
  ]);

  // =========================================================================
  // SECONDARY INSET RENDERER (35% width)
  // =========================================================================
  useEffect(() => {
    const canvas = insetCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = rect.width;
    const h = rect.height;
    if (w <= 0 || h <= 0) return;
    const targetW = Math.floor(w * dpr);
    const targetH = Math.floor(h * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = isCarbon ? '#0F0F0E' : '#F5F4EE';
    ctx.fillRect(0, 0, w, h);

    // 1. DENDROGRAM INSET (Hierarchical Clustering)
    if (algorithm.id === 'hierarchical-clustering' && dendrogramArtifact) {
      const { nodes, cutHeight, maxDistance } = dendrogramArtifact;
      const padL = 24;
      const padR = 24;
      const padT = 30;
      const padB = 25;
      const cw = w - padL - padR;
      const ch = h - padT - padB;

      const toDendroX = (relX: number) => padL + relX * cw;
      const toDendroY = (d: number) => padT + ch - (d / Math.max(0.1, maxDistance)) * ch;

      // Draw U-shaped merge branches
      ctx.save();
      ctx.strokeStyle = isCarbon ? '#4B5563' : '#9CA3AF';
      ctx.lineWidth = 1.2;

      for (const node of nodes) {
        if (node.left !== undefined && node.right !== undefined) {
          const leftNode = nodes[node.left];
          const rightNode = nodes[node.right];
          if (leftNode && rightNode) {
            const x1 = toDendroX(leftNode.x);
            const y1 = toDendroY(leftNode.distance);
            const x2 = toDendroX(rightNode.x);
            const y2 = toDendroY(rightNode.distance);
            const yTop = toDendroY(node.distance);

            // Left vertical riser
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x1, yTop);
            // Horizontal crossbar
            ctx.lineTo(x2, yTop);
            // Right vertical riser
            ctx.lineTo(x2, y2);
            ctx.stroke();
          }
        }
      }
      ctx.restore();

      // Draw Draggable Cut-Height Line
      const cutY = toDendroY(cutHeight);
      ctx.save();
      ctx.strokeStyle = '#DC2626'; // Crimson
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(padL - 10, cutY);
      ctx.lineTo(w - padR + 10, cutY);
      ctx.stroke();

      // Cut handle tag badge
      ctx.fillStyle = '#DC2626';
      ctx.font = 'bold 9px "JetBrains Mono", monospace';
      ctx.fillRect(w - padR - 85, cutY - 14, 85, 14);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(`Cut h=${cutHeight.toFixed(2)}`, w - padR - 80, cutY - 4);
      ctx.restore();
      return;
    }

    // 2. WCSS INERTIA ELBOW / SILHOUETTE PROFILE (K-Means)
    if (isClustering && algorithm.id === 'k-means') {
      const profiles = clusteringArtifact?.clusterProfiles ?? [];
      if (profiles.length === 0) {
        ctx.save();
        ctx.fillStyle = isCarbon ? '#94A3B8' : '#64748B';
        ctx.font = '10px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('Precomputing WCSS & Silhouette for K=1..8…', w / 2, h / 2);
        ctx.restore();
        return;
      }

      const padL = 42;
      const padR = 36;
      const padT = 32;
      const padB = 30;
      const cw = Math.max(20, w - padL - padR);
      const ch = Math.max(20, h - padT - padB);

      const maxInertia = Math.max(...profiles.map((p) => p.inertia ?? 0), 10);
      const minSil = Math.min(0, ...profiles.map((p) => p.silhouette ?? 0));
      const maxSil = 1.0;
      const silSpan = Math.max(0.1, maxSil - minSil);

      const toProfX = (idx: number) => padL + (idx / Math.max(1, profiles.length - 1)) * cw;
      const toProfY = (val: number) => padT + (1 - val / maxInertia) * ch;
      const toSilY = (sil: number) => padT + (1 - (sil - minSil) / silSpan) * ch;

      // Gridlines & Left/Right Y-Axis Ticks
      ctx.save();
      ctx.strokeStyle = isCarbon ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
      ctx.lineWidth = 1;
      [0, 0.5, 1].forEach((frac) => {
        const gy = padT + (1 - frac) * ch;
        ctx.beginPath();
        ctx.moveTo(padL, gy);
        ctx.lineTo(padL + cw, gy);
        ctx.stroke();

        // Left WCSS ticks
        const wcssTick = maxInertia * frac;
        ctx.fillStyle = '#2563EB';
        ctx.font = '8px "JetBrains Mono", monospace';
        ctx.textAlign = 'right';
        ctx.fillText(wcssTick >= 100 ? wcssTick.toFixed(0) : wcssTick.toFixed(1), padL - 5, gy + 3);

        // Right Silhouette ticks
        const silTick = minSil + silSpan * frac;
        ctx.fillStyle = '#D97706';
        ctx.textAlign = 'left';
        ctx.fillText(silTick.toFixed(1), padL + cw + 5, gy + 3);
      });

      // Top-right mini legend
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.textAlign = 'right';
      ctx.fillStyle = '#2563EB';
      ctx.fillText('— WCSS', padL + cw - 42, padT - 12);
      ctx.fillStyle = '#D97706';
      ctx.fillText('-- Sil', padL + cw, padT - 12);

      // Hover vertical guide
      if (elbowHoverIdx !== null && profiles[elbowHoverIdx]) {
        const hx = toProfX(elbowHoverIdx);
        ctx.strokeStyle = isCarbon ? 'rgba(226,232,240,0.25)' : 'rgba(30,41,59,0.2)';
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(hx, padT);
        ctx.lineTo(hx, padT + ch);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Inertia Curve (Blue solid)
      ctx.strokeStyle = '#2563EB';
      ctx.lineWidth = 2;
      ctx.beginPath();
      profiles.forEach((p, idx) => {
        const px = toProfX(idx);
        const py = toProfY(p.inertia ?? 0);
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();

      // Silhouette Curve (Amber dashed)
      ctx.strokeStyle = '#D97706';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      profiles.forEach((p, idx) => {
        const px = toProfX(idx);
        const sy = toSilY(p.silhouette ?? 0);
        if (idx === 0) ctx.moveTo(px, sy);
        else ctx.lineTo(px, sy);
      });
      ctx.stroke();
      ctx.setLineDash([]);

      // Data Points & K-labels
      profiles.forEach((p, idx) => {
        const px = toProfX(idx);
        const py = toProfY(p.inertia ?? 0);
        const sy = toSilY(p.silhouette ?? 0);
        const isCurrentK = p.k === Number(hyperparams.k ?? 3);
        const isHovered = elbowHoverIdx === idx;

        // WCSS dot
        ctx.fillStyle = isCurrentK ? '#DC2626' : '#2563EB';
        ctx.beginPath();
        ctx.arc(px, py, isHovered ? 5.5 : isCurrentK ? 5 : 3, 0, Math.PI * 2);
        ctx.fill();

        // Silhouette dot
        ctx.fillStyle = '#D97706';
        ctx.beginPath();
        ctx.arc(px, sy, isHovered ? 4.5 : 2.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = isHovered
          ? (isCarbon ? '#F8FAFC' : '#0F172A')
          : (isCarbon ? '#94A3B8' : '#64748B');
        ctx.font = isHovered ? 'bold 9px "JetBrains Mono", monospace' : '9px "JetBrains Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`K=${p.k}`, px, h - padB + 14);
      });

      // Interactive Hover Tooltip Box
      if (elbowHoverIdx !== null && profiles[elbowHoverIdx]) {
        const hp = profiles[elbowHoverIdx];
        const hx = toProfX(elbowHoverIdx);
        const lines = [
          `K = ${hp.k}`,
          `WCSS: ${(hp.inertia ?? 0).toFixed(1)}`,
          `Sil:  ${(hp.silhouette ?? 0).toFixed(3)}`,
        ];
        const boxW = 96;
        const boxH = 44;
        const boxX = hx + boxW + 10 > w - 6 ? hx - boxW - 8 : hx + 8;
        const boxY = padT + 4;

        ctx.fillStyle = isCarbon ? 'rgba(24, 24, 23, 0.94)' : 'rgba(255, 255, 255, 0.95)';
        ctx.strokeStyle = isCarbon ? '#3F3F3B' : '#CBD5E1';
        ctx.lineWidth = 1;
        ctx.fillRect(boxX, boxY, boxW, boxH);
        ctx.strokeRect(boxX, boxY, boxW, boxH);

        ctx.textAlign = 'left';
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillStyle = isCarbon ? '#F1F5F9' : '#0F172A';
        ctx.fillText(lines[0], boxX + 7, boxY + 13);
        ctx.fillStyle = '#2563EB';
        ctx.fillText(lines[1], boxX + 7, boxY + 25);
        ctx.fillStyle = '#D97706';
        ctx.fillText(lines[2], boxX + 7, boxY + 37);
      }
      ctx.restore();
      return;
    }

    // 3. SCREE PLOT & 1D EMBEDDING STRIP (PCA / LDA / ICA)
    if (isManifold && projectionArtifact?.projected1D) {
      const proj = projectionArtifact.projected1D;
      const padL = 24;
      const padR = 24;
      const stripY = h * 0.65;
      const cw = w - padL - padR;

      // A. Scree Plot Bars (Top half)
      if (projectionArtifact.screeData) {
        const scree = projectionArtifact.screeData;
        const barW = 36;
        scree.forEach((sc, idx) => {
          const bx = padL + 30 + idx * 60;
          const barH = sc.varianceRatio * (h * 0.45);
          const by = h * 0.5 - barH;

          ctx.fillStyle = idx === 0 ? '#2563EB' : '#7C3AED';
          ctx.fillRect(bx, by, barW, barH);

          ctx.fillStyle = isCarbon ? '#E2E8F0' : '#1E293B';
          ctx.font = '9px "JetBrains Mono", monospace';
          ctx.textAlign = 'center';
          ctx.fillText(`PC${sc.component}`, bx + barW / 2, h * 0.5 + 12);
          ctx.fillText(`${(sc.varianceRatio * 100).toFixed(0)}%`, bx + barW / 2, by - 4);
        });
      }

      // B. 1D Projected Point Rug Plot (Bottom half)
      if (proj.length > 0) {
        const minZ = Math.min(...proj.map((p) => p.value));
        const maxZ = Math.max(...proj.map((p) => p.value));
        const span = Math.max(1e-4, maxZ - minZ);

        // Axis line
        ctx.strokeStyle = isCarbon ? '#4B5563' : '#9CA3AF';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(padL, stripY);
        ctx.lineTo(w - padR, stripY);
        ctx.stroke();

        // Points along 1D axis
        for (const p of proj) {
          const px = padL + ((p.value - minZ) / span) * cw;
          ctx.fillStyle = getGroundTruthColor(p.label);
          ctx.beginPath();
          ctx.arc(px, stripY + (Math.random() - 0.5) * 8, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      return;
    }

    // 4. LIVE 2D EMBEDDING SCATTER CANVAS (t-SNE & UMAP)
    if (isManifold && manifoldArtifact?.embeddedPoints) {
      const pts = manifoldArtifact.embeddedPoints;
      if (pts.length > 0) {
        const minX = Math.min(...pts.map((p) => p.x));
        const maxX = Math.max(...pts.map((p) => p.x));
        const minY = Math.min(...pts.map((p) => p.y));
        const maxY = Math.max(...pts.map((p) => p.y));
        const spanX = Math.max(1e-4, maxX - minX);
        const spanY = Math.max(1e-4, maxY - minY);

        const pad = 24;
        const cw = w - pad * 2;
        const ch = h - pad * 2;

        for (const p of pts) {
          const px = pad + ((p.x - minX) / spanX) * cw;
          const py = pad + ((maxY - p.y) / spanY) * ch;

          ctx.fillStyle = getGroundTruthColor(p.label);
          ctx.beginPath();
          ctx.arc(px, py, 3.5, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }, [
    theme,
    artifacts,
    dendrogramArtifact,
    clusteringArtifact,
    projectionArtifact,
    manifoldArtifact,
    algorithm.id,
    isClustering,
    isManifold,
    isCarbon,
    hyperparams.k,
    isRegressionDataset,
    elbowHoverIdx,
    renderTick,
  ]);

  // =========================================================================
  // MOUSE & DRAG INTERACTION (Centroids, Dendrogram Cut, Data Points)
  // =========================================================================
  const handleMainMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const coordX = toCoordX(mx, rect.width);
    const coordY = toCoordY(my, rect.height);

    // 1. Check if clicking on a Centroid (radius ~14px)
    if (isClustering && localCentroids.length > 0) {
      for (let i = 0; i < localCentroids.length; i++) {
        const cx = toPxX(localCentroids[i][0], rect.width);
        const cy = toPxY(localCentroids[i][1], rect.height);
        if (Math.hypot(mx - cx, my - cy) <= 14) {
          setDraggingCentroidIdx(i);
          if (onDragStart) onDragStart();
          return;
        }
      }
    }

    // 2. Check if clicking on a Data Point
    for (const pt of trainData) {
      const px = toPxX(pt.x1, rect.width);
      const py = toPxY(pt.x2, rect.height);
      if (Math.hypot(mx - px, my - py) <= 8) {
        if (pointerMode === 'delete') {
          onDeletePoint(pt.id);
          return;
        }
        if (pointerMode === 'flip') {
          onUpdatePoint({ ...pt, y: pt.y === 1 ? -1 : 1 });
          return;
        }
        if (pointerMode === 'add-drag') {
          setDraggingPointId(pt.id);
          if (onDragStart) onDragStart();
          return;
        }
      }
    }

    // 3. Otherwise, add point in add-drag mode
    if (pointerMode === 'add-drag') {
      onAddPoint({ x1: coordX, x2: coordY, y: activeClass });
    }
  };

  const handleMainMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const coordX = toCoordX(mx, rect.width);
    const coordY = toCoordY(my, rect.height);

    // Dragging Centroid
    if (draggingCentroidIdx !== null) {
      const updated = localCentroids.map((c, i) =>
        i === draggingCentroidIdx ? ([coordX, coordY] as [number, number]) : c
      );
      setLocalCentroids(updated);
      if (onSyncCentroids) {
        onSyncCentroids(updated);
      }
      dirtyManager.markDirty();
      return;
    }

    // Dragging Data Point
    if (draggingPointId !== null) {
      const pt = trainData.find((p) => p.id === draggingPointId);
      if (pt) {
        onUpdatePoint({ ...pt, x1: coordX, x2: coordY });
      }
      dirtyManager.markDirty();
    }
  };

  const handleMainMouseUp = () => {
    if (draggingCentroidIdx !== null) {
      setDraggingCentroidIdx(null);
      if (onDragEnd) onDragEnd();
    }
    if (draggingPointId !== null) {
      setDraggingPointId(null);
      if (onDragEnd) onDragEnd();
    }
  };

  // Inset Mouse Handlers (Dendrogram Cut-Height & k-Means Elbow Tooltip)
  const handleInsetMouseDown = () => {
    if (algorithm.id !== 'hierarchical-clustering' || !dendrogramArtifact) return;
    setDraggingCutHeight(true);
  };

  const handleInsetMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = insetCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (algorithm.id === 'k-means' && clusteringArtifact?.clusterProfiles?.length) {
      const profiles = clusteringArtifact.clusterProfiles;
      const padL = 42;
      const padR = 36;
      const cw = Math.max(20, rect.width - padL - padR);
      if (mx >= padL - 12 && mx <= padL + cw + 12) {
        const frac = Math.max(0, Math.min(1, (mx - padL) / cw));
        const idx = Math.round(frac * Math.max(0, profiles.length - 1));
        setElbowHoverIdx(idx);
      } else {
        setElbowHoverIdx(null);
      }
      return;
    }

    if (!draggingCutHeight || !dendrogramArtifact) return;
    const padT = 30;
    const padB = 25;
    const ch = rect.height - padT - padB;
    const relH = 1.0 - (my - padT) / ch;
    const newCut = Math.max(0.1, relH * dendrogramArtifact.maxDistance);

    if (onUpdateHyperparam) {
      onUpdateHyperparam('cutHeight', Number(newCut.toFixed(2)));
    }
    dirtyManager.markDirty();
  };

  const handleInsetMouseUp = () => {
    setDraggingCutHeight(false);
    setElbowHoverIdx(null);
  };

  const footerLegendInfo = (() => {
    switch (algorithm.id) {
      case 'dbscan':
        return {
          title: 'Density clusters: Core (solid), Border (ring), Noise (dim):',
          tooltip: 'Density clusters: Core (solid), Border (ring), Noise (dim)',
          field: 'Density Field (ε-reachability)',
          mode: 'dbscan' as const,
        };
      case 'gmm':
        return {
          title: 'Colors show GMM component responsibilities:',
          tooltip: 'Colors show soft Gaussian component mixture responsibilities',
          field: 'Soft GMM Responsibility Field',
          mode: 'cluster-gradient' as const,
        };
      case 'hierarchical-clustering':
        return {
          title: 'Colors show agglomerative cluster assignments:',
          tooltip: 'Colors show hierarchical cluster assignments at current cut height',
          field: 'Hierarchical Linkage Partition',
          mode: 'cluster-gradient' as const,
        };
      case 'pca':
        return {
          title: 'Colors show class/target labels; dashed lines project onto PC1:',
          tooltip: 'Colors show class/target labels with orthogonal projection drop-lines onto PC1',
          field: 'Principal Component Subspace',
          mode: 'class-labels' as const,
        };
      case 'lda':
        return {
          title: 'Colors show class labels; ellipses show within-class scatter (S_W):',
          tooltip: 'Colors show class labels with Fisher discriminant axis w and S_W ellipses',
          field: 'Fisher Discriminant Projection',
          mode: 'class-labels' as const,
        };
      case 'ica':
        return {
          title: 'Colors show class labels; axes show independent components:',
          tooltip: 'Colors show class labels with non-orthogonal FastICA independent component axes',
          field: 'Independent Component Basis',
          mode: 'class-labels' as const,
        };
      case 't-sne':
        return {
          title: 'Colors show class labels; right panel shows 2D Student-t embedding:',
          tooltip: 'Colors show class/target labels; inset shows live 2D t-SNE embedding',
          field: 'KL-Divergence Manifold Embedding',
          mode: 'class-labels' as const,
        };
      case 'umap':
        return {
          title: 'Colors show class labels; edges show fuzzy simplicial graph:',
          tooltip: 'Colors show class/target labels with fuzzy simplicial k-NN graph edges',
          field: 'Fuzzy Simplicial Manifold Graph',
          mode: 'class-labels' as const,
        };
      default:
        return {
          title: 'Colors show cluster assignments & Voronoi boundaries:',
          tooltip: 'Colors show cluster assignments & Voronoi partition boundaries',
          field: 'Voronoi Partition Field',
          mode: 'cluster-gradient' as const,
        };
    }
  })();

  return (
    <div ref={containerRef} className="relative flex w-full h-full min-h-0 min-w-0 bg-[var(--bg-canvas)]">
      {/* 1. Main 2D Canvas (65% width when inset is shown, 100% otherwise) */}
      <div
        className={`relative flex flex-col h-full min-w-0 ${showInset ? 'border-r border-[var(--border-hairline)]' : ''}`}
        style={{ flex: showInset ? '65 1 0%' : '1 1 0%' }}
      >
        {/* Canvas HUD Bar */}
        <div className="absolute top-3 left-4 z-10 flex flex-wrap items-center gap-1.5 bg-[var(--bg-elevated)]/90 backdrop-blur-sm px-2.5 py-1 border border-[var(--border-hairline)] rounded-sm font-mono text-[10px] max-w-[calc(100%-2rem)]">
          <span className="text-[var(--text-muted)] uppercase tracking-wider shrink-0">View:</span>
          <button
            onClick={() => setViewMode('neutral')}
            className={`px-1.5 py-0.5 rounded shrink-0 ${viewMode === 'neutral' ? 'bg-[var(--bg-subtle)] text-[var(--text-primary)] font-semibold' : 'text-[var(--text-secondary)]'}`}
          >
            Neutral Unlabeled
          </button>
          <button
            onClick={() => setViewMode('ground-truth')}
            className={`px-1.5 py-0.5 rounded shrink-0 ${viewMode === 'ground-truth' ? 'bg-[var(--bg-subtle)] text-[var(--text-primary)] font-semibold' : 'text-[var(--text-secondary)]'}`}
          >
            Ground Truth
          </button>
          <button
            onClick={() => setViewMode('clusters')}
            className={`px-1.5 py-0.5 rounded shrink-0 ${viewMode === 'clusters' ? 'bg-[var(--bg-subtle)] text-[var(--text-primary)] font-semibold' : 'text-[var(--text-secondary)]'}`}
          >
            Discovered Clusters
          </button>
        </div>

        <div className="relative flex-1 min-h-0 min-w-0 overflow-hidden">
          <canvas
            ref={mainCanvasRef}
            className="canvas-main block w-full h-full cursor-crosshair"
            onMouseDown={handleMainMouseDown}
            onMouseMove={handleMainMouseMove}
            onMouseUp={handleMainMouseUp}
            onMouseLeave={handleMainMouseUp}
          />
        </div>

        {/* Bottom Bar: Swiss Color Legend & Discretize Toggle */}
        <div className="flex flex-col px-3 py-2 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] gap-2 select-none shrink-0">
          <div className="flex flex-wrap items-center justify-between text-[var(--text-muted)] gap-1.5">
            <span className="text-[9px] truncate mr-2" title={footerLegendInfo.tooltip}>
              {footerLegendInfo.title}
            </span>
            {footerLegendInfo.mode === 'dbscan' ? (
              <div className="flex items-center gap-2 font-mono text-[9px] shrink-0">
                <span className="text-[var(--text-muted)]">Noise</span>
                <div className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-[#3B82F6]" />
                  <span className="w-2 h-2 rounded-full bg-[#10B981]" />
                  <span className="w-2 h-2 rounded-full bg-[#F59E0B]" />
                  <span className="w-2 h-2 rounded-full bg-[#EC4899]" />
                </div>
                <span className="text-[var(--text-primary)]">Clusters</span>
              </div>
            ) : footerLegendInfo.mode === 'class-labels' ? (
              <div className="flex items-center gap-2 font-mono font-bold text-[9px] shrink-0">
                <span className="flex items-center gap-1">
                  <span className={`w-2 h-2 rounded-full ${isCarbon ? 'bg-[#5276B8]' : 'bg-[#DC2626]'}`} />
                  <span className={isCarbon ? 'text-[#5276B8]' : 'text-[#DC2626]'}>y&lt;0</span>
                </span>
                <span className="flex items-center gap-1">
                  <span className={`w-2 h-2 rounded-full ${isCarbon ? 'bg-[#C26752]' : 'bg-[#2563EB]'}`} />
                  <span className={isCarbon ? 'text-[#C26752]' : 'text-[#2563EB]'}>y≥0</span>
                </span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5 font-mono font-bold text-[9px] shrink-0">
                <span className={isCarbon ? 'text-[#4E89B8]' : 'text-[#1E40AF]'}>C0</span>
                <div
                  className="w-20 h-2 rounded-xs border border-[var(--border-hairline)]"
                  style={{
                    background: isCarbon
                      ? 'linear-gradient(to right, rgb(78, 137, 184), rgb(24, 24, 23), rgb(229, 131, 68))'
                      : 'linear-gradient(to right, rgb(43, 91, 132), rgb(250, 249, 245), rgb(200, 90, 23))',
                  }}
                />
                <span className={isCarbon ? 'text-[#E58344]' : 'text-[#D94E34]'}>C1+</span>
              </div>
            )}
          </div>
          <div className="flex flex-wrap items-center justify-between pt-0.5 border-t border-[var(--border-hairline)]/50 gap-2">
            <span className="text-[9px] text-[var(--text-muted)] italic truncate">
              {footerLegendInfo.field}
            </span>
            <label className="flex items-center gap-1.5 cursor-pointer text-[var(--text-primary)] hover:text-[var(--text-primary)] shrink-0">
              <input
                type="checkbox"
                checked={discretize}
                onChange={() => {
                  setDiscretize((prev) => !prev);
                  dirtyManager.markDirty();
                }}
                className="rounded-xs border-[var(--border-hairline)] text-[var(--accent-amber)] focus:ring-0 cursor-pointer"
              />
              <span>Discretize output</span>
            </label>
          </div>
        </div>
      </div>

      {/* 2. Secondary Inset Canvas (35% width, hidden for GMM & DBSCAN) */}
      {showInset && (
        <div className="relative h-full flex flex-col bg-[var(--bg-elevated)] min-w-0" style={{ flex: '35 1 0%' }}>
          {/* Inset Header */}
          <div className="px-3 py-2 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] flex flex-wrap items-center justify-between font-mono text-[10px] gap-1 min-w-0">
            <span className="font-semibold text-[var(--text-primary)] uppercase tracking-wider min-w-0 truncate">
              {algorithm.id === 'hierarchical-clustering'
                ? 'Dendrogram Tree & Cut'
                : isClustering
                ? 'Inertia Elbow & Silhouette'
                : algorithm.id === 't-sne' || algorithm.id === 'umap'
                ? '2D Manifold Embedding'
                : '1D Projected Embedding & Scree'}
            </span>
            {algorithm.id === 'hierarchical-clustering' && (
              <span className="text-[var(--text-muted)] shrink-0 text-[9px]">Drag red line</span>
            )}
          </div>

          <div className="relative flex-1 min-h-0">
            <canvas
              ref={insetCanvasRef}
              className={`block w-full h-full ${algorithm.id === 'hierarchical-clustering' ? 'cursor-row-resize' : 'cursor-default'}`}
              onMouseDown={handleInsetMouseDown}
              onMouseMove={handleInsetMouseMove}
              onMouseUp={handleInsetMouseUp}
              onMouseLeave={handleInsetMouseUp}
            />
          </div>
        </div>
      )}
    </div>
  );
};
