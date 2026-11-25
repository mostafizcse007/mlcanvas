import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { DataPoint2D, PointerToolMode } from '../datasets/tabular2d';
import { AlgorithmPlugin, VisualArtifact, TreeNodeData, TaskType } from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { DirtyManager } from './dirty-manager';
import { populateHeatmapImageData } from './color-maps';
import { predictTree } from '../algorithms/stage02/treeUtils';
import { getVisibleDataPoints, hitTestDataPoints, renderDataPoints2D } from './pointUtils';

export interface Stage02CanvasProps {
  theme: ThemeMode;
  trainData: DataPoint2D[];
  testData: DataPoint2D[];
  artifacts: VisualArtifact[];
  gridValues?: Float32Array | null;
  gridResolution: number;
  dirtyManager: DirtyManager;
  algorithm: AlgorithmPlugin;
  datasetTaskType?: TaskType;
  hyperparams?: Record<string, number | string>;
  pointerMode: PointerToolMode;
  activeClass: number;
  showTestData?: boolean;
  onToggleShowTestData?: () => void;
  onAddPoint: (pt: { x1: number; x2: number; y: number }) => void;
  onUpdatePoint: (pt: DataPoint2D) => void;
  onDeletePoint: (id: number) => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
  predictValueAtPoint: (x1: number, x2: number) => number;
  onUpdateHyperparam?: (key: string, value: number | string) => void;
}

const VIEWPORT = { minX: -6.0, maxX: 6.0, minY: -6.0, maxY: 6.0 };

export const Stage02Canvas: React.FC<Stage02CanvasProps> = ({
  theme,
  trainData,
  testData,
  artifacts,
  gridValues,
  gridResolution: _gridResolution,
  dirtyManager,
  algorithm,
  datasetTaskType,
  hyperparams = {},
  pointerMode,
  activeClass,
  showTestData,
  onToggleShowTestData,
  onAddPoint,
  onUpdatePoint,
  onDeletePoint,
  onDragStart,
  onDragEnd,
  predictValueAtPoint,
  onUpdateHyperparam,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement>(document.createElement('canvas'));
  const cachedImageDataRef = useRef<ImageData | null>(null);

  // Bidirectional hover state
  const [hoveredNodeId, setHoveredNodeId] = useState<number | null>(null);
  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number } | null>(null);
  const [draggingPointId, setDraggingPointId] = useState<number | null>(null);

  // Missing options from TF Playground
  const [localShowTestData, setLocalShowTestData] = useState<boolean>(true);
  const effectiveShowTestData = showTestData ?? localShowTestData;
  const handleToggleShowTestData = useCallback(() => {
    if (onToggleShowTestData) {
      onToggleShowTestData();
    } else if (showTestData === undefined) {
      setLocalShowTestData((prev) => !prev);
    }
    dirtyManager.markDirty();
  }, [onToggleShowTestData, showTestData, dirtyManager]);

  useEffect(() => {
    dirtyManager.markDirty();
  }, [effectiveShowTestData, dirtyManager]);

  const [discretize, setDiscretize] = useState<boolean>(false);

  const isCarbon = theme === 'carbon';

  const isRegressionDataset = datasetTaskType !== undefined
    ? datasetTaskType === 'regression'
    : (algorithm?.taskType === 'regression');
  const isRegression = isRegressionDataset;
  const is1DRegression =
    isRegression &&
    trainData.length > 0 &&
    trainData.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);

  // Extract artifacts
  const treePartitionArtifact = artifacts.find(
    (a) => a.type === 'tree-partition'
  ) as Extract<VisualArtifact, { type: 'tree-partition' }> | undefined;

  const adaboostWeightsArtifact = artifacts.find(
    (a) => a.type === 'adaboost-weights'
  ) as Extract<VisualArtifact, { type: 'adaboost-weights' }> | undefined;

  const baggingSamplesArtifact = artifacts.find(
    (a) => a.type === 'bagging-samples'
  ) as Extract<VisualArtifact, { type: 'bagging-samples' }> | undefined;

  const residualBarsArtifact = artifacts.find(
    (a) => a.type === 'residual-bars'
  ) as Extract<VisualArtifact, { type: 'residual-bars' }> | undefined;

  const ensembleOverlayArtifact = artifacts.find(
    (a) => a.type === 'ensemble-overlay'
  ) as Extract<VisualArtifact, { type: 'ensemble-overlay' }> | undefined;

  const histogramGridArtifact = artifacts.find(
    (a) => a.type === 'histogram-grid'
  ) as Extract<VisualArtifact, { type: 'histogram-grid' }> | undefined;

  const metaFeatureArtifact = artifacts.find(
    (a) => a.type === 'meta-feature-projection'
  ) as Extract<VisualArtifact, { type: 'meta-feature-projection' }> | undefined;

  const votingBoundariesArtifact = artifacts.find(
    (a) => a.type === 'voting-boundaries'
  ) as Extract<VisualArtifact, { type: 'voting-boundaries' }> | undefined;

  // Selected tree index for ensemble inspector
  const currentSelectedTree = Number(hyperparams.selectedTree ?? -1);

  // Coordinates converters
  const toPxX = (x: number, w: number) => ((x - VIEWPORT.minX) / (VIEWPORT.maxX - VIEWPORT.minX)) * w;
  const toPxY = (y: number, h: number) => h - ((y - VIEWPORT.minY) / (VIEWPORT.maxY - VIEWPORT.minY)) * h;
  const toCoordX = (px: number, w: number) => VIEWPORT.minX + (px / w) * (VIEWPORT.maxX - VIEWPORT.minX);
  const toCoordY = (py: number, h: number) => VIEWPORT.maxY - (py / h) * (VIEWPORT.maxY - VIEWPORT.minY);

  // Resize canvas handling with devicePixelRatio cap of 2
  const resizeCanvas = useCallback(() => {
    const canvas = mainCanvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const rect = container.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.scale(dpr, dpr);
    }
    dirtyManager.markDirty();
  }, [dirtyManager]);

  useEffect(() => {
    resizeCanvas();
    const ro = new ResizeObserver(() => resizeCanvas());
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, [resizeCanvas]);

  // Update heatmap offscreen buffer
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

  // Find active root-to-leaf decision path when cursor hovers on 2D partition canvas
  const activeHoverPathNodeIds = useMemo(() => {
    if (!hoverCoord || !treePartitionArtifact || !treePartitionArtifact.allNodes) return [];
    const all = treePartitionArtifact.allNodes;
    if (all.length === 0) return [];

    const nodeMap = new Map<number, TreeNodeData>();
    for (const nd of all) nodeMap.set(nd.id, nd);

    const path: number[] = [];
    let current: TreeNodeData | undefined = treePartitionArtifact.root;

    while (current) {
      path.push(current.id);
      if (current.isLeaf || current.featureIndex === undefined || current.threshold === undefined) {
        break;
      }
      const val = current.featureIndex === 0 ? hoverCoord.x : hoverCoord.y;
      if (val <= current.threshold && current.leftId !== undefined) {
        current = nodeMap.get(current.leftId);
      } else if (current.rightId !== undefined) {
        current = nodeMap.get(current.rightId);
      } else {
        break;
      }
    }
    return path;
  }, [hoverCoord, treePartitionArtifact]);

  // Map of sample weights for AdaBoost point radius scaling
  const sampleWeightMap = useMemo(() => {
    const map = new Map<number, { scale: number; isMisclassified: boolean }>();
    if (!adaboostWeightsArtifact) return map;
    for (const w of adaboostWeightsArtifact.weights) {
      map.set(w.id, {
        scale: Math.max(0.6, Math.min(2.8, Math.sqrt(w.normalizedWeight))),
        isMisclassified: w.isMisclassified,
      });
    }
    return map;
  }, [adaboostWeightsArtifact]);

  // Map of bagging sample counts for badges
  const baggingSampleMap = useMemo(() => {
    const map = new Map<number, { count: number; isOOB: boolean }>();
    if (!baggingSamplesArtifact) return map;
    for (const s of baggingSamplesArtifact.samples) {
      map.set(s.id, { count: s.count, isOOB: s.isOOB });
    }
    return map;
  }, [baggingSamplesArtifact]);

  // Hovered node region for 2D highlight overlay
  const hoveredNodeRegion = useMemo(() => {
    if (hoveredNodeId === null || !treePartitionArtifact) return null;
    const node = treePartitionArtifact.allNodes?.find((n) => n.id === hoveredNodeId);
    return node ? node.region : null;
  }, [hoveredNodeId, treePartitionArtifact]);

  // Render Loop
  useEffect(() => {
    let animId: number;

    const render = () => {
      if (dirtyManager.checkAndClear()) {
        const canvas = mainCanvasRef.current;
        const ctx = canvas?.getContext('2d');
        if (canvas && ctx) {
          const w = parseFloat(canvas.style.width) || canvas.width;
          const h = parseFloat(canvas.style.height) || canvas.height;

        // 1. Clear background
        ctx.fillStyle = isCarbon ? '#131312' : '#FAF9F5';
        ctx.fillRect(0, 0, w, h);

        // 2. Decision boundary heatmap buffer (Only for classification and 2D surface regression)
        const shouldDrawHeatmap =
          !is1DRegression &&
          gridValues &&
          gridValues.length > 0 &&
          offscreenCanvasRef.current.width > 0;

        if (shouldDrawHeatmap) {
          ctx.save();
          ctx.imageSmoothingEnabled = !discretize;
          ctx.imageSmoothingQuality = 'high';
          ctx.drawImage(offscreenCanvasRef.current, 0, 0, w, h);
          ctx.restore();
        }

        // 3. Exposed Swiss Grid Hairlines
        ctx.save();
        ctx.lineWidth = 1;
        ctx.strokeStyle = isCarbon ? 'rgba(46, 46, 43, 0.7)' : 'rgba(220, 218, 211, 0.8)';
        for (let x = Math.ceil(VIEWPORT.minX); x <= Math.floor(VIEWPORT.maxX); x += 2) {
          const px = Math.floor(toPxX(x, w)) + 0.5;
          ctx.beginPath();
          ctx.moveTo(px, 0);
          ctx.lineTo(px, h);
          ctx.stroke();
        }
        for (let y = Math.ceil(VIEWPORT.minY); y <= Math.floor(VIEWPORT.maxY); y += 2) {
          const py = Math.floor(toPxY(y, h)) + 0.5;
          ctx.beginPath();
          ctx.moveTo(0, py);
          ctx.lineTo(w, py);
          ctx.stroke();
        }

        // Axes at 0
        ctx.strokeStyle = isCarbon ? '#444440' : '#B8B5AB';
        const zx = Math.floor(toPxX(0, w)) + 0.5;
        const zy = Math.floor(toPxY(0, h)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(zx, 0);
        ctx.lineTo(zx, h);
        ctx.moveTo(0, zy);
        ctx.lineTo(w, zy);
        ctx.stroke();
        ctx.restore();

        // 4. Histogram Grid Ticks (LightGBM)
        if (histogramGridArtifact) {
          ctx.save();
          ctx.lineWidth = 0.8;
          ctx.setLineDash([2, 4]);
          ctx.strokeStyle = isCarbon ? 'rgba(80, 80, 75, 0.4)' : 'rgba(180, 178, 170, 0.5)';
          for (const bx of histogramGridArtifact.binsX1) {
            const px = Math.floor(toPxX(bx, w)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.stroke();
          }
          {!is1DRegression && histogramGridArtifact.binsX2.map((by) => {
            const py = Math.floor(toPxY(by, h)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(0, py);
            ctx.lineTo(w, py);
            ctx.stroke();
          })}
          ctx.restore();
        }

        // 5. Ensemble Overlay Faint Lines (Random Forest)
        if (ensembleOverlayArtifact && currentSelectedTree === -1) {
          ctx.save();
          ctx.lineWidth = 1;
          ctx.strokeStyle = isCarbon ? 'rgba(217, 78, 52, 0.18)' : 'rgba(217, 78, 52, 0.22)';
          for (const ot of ensembleOverlayArtifact.trees) {
            for (const sp of ot.splits) {
              if (is1DRegression && sp.featureIndex === 1) continue;
              if (sp.featureIndex === 0) {
                const px = toPxX(sp.threshold, w);
                const py1 = toPxY(sp.range[1], h);
                const py2 = toPxY(sp.range[0], h);
                ctx.beginPath();
                ctx.moveTo(px, py1);
                ctx.lineTo(px, py2);
                ctx.stroke();
              } else {
                const py = toPxY(sp.threshold, h);
                const px1 = toPxX(sp.range[0], w);
                const px2 = toPxX(sp.range[1], w);
                ctx.beginPath();
                ctx.moveTo(px1, py);
                ctx.lineTo(px2, py);
                ctx.stroke();
              }
            }
          }
          ctx.restore();
        }

        // 6. Active Tree Splits (Axis-Aligned Orthogonal Lines)
        if (treePartitionArtifact && treePartitionArtifact.splits) {
          ctx.save();
          for (const sp of treePartitionArtifact.splits) {
            if (is1DRegression && sp.featureIndex === 1) continue;
            ctx.lineWidth = sp.isPlaid ? 1.8 : 1.6;
            ctx.strokeStyle = isCarbon ? '#E0DFD8' : '#141413';

            if (sp.featureIndex === 0) {
              // Vertical split line x1 < t
              const px = Math.floor(toPxX(sp.threshold, w)) + 0.5;
              const pyTop = toPxY(sp.range[1], h);
              const pyBot = toPxY(sp.range[0], h);
              ctx.beginPath();
              ctx.moveTo(px, pyTop);
              ctx.lineTo(px, pyBot);
              ctx.stroke();

              // Pill label
              ctx.fillStyle = isCarbon ? '#252523' : '#EBE9E1';
              const midY = (pyTop + pyBot) / 2;
              ctx.fillRect(px - 28, midY - 7, 56, 14);
              ctx.strokeStyle = isCarbon ? '#444440' : '#B8B5AB';
              ctx.lineWidth = 0.8;
              ctx.strokeRect(px - 28, midY - 7, 56, 14);

              ctx.fillStyle = isCarbon ? '#FAF9F5' : '#141413';
              ctx.font = '9px "JetBrains Mono"';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(sp.label || `X₁ < ${sp.threshold.toFixed(2)}`, px, midY);
            } else {
              // Horizontal split line x2 < t
              const py = Math.floor(toPxY(sp.threshold, h)) + 0.5;
              const pxLeft = toPxX(sp.range[0], w);
              const pxRight = toPxX(sp.range[1], w);
              ctx.beginPath();
              ctx.moveTo(pxLeft, py);
              ctx.lineTo(pxRight, py);
              ctx.stroke();

              // Pill label
              ctx.fillStyle = isCarbon ? '#252523' : '#EBE9E1';
              const midX = (pxLeft + pxRight) / 2;
              ctx.fillRect(midX - 28, py - 7, 56, 14);
              ctx.strokeStyle = isCarbon ? '#444440' : '#B8B5AB';
              ctx.lineWidth = 0.8;
              ctx.strokeRect(midX - 28, py - 7, 56, 14);

              ctx.fillStyle = isCarbon ? '#FAF9F5' : '#141413';
              ctx.font = '9px "JetBrains Mono"';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(sp.label || `X₂ < ${sp.threshold.toFixed(2)}`, midX, py);
            }
          }
          ctx.restore();
        }

        // 7. Hovered Tree Node 2D Sub-Region Highlight
        if (hoveredNodeRegion) {
          ctx.save();
          const [rxMin, rxMax, ryMin, ryMax] = hoveredNodeRegion;
          const px1 = toPxX(rxMin, w);
          const py1 = toPxY(ryMax, h);
          const rw = toPxX(rxMax, w) - px1;
          const rh = toPxY(ryMin, h) - py1;

          ctx.fillStyle = isCarbon ? 'rgba(217, 78, 52, 0.16)' : 'rgba(217, 78, 52, 0.18)';
          ctx.fillRect(px1, py1, rw, rh);

          ctx.strokeStyle = isCarbon ? '#D94E34' : '#B91C1C';
          ctx.lineWidth = 2.0;
          ctx.setLineDash([4, 4]);
          ctx.strokeRect(px1, py1, rw, rh);
          ctx.restore();
        }

        // 8. Residual Error Vectors (GBM / XGBoost)
        if (residualBarsArtifact) {
          ctx.save();
          for (const res of residualBarsArtifact.residuals) {
            const px = toPxX(res.x1, w);
            const py = toPxY(res.x2, h);
            const barLen = Math.max(-28, Math.min(28, res.error * 35));

            ctx.lineWidth = 1.6;
            ctx.strokeStyle = res.error >= 0 ? '#10B981' : '#EF4444';
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(px, py - barLen);
            ctx.stroke();

            // Arrow micro-cap
            ctx.beginPath();
            ctx.arc(px, py - barLen, 2.0, 0, Math.PI * 2);
            ctx.fillStyle = ctx.strokeStyle;
            ctx.fill();
          }
          ctx.restore();
        }

        // 8b. 1D Regression Fitted Step-wise Curve & Residual Pins
        if (is1DRegression && predictValueAtPoint) {
          ctx.save();
          // 1. Dotted vertical residual pins
          for (let i = 0; i < trainData.length; i++) {
            const p = trainData[i];
            const px = toPxX(p.x1, w);
            const py = toPxY(p.x2, h);
            const yHat = predictValueAtPoint(p.x1, p.x2);
            const pyHat = toPxY(yHat, h);

            ctx.strokeStyle = isCarbon ? 'rgba(194, 103, 82, 0.45)' : 'rgba(217, 78, 52, 0.4)';
            ctx.lineWidth = 1.0;
            ctx.setLineDash([2, 2]);
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(px, pyHat);
            ctx.stroke();

            // Small tick anchor notch at yHat
            ctx.setLineDash([]);
            ctx.beginPath();
            ctx.moveTo(px - 2.5, pyHat);
            ctx.lineTo(px + 2.5, pyHat);
            ctx.stroke();
          }

          // 2. Continuous 1D piecewise step curve
          const numSamples = Math.min(400, Math.floor(w));
          ctx.strokeStyle = isCarbon ? '#FAF9F5' : '#141413';
          ctx.lineWidth = 2.0;
          ctx.setLineDash([]);
          ctx.beginPath();
          for (let s = 0; s <= numSamples; s++) {
            const currX = VIEWPORT.minX + (s / numSamples) * (VIEWPORT.maxX - VIEWPORT.minX);
            const currYHat = predictValueAtPoint(currX, 0);
            const px = toPxX(currX, w);
            const py = toPxY(currYHat, h);
            if (s === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          ctx.stroke();
          ctx.restore();
        }

        // 9. Draw Data Points (Dynamic Radius for AdaBoost, Badges for Bagging)
        const getPointColor = (pt: DataPoint2D): string => {
          if (isRegressionDataset) {
            return isCarbon ? '#C26752' : '#D94E34';
          }
          if (pt.y === 2) {
            return isCarbon ? '#4D8F68' : '#15803D'; // Class +2
          }
          if (pt.y < 0) {
            return isCarbon ? '#5276B8' : '#1E40AF'; // Negative / Class -1
          }
          return isCarbon ? '#C26752' : '#D94E34'; // Positive / Class +1
        };

        renderDataPoints2D(ctx, {
          trainData,
          testData,
          showTestData: effectiveShowTestData,
          width: w,
          height: h,
          toPxX,
          toPxY,
          getColor: getPointColor,
          isCarbon,
          beforeDrawPoint: (pt, px, py, color) => {
            // Dynamic radius scaling (AdaBoost)
            const weightInfo = sampleWeightMap.get(pt.id);
            const baseRadius = 4.0;
            const radius = weightInfo ? baseRadius * weightInfo.scale : baseRadius;

            // Bagging Out-of-Bag or Multiplicity badge
            const baggingInfo = baggingSampleMap.get(pt.id);
            const isOOB = baggingInfo ? baggingInfo.isOOB : false;
            const sampleCount = baggingInfo ? baggingInfo.count : 1;

            if (isOOB) {
              // Out-of-Bag sample: hollow dashed ring
              ctx.save();
              ctx.beginPath();
              ctx.arc(px, py, radius + 2.5, 0, Math.PI * 2);
              ctx.setLineDash([2, 2]);
              ctx.strokeStyle = color;
              ctx.lineWidth = 1.4;
              ctx.stroke();
              ctx.restore();
            } else if (sampleCount >= 2) {
              // Concentric multiplicity ring badge (2x, 3x)
              ctx.save();
              ctx.beginPath();
              ctx.arc(px, py, radius + 3.0, 0, Math.PI * 2);
              ctx.strokeStyle = color;
              ctx.lineWidth = 1.2;
              ctx.stroke();

              ctx.fillStyle = isCarbon ? '#FAF9F5' : '#141413';
              ctx.font = '8px "JetBrains Mono"';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText(`${sampleCount}x`, px, py - radius - 5);
              ctx.restore();
            }

            if (weightInfo && weightInfo.isMisclassified) {
              // Heavy misclassified point halo glow
              ctx.save();
              ctx.beginPath();
              ctx.arc(px, py, radius + 4.5, 0, Math.PI * 2);
              ctx.fillStyle = isCarbon ? 'rgba(217, 78, 52, 0.22)' : 'rgba(217, 78, 52, 0.25)';
              ctx.fill();
              ctx.restore();
            }

            return { trainRadius: radius, testRadius: radius + 0.5 };
          },
        });
        }
      }
      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [
    dirtyManager,
    isCarbon,
    isRegressionDataset,
    is1DRegression,
    predictValueAtPoint,
    gridValues,
    trainData,
    testData,
    treePartitionArtifact,
    adaboostWeightsArtifact,
    sampleWeightMap,
    baggingSampleMap,
    residualBarsArtifact,
    ensembleOverlayArtifact,
    histogramGridArtifact,
    hoveredNodeRegion,
    currentSelectedTree,
    effectiveShowTestData,
    discretize,
  ]);

  // Pointer Interaction
  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const pxClick = e.clientX - rect.left;
    const pyClick = e.clientY - rect.top;
    const x = toCoordX(pxClick, rect.width);
    const y = toCoordY(pyClick, rect.height);

    // Hit test existing visible point (radius 12px)
    const visiblePoints = getVisibleDataPoints(trainData, testData, effectiveShowTestData);
    const hitPoint = hitTestDataPoints(visiblePoints, {
      px: pxClick,
      py: pyClick,
      width: rect.width,
      height: rect.height,
      hitRadiusPx: 12,
      toPxX,
      toPxY,
      searchFromEnd: true,
    });

    if (pointerMode === 'delete' && hitPoint !== null) {
      onDeletePoint(hitPoint.id);
      dirtyManager.markDirty();
      return;
    }

    if (pointerMode === 'flip' && hitPoint !== null) {
      onUpdatePoint({ ...hitPoint, y: hitPoint.y === 1 ? -1 : 1 });
      dirtyManager.markDirty();
      return;
    }

    if (pointerMode === 'add-drag') {
      if (hitPoint !== null) {
        setDraggingPointId(hitPoint.id);
        onDragStart?.();
      } else {
        if (is1DRegression) {
          onAddPoint({ x1: x, x2: y, y: y });
        } else {
          onAddPoint({ x1: x, x2: y, y: activeClass });
        }
        dirtyManager.markDirty();
      }
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = toCoordX(e.clientX - rect.left, rect.width);
    const y = toCoordY(e.clientY - rect.top, rect.height);

    setHoverCoord({ x, y });

    if (draggingPointId !== null) {
      const visiblePoints = getVisibleDataPoints(trainData, testData, effectiveShowTestData);
      const target = visiblePoints.find((p) => p.id === draggingPointId);
      if (target) {
        if (is1DRegression) {
          onUpdatePoint({ ...target, x1: x, x2: y, y: y });
        } else {
          onUpdatePoint({ ...target, x1: x, x2: y });
        }
        dirtyManager.markDirty();
      }
    }
  };

  const handleMouseUp = () => {
    if (draggingPointId !== null) {
      setDraggingPointId(null);
      onDragEnd?.();
    }
  };

  const handleMouseLeave = () => {
    setHoverCoord(null);
    if (draggingPointId !== null) {
      setDraggingPointId(null);
      onDragEnd?.();
    }
  };

  // Tree nodes layout calculation for the SVG Inspector
  const svgTreeLayout = useMemo(() => {
    if (!treePartitionArtifact || !treePartitionArtifact.allNodes || treePartitionArtifact.allNodes.length === 0) {
      return null;
    }
    const nodes = treePartitionArtifact.allNodes;
    const root = treePartitionArtifact.root;
    if (!root) return null;

    // Calculate depth levels and tree layout
    const levelMap = new Map<number, TreeNodeData[]>();
    for (const nd of nodes) {
      const list = levelMap.get(nd.depth) || [];
      list.push(nd);
      levelMap.set(nd.depth, list);
    }

    const maxLevelCount = Math.max(...Array.from(levelMap.values()).map((l) => l.length), 1);
    const minStep = 96;
    const svgWidth = Math.max(380, (maxLevelCount + 1) * minStep);
    const levelHeight = 64;
    const nodeCoords = new Map<number, { x: number; y: number }>();

    for (const [depth, levelNodes] of levelMap.entries()) {
      const count = levelNodes.length;
      const step = svgWidth / (count + 1);
      for (let i = 0; i < count; i++) {
        const nd = levelNodes[i];
        nodeCoords.set(nd.id, {
          x: step * (i + 1),
          y: 36 + depth * levelHeight,
        });
      }
    }

    const maxDepth = Math.max(...Array.from(levelMap.keys()));
    const svgHeight = Math.max(260, 60 + (maxDepth + 1) * levelHeight);

    return { nodeCoords, svgWidth, svgHeight, maxDepth };
  }, [treePartitionArtifact]);

  return (
    <div className="flex flex-row w-full h-full min-h-0 min-w-0 overflow-hidden bg-[var(--bg-primary)]">
      {/* 1. Main 2D Partition Canvas (Left, 65% width) */}
      <div
        className="relative flex flex-col h-full min-w-0 overflow-hidden border-r border-[var(--border-hairline)]"
        style={{ flex: '65 1 0%' }}
      >
        <div ref={containerRef} className="relative flex-1 min-h-0 min-w-0 overflow-hidden">
          <canvas
            ref={mainCanvasRef}
            className="canvas-main block w-full h-full cursor-crosshair"
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
          />

          {/* Hover Coordinate Tag & Scale Invariance badge */}
          <div className="absolute top-2 left-3 flex items-center gap-2 pointer-events-none font-mono text-[9px]">
            <span className="bg-[var(--bg-canvas)]/90 backdrop-blur-sm border border-[var(--border-hairline)] px-2 py-0.5 text-[var(--text-secondary)] shadow-sm">
              {hoverCoord
                ? `Cursor (${hoverCoord.x.toFixed(2)}, ${hoverCoord.y.toFixed(2)})`
                : algorithm.id === 'voting-classifier' || algorithm.id === 'stacking'
                ? 'Hover canvas for coordinates'
                : 'Hover to trace active leaf'}
            </span>
            <span className="bg-emerald-950/20 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 px-1.5 py-0.5">
              Monotonic Scale Invariant
            </span>
          </div>
        </div>

        {/* Bottom Bar: Swiss Color Legend & Discretize Toggle */}
        <div className="flex flex-col px-3 py-2 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] gap-2 select-none shrink-0">
          <div className="flex flex-wrap items-center justify-between text-[var(--text-muted)] gap-1.5">
            {is1DRegression ? (
              <div className="flex items-center gap-1.5 font-mono text-[9px] text-[var(--text-secondary)]">
                <span
                  className="w-2 h-2 rounded-full inline-block shrink-0"
                  style={{ backgroundColor: isCarbon ? '#C26752' : '#D94E34' }}
                />
                <span>Target y ∈ ℝ (Continuous 1D response curve)</span>
              </div>
            ) : isRegressionDataset ? (
              <>
                <span className="text-[9px] truncate mr-2" title="Predicted continuous surface elevation across 2D plane:">
                  Predicted surface elevation:
                </span>
                <div className="flex items-center gap-1.5 font-mono font-bold text-[9px] shrink-0">
                  <span className={isCarbon ? 'text-[#4E89B8]' : 'text-[#1E40AF]'}>ŷ &lt; 0</span>
                  <div
                    className="w-20 h-2 rounded-xs border border-[var(--border-hairline)]"
                    style={{
                      background: isCarbon
                        ? 'linear-gradient(to right, rgb(78, 137, 184), rgb(24, 24, 23), rgb(229, 131, 68))'
                        : 'linear-gradient(to right, rgb(43, 91, 132), rgb(250, 249, 245), rgb(200, 90, 23))',
                    }}
                  />
                  <span className={isCarbon ? 'text-[#E58344]' : 'text-[#B91C1C]'}>ŷ &gt; 0</span>
                </div>
              </>
            ) : (
              <>
                <span className="text-[9px] truncate mr-2" title="Colors show data, decision boundary confidence & partition values:">
                  Colors show data & partition confidence:
                </span>
                <div className="flex items-center gap-1.5 font-mono font-bold text-[9px] shrink-0">
                  <span className={isCarbon ? 'text-[#4E89B8]' : 'text-[#1E40AF]'}>-1</span>
                  <div
                    className="w-20 h-2 rounded-xs border border-[var(--border-hairline)]"
                    style={{
                      background: isCarbon
                        ? 'linear-gradient(to right, rgb(78, 137, 184), rgb(24, 24, 23), rgb(229, 131, 68))'
                        : 'linear-gradient(to right, rgb(43, 91, 132), rgb(250, 249, 245), rgb(200, 90, 23))',
                    }}
                  />
                  <span className={isCarbon ? 'text-[#E58344]' : 'text-[#B91C1C]'}>+1</span>
                </div>
              </>
            )}
          </div>
          <div className="flex flex-wrap items-center justify-between pt-0.5 border-t border-[var(--border-hairline)]/50 gap-2">
            <label className="flex items-center gap-1.5 cursor-pointer text-[var(--text-primary)] hover:text-[var(--text-primary)] shrink-0">
              <input
                type="checkbox"
                checked={effectiveShowTestData}
                onChange={handleToggleShowTestData}
                className="rounded-xs border-[var(--border-hairline)] text-[var(--accent-amber)] focus:ring-0 cursor-pointer"
              />
              <span>Show test data</span>
            </label>
            {!is1DRegression && (
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
            )}
          </div>
        </div>
      </div>

      {/* 2. Interactive Tree Topology & Ensemble Inspector Inset (Right, 35% width) */}
      <div
        className="relative flex flex-col h-full bg-[var(--bg-elevated)] min-w-[280px] overflow-hidden"
        style={{ flex: '35 1 0%' }}
      >
        {/* Inset Header */}
        <div className="px-4 py-2 border-b border-[var(--border-hairline)] flex items-center justify-between bg-[var(--bg-subtle)]">
          <span className="font-mono text-[10px] uppercase font-semibold text-[var(--text-primary)] tracking-wider">
            {algorithm.id === 'stacking'
              ? 'Meta-Feature Projection (Level-1)'
              : algorithm.id === 'voting-classifier'
              ? 'Base Model Consensus Breakdown'
              : 'Tree Topology & Node Inspector'}
          </span>
          <span className="font-mono text-[9px] text-[var(--text-muted)]">
            Stage 02
          </span>
        </div>

        {/* Multi-Tree Ensemble Progression Filmstrip */}
        {algorithm.id !== 'decision-tree' && algorithm.id !== 'stacking' && algorithm.id !== 'voting-classifier' && (
          <div className="px-3 py-1.5 border-b border-[var(--border-hairline)] bg-[var(--bg-canvas)] flex items-center gap-2 overflow-x-auto no-scrollbar font-mono text-[10px]">
            <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider shrink-0 mr-0.5">
              Filmstrip:
            </span>
            {(treePartitionArtifact?.totalTreesCount ?? (treePartitionArtifact?.allNodes && treePartitionArtifact.allNodes.length > 0 ? 1 : 0)) <= 0 ? (
              <span className="text-[10px] text-[var(--text-muted)] italic px-1">
                Ready to Train (0 Trees Built)
              </span>
            ) : (
              <>
                <button
                  className={`px-2 py-1 border shrink-0 transition-all flex flex-col items-center gap-0.5 rounded-xs ${
                    currentSelectedTree === -1
                      ? 'border-[var(--accent-amber)] ring-1 ring-[var(--accent-amber)]/60 bg-[var(--bg-active)]'
                      : 'border-[var(--border-hairline)] bg-[var(--bg-canvas)] hover:border-[var(--text-muted)]'
                  }`}
                  onClick={() => onUpdateHyperparam?.('selectedTree', -1)}
                  title="Inspect Ensemble Combined Surface"
                >
                  <div className="w-6 h-6 border border-[var(--border-hairline)]/50 rounded-2xs overflow-hidden bg-neutral-900/10 flex items-center justify-center font-mono text-[8px] font-bold text-[var(--accent-amber)]">
                    ALL
                  </div>
                  <span className="text-[8px] font-mono font-medium text-[var(--text-secondary)] whitespace-nowrap">
                    Ensemble ({treePartitionArtifact?.totalTreesCount ?? 1})
                  </span>
                </button>

                {/* Individual Tree Mini Previews */}
                {(treePartitionArtifact?.treesRoots && treePartitionArtifact.treesRoots.length > 0) ? (
                  treePartitionArtifact.treesRoots.slice(0, 12).map((rootNode, i) => (
                    <TreeMiniPreview
                      key={i}
                      root={rootNode}
                      isCarbon={isCarbon}
                      isSelected={currentSelectedTree === i}
                      label={`Tree #${i + 1}`}
                      onClick={() => onUpdateHyperparam?.('selectedTree', i)}
                    />
                  ))
                ) : (
                  Array.from({ length: Math.min(12, treePartitionArtifact?.totalTreesCount ?? 1) }, (_, i) => (
                    <button
                      key={i}
                      className={`px-1.5 py-0.5 border shrink-0 transition-colors ${
                        currentSelectedTree === i
                          ? 'bg-[var(--accent-amber)] text-white font-medium border-[var(--accent-amber)]'
                          : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] border-[var(--border-hairline)] hover:text-[var(--text-primary)]'
                      }`}
                      onClick={() => onUpdateHyperparam?.('selectedTree', i)}
                    >
                      Tree #{i + 1}
                    </button>
                  ))
                )}
              </>
            )}
          </div>
        )}

        {/* Inset Main View */}
        <div className="flex-1 overflow-auto p-3">
          {/* A. Stacking Level-1 Meta-Feature Projection Space */}
          {algorithm.id === 'stacking' && (
            metaFeatureArtifact ? (
              <div className="flex flex-col gap-3 h-full">
                <div className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                  {isRegression ? (
                    <>
                      Projected base predictions <span className="font-mono font-semibold">[ŷ(Tree), ŷ(KNN)]</span> with Level-1 meta-regression:
                    </>
                  ) : (
                    <>
                      Projected base probabilities <span className="font-mono font-semibold">[P(Tree), P(KNN)]</span> with Level-1 meta-learner separating line:
                    </>
                  )}
                </div>
                <div className="relative w-full aspect-square bg-[var(--bg-canvas)] border border-[var(--border-hairline)] p-2">
                  <svg className="w-full h-full" viewBox="0 0 100 100">
                    {/* Axis lines */}
                    <line x1="5" y1="95" x2="95" y2="95" stroke={isCarbon ? '#444' : '#BBB'} strokeWidth="0.8" />
                    <line x1="5" y1="5" x2="5" y2="95" stroke={isCarbon ? '#444' : '#BBB'} strokeWidth="0.8" />

                    {(() => {
                      let minP1 = 0, maxP1 = 1, minP2 = 0, maxP2 = 1;
                      if (isRegression && metaFeatureArtifact.points.length > 0) {
                        minP1 = Infinity; maxP1 = -Infinity;
                        minP2 = Infinity; maxP2 = -Infinity;
                        for (const pt of metaFeatureArtifact.points) {
                          if (pt.p1 < minP1) minP1 = pt.p1;
                          if (pt.p1 > maxP1) maxP1 = pt.p1;
                          if (pt.p2 < minP2) minP2 = pt.p2;
                          if (pt.p2 > maxP2) maxP2 = pt.p2;
                        }
                        const pad1 = Math.max(0.25, (maxP1 - minP1) * 0.1);
                        const pad2 = Math.max(0.25, (maxP2 - minP2) * 0.1);
                        minP1 -= pad1; maxP1 += pad1;
                        minP2 -= pad2; maxP2 += pad2;
                      }
                      const span1 = Math.max(1e-6, maxP1 - minP1);
                      const span2 = Math.max(1e-6, maxP2 - minP2);

                      return (
                        <>
                          {/* Level-1 separating hyperplane in classification mode */}
                          {!isRegression && metaFeatureArtifact.weights[1] !== 0 && (
                            <line
                              x1={5}
                              y1={95 - (-(metaFeatureArtifact.weights[0] * 0 + metaFeatureArtifact.bias) / metaFeatureArtifact.weights[1]) * 90}
                              x2={95}
                              y2={95 - (-(metaFeatureArtifact.weights[0] * 1 + metaFeatureArtifact.bias) / metaFeatureArtifact.weights[1]) * 90}
                              stroke={isCarbon ? '#F59E0B' : '#D97706'}
                              strokeWidth="1.6"
                            />
                          )}

                          {/* Projected Points */}
                          {metaFeatureArtifact.points.map((pt) => {
                            const isPositive = isRegression ? pt.y >= 0 : pt.y === 1;
                            const normX = Math.max(0, Math.min(1, (pt.p1 - minP1) / span1));
                            const normY = Math.max(0, Math.min(1, (pt.p2 - minP2) / span2));
                            return (
                              <circle
                                key={pt.id}
                                cx={5 + normX * 90}
                                cy={95 - normY * 90}
                                r={2.2}
                                fill={isPositive ? (isCarbon ? '#C26752' : '#D94E34') : (isCarbon ? '#5276B8' : '#1E40AF')}
                              />
                            );
                          })}
                        </>
                      );
                    })()}
                  </svg>
                  <span className="absolute bottom-1 right-2 font-mono text-[8px] text-[var(--text-muted)]">
                    {isRegression ? 'ŷ(Tree) →' : 'P(Tree) →'}
                  </span>
                  <span className="absolute top-1 left-2 font-mono text-[8px] text-[var(--text-muted)]">
                    {isRegression ? '↑ ŷ(KNN)' : '↑ P(KNN)'}
                  </span>
                </div>
                <div className="font-mono text-[10px] bg-[var(--bg-subtle)] p-2 border border-[var(--border-hairline)] flex flex-col gap-1">
                  <span>w_tree: {metaFeatureArtifact.weights[0].toFixed(3)}</span>
                  <span>w_knn: {metaFeatureArtifact.weights[1].toFixed(3)}</span>
                  <span>bias: {metaFeatureArtifact.bias.toFixed(3)}</span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center text-[var(--text-muted)] border border-dashed border-[var(--border-hairline)] rounded-sm my-auto min-h-[220px]">
                <div className="w-9 h-9 rounded-full bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex items-center justify-center mb-2.5 text-[var(--text-secondary)] font-mono text-[13px] font-semibold">
                  0
                </div>
                <span className="font-mono text-[12px] font-semibold text-[var(--text-primary)]">
                  Ready to Train (Untrained State)
                </span>
                <span className="text-[10px] text-[var(--text-secondary)] mt-1.5 max-w-[220px] leading-relaxed">
                  Click <kbd className="px-1 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono text-[9px] rounded">Play</kbd> or <kbd className="px-1 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono text-[9px] rounded">Step</kbd> to train base models and project meta-features.
                </span>
              </div>
            )
          )}

          {/* B. Voting Classifier Breakdown View */}
          {algorithm.id === 'voting-classifier' && (
            votingBoundariesArtifact ? (
              <div className="flex flex-col gap-3">
                <div className="text-[11px] text-[var(--text-secondary)]">
                  Base Model Performance & Consensus Weights:
                </div>
                {votingBoundariesArtifact.baseModels.map((bm, idx) => (
                  <div key={idx} className="p-2.5 bg-[var(--bg-canvas)] border border-[var(--border-hairline)] flex flex-col gap-1 font-mono text-[11px]">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-semibold text-[var(--text-primary)] flex items-center gap-1.5 truncate min-w-0">
                        <span className="w-2.5 h-2.5 rounded-full inline-block shrink-0" style={{ backgroundColor: bm.color }}></span>
                        <span className="truncate">{bm.name}</span>
                      </span>
                      <span className="text-[var(--text-secondary)] shrink-0">Weight: {bm.weight.toFixed(1)}x</span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-[var(--text-muted)] mt-1">
                      <span>{isRegression ? 'Model MSE' : 'Individual Accuracy'}</span>
                      <span className="font-semibold text-[var(--text-primary)]">
                        {isRegression ? bm.accuracy.toFixed(4) : `${bm.accuracy.toFixed(1)}%`}
                      </span>
                    </div>
                    <div className="w-full bg-[var(--bg-subtle)] h-1.5 overflow-hidden">
                      <div
                        className="h-full"
                        style={{
                          width: isRegression
                            ? `${Math.max(5, Math.min(100, Math.max(0, 1 - bm.accuracy) * 100))}%`
                            : `${bm.accuracy}%`,
                          backgroundColor: bm.color,
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center text-[var(--text-muted)] border border-dashed border-[var(--border-hairline)] rounded-sm my-auto min-h-[220px]">
                <div className="w-9 h-9 rounded-full bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex items-center justify-center mb-2.5 text-[var(--text-secondary)] font-mono text-[13px] font-semibold">
                  0
                </div>
                <span className="font-mono text-[12px] font-semibold text-[var(--text-primary)]">
                  Ready to Train (Untrained State)
                </span>
                <span className="text-[10px] text-[var(--text-secondary)] mt-1.5 max-w-[220px] leading-relaxed">
                  Click <kbd className="px-1 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono text-[9px] rounded">Play</kbd> or <kbd className="px-1 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono text-[9px] rounded">Step</kbd> to fit base models and evaluate consensus weights.
                </span>
              </div>
            )
          )}

          {/* C. Binary Decision Tree SVG Graph */}
          {algorithm.id !== 'stacking' && algorithm.id !== 'voting-classifier' && (
            svgTreeLayout ? (
              <div className="w-full overflow-x-auto">
                <svg
                  width={svgTreeLayout.svgWidth}
                  height={svgTreeLayout.svgHeight}
                  className="overflow-visible select-none"
                >
                  {/* Connectors / Edges */}
                  {treePartitionArtifact?.allNodes?.map((nd) => {
                    if (nd.isLeaf) return null;
                    const parentCoord = svgTreeLayout.nodeCoords.get(nd.id);
                    if (!parentCoord) return null;

                    const leftCoord = nd.leftId !== undefined ? svgTreeLayout.nodeCoords.get(nd.leftId) : undefined;
                    const rightCoord = nd.rightId !== undefined ? svgTreeLayout.nodeCoords.get(nd.rightId) : undefined;

                    const isLeftActive = nd.leftId !== undefined && activeHoverPathNodeIds.includes(nd.leftId);
                    const isRightActive = nd.rightId !== undefined && activeHoverPathNodeIds.includes(nd.rightId);

                    return (
                      <g key={`edge-${nd.id}`}>
                        {leftCoord && (
                          <path
                            d={`M ${parentCoord.x} ${parentCoord.y + 12} L ${leftCoord.x} ${leftCoord.y - 12}`}
                            stroke={isLeftActive ? (isCarbon ? '#F59E0B' : '#D97706') : (isCarbon ? '#383834' : '#DCDAD3')}
                            strokeWidth={isLeftActive ? 2.2 : 1.2}
                            fill="none"
                          />
                        )}
                        {rightCoord && (
                          <path
                            d={`M ${parentCoord.x} ${parentCoord.y + 12} L ${rightCoord.x} ${rightCoord.y - 12}`}
                            stroke={isRightActive ? (isCarbon ? '#F59E0B' : '#D97706') : (isCarbon ? '#383834' : '#DCDAD3')}
                            strokeWidth={isRightActive ? 2.2 : 1.2}
                            fill="none"
                          />
                        )}
                      </g>
                    );
                  })}

                  {/* Nodes */}
                  {treePartitionArtifact?.allNodes?.map((nd) => {
                    const coord = svgTreeLayout.nodeCoords.get(nd.id);
                    if (!coord) return null;

                    const isHovered = hoveredNodeId === nd.id;
                    const isInActivePath = activeHoverPathNodeIds.includes(nd.id);

                    if (nd.isLeaf) {
                      const isBoostingLogOdds =
                        algorithm.id === 'gradient-boosting' ||
                        algorithm.id === 'xgboost' ||
                        algorithm.id === 'lightgbm' ||
                        algorithm.id === 'catboost';

                      // Leaf Node: Pill with Class/Weight/Value Badge and Mini 2D Partition Glyph
                      const leafColor = isRegression
                        ? (isCarbon ? '#C26752' : '#D94E34')
                        : isBoostingLogOdds
                        ? (nd.value >= 0 ? (isCarbon ? '#C26752' : '#D94E34') : (isCarbon ? '#5276B8' : '#1E40AF'))
                        : nd.value === 1
                        ? (isCarbon ? '#C26752' : '#D94E34')
                        : nd.value === 2
                        ? (isCarbon ? '#4D8F68' : '#15803D')
                        : (isCarbon ? '#5276B8' : '#1E40AF');

                      const leafLabel = isRegression
                        ? `ŷ=${typeof nd.value === 'number' ? nd.value.toFixed(2) : nd.value}`
                        : isBoostingLogOdds
                        ? `w=${nd.value >= 0 ? '+' : ''}${typeof nd.value === 'number' ? nd.value.toFixed(2) : nd.value}`
                        : nd.value === 1
                        ? 'Class +1'
                        : nd.value === 2
                        ? 'Class +2'
                        : typeof nd.value === 'number' && Math.abs(nd.value) > 1.5
                        ? `Val ${nd.value.toFixed(1)}`
                        : 'Class -1';

                      return (
                        <g
                          key={`node-${nd.id}`}
                          transform={`translate(${coord.x}, ${coord.y})`}
                          className="cursor-pointer transition-transform"
                          onMouseEnter={() => setHoveredNodeId(nd.id)}
                          onMouseLeave={() => setHoveredNodeId(null)}
                        >
                          {/* Mini 2D partition glyph */}
                          <rect
                            x={-42}
                            y={-10}
                            width={20}
                            height={20}
                            rx={2}
                            fill={leafColor}
                            opacity={0.35}
                            stroke={leafColor}
                            strokeWidth={1.2}
                          />
                          <rect
                            x={-18}
                            y={-12}
                            width={54}
                            height={24}
                            rx={3}
                            fill={isHovered ? leafColor : (isCarbon ? '#1D1D1B' : '#F2EFE8')}
                            stroke={isHovered || isInActivePath ? leafColor : (isCarbon ? '#3E3E3A' : '#CCC9C0')}
                            strokeWidth={isHovered || isInActivePath ? 1.8 : 1.0}
                          />
                          <text
                            x={9}
                            textAnchor="middle"
                            dy={-1}
                            fontFamily="JetBrains Mono"
                            fontSize="9"
                            fontWeight="600"
                            fill={isHovered ? '#FFF' : leafColor}
                          >
                            {leafLabel}
                          </text>
                          <text
                            x={9}
                            textAnchor="middle"
                            dy={8}
                            fontFamily="JetBrains Mono"
                            fontSize="7"
                            fill={isHovered ? '#FFF' : (isCarbon ? '#888' : '#666')}
                          >
                            N = {nd.samples}
                          </text>
                        </g>
                      );
                    }

                    // Internal Decision Node
                    const splitLabel = `${nd.featureIndex === 0 ? 'X₁' : 'X₂'} < ${nd.threshold?.toFixed(2) ?? '0'}`;
                    const normSplit = nd.threshold !== undefined
                      ? Math.max(0.15, Math.min(0.85, (nd.threshold + 6) / 12))
                      : 0.5;

                    return (
                      <g
                        key={`node-${nd.id}`}
                        transform={`translate(${coord.x}, ${coord.y})`}
                        className="cursor-pointer transition-transform"
                        onMouseEnter={() => setHoveredNodeId(nd.id)}
                        onMouseLeave={() => setHoveredNodeId(null)}
                      >
                        {/* Mini 2D partition glyph showing feature split */}
                        <g>
                          <rect
                            x={-46}
                            y={-11}
                            width={22}
                            height={22}
                            rx={2}
                            fill={isCarbon ? '#22221F' : '#ECE9E0'}
                            stroke={isCarbon ? '#444' : '#CCC'}
                            strokeWidth={1}
                          />
                          {nd.featureIndex === 0 ? (
                            <line
                              x1={-46 + 22 * normSplit}
                              y1={-11}
                              x2={-46 + 22 * normSplit}
                              y2={11}
                              stroke={isCarbon ? '#F59E0B' : '#D97706'}
                              strokeWidth={1.8}
                            />
                          ) : (
                            <line
                              x1={-46}
                              y1={11 - 22 * normSplit}
                              x2={-24}
                              y2={11 - 22 * normSplit}
                              stroke={isCarbon ? '#F59E0B' : '#D97706'}
                              strokeWidth={1.8}
                            />
                          )}
                        </g>

                        <rect
                          x={-20}
                          y={-14}
                          width={66}
                          height={28}
                          rx={2}
                          fill={isHovered ? (isCarbon ? '#2A2A26' : '#E8E5DD') : (isCarbon ? '#181816' : '#FAF9F5')}
                          stroke={isHovered || isInActivePath ? (isCarbon ? '#F59E0B' : '#D97706') : (isCarbon ? '#3E3E3A' : '#CCC9C0')}
                          strokeWidth={isHovered || isInActivePath ? 1.8 : 1.0}
                        />
                        <text
                          x={13}
                          textAnchor="middle"
                          dy={-2}
                          fontFamily="JetBrains Mono"
                          fontSize="9"
                          fontWeight="600"
                          fill={isCarbon ? '#FAF9F5' : '#141413'}
                        >
                          {splitLabel}
                        </text>
                        <text
                          x={13}
                          textAnchor="middle"
                          dy={8}
                          fontFamily="JetBrains Mono"
                          fontSize="7"
                          fill={isCarbon ? '#777' : '#888'}
                        >
                          {nd.gain !== undefined
                            ? `${
                                isRegression || hyperparams.criterion === 'mse'
                                  ? 'ΔMSE'
                                  : hyperparams.criterion === 'entropy'
                                  ? 'IG'
                                  : 'ΔG'
                              } ${nd.gain.toFixed(2)}`
                            : `Imp ${nd.impurity.toFixed(2)}`} · N {nd.samples}
                        </text>
                      </g>
                    );
                  })}
                </svg>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center p-8 text-center text-[var(--text-muted)] border border-dashed border-[var(--border-hairline)] rounded-sm my-auto min-h-[220px]">
                <div className="w-9 h-9 rounded-full bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex items-center justify-center mb-2.5 text-[var(--text-secondary)] font-mono text-[13px] font-semibold">
                  0
                </div>
                <span className="font-mono text-[12px] font-semibold text-[var(--text-primary)]">
                  Ready to Train (Untrained State)
                </span>
                <span className="text-[10px] text-[var(--text-secondary)] mt-1.5 max-w-[220px] leading-relaxed">
                  0 Trees constructed. Click <kbd className="px-1 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono text-[9px] rounded">Play</kbd> or <kbd className="px-1 py-0.5 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] font-mono text-[9px] rounded">Step</kbd> to begin building trees.
                </span>
              </div>
            )
          )}
        </div>
      </div>
    </div>
  );
};

// Mini 24x24 Preview Canvas for each tree in the ensemble filmstrip
interface TreeMiniPreviewProps {
  root?: TreeNodeData;
  isCarbon: boolean;
  isSelected?: boolean;
  label: string;
  onClick: () => void;
}

const TreeMiniPreview: React.FC<TreeMiniPreviewProps> = React.memo(
  ({ root, isCarbon, isSelected, label, onClick }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas || !root) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const res = 24;
      canvas.width = res;
      canvas.height = res;

      const imgData = ctx.createImageData(res, res);
      const data = imgData.data;
      const step = 12.0 / (res - 1);
      let idx = 0;

      for (let r = 0; r < res; r++) {
        const x2 = 6.0 - r * step;
        for (let c = 0; c < res; c++) {
          const x1 = -6.0 + c * step;
          const v = predictTree(root as any, x1, x2);
          const t = Math.tanh(v);
          const isPos = t > 0;
          const alpha = Math.min(0.85, Math.abs(t) * 0.85);

          const rCol = isPos ? (isCarbon ? 229 : 200) : isCarbon ? 78 : 43;
          const gCol = isPos ? (isCarbon ? 131 : 90) : isCarbon ? 137 : 91;
          const bCol = isPos ? (isCarbon ? 68 : 23) : isCarbon ? 184 : 132;

          const off = idx * 4;
          data[off] = rCol;
          data[off + 1] = gCol;
          data[off + 2] = bCol;
          data[off + 3] = Math.floor(alpha * 255);
          idx++;
        }
      }
      ctx.putImageData(imgData, 0, 0);
    }, [root, isCarbon]);

    return (
      <button
        onClick={onClick}
        className={`cursor-pointer flex flex-col items-center p-1 rounded-xs border shrink-0 transition-all ${
          isSelected
            ? 'border-[var(--accent-amber)] ring-1 ring-[var(--accent-amber)]/60 bg-[var(--bg-active)]'
            : 'border-[var(--border-hairline)] bg-[var(--bg-canvas)] hover:border-[var(--text-muted)]'
        }`}
        title={`Inspect ${label}`}
      >
        <div className="w-6 h-6 border border-[var(--border-hairline)]/50 rounded-2xs overflow-hidden bg-neutral-900/10">
          <canvas ref={canvasRef} className="w-full h-full block" />
        </div>
        <span className="text-[8px] font-mono mt-0.5 font-medium text-[var(--text-secondary)] whitespace-nowrap">
          {label}
        </span>
      </button>
    );
  }
);

