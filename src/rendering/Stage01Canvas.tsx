import React, { useRef, useEffect, useState, useCallback } from 'react';
import { DataPoint2D, PointerToolMode } from '../datasets/tabular2d';
import { VisualArtifact, AlgorithmPlugin, TaskType } from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { DirtyManager } from './dirty-manager';
import {
  observeElementSize,
  subscribeDirtyRender,
  syncCanvasBufferSize,
} from './canvasUtils';
import { populateHeatmapImageData } from './color-maps';
import { gaussianPdf, euclideanDist, manhattanDist } from '../utils/math';
import { getVisibleDataPoints, hitTestDataPoints, renderDataPoints2D } from './pointUtils';

export interface Stage01CanvasProps {
  algorithm?: AlgorithmPlugin;
  taskType?: TaskType;
  datasetTaskType?: TaskType;
  theme: ThemeMode;
  trainData: DataPoint2D[];
  testData: DataPoint2D[];
  artifacts: VisualArtifact[];
  gridValues?: Float32Array | null;
  gridResolution: number;
  dirtyManager: DirtyManager;
  highlightTarget?: string | null;
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
}

const VIEWPORT = { minX: -6.0, maxX: 6.0, minY: -6.0, maxY: 6.0 };

function strokeMarchingSquaresContour(
  ctx: CanvasRenderingContext2D,
  field: Float32Array,
  res: number,
  bounds: [number, number, number, number],
  level: number,
  toPxX: (x: number, w: number) => number,
  toPxY: (y: number, h: number) => number,
  w: number,
  h: number
): void {
  if (res < 2 || field.length < res * res) return;
  const [minX, maxX, minY, maxY] = bounds;
  const stepX = (maxX - minX) / (res - 1);
  const stepY = (maxY - minY) / (res - 1);

  const lerpT = (a: number, b: number) => {
    const denom = a - b;
    if (Math.abs(denom) < 1e-12) return 0.5;
    return Math.max(0, Math.min(1, (a - level) / denom));
  };

  ctx.beginPath();
  for (let r = 0; r < res - 1; r++) {
    const yTop = maxY - r * stepY;
    const yBot = yTop - stepY;
    const rowTop = r * res;
    const rowBot = (r + 1) * res;

    for (let c = 0; c < res - 1; c++) {
      const vTL = field[rowTop + c];
      const vTR = field[rowTop + c + 1];
      const vBR = field[rowBot + c + 1];
      const vBL = field[rowBot + c];

      const mask =
        (vTL >= level ? 8 : 0) |
        (vTR >= level ? 4 : 0) |
        (vBR >= level ? 2 : 0) |
        (vBL >= level ? 1 : 0);

      if (mask === 0 || mask === 15) continue;

      const xLeft = minX + c * stepX;
      const xRight = xLeft + stepX;

      const topPt: [number, number] = [xLeft + lerpT(vTL, vTR) * stepX, yTop];
      const rightPt: [number, number] = [xRight, yTop - lerpT(vTR, vBR) * stepY];
      const bottomPt: [number, number] = [xLeft + lerpT(vBL, vBR) * stepX, yBot];
      const leftPt: [number, number] = [xLeft, yTop - lerpT(vTL, vBL) * stepY];

      const addSeg = (pA: [number, number], pB: [number, number]) => {
        ctx.moveTo(toPxX(pA[0], w), toPxY(pA[1], h));
        ctx.lineTo(toPxX(pB[0], w), toPxY(pB[1], h));
      };

      switch (mask) {
        case 1:
        case 14:
          addSeg(bottomPt, leftPt);
          break;
        case 2:
        case 13:
          addSeg(rightPt, bottomPt);
          break;
        case 3:
        case 12:
          addSeg(rightPt, leftPt);
          break;
        case 4:
        case 11:
          addSeg(topPt, rightPt);
          break;
        case 6:
        case 9:
          addSeg(topPt, bottomPt);
          break;
        case 7:
        case 8:
          addSeg(topPt, leftPt);
          break;
        case 5: {
          const center = 0.25 * (vTL + vTR + vBR + vBL);
          if (center >= level) {
            addSeg(topPt, leftPt);
            addSeg(rightPt, bottomPt);
          } else {
            addSeg(topPt, rightPt);
            addSeg(bottomPt, leftPt);
          }
          break;
        }
        case 10: {
          const center = 0.25 * (vTL + vTR + vBR + vBL);
          if (center >= level) {
            addSeg(topPt, rightPt);
            addSeg(bottomPt, leftPt);
          } else {
            addSeg(topPt, leftPt);
            addSeg(rightPt, bottomPt);
          }
          break;
        }
      }
    }
  }
  ctx.stroke();
}

function compute2DGpUncertaintyField(
  trainingX: [number, number][],
  lengthScale: number,
  res: number,
  bounds: [number, number, number, number]
): Float32Array {
  const field = new Float32Array(res * res);
  const [minX, maxX, minY, maxY] = bounds;
  const stepX = (maxX - minX) / (res - 1);
  const stepY = (maxY - minY) / (res - 1);
  const l = Math.max(0.05, lengthScale);
  const invL2 = 1.0 / (l * l);

  for (let r = 0; r < res; r++) {
    const x2 = maxY - r * stepY;
    const rowOffset = r * res;
    for (let c = 0; c < res; c++) {
      const x1 = minX + c * stepX;
      let kStarSum = 0;
      for (let i = 0; i < trainingX.length; i++) {
        const dx1 = x1 - trainingX[i][0];
        const dx2 = x2 - trainingX[i][1];
        const dSq = dx1 * dx1 + dx2 * dx2;
        kStarSum += Math.exp(-dSq * invL2);
      }
      // Normalized epistemic variance ratio: 0 near observations, 1 in data voids
      field[rowOffset + c] = Math.max(0, 1.0 - Math.min(1.0, kStarSum));
    }
  }
  return field;
}

/**
 * Evaluates explicit x2 = -(w0 / w1) * x1 - (b / w1) along hyperplane w0*x1 + w1*x2 + b = 0.
 */
export function evaluateHyperplaneY(
  normal: [number, number],
  bias: number,
  x1: number
): number {
  const [w0, w1] = normal;
  const denom = Math.abs(w1) < 1e-6 ? 1e-6 : w1;
  return -bias / denom - (w0 / denom) * x1;
}

function formatFixedSigned(val: number): string {
  return Math.abs(val) < 0.005 ? '0.00' : val.toFixed(2);
}

/**
 * Formats the displayed hyperplane / fit equation consistently with evaluateHyperplaneY.
 */
export function formatHyperplaneDiagnostic(
  normal: [number, number],
  bias: number,
  options: {
    is1DRegression: boolean;
    isRegressionDataset: boolean;
    isCrossTaskClassification: boolean;
  }
): string {
  const [w0, w1] = normal;
  const { is1DRegression, isRegressionDataset, isCrossTaskClassification } = options;

  const slope = -w0 / (Math.abs(w1) < 1e-6 ? 1e-6 : w1);
  const intercept = -bias / (Math.abs(w1) < 1e-6 ? 1e-6 : w1);
  const mStr = formatFixedSigned(slope);
  const signI = intercept >= 0 ? '+' : '-';
  const bStr = Math.abs(intercept).toFixed(2);

  if (is1DRegression || (!isCrossTaskClassification && Math.abs(w1 + 1.0) < 1e-4)) {
    return `Fit: y = ${mStr}x₁ ${signI} ${bStr}`;
  }

  const s0 = formatFixedSigned(w0);
  const sign1 = w1 >= 0 ? '+' : '-';
  const s1 = Math.abs(w1).toFixed(2);
  const signB = bias >= 0 ? '+' : '-';
  const sB = Math.abs(bias).toFixed(2);

  if (isRegressionDataset && !is1DRegression && !isCrossTaskClassification) {
    return `Fit Surface: ŷ = ${s0}x₁ ${sign1} ${s1}x₂ ${signB} ${sB}`;
  }

  if (isCrossTaskClassification) {
    return `Boundary: ${s0}x₁ ${sign1} ${s1}x₂ ${signB} ${sB} = 0  (x₂ = ${mStr}x₁ ${signI} ${bStr})`;
  }

  return `Boundary: ${s0}x₁ ${sign1} ${s1}x₂ ${signB} ${sB} = 0`;
}

export function is1DRegressionData(points: DataPoint2D[]): boolean {
  if (!points || points.length < 2) return false;
  const sample = points.slice(0, Math.min(points.length, 25));
  if (sample.every((p) => Math.abs(p.x2 - p.y) < 1e-3)) return true;
  if (sample.every((p) => Math.abs(p.x2 - 10 * p.y) < 1e-3)) return true;

  let meanX2 = 0, meanY = 0;
  for (let i = 0; i < sample.length; i++) {
    meanX2 += sample[i].x2;
    meanY += sample[i].y;
  }
  meanX2 /= sample.length;
  meanY /= sample.length;
  let varX2 = 0, varY = 0, cov = 0;
  for (let i = 0; i < sample.length; i++) {
    const dx = sample[i].x2 - meanX2;
    const dy = sample[i].y - meanY;
    varX2 += dx * dx;
    varY += dy * dy;
    cov += dx * dy;
  }
  if (varX2 > 1e-8 && varY > 1e-8) {
    const r2 = (cov * cov) / (varX2 * varY);
    if (r2 > 0.999) return true;
  }
  return false;
}

export const Stage01Canvas: React.FC<Stage01CanvasProps> = ({
  algorithm,
  taskType,
  datasetTaskType,
  theme,
  trainData,
  testData,
  artifacts,
  gridValues,
  gridResolution: _gridResolution,
  dirtyManager,
  highlightTarget = null,
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
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const mainCanvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const uncertaintyCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const getOffscreenCanvas = () => {
    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement('canvas');
    }
    return offscreenCanvasRef.current;
  };
  const getUncertaintyCanvas = () => {
    if (!uncertaintyCanvasRef.current) {
      uncertaintyCanvasRef.current = document.createElement('canvas');
    }
    return uncertaintyCanvasRef.current;
  };
  const canvasSizeRef = useRef<{ width: number; height: number }>({ width: 0, height: 0 });
  const cachedImageDataRef = useRef<ImageData | null>(null);

  const [hoverCoord, setHoverCoord] = useState<{ x: number; y: number } | null>(null);
  const [probeScore, setProbeScore] = useState<number | null>(null);
  const [draggingPointId, setDraggingPointId] = useState<number | null>(null);
  const dragStateRef = useRef<{
    pointId: number;
    startPx: { x: number; y: number };
    hasMoved: boolean;
  } | null>(null);

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
    : (taskType === 'regression' || algorithm?.taskType === 'regression');
  const isNativeRegressor =
    algorithm?.taskType === 'regression' ||
    algorithm?.id === 'knn' ||
    artifacts.some((a) => a.type === 'curve-fit' || a.type === 'epsilon-tube');
  const isCrossTaskClassification =
    isRegressionDataset && Boolean(algorithm) && !isNativeRegressor;
  const has1DRegressionCoords =
    trainData.length >= 2 ? is1DRegressionData(trainData) : is1DRegressionData(testData);
  const is1DRegression =
    !isCrossTaskClassification &&
    isRegressionDataset &&
    has1DRegressionCoords;

  // Resize handling via ResizeObserver contentRect (zero forced reflow or inline style writes)
  useEffect(() => {
    return observeElementSize(containerRef.current, ({ width, height }) => {
      canvasSizeRef.current = { width, height };
      const canvas = mainCanvasRef.current;
      if (canvas) {
        syncCanvasBufferSize(canvas, width, height, 2);
      }
      dirtyManager.markDirty();
    });
  }, [dirtyManager]);

  // Update low-res offscreen heatmap buffer
  useEffect(() => {
    const offCanvas = getOffscreenCanvas();
    if (!gridValues || gridValues.length === 0) {
      if (offCanvas.width > 0 && offCanvas.height > 0) {
        const offCtx = offCanvas.getContext('2d');
        if (offCtx) {
          offCtx.clearRect(0, 0, offCanvas.width, offCanvas.height);
        }
      }
      dirtyManager.markDirty();
      return;
    }

    // Mathematically derive buffer resolution from the actual float array length
    // to strictly prevent stride mismatch zebra/Venetian-blind moiré striping
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

  const toPxX = (x: number, w: number) => ((x - VIEWPORT.minX) / (VIEWPORT.maxX - VIEWPORT.minX)) * w;
  const toPxY = (y: number, h: number) => h - ((y - VIEWPORT.minY) / (VIEWPORT.maxY - VIEWPORT.minY)) * h;
  const toCoordX = (px: number, w: number) => VIEWPORT.minX + (px / w) * (VIEWPORT.maxX - VIEWPORT.minX);
  const toCoordY = (py: number, h: number) => VIEWPORT.maxY - (py / h) * (VIEWPORT.maxY - VIEWPORT.minY);

  // On-demand Dirty Render Loop (true 0% Idle CPU)
  useEffect(() => {
    const render = () => {
      const canvas = mainCanvasRef.current;
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) return;

      const w = canvasSizeRef.current.width;
      const h = canvasSizeRef.current.height;
      if (w <= 0 || h <= 0) return;

      // 1. Clear Background
      ctx.fillStyle = isCarbon ? '#131312' : '#FAF9F5';
      ctx.fillRect(0, 0, w, h);

      // 2. Draw Upscaled Heatmap Buffer (Only for classification and 2D surface regression)
      const offCanvas = offscreenCanvasRef.current;
      const shouldDrawHeatmap =
        !is1DRegression &&
        gridValues &&
        gridValues.length > 0 &&
        offCanvas !== null &&
        offCanvas.width > 0;

      if (shouldDrawHeatmap && offCanvas) {
        ctx.save();
        ctx.imageSmoothingEnabled = !discretize;
        ctx.imageSmoothingQuality = 'high';
        ctx.drawImage(offCanvas, 0, 0, w, h);
        ctx.restore();
      }

          // 3. Exposed 1px Swiss Grid Hairlines
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

          // Clear axis labels & units for supported dimensionality
          ctx.font = '10px "JetBrains Mono", monospace';
          ctx.fillStyle = isCarbon ? '#A8A69E' : '#575653';
          ctx.fillText('x₁', w - 18, zy - 6);
          ctx.fillText(is1DRegression ? 'y (Target)' : 'x₂', zx + 6, 16);
          ctx.restore();

          // 4. Render Active Visual Artifacts
          if (artifacts && artifacts.length > 0) {
            renderArtifacts(ctx, artifacts, w, h);
          }

          // 6. Draw Data Points (Vector Circles & Shapes)
          // Find support vector point IDs if any
          const svArtifact = artifacts.find((a) => a.type === 'support-vector-rings') as
            | Extract<VisualArtifact, { type: 'support-vector-rings' }>
            | undefined;
          const svSet = new Set(svArtifact?.pointIds || []);

          const getPointColor = (pt: DataPoint2D): string => {
            if (isRegressionDataset && !isCrossTaskClassification) {
              return isCarbon ? '#C26752' : '#D94E34';
            }
            if (!isRegressionDataset && pt.y === 2) {
              return isCarbon ? '#4D8F68' : '#15803D'; // Class +2
            }
            if (pt.y < 0) {
              return isCarbon ? '#5276B8' : '#1E40AF'; // Negative / Class -1 (y < 0)
            }
            return isCarbon ? '#C26752' : '#D94E34'; // Positive / Class +1 (y >= 0)
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
            beforeDrawPoint: (pt, px, py) => {
              const isSV = svSet.has(pt.id);
              // Support Vector Target Ring: ONLY around active support vectors!
              if (isSV || highlightTarget === 'support-vectors') {
                ctx.beginPath();
                ctx.arc(px, py, 8.0, 0, Math.PI * 2);
                ctx.strokeStyle = isCarbon ? '#D4D3CD' : '#141413';
                ctx.lineWidth = 1.6;
                ctx.stroke();
              }
            },
          });

          // 7. Reticle Probe Cursor
          if (hoverCoord) {
            const hx = toPxX(hoverCoord.x, w);
            const hy = toPxY(hoverCoord.y, h);

            ctx.save();
            ctx.strokeStyle = isCarbon ? 'rgba(212, 211, 205, 0.45)' : 'rgba(20, 20, 19, 0.35)';
            ctx.lineWidth = 1;
            ctx.setLineDash([3, 3]);

            ctx.beginPath();
            ctx.moveTo(hx, 0); ctx.lineTo(hx, h);
            ctx.moveTo(0, hy); ctx.lineTo(w, hy);
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(hx, hy, 4, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
          }

          // 8. Numeric Diagnostics Micro-Badge (Mathematical Equations & Model Metrics)
          if (artifacts && artifacts.length > 0) {
            renderNumericDiagnosticsOverlay(ctx, artifacts, w, h, isCarbon);
          }
    };

    return subscribeDirtyRender(dirtyManager, render);
  }, [
    dirtyManager,
    theme,
    trainData,
    testData,
    artifacts,
    hoverCoord,
    highlightTarget,
    isCarbon,
    effectiveShowTestData,
    discretize,
    gridValues,
    isRegressionDataset,
    isCrossTaskClassification,
    is1DRegression,
  ]);

  const renderArtifacts = (
    ctx: CanvasRenderingContext2D,
    arts: VisualArtifact[],
    w: number,
    h: number
  ) => {
    ctx.save();

    for (const art of arts) {
      if (art.type === 'hyperplane') {
        const [w0, w1] = art.normal;
        if (Math.abs(w0) < 1e-6 && Math.abs(w1) < 1e-6) continue;
        const b = art.bias;

        const calcY = (x: number) => evaluateHyperplaneY(art.normal, b, x);

        ctx.strokeStyle = art.color || (isCarbon ? '#D4D3CD' : '#141413');
        ctx.lineWidth = 2.0;
        ctx.setLineDash(art.style === 'dashed' ? [5, 4] : []);

        const yLeft = calcY(VIEWPORT.minX);
        const yRight = calcY(VIEWPORT.maxX);

        ctx.beginPath();
        ctx.moveTo(toPxX(VIEWPORT.minX, w), toPxY(yLeft, h));
        ctx.lineTo(toPxX(VIEWPORT.maxX, w), toPxY(yRight, h));
        ctx.stroke();

        const labelText =
          art.label ||
          (isCrossTaskClassification || !isRegressionDataset
            ? 'Decision Boundary'
            : !is1DRegression
            ? 'ŷ = 0 Contour'
            : undefined);
        if (labelText) {
          ctx.fillStyle = isCarbon ? '#D4D3CD' : '#141413';
          ctx.font = '9px "JetBrains Mono"';
          const midY = calcY(0);
          ctx.fillText(labelText, toPxX(0.2, w), toPxY(midY, h) - 4);
        }
      } else if (art.type === 'margin-line') {
        const [w0, w1] = art.normal;
        if (Math.abs(w0) < 1e-6 && Math.abs(w1) < 1e-6) continue;
        const b = art.bias;
        const offset = art.offset;

        const calcY = (x: number) =>
          (offset - b - w0 * x) / (Math.abs(w1) < 1e-6 ? 1e-6 : w1);

        ctx.strokeStyle = isCarbon ? '#94938E' : '#575653';
        ctx.lineWidth = 1.0;
        ctx.setLineDash([4, 4]);

        const yLeft = calcY(VIEWPORT.minX);
        const yRight = calcY(VIEWPORT.maxX);

        ctx.beginPath();
        ctx.moveTo(toPxX(VIEWPORT.minX, w), toPxY(yLeft, h));
        ctx.lineTo(toPxX(VIEWPORT.maxX, w), toPxY(yRight, h));
        ctx.stroke();

        if (art.label) {
          ctx.fillStyle = isCarbon ? '#94938E' : '#575653';
          ctx.font = '8px "JetBrains Mono"';
          ctx.fillText(art.label, toPxX(VIEWPORT.minX + 0.3, w), toPxY(calcY(VIEWPORT.minX + 0.3), h) - 3);
        }
      } else if (art.type === 'residual-bars') {
        if (!is1DRegression) continue;
        const regBarColor = isCarbon ? 'rgba(194, 103, 82, 0.85)' : 'rgba(217, 78, 52, 0.85)';
        for (const res of art.residuals) {
          const px = toPxX(res.x1, w);
          const py = toPxY(res.y, h);
          const pyHat = toPxY(res.yHat, h);

          ctx.strokeStyle = isRegressionDataset
            ? regBarColor
            : (res.y === 2
              ? (isCarbon ? '#4D8F68' : '#15803D')
              : res.y < 0
              ? (isCarbon ? '#5276B8' : '#1E40AF')
              : (isCarbon ? '#C26752' : '#D94E34'));
          ctx.lineWidth = 1.2;
          ctx.setLineDash([2, 2]);

          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(px, pyHat);
          ctx.stroke();

          // Anchor notch at yHat
          ctx.setLineDash([]);
          ctx.beginPath();
          ctx.moveTo(px - 2.5, pyHat);
          ctx.lineTo(px + 2.5, pyHat);
          ctx.stroke();
        }
      } else if (art.type === 'curve-fit') {
        if (!is1DRegression) continue;
        if (art.points.length > 1) {
          ctx.strokeStyle = art.color || (isCarbon ? '#D4D3CD' : '#141413');
          ctx.lineWidth = 2.0;
          ctx.setLineDash(art.style === 'dashed' ? [4, 4] : []);

          ctx.beginPath();
          for (let i = 0; i < art.points.length; i++) {
            const [cx, cy] = art.points[i];
            const px = toPxX(cx, w);
            const py = toPxY(cy, h);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          ctx.stroke();

          if (art.rungeWarning) {
            ctx.fillStyle = isCarbon ? '#C26752' : '#D94E34';
            ctx.font = '10px "JetBrains Mono"';
            ctx.fillText("⚠️ Runge's Phenomenon: Boundary Oscillation", 14, 24);
          }
        }
      } else if (art.type === 'epsilon-tube') {
        if (!is1DRegression) continue;
        if (art.pointsUpper.length > 1 && art.pointsLower.length > 1) {
          ctx.fillStyle = isCarbon ? 'rgba(82, 118, 184, 0.12)' : 'rgba(30, 64, 175, 0.08)';
          ctx.beginPath();
          for (let i = 0; i < art.pointsUpper.length; i++) {
            const [cx, cy] = art.pointsUpper[i];
            const px = toPxX(cx, w);
            const py = toPxY(cy, h);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          for (let i = art.pointsLower.length - 1; i >= 0; i--) {
            const [cx, cy] = art.pointsLower[i];
            const px = toPxX(cx, w);
            const py = toPxY(cy, h);
            ctx.lineTo(px, py);
          }
          ctx.closePath();
          ctx.fill();

          ctx.strokeStyle = isCarbon ? 'rgba(82, 118, 184, 0.45)' : 'rgba(30, 64, 175, 0.4)';
          ctx.lineWidth = 1.0;
          ctx.setLineDash([3, 3]);

          ctx.beginPath();
          for (let i = 0; i < art.pointsUpper.length; i++) {
            const px = toPxX(art.pointsUpper[i][0], w);
            const py = toPxY(art.pointsUpper[i][1], h);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();

          ctx.beginPath();
          for (let i = 0; i < art.pointsLower.length; i++) {
            const px = toPxX(art.pointsLower[i][0], w);
            const py = toPxY(art.pointsLower[i][1], h);
            if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
          }
          ctx.stroke();
        }
      } else if (art.type === 'uncertainty-band') {
        if (!is1DRegression) continue;
        if (art.upperBand.length > 1 && art.lowerBand.length > 1) {
          ctx.fillStyle = isCarbon ? 'rgba(194, 103, 82, 0.14)' : 'rgba(217, 78, 52, 0.10)';
          ctx.beginPath();
          for (let i = 0; i < art.upperBand.length; i++) {
            const px = toPxX(art.upperBand[i][0], w);
            const py = toPxY(art.upperBand[i][1], h);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          for (let i = art.lowerBand.length - 1; i >= 0; i--) {
            const px = toPxX(art.lowerBand[i][0], w);
            const py = toPxY(art.lowerBand[i][1], h);
            ctx.lineTo(px, py);
          }
          ctx.closePath();
          ctx.fill();

          if (art.samples) {
            ctx.lineWidth = 0.8;
            ctx.setLineDash([2, 3]);
            for (const s of art.samples) {
              ctx.strokeStyle = isCarbon ? 'rgba(212, 211, 205, 0.28)' : 'rgba(20, 20, 19, 0.22)';
              ctx.beginPath();
              for (let i = 0; i < s.length; i++) {
                const px = toPxX(s[i][0], w);
                const py = toPxY(s[i][1], h);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
              }
              ctx.stroke();
            }
          }
        }
      } else if (art.type === 'uncertainty-surface') {
        if (is1DRegression) continue;

        // 1. 2D epistemic uncertainty shading + horizon contour where data support fades
        if (art.trainingX.length > 0) {
          const uncRes = 40;
          const bounds: [number, number, number, number] = [VIEWPORT.minX, VIEWPORT.maxX, VIEWPORT.minY, VIEWPORT.maxY];
          const uncField = compute2DGpUncertaintyField(art.trainingX, art.lengthScale, uncRes, bounds);

          const uncCanvas = getUncertaintyCanvas();
          if (uncCanvas.width !== uncRes || uncCanvas.height !== uncRes) {
            uncCanvas.width = uncRes;
            uncCanvas.height = uncRes;
          }
          const uncCtx = uncCanvas.getContext('2d');
          if (uncCtx) {
            const img = uncCtx.createImageData(uncRes, uncRes);
            const data = img.data;
            const [r, g, b] = isCarbon ? [19, 19, 18] : [250, 249, 245];
            for (let i = 0; i < uncField.length; i++) {
              const u = uncField[i];
              const alpha = u > 0.65 ? Math.min(0.22, ((u - 0.65) / 0.35) * 0.22) : 0;
              const off = i * 4;
              data[off] = r;
              data[off + 1] = g;
              data[off + 2] = b;
              data[off + 3] = Math.floor(alpha * 255);
            }
            uncCtx.putImageData(img, 0, 0);
            ctx.save();
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            ctx.drawImage(uncCanvas, 0, 0, w, h);
            ctx.restore();
          }

          ctx.save();
          ctx.strokeStyle = isCarbon ? 'rgba(212, 211, 205, 0.32)' : 'rgba(20, 20, 19, 0.25)';
          ctx.lineWidth = 1.0;
          ctx.setLineDash([4, 4]);
          strokeMarchingSquaresContour(ctx, uncField, uncRes, bounds, 0.68, toPxX, toPxY, w, h);
          ctx.restore();
        }

        // 2. Crisp 2D posterior mean zero-contour μ(x₁, x₂) = 0
        if (gridValues && gridValues.length >= 4) {
          const gridRes = Math.round(Math.sqrt(gridValues.length));
          if (gridRes * gridRes === gridValues.length && gridRes >= 2) {
            ctx.save();
            ctx.strokeStyle = isCarbon ? '#D4D3CD' : '#141413';
            ctx.lineWidth = 2.0;
            ctx.setLineDash([]);
            strokeMarchingSquaresContour(
              ctx,
              gridValues,
              gridRes,
              [VIEWPORT.minX, VIEWPORT.maxX, VIEWPORT.minY, VIEWPORT.maxY],
              0,
              toPxX,
              toPxY,
              w,
              h
            );
            ctx.restore();
          }
        }
      } else if (art.type === 'decision-contour') {
        if (is1DRegression || !gridValues || gridValues.length < 4) continue;
        const gridRes = Math.round(Math.sqrt(gridValues.length));
        if (gridRes * gridRes !== gridValues.length || gridRes < 2) continue;

        const bounds: [number, number, number, number] = [VIEWPORT.minX, VIEWPORT.maxX, VIEWPORT.minY, VIEWPORT.maxY];
        const level = art.level ?? 0;

        if (art.showMargins) {
          ctx.save();
          ctx.strokeStyle = isCarbon ? 'rgba(148, 147, 142, 0.7)' : 'rgba(87, 86, 83, 0.7)';
          ctx.lineWidth = 1.0;
          ctx.setLineDash([4, 4]);
          strokeMarchingSquaresContour(ctx, gridValues, gridRes, bounds, level + 1, toPxX, toPxY, w, h);
          strokeMarchingSquaresContour(ctx, gridValues, gridRes, bounds, level - 1, toPxX, toPxY, w, h);
          ctx.restore();
        }

        ctx.save();
        ctx.strokeStyle = isCarbon ? '#D4D3CD' : '#141413';
        ctx.lineWidth = 2.0;
        ctx.setLineDash([]);
        strokeMarchingSquaresContour(ctx, gridValues, gridRes, bounds, level, toPxX, toPxY, w, h);
        ctx.restore();
      } else if (art.type === 'knn-probe-rays') {
        const probe = hoverCoord || { x: art.probePoint[0], y: art.probePoint[1] };
        const px = toPxX(probe.x, w);
        let py = toPxY(probe.y, h);

        let neighbors = art.neighbors;
        let radius = art.radius;

        if (hoverCoord && trainData.length > 0) {
          const distFn = art.metric === 'euclidean' ? euclideanDist : manhattanDist;
          const k = art.neighbors.length || 5;
          const sorted = trainData.map((pt) => ({
            id: pt.id,
            point: [pt.x1, pt.x2] as [number, number],
            distance: is1DRegression
              ? Math.abs(hoverCoord.x - pt.x1)
              : distFn(hoverCoord.x, hoverCoord.y, pt.x1, pt.x2),
            weight: 1.0,
            label: pt.y,
          })).sort((a, b) => a.distance - b.distance);
          neighbors = sorted.slice(0, Math.min(k, sorted.length));
          radius = neighbors.length > 0 ? neighbors[neighbors.length - 1].distance : 1.0;
        }

        if (is1DRegression && neighbors.length > 0) {
          const yHatProbe = hoverCoord
            ? predictValueAtPoint(probe.x, probe.y)
            : neighbors.reduce((acc, n) => acc + n.label * (n.weight || 1), 0) /
              Math.max(1e-6, neighbors.reduce((acc, n) => acc + (n.weight || 1), 0));
          py = toPxY(yHatProbe, h);
        }

        ctx.strokeStyle = isCarbon ? 'rgba(212, 211, 205, 0.35)' : 'rgba(20, 20, 19, 0.3)';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);

        if (is1DRegression) {
          const leftPx = toPxX(probe.x - radius, w);
          const rightPx = toPxX(probe.x + radius, w);
          ctx.beginPath();
          ctx.moveTo(leftPx, 0);
          ctx.lineTo(leftPx, h);
          ctx.moveTo(rightPx, 0);
          ctx.lineTo(rightPx, h);
          ctx.stroke();
        } else {
          const radPx = (radius / 12.0) * w;
          ctx.beginPath();
          if (art.metric === 'euclidean') {
            ctx.arc(px, py, radPx, 0, Math.PI * 2);
          } else {
            ctx.moveTo(px, py - radPx);
            ctx.lineTo(px + radPx, py);
            ctx.lineTo(px, py + radPx);
            ctx.lineTo(px - radPx, py);
            ctx.closePath();
          }
          ctx.stroke();
        }

        for (const n of neighbors) {
          const nx = toPxX(n.point[0], w);
          const ny = toPxY(n.point[1], h);
          if (is1DRegression) {
            ctx.strokeStyle = isCarbon ? '#C26752' : '#D94E34';
          } else {
            ctx.strokeStyle = n.label >= 0
              ? isCarbon ? '#C26752' : '#D94E34'
              : isCarbon ? '#5276B8' : '#1E40AF';
          }
          ctx.lineWidth = 1.2;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.moveTo(px, py);
          ctx.lineTo(nx, ny);
          ctx.stroke();

          const mx = (px + nx) / 2;
          const my = (py + ny) / 2;
          ctx.fillStyle = isCarbon ? '#D4D3CD' : '#141413';
          ctx.font = '8px "JetBrains Mono"';
          ctx.fillText(n.distance.toFixed(2), mx + 2, my - 2);
        }
      } else if (art.type === 'marginal-gaussians') {
        // Calculate maximum 1D density peaks to smoothly normalize curve heights without flat clipping
        let maxPeakX1 = 0.01;
        let maxPeakX2 = 0.01;
        for (const cls of art.classes) {
          const peak1 = gaussianPdf(cls.mean[0], cls.mean[0], cls.variance[0]);
          const peak2 = gaussianPdf(cls.mean[1], cls.mean[1], cls.variance[1]);
          if (peak1 > maxPeakX1) maxPeakX1 = peak1;
          if (peak2 > maxPeakX2) maxPeakX2 = peak2;
        }

        const topBaselineY = 32;
        const rightBaselineX = w - 32;

        // Subtle baseline dividers for marginal projection tracks
        ctx.save();
        ctx.strokeStyle = isCarbon ? 'rgba(212, 211, 205, 0.25)' : 'rgba(20, 20, 19, 0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);

        ctx.beginPath();
        ctx.moveTo(0, topBaselineY);
        ctx.lineTo(w, topBaselineY);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(rightBaselineX, 0);
        ctx.lineTo(rightBaselineX, h);
        ctx.stroke();

        // Sleek axis track indicators
        ctx.setLineDash([]);
        ctx.font = '8px "JetBrains Mono", monospace';
        ctx.fillStyle = isCarbon ? 'rgba(212, 211, 205, 0.6)' : 'rgba(20, 20, 19, 0.55)';
        ctx.textAlign = 'right';
        ctx.fillText('P(X₁|Y)', w - 38, topBaselineY - 5);

        ctx.save();
        ctx.translate(rightBaselineX + 16, h - 20);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'left';
        ctx.fillText('P(X₂|Y)', 0, 0);
        ctx.restore();
        ctx.restore();

        // Render curves and covariance ellipses for each class
        for (const cls of art.classes) {
          const strokeColor = cls.label === 1
            ? isCarbon ? '#E58344' : '#D94E34'
            : isCarbon ? '#4E89B8' : '#1E40AF';
          const fillColor = cls.label === 1
            ? isCarbon ? 'rgba(229, 131, 68, 0.20)' : 'rgba(217, 78, 52, 0.16)'
            : isCarbon ? 'rgba(78, 137, 184, 0.20)' : 'rgba(30, 64, 175, 0.16)';

          // 1. Top axis 1D Gaussian (X₁)
          // Filled area under curve down to baseline
          ctx.beginPath();
          ctx.moveTo(toPxX(-6.0, w), topBaselineY);
          for (let i = 0; i <= 80; i++) {
            const x = -6.0 + (i / 80) * 12.0;
            const p = gaussianPdf(x, cls.mean[0], cls.variance[0]);
            const gpx = toPxX(x, w);
            const gpy = topBaselineY - (p / maxPeakX1) * 26;
            ctx.lineTo(gpx, gpy);
          }
          ctx.lineTo(toPxX(6.0, w), topBaselineY);
          ctx.closePath();
          ctx.fillStyle = fillColor;
          ctx.fill();

          // Stroke line
          ctx.beginPath();
          for (let i = 0; i <= 80; i++) {
            const x = -6.0 + (i / 80) * 12.0;
            const p = gaussianPdf(x, cls.mean[0], cls.variance[0]);
            const gpx = toPxX(x, w);
            const gpy = topBaselineY - (p / maxPeakX1) * 26;
            if (i === 0) ctx.moveTo(gpx, gpy); else ctx.lineTo(gpx, gpy);
          }
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.4;
          ctx.stroke();

          // 2. Right axis 1D Gaussian (X₂)
          // Filled area under curve to baseline
          ctx.beginPath();
          ctx.moveTo(rightBaselineX, toPxY(-6.0, h));
          for (let i = 0; i <= 80; i++) {
            const y = -6.0 + (i / 80) * 12.0;
            const p = gaussianPdf(y, cls.mean[1], cls.variance[1]);
            const gpy = toPxY(y, h);
            const gpx = rightBaselineX + (p / maxPeakX2) * 26;
            ctx.lineTo(gpx, gpy);
          }
          ctx.lineTo(rightBaselineX, toPxY(6.0, h));
          ctx.closePath();
          ctx.fillStyle = fillColor;
          ctx.fill();

          // Stroke line
          ctx.beginPath();
          for (let i = 0; i <= 80; i++) {
            const y = -6.0 + (i / 80) * 12.0;
            const p = gaussianPdf(y, cls.mean[1], cls.variance[1]);
            const gpy = toPxY(y, h);
            const gpx = rightBaselineX + (p / maxPeakX2) * 26;
            if (i === 0) ctx.moveTo(gpx, gpy); else ctx.lineTo(gpx, gpy);
          }
          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.4;
          ctx.stroke();

          // 3. Axis-aligned covariance ellipses (1σ and 2σ equiprobability contours)
          const cx = toPxX(cls.mean[0], w);
          const cy = toPxY(cls.mean[1], h);
          const rx = (Math.sqrt(cls.variance[0]) / 12.0) * w;
          const ry = (Math.sqrt(cls.variance[1]) / 12.0) * h;

          ctx.strokeStyle = strokeColor;
          ctx.lineWidth = 1.2;

          // 1-sigma ellipse (solid dashed)
          ctx.setLineDash([4, 3]);
          ctx.beginPath();
          ctx.ellipse(cx, cy, Math.max(2, rx), Math.max(2, ry), 0, 0, Math.PI * 2);
          ctx.stroke();

          // 2-sigma ellipse (fainter dotted)
          ctx.setLineDash([2, 4]);
          ctx.beginPath();
          ctx.ellipse(cx, cy, Math.max(4, rx * 2), Math.max(4, ry * 2), 0, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      } else if (art.type === 'vector-addition') {
        const cx = w / 2;
        const cy = h / 2;
        const scale = w / 6.0;

        ctx.strokeStyle = isCarbon ? '#D4D3CD' : '#141413';
        ctx.lineWidth = 2.0;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + art.wVector[0] * scale, cy - art.wVector[1] * scale);
        ctx.stroke();

        if (Math.hypot(art.deltaVector[0], art.deltaVector[1]) > 1e-4) {
          ctx.strokeStyle = isCarbon ? '#C26752' : '#D94E34';
          ctx.lineWidth = 1.4;
          ctx.setLineDash([3, 2]);
          ctx.beginPath();
          ctx.moveTo(cx + art.wVector[0] * scale, cy - art.wVector[1] * scale);
          ctx.lineTo(
            cx + (art.wVector[0] + art.deltaVector[0]) * scale,
            cy - (art.wVector[1] + art.deltaVector[1]) * scale
          );
          ctx.stroke();
        }
      } else if (art.type === 'weight-constraint') {
        // Weight-Space (w1, w2) Inset Box (Top-Right)
        const boxW = 120;
        const boxH = 120;
        const boxX = w - boxW - 14;
        const boxY = 14;
        const cx = boxX + boxW / 2;
        const cy = boxY + boxH / 2;
        const rMax = Math.max(1.3, art.radius * 1.25);
        const scale = (boxW / 2 - 16) / rMax;

        ctx.save();
        // Inset Panel Background
        ctx.fillStyle = isCarbon ? 'rgba(24, 24, 23, 0.90)' : 'rgba(250, 249, 245, 0.92)';
        ctx.fillRect(boxX, boxY, boxW, boxH);
        ctx.strokeStyle = isCarbon ? '#3E3E3A' : '#DCDAD3';
        ctx.lineWidth = 1;
        ctx.strokeRect(boxX, boxY, boxW, boxH);

        // Header Title
        ctx.fillStyle = isCarbon ? '#A8A69E' : '#6A6862';
        ctx.font = '8px "JetBrains Mono"';
        ctx.fillText('WEIGHT SPACE (w₁, w₂)', boxX + 6, boxY + 12);

        // Coordinate Axes (w1 horizontal, w2 vertical)
        ctx.strokeStyle = isCarbon ? 'rgba(212, 211, 205, 0.25)' : 'rgba(20, 20, 19, 0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(boxX + 6, cy);
        ctx.lineTo(boxX + boxW - 6, cy);
        ctx.moveTo(cx, boxY + 18);
        ctx.lineTo(cx, boxY + boxH - 6);
        ctx.stroke();

        // Constraint Ball / Diamond
        const rPx = art.radius * scale;
        if (art.constraintType === 'l2-circle') {
          ctx.fillStyle = isCarbon ? 'rgba(82, 118, 184, 0.18)' : 'rgba(30, 64, 175, 0.12)';
          ctx.strokeStyle = isCarbon ? '#5276B8' : '#1E40AF';
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(cx, cy, Math.max(2, rPx), 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        } else if (art.constraintType === 'l1-diamond') {
          ctx.fillStyle = isCarbon ? 'rgba(194, 103, 82, 0.18)' : 'rgba(217, 78, 52, 0.12)';
          ctx.strokeStyle = isCarbon ? '#C26752' : '#D94E34';
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.moveTo(cx + rPx, cy);
          ctx.lineTo(cx, cy - rPx);
          ctx.lineTo(cx - rPx, cy);
          ctx.lineTo(cx, cy + rPx);
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        } else if (art.constraintType === 'elastic-net') {
          const alpha = art.l1Ratio ?? 0.5;
          ctx.fillStyle = isCarbon ? 'rgba(200, 140, 60, 0.18)' : 'rgba(180, 110, 30, 0.12)';
          ctx.strokeStyle = isCarbon ? '#D97706' : '#B45309';
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          const steps = 48;
          for (let s = 0; s <= steps; s++) {
            const theta = (s / steps) * Math.PI * 2;
            const cosT = Math.cos(theta);
            const sinT = Math.sin(theta);
            const rDiamond = 1.0 / (Math.abs(cosT) + Math.abs(sinT));
            const rCircle = 1.0;
            const rMorph = (1.0 - alpha) * rCircle + alpha * rDiamond;
            const curR = art.radius * rMorph * scale;
            const px = cx + curR * cosT;
            const py = cy - curR * sinT;
            if (s === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          }
          ctx.closePath();
          ctx.fill();
          ctx.stroke();
        }

        // Current Weight Vector (w1, w2)
        const [w0, w1] = art.currentWeight;
        const wx = cx + w0 * scale;
        const wy = cy - w1 * scale;

        ctx.strokeStyle = isCarbon ? '#FAF9F5' : '#141413';
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(wx, wy);
        ctx.stroke();

        ctx.fillStyle = isCarbon ? '#FAF9F5' : '#141413';
        ctx.beginPath();
        ctx.arc(wx, wy, 3.2, 0, Math.PI * 2);
        ctx.fill();

        // Footer readout
        ctx.fillStyle = isCarbon ? '#A8A69E' : '#6A6862';
        ctx.font = '7.5px "JetBrains Mono"';
        ctx.fillText(`w:[${w0.toFixed(2)}, ${w1.toFixed(2)}]`, boxX + 6, boxY + boxH - 4);

        ctx.restore();
      }
    }

    ctx.restore();
  };

  const renderNumericDiagnosticsOverlay = (
    ctx: CanvasRenderingContext2D,
    arts: VisualArtifact[],
    _w: number,
    _h: number,
    isDark: boolean
  ) => {
    if (!arts || arts.length === 0) return;

    const lines: string[] = [];

    const hp = arts.find((a) => a.type === 'hyperplane') as Extract<VisualArtifact, { type: 'hyperplane' }> | undefined;
    if (hp) {
      lines.push(
        formatHyperplaneDiagnostic(hp.normal, hp.bias, {
          is1DRegression,
          isRegressionDataset,
          isCrossTaskClassification,
        })
      );
    }

    const eps = is1DRegression
      ? (arts.find((a) => a.type === 'epsilon-tube') as Extract<VisualArtifact, { type: 'epsilon-tube' }> | undefined)
      : undefined;
    if (eps) {
      lines.push(`Corridor: |y - f(x)| ≤ ${eps.epsilon.toFixed(2)}`);
    }

    const dc = arts.find((a) => a.type === 'decision-contour') as Extract<VisualArtifact, { type: 'decision-contour' }> | undefined;
    if (dc) {
      lines.push(dc.label || 'Boundary: f(x) = 0');
    }

    const sv = arts.find((a) => a.type === 'support-vector-rings') as Extract<VisualArtifact, { type: 'support-vector-rings' }> | undefined;
    if (sv && sv.pointIds.length > 0) {
      lines.push(`Support Vectors: N_sv = ${sv.pointIds.length}`);
    }

    const gp = is1DRegression
      ? (arts.find((a) => a.type === 'uncertainty-band') as Extract<VisualArtifact, { type: 'uncertainty-band' }> | undefined)
      : undefined;
    if (gp) {
      lines.push(`Posterior: μ(x*) ± 2σ(x*) [95.4% CI]`);
    }

    const gp2d = !is1DRegression
      ? (arts.find((a) => a.type === 'uncertainty-surface') as Extract<VisualArtifact, { type: 'uncertainty-surface' }> | undefined)
      : undefined;
    if (gp2d) {
      lines.push(
        isRegressionDataset && !isCrossTaskClassification
          ? `Fit Surface: GP μ(x₁, x₂) · ±σ(x₁, x₂)`
          : `Boundary: GP Posterior μ(x₁, x₂) = 0`
      );
      lines.push(`Epistemic σ(x₁, x₂): l=${gp2d.lengthScale.toFixed(2)}, σ_f²=${gp2d.signalVar.toFixed(2)}`);
    }

    const cf = arts.find((a) => a.type === 'curve-fit') as Extract<VisualArtifact, { type: 'curve-fit' }> | undefined;
    if (cf && cf.label && !lines.some((l) => l.startsWith('Fit:')) && (is1DRegression || cf.points.length === 0)) {
      lines.push(cf.label);
    }

    const knn = arts.find((a) => a.type === 'knn-probe-rays') as Extract<VisualArtifact, { type: 'knn-probe-rays' }> | undefined;
    if (knn) {
      lines.push(
        is1DRegression
          ? `k-NN: 1D |Δx₁|, Radius=${knn.radius.toFixed(2)}`
          : `k-NN: Metric=${knn.metric}, Radius=${knn.radius.toFixed(2)}`
      );
    }

    const mg = arts.find((a) => a.type === 'marginal-gaussians') as Extract<VisualArtifact, { type: 'marginal-gaussians' }> | undefined;
    if (mg && mg.classes.length > 0) {
      lines.push(`Bayes: ${mg.classes.map((c) => `C${c.label}: μ=(${c.mean[0].toFixed(1)}, ${c.mean[1].toFixed(1)})`).join(' | ')}`);
    }

    if (lines.length === 0) return;

    ctx.save();
    ctx.font = '10px "JetBrains Mono", monospace';
    const padX = 8;
    const padY = 5;
    const lineHeight = 14;
    const badgeH = lines.length * lineHeight + padY * 2;
    let maxW = 0;
    for (const l of lines) {
      const tw = ctx.measureText(l).width;
      if (tw > maxW) maxW = tw;
    }
    const badgeW = maxW + padX * 2 + 16;
    const badgeX = 14;
    const badgeY = 14;

    ctx.fillStyle = isDark ? 'rgba(26, 26, 25, 0.88)' : 'rgba(250, 249, 245, 0.92)';
    ctx.strokeStyle = isDark ? 'rgba(212, 211, 205, 0.22)' : 'rgba(20, 20, 19, 0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(badgeX, badgeY, badgeW, badgeH, 3);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = isDark ? '#C26752' : '#D94E34';
    ctx.beginPath();
    ctx.arc(badgeX + 10, badgeY + padY + 7, 2.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = isDark ? '#D4D3CD' : '#141413';
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], badgeX + 18, badgeY + padY + 10 + i * lineHeight);
    }
    ctx.restore();
  };

  const handlePointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const x = toCoordX(px, rect.width);
    const y = toCoordY(py, rect.height);

    const visiblePoints = getVisibleDataPoints(trainData, testData, effectiveShowTestData);
    const hit = hitTestDataPoints(visiblePoints, {
      px,
      py,
      width: rect.width,
      height: rect.height,
      hitRadiusPx: 10,
      toPxX,
      toPxY,
    });

    const is1D = is1DRegression;

    if (e.shiftKey || pointerMode === 'delete') {
      if (hit) {
        onDeletePoint(hit.id);
        dirtyManager.markDirty();
      }
      return;
    }

    if (e.altKey || pointerMode === 'flip') {
      if (hit) {
        if (is1D) {
          // In 1D regression, invert both target y and vertical coordinate x2
          onUpdatePoint({ ...hit, x2: -hit.x2, y: -hit.y });
        } else {
          const nextY = hit.y === 1 ? -1 : hit.y === -1 ? 2 : 1;
          onUpdatePoint({ ...hit, y: nextY });
        }
        dirtyManager.markDirty();
      }
      return;
    }

    const isForceAdd = e.ctrlKey || e.metaKey;

    if (hit && !isForceAdd) {
      dragStateRef.current = {
        pointId: hit.id,
        startPx: { x: px, y: py },
        hasMoved: false,
      };
      setDraggingPointId(hit.id);
    } else {
      let addX1 = x;
      let addX2 = y;

      // Small Jitter Offset: when adding on top of / within proximity of an existing point (or via Ctrl/Cmd),
      // apply a micro-offset (±0.08 coordinate units) so both points are visibly side-by-side
      const nearby = hit || visiblePoints.find((p) => Math.hypot(p.x1 - x, p.x2 - y) < 0.25);
      if (nearby) {
        const offset = 0.08;
        const dx = (Math.random() > 0.5 ? 1 : -1) * offset;
        const dy = (Math.random() > 0.5 ? 1 : -1) * offset;
        addX1 = nearby.x1 + dx;
        addX2 = is1D ? nearby.x2 : nearby.x2 + dy;
      }

      if (is1D) {
        // In 1D regression curves, vertical coordinate represents target y.
        // Match sign to activeClass:
        // -1 -> blue (negative)
        // +1 -> orange (positive)
        // +2 -> green (positive)
        let targetY = addX2;
        if (activeClass === -1) {
          targetY = -Math.abs(addX2 === 0 ? 1 : addX2);
        } else {
          targetY = Math.abs(addX2 === 0 ? 1 : addX2);
        }
        onAddPoint({ x1: addX1, x2: targetY, y: targetY });
      } else {
        onAddPoint({ x1: addX1, x2: addX2, y: activeClass });
      }
      dirtyManager.markDirty();
    }
  };

  const handlePointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = mainCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;

    const x = toCoordX(px, rect.width);
    const y = toCoordY(py, rect.height);

    setHoverCoord({ x, y });
    setProbeScore(predictValueAtPoint(x, y));

    if (dragStateRef.current) {
      const dist = Math.hypot(px - dragStateRef.current.startPx.x, py - dragStateRef.current.startPx.y);
      if (!dragStateRef.current.hasMoved && dist > 3) {
        dragStateRef.current.hasMoved = true;
        if (onDragStart) onDragStart();
      }

      if (dragStateRef.current.hasMoved) {
        const visiblePoints = getVisibleDataPoints(trainData, testData, effectiveShowTestData);
        const existing = visiblePoints.find((p) => p.id === dragStateRef.current?.pointId);
        if (existing) {
          const is1D = is1DRegression;
          if (is1D) {
            onUpdatePoint({ ...existing, x1: x, x2: y, y: y });
          } else {
            onUpdatePoint({ ...existing, x1: x, x2: y });
          }
          dirtyManager.markDirty();
        }
      }
    }
    dirtyManager.markDirty();
  };

  const handlePointerUp = () => {
    if (dragStateRef.current) {
      if (dragStateRef.current.hasMoved && onDragEnd) {
        onDragEnd();
      }
      dragStateRef.current = null;
    }
    if (draggingPointId !== null) {
      setDraggingPointId(null);
    }
  };

  const handlePointerLeave = () => {
    setHoverCoord(null);
    setProbeScore(null);
    if (dragStateRef.current) {
      if (dragStateRef.current.hasMoved && onDragEnd) {
        onDragEnd();
      }
      dragStateRef.current = null;
    }
    if (draggingPointId !== null) {
      setDraggingPointId(null);
    }
    dirtyManager.markDirty();
  };

  return (
    <div className="flex flex-col w-full h-full min-h-0 min-w-0 overflow-hidden bg-[var(--bg-primary)]">
      {/* 1. Main Canvas Area */}
      <div ref={containerRef} className="canvas-container relative flex-1 min-h-0 min-w-0 overflow-hidden">
        <canvas
          ref={mainCanvasRef}
          className="canvas-main block w-full h-full cursor-crosshair"
          role="img"
          aria-label="Interactive 2D Continuous Machine Learning Data Plane"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerLeave={handlePointerLeave}
        />

        {/* Hover Coordinate Probe HUD (Docked top-right to prevent overlap with top-left model diagnostics) */}
        {hoverCoord && (
          <div
            className={`absolute ${
              hoverCoord.x > 2.0 && hoverCoord.y > 2.0 ? 'bottom-3 left-3' : 'top-3 right-3'
            } pointer-events-none z-10 flex items-center gap-2 font-mono text-[11px] bg-[var(--bg-elevated)]/95 backdrop-blur-sm px-2.5 py-1 border border-[var(--border-hairline)] text-[var(--text-primary)] shadow-sm rounded-xs transition-all`}
          >
            <span>X₁: {hoverCoord.x.toFixed(2)}</span>
            <span className="text-[var(--text-muted)]">|</span>
            <span>{is1DRegression ? 'y' : 'X₂'}: {hoverCoord.y.toFixed(2)}</span>
            {probeScore !== null && (
              <>
                <span className="text-[var(--text-muted)]">|</span>
                <span>{is1DRegression ? 'ŷ' : 'f(x)'}: {probeScore.toFixed(3)}</span>
                {is1DRegression && (
                  <>
                    <span className="text-[var(--text-muted)]">|</span>
                    <span>e: {(hoverCoord.y - probeScore).toFixed(3)}</span>
                  </>
                )}
                {(!isRegressionDataset || isCrossTaskClassification) && (
                  <>
                    <span className="text-[var(--text-muted)]">|</span>
                    <span
                      style={{
                        color:
                          probeScore >= 0
                            ? isCarbon ? '#C26752' : '#D94E34'
                            : isCarbon ? '#5276B8' : '#1E40AF',
                      }}
                    >
                      Class {probeScore >= 0 ? '+1' : '-1'}
                    </span>
                  </>
                )}
              </>
            )}
          </div>
        )}
      </div>

      {/* 2. Bottom Bar: Swiss Color Legend & Discretize Toggle */}
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
          ) : isRegressionDataset && !isCrossTaskClassification ? (
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
              <span className="text-[9px] truncate mr-2" title="Colors show data & decision boundary confidence values:">
                Data & boundary confidence:
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
  );
};
