import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { DataPoint2D, PointerToolMode } from '../datasets/tabular2d';
import { AlgorithmPlugin, VisualArtifact, TaskType } from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { DirtyManager } from './dirty-manager';
import {
  observeElementSize,
  subscribeDirtyRender,
  syncCanvasBufferSize,
} from './canvasUtils';
import { populateHeatmapImageData } from './color-maps';
import { Plus, Minus, Zap, Compass, Move } from 'lucide-react';
import { BASIS_FEATURES, BasisFeature } from '../algorithms/stage04/neuralUtils';
import { getVisibleDataPoints, hitTestDataPoints, renderDataPoints2D } from './pointUtils';

export interface Stage04CanvasProps {
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
const DEFAULT_LATENT_BOUND = 3.0;
const X2_DEPENDENT_FEATURES: readonly BasisFeature[] = ['x2', 'x2_sq', 'x1_x2', 'sin_x2'];

export const Stage04Canvas: React.FC<Stage04CanvasProps> = ({
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
  const insetCanvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const getOffscreenCanvas = () => {
    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement('canvas');
    }
    return offscreenCanvasRef.current;
  };
  const mainSizeRef = useRef<{ width: number; height: number }>({ width: 0, height: 0 });
  const insetSizeRef = useRef<{ width: number; height: number }>({ width: 0, height: 0 });
  const cachedImageDataRef = useRef<ImageData | null>(null);

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

  // Interactive Synapse hover tooltip state
  const [hoveredSynapse, setHoveredSynapse] = useState<{
    syn: Extract<VisualArtifact, { type: 'neural-graph-overlay' }>['synapses'][0];
    x: number;
    y: number;
  } | null>(null);

  // Magnified neuron overlay
  const [magnifiedNeuron, setMagnifiedNeuron] = useState<{ layer: number; neuron: number } | null>(null);

  // Latent probe state
  const [probeZ, setProbeZ] = useState<[number, number]>([0, 0]);
  const [isDraggingProbe, setIsDraggingProbe] = useState(false);
  const [draggingPointId, setDraggingPointId] = useState<number | null>(null);

  const isCarbon = theme === 'carbon';
  const isMLP = algorithm.id === 'mlp';
  const isAutoencoder = algorithm.id === 'autoencoder';
  const isVAE = algorithm.id === 'vae';
  const isBottleneck = isAutoencoder || isVAE;
  const isGAN = algorithm.id === 'gan';

  const isRegressionDataset = datasetTaskType !== undefined
    ? datasetTaskType === 'regression'
    : trainData.some((p) => p.y !== 1 && p.y !== -1 && p.y !== 2);
  const is1DRegression =
    isRegressionDataset &&
    trainData.length > 0 &&
    trainData.slice(0, 10).every((p) => Math.abs(p.x2 - p.y) < 1e-3);

  const getPointColor = useCallback((y: number): string => {
    if (isRegressionDataset) {
      return isCarbon ? '#C26752' : '#D94E34'; // Uniform terracotta matching Stages 01 and 02
    }
    if (y === 2) return isCarbon ? '#4D8F68' : '#15803D'; // 3-Class green
    if (y < 0) return isCarbon ? '#4E89B8' : '#1E40AF'; // Class -1 blue
    return isCarbon ? '#E58344' : '#D94E34'; // Class +1 orange
  }, [isRegressionDataset, isCarbon]);

  // Extract Visual Artifacts
  const neuralArtifact = useMemo(
    () =>
      artifacts.find((a) => a.type === 'neural-graph-overlay') as
        | Extract<VisualArtifact, { type: 'neural-graph-overlay' }>
        | undefined,
    [artifacts]
  );

  const latentArtifact = useMemo(
    () =>
      artifacts.find((a) => a.type === 'latent-bottleneck-overlay') as
        | Extract<VisualArtifact, { type: 'latent-bottleneck-overlay' }>
        | undefined,
    [artifacts]
  );

  const ganArtifact = useMemo(
    () =>
      artifacts.find((a) => a.type === 'gan-minimax-overlay') as
        | Extract<VisualArtifact, { type: 'gan-minimax-overlay' }>
        | undefined,
    [artifacts]
  );

  // Active features list parsed
  const activeFeatures = useMemo(() => {
    return (hyperparams.features ? String(hyperparams.features).split(',') : ['x1', 'x2'])
      .map((s) => s.trim().toLowerCase())
      .filter((s): s is BasisFeature => (BASIS_FEATURES as readonly string[]).includes(s));
  }, [hyperparams.features]);

  // Coordinates helper (Full Canvas Mapping matching Stages 01-03)
  const toPxX = (x: number, w: number, _h: number = w) => {
    return ((x - VIEWPORT.minX) / (VIEWPORT.maxX - VIEWPORT.minX)) * w;
  };
  const toPxY = (y: number, _w: number, h: number = _w) => {
    return ((VIEWPORT.maxY - y) / (VIEWPORT.maxY - VIEWPORT.minY)) * h;
  };
  const toCoordX = (px: number, w: number, _h: number = w) => {
    return VIEWPORT.minX + (px / w) * (VIEWPORT.maxX - VIEWPORT.minX);
  };
  const toCoordY = (py: number, _w: number, h: number = _w) => {
    return VIEWPORT.maxY - (py / h) * (VIEWPORT.maxY - VIEWPORT.minY);
  };

  // Dynamic auto-scaled 2D Latent Bottleneck viewport so encoded points never clip
  const latentViewport = useMemo(() => {
    let maxAbs = DEFAULT_LATENT_BOUND;
    if (latentArtifact) {
      for (const ep of latentArtifact.encodedPoints) {
        const padX = latentArtifact.isVAE && ep.sigma1 ? ep.sigma1 : 0;
        const padY = latentArtifact.isVAE && ep.sigma2 ? ep.sigma2 : 0;
        if (Number.isFinite(ep.z1)) maxAbs = Math.max(maxAbs, Math.abs(ep.z1) + padX);
        if (Number.isFinite(ep.z2)) maxAbs = Math.max(maxAbs, Math.abs(ep.z2) + padY);
      }
    }
    const bound = maxAbs > DEFAULT_LATENT_BOUND ? maxAbs * 1.15 : DEFAULT_LATENT_BOUND;
    return { minZ: -bound, maxZ: bound };
  }, [latentArtifact]);

  const toLatentPxX = useCallback(
    (z1: number, w: number, _h: number = w) => {
      return ((z1 - latentViewport.minZ) / (latentViewport.maxZ - latentViewport.minZ)) * w;
    },
    [latentViewport]
  );
  const toLatentPxY = useCallback(
    (z2: number, _w: number, h: number = _w) => {
      return ((latentViewport.maxZ - z2) / (latentViewport.maxZ - latentViewport.minZ)) * h;
    },
    [latentViewport]
  );
  const toLatentCoordX = useCallback(
    (px: number, w: number, _h: number = w) => {
      return latentViewport.minZ + (px / w) * (latentViewport.maxZ - latentViewport.minZ);
    },
    [latentViewport]
  );
  const toLatentCoordY = useCallback(
    (py: number, _w: number, h: number = _w) => {
      return latentViewport.maxZ - (py / h) * (latentViewport.maxZ - latentViewport.minZ);
    },
    [latentViewport]
  );

  // Resize Handling via ResizeObserver contentRect (zero forced reflow)
  useEffect(() => {
    const cleanups = [
      observeElementSize(mainCanvasRef.current, ({ width, height }) => {
        mainSizeRef.current = { width, height };
        const c = mainCanvasRef.current;
        if (c) syncCanvasBufferSize(c, width, height, 2);
        dirtyManager.markDirty();
      }),
      observeElementSize(insetCanvasRef.current, ({ width, height }) => {
        insetSizeRef.current = { width, height };
        const c = insetCanvasRef.current;
        if (c) syncCanvasBufferSize(c, width, height, 2);
        dirtyManager.markDirty();
      }),
    ];
    return () => cleanups.forEach((fn) => fn());
  }, [dirtyManager, algorithm.id, isBottleneck, isGAN, isMLP]);

  // Update Low-Res Offscreen Canvas Heatmap
  useEffect(() => {
    const off = getOffscreenCanvas();
    if (!gridValues || gridValues.length === 0) {
      if (off.width > 0 && off.height > 0) {
        const offCtx = off.getContext('2d');
        offCtx?.clearRect(0, 0, off.width, off.height);
      }
      dirtyManager.markDirty();
      return;
    }

    const actualRes = Math.max(1, Math.round(Math.sqrt(gridValues.length)));

    if (off.width !== actualRes || off.height !== actualRes) {
      off.width = actualRes;
      off.height = actualRes;
      cachedImageDataRef.current = off.getContext('2d')?.createImageData(actualRes, actualRes) || null;
    }

    const offCtx = off.getContext('2d');
    if (!offCtx) return;

    if (!cachedImageDataRef.current) {
      cachedImageDataRef.current = offCtx.createImageData(actualRes, actualRes);
    }

    populateHeatmapImageData(cachedImageDataRef.current, gridValues, theme, discretize);
    offCtx.putImageData(cachedImageDataRef.current, 0, 0);
    dirtyManager.markDirty();
  }, [gridValues, theme, discretize, dirtyManager]);

  // On-demand Dirty Render Loop (0% Idle CPU)
  useEffect(() => {
    const render = () => {
      // 1. Render Main Output Canvas
      const mainCanvas = mainCanvasRef.current;
      if (mainCanvas) {
        const w = mainSizeRef.current.width;
        const h = mainSizeRef.current.height;
        const ctx = mainCanvas.getContext('2d');

        if (ctx && w > 0 && h > 0) {
          ctx.clearRect(0, 0, w, h);

          // Background
          ctx.fillStyle = isCarbon ? '#131312' : '#FAF9F5';
          ctx.fillRect(0, 0, w, h);

          // Heatmap from offscreen canvas (Only for classification and 2D surface regression; suppressed for 1D regression and unsupervised AE/VAE)
          const shouldDrawHeatmap = !is1DRegression && !isBottleneck;
          const off = offscreenCanvasRef.current;
          if (shouldDrawHeatmap && off && off.width > 0 && off.height > 0) {
            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = 'high';
            // Heatmap fills full canvas matching Stages 01-03
            ctx.drawImage(off, 0, 0, w, h);
          }

          // Gridlines across the entire canvas
          ctx.strokeStyle = isCarbon ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
          ctx.lineWidth = 1;
          for (let x = -4; x <= 4; x += 2) {
            if (x === 0) continue;
            const px = Math.floor(toPxX(x, w, h)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(px, 0);
            ctx.lineTo(px, h);
            ctx.stroke();
          }
          for (let y = -4; y <= 4; y += 2) {
            if (y === 0) continue;
            const py = Math.floor(toPxY(y, w, h)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(0, py);
            ctx.lineTo(w, py);
            ctx.stroke();
          }

          // Axes
          ctx.strokeStyle = isCarbon ? '#2E2E2B' : '#DCDAD3';
          ctx.lineWidth = 1;
          const zeroX = Math.floor(toPxX(0, w, h)) + 0.5;
          const zeroY = Math.floor(toPxY(0, w, h)) + 0.5;
          ctx.beginPath();
          ctx.moveTo(zeroX, 0);
          ctx.lineTo(zeroX, h);
          ctx.moveTo(0, zeroY);
          ctx.lineTo(w, zeroY);
          ctx.stroke();

          // Overlay Magnified Neuron Activation if hovering a hidden node
          if (magnifiedNeuron && neuralArtifact) {
            const { layer, neuron } = magnifiedNeuron;
            const targetNeuron = neuralArtifact.layers[layer]?.neurons[neuron];
            if (targetNeuron?.activationPreview) {
              const preview = targetNeuron.activationPreview;
              const res = 24;
              const cellW = w / res;
              const cellH = h / res;
              for (let r = 0; r < res; r++) {
                for (let c = 0; c < res; c++) {
                  const val = preview[r * res + c];
                  const alpha = Math.min(0.4, Math.abs(val) * 0.4);
                  ctx.fillStyle =
                    val > 0
                      ? isCarbon
                        ? `rgba(229, 131, 68, ${alpha})`
                        : `rgba(200, 90, 23, ${alpha})`
                      : isCarbon
                      ? `rgba(78, 137, 184, ${alpha})`
                      : `rgba(43, 91, 132, ${alpha})`;
                  ctx.fillRect(c * cellW, r * cellH, cellW + 0.5, cellH + 0.5);
                }
              }
            }
          }

          // Overlay Mode B (Autoencoder / VAE) Reconstruction Vectors & PCA Axis
          if (latentArtifact) {
            // PCA Axis if Linear AE 1D
            if (latentArtifact.pca1Axis) {
              const { center, direction, label } = latentArtifact.pca1Axis;
              const span = 9.0;
              const p1x = toPxX(center[0] - direction[0] * span, w, h);
              const p1y = toPxY(center[1] - direction[1] * span, w, h);
              const p2x = toPxX(center[0] + direction[0] * span, w, h);
              const p2y = toPxY(center[1] + direction[1] * span, w, h);

              ctx.save();
              ctx.strokeStyle = isCarbon ? '#D97706' : '#B45309';
              ctx.lineWidth = 1.5;
              ctx.setLineDash([4, 4]);
              ctx.beginPath();
              ctx.moveTo(p1x, p1y);
              ctx.lineTo(p2x, p2y);
              ctx.stroke();

              ctx.fillStyle = isCarbon ? '#D97706' : '#B45309';
              ctx.font = '9px "JetBrains Mono"';
              ctx.fillText(label, p2x - 120, p2y - 6);
              ctx.restore();
            }

            // Reconstruction Vectors
            ctx.save();
            for (const vec of latentArtifact.reconstructionVectors) {
              const ox = toPxX(vec.xOriginal[0], w, h);
              const oy = toPxY(vec.xOriginal[1], w, h);
              const rx = toPxX(vec.xReconstructed[0], w, h);
              const ry = toPxY(vec.xReconstructed[1], w, h);

              ctx.strokeStyle = isCarbon ? 'rgba(255,255,255,0.22)' : 'rgba(0,0,0,0.22)';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(ox, oy);
              ctx.lineTo(rx, ry);
              ctx.stroke();

              // Decoded endpoint dot
              ctx.fillStyle = isCarbon ? '#A8A7A2' : '#575653';
              ctx.beginPath();
              ctx.arc(rx, ry, 2, 0, Math.PI * 2);
              ctx.fill();
            }
            ctx.restore();

            // Decoded Live Probe marker (✦)
            if (latentArtifact.probePoint) {
              const dx = toPxX(latentArtifact.probePoint.decoded[0], w, h);
              const dy = toPxY(latentArtifact.probePoint.decoded[1], w, h);

              ctx.save();
              ctx.fillStyle = isCarbon ? '#F59E0B' : '#D97706';
              ctx.strokeStyle = isCarbon ? '#131312' : '#FFFFFF';
              ctx.lineWidth = 2;
              ctx.beginPath();
              ctx.arc(dx, dy, 7, 0, Math.PI * 2);
              ctx.fill();
              ctx.stroke();

              // Crosshair
              ctx.strokeStyle = isCarbon ? '#F59E0B' : '#D97706';
              ctx.lineWidth = 1.5;
              ctx.beginPath();
              ctx.moveTo(dx - 12, dy);
              ctx.lineTo(dx + 12, dy);
              ctx.moveTo(dx, dy - 12);
              ctx.lineTo(dx, dy + 12);
              ctx.stroke();

              ctx.font = '10px "JetBrains Mono"';
              ctx.fillText(
                `D(z) = (${latentArtifact.probePoint.decoded[0].toFixed(2)}, ${latentArtifact.probePoint.decoded[1].toFixed(2)})`,
                dx + 10,
                dy - 8
              );
              ctx.restore();
            }
          }

          // Overlay Mode C (GAN) Gradient Pull Vectors & Fake Points
          if (ganArtifact) {
            ctx.save();
            for (const fp of ganArtifact.fakePoints) {
              const fx = toPxX(fp.x1, w, h);
              const fy = toPxY(fp.x2, w, h);

              // Motion trail
              if (fp.prevX1 !== undefined && fp.prevX2 !== undefined) {
                const px = toPxX(fp.prevX1, w, h);
                const py = toPxY(fp.prevX2, w, h);
                ctx.strokeStyle = isCarbon ? 'rgba(217, 119, 6, 0.4)' : 'rgba(180, 83, 9, 0.4)';
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(px, py);
                ctx.lineTo(fx, fy);
                ctx.stroke();
              }

              // Gradient Pull Vector
              const arrowScale = 1.2;
              const vx = fp.pullVector[0] * arrowScale;
              const vy = fp.pullVector[1] * arrowScale;
              const endX = toPxX(fp.x1 + vx, w, h);
              const endY = toPxY(fp.x2 + vy, w, h);

              ctx.strokeStyle = isCarbon ? '#10B981' : '#059669';
              ctx.lineWidth = 1.2;
              ctx.beginPath();
              ctx.moveTo(fx, fy);
              ctx.lineTo(endX, endY);
              ctx.stroke();

              // Fake diamond marker
              ctx.fillStyle = isCarbon ? '#D97706' : '#B45309';
              ctx.strokeStyle = isCarbon ? '#FFFFFF' : '#141413';
              ctx.lineWidth = 1;
              ctx.beginPath();
              ctx.moveTo(fx, fy - 4);
              ctx.lineTo(fx + 4, fy);
              ctx.lineTo(fx, fy + 4);
              ctx.lineTo(fx - 4, fy);
              ctx.closePath();
              ctx.fill();
              ctx.stroke();
            }
            ctx.restore();
          }

          // Data Points (Train Solid Dots & Test Hollow Rings, if showTestData is enabled)
          renderDataPoints2D(ctx, {
            trainData,
            testData,
            showTestData: effectiveShowTestData,
            width: w,
            height: h,
            toPxX: (x1, width) => toPxX(x1, width, h),
            toPxY: (x2, height) => toPxY(x2, w, height),
            getColor: (pt) => getPointColor(pt.y),
            isCarbon,
            trainRadius: 4.5,
            testRadius: 4.0,
            trainStrokeColor: isCarbon ? '#131312' : '#FAF9F5',
            trainStrokeWidth: 1.0,
            testStrokeWidth: 1.5,
            innerDotRadius: 0,
          });

          // 1D Regression Fitted Response Curve & Residual Pins (for MLP)
          if (is1DRegression && isMLP && predictValueAtPoint) {
            ctx.save();
            // 1. Dotted vertical residual pins
            for (let i = 0; i < trainData.length; i++) {
              const p = trainData[i];
              const px = toPxX(p.x1, w, h);
              const py = toPxY(p.x2, w, h);
              const yHat = predictValueAtPoint(p.x1, 0);
              const pyHat = toPxY(yHat, w, h);

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

            // 2. Continuous 1D regression curve
            const numSamples = Math.min(400, Math.floor(w));
            ctx.strokeStyle = isCarbon ? '#FAF9F5' : '#141413';
            ctx.lineWidth = 2.0;
            ctx.setLineDash([]);
            ctx.beginPath();
            for (let s = 0; s <= numSamples; s++) {
              const currX = VIEWPORT.minX + (s / numSamples) * (VIEWPORT.maxX - VIEWPORT.minX);
              const currYHat = predictValueAtPoint(currX, 0);
              const px = toPxX(currX, w, h);
              const py = toPxY(currYHat, w, h);
              if (s === 0) ctx.moveTo(px, py);
              else ctx.lineTo(px, py);
            }
            ctx.stroke();
            ctx.restore();
          }
        }
      }

      // 2. Render Inset Canvas (Mode B Latent Pad OR Mode C GAN Inset)
      const insetCanvas = insetCanvasRef.current;
      if (insetCanvas) {
        const iw = insetSizeRef.current.width;
        const ih = insetSizeRef.current.height;
        const ctx = insetCanvas.getContext('2d');

        if (ctx && iw > 0 && ih > 0) {
          ctx.clearRect(0, 0, iw, ih);

          // Background
          ctx.fillStyle = isCarbon ? '#181817' : '#EFECE6';
          ctx.fillRect(0, 0, iw, ih);

          if (isBottleneck) {
            // Mode B: 2D Latent Bottleneck Pad (Auto-scaled symmetric viewport)
            const lcx = iw / 2;
            const lcy = ih / 2;
            const spanZ = latentViewport.maxZ - latentViewport.minZ;

            // Prior N(0, I) unit contours (1-sigma and 2-sigma)
            ctx.save();
            ctx.strokeStyle = isCarbon ? 'rgba(255,255,255,0.10)' : 'rgba(0,0,0,0.10)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.ellipse(lcx, lcy, (1 / spanZ) * iw, (1 / spanZ) * ih, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.beginPath();
            ctx.ellipse(lcx, lcy, (2 / spanZ) * iw, (2 / spanZ) * ih, 0, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();

            // Gridlines across the entire latent pane
            ctx.strokeStyle = isCarbon ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.04)';
            ctx.lineWidth = 1;
            const gridStep = spanZ > 12 ? 2 : 1;
            const gridLimit = Math.floor(latentViewport.maxZ);
            for (let z1 = -gridLimit; z1 <= gridLimit; z1 += gridStep) {
              if (z1 === 0) continue;
              const px = Math.floor(toLatentPxX(z1, iw, ih)) + 0.5;
              ctx.beginPath();
              ctx.moveTo(px, 0);
              ctx.lineTo(px, ih);
              ctx.stroke();
            }

            for (let z2 = -gridLimit; z2 <= gridLimit; z2 += gridStep) {
              if (z2 === 0) continue;
              const py = Math.floor(toLatentPxY(z2, iw, ih)) + 0.5;
              ctx.beginPath();
              ctx.moveTo(0, py);
              ctx.lineTo(iw, py);
              ctx.stroke();
            }

            // Axes
            ctx.strokeStyle = isCarbon ? '#2E2E2B' : '#DCDAD3';
            ctx.lineWidth = 1;
            const zeroZX = Math.floor(toLatentPxX(0, iw, ih)) + 0.5;
            const zeroZY = Math.floor(toLatentPxY(0, iw, ih)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(zeroZX, 0);
            ctx.lineTo(zeroZX, ih);
            ctx.moveTo(0, zeroZY);
            ctx.lineTo(iw, zeroZY);
            ctx.stroke();

            // Plot Encoded Points / Ellipses if latentArtifact is present
            if (latentArtifact) {
              for (const ep of latentArtifact.encodedPoints) {
                const ex = toLatentPxX(ep.z1, iw, ih);
                const ey = toLatentPxY(ep.z2, iw, ih);
                const color = getPointColor(ep.label);

                // VAE 1-sigma uncertainty ellipse
                if (latentArtifact.isVAE && ep.sigma1 && ep.sigma2) {
                  const rx = ep.sigma1 * (iw / spanZ);
                  const ry = ep.sigma2 * (ih / spanZ);
                  ctx.strokeStyle = isRegressionDataset
                    ? (isCarbon ? 'rgba(194, 103, 82, 0.45)' : 'rgba(217, 78, 52, 0.4)')
                    : ep.label === 2
                    ? (isCarbon ? 'rgba(77, 143, 104, 0.4)' : 'rgba(21, 128, 61, 0.4)')
                    : ep.label < 0
                    ? (isCarbon ? 'rgba(78, 137, 184, 0.4)' : 'rgba(43, 91, 132, 0.4)')
                    : (isCarbon ? 'rgba(229, 131, 68, 0.4)' : 'rgba(200, 90, 23, 0.4)');
                  ctx.lineWidth = 1;
                  ctx.beginPath();
                  ctx.ellipse(ex, ey, Math.max(1, rx), Math.max(1, ry), 0, 0, Math.PI * 2);
                  ctx.stroke();
                }

                // Center point
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(ex, ey, 3, 0, Math.PI * 2);
                ctx.fill();
              }
            }

            // Draggable Latent Cursor Probe (✦)
            const px = toLatentPxX(probeZ[0], iw, ih);
            const py = toLatentPxY(probeZ[1], iw, ih);

            ctx.fillStyle = isCarbon ? '#F59E0B' : '#D97706';
            ctx.strokeStyle = isCarbon ? '#131312' : '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(px, py, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            ctx.font = '9px "JetBrains Mono"';
            ctx.fillText(`z = (${probeZ[0].toFixed(2)}, ${probeZ[1].toFixed(2)})`, px + 8, py - 4);
            ctx.restore();
          } else if (isGAN && ganArtifact) {
            // Mode C: 9x9 Latent Mesh Warp & Dual Losses
            // Centered axes
            const gcx = toPxX(0, iw, ih);
            const gcy = toPxY(0, iw, ih);
            ctx.save();
            ctx.strokeStyle = isCarbon ? '#2E2E2B' : '#DCDAD3';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(gcx, 0);
            ctx.lineTo(gcx, ih);
            ctx.moveTo(0, gcy);
            ctx.lineTo(iw, gcy);
            ctx.stroke();
            ctx.restore();

            const warp = ganArtifact.latentGridWarp;
            const res = ganArtifact.gridResolution || 9;

            if (warp && warp.length === res * res) {
              ctx.save();
              ctx.strokeStyle = isCarbon ? 'rgba(217, 119, 6, 0.35)' : 'rgba(180, 83, 9, 0.35)';
              ctx.lineWidth = 1;

              // Draw horizontal mesh lines
              for (let r = 0; r < res; r++) {
                ctx.beginPath();
                for (let c = 0; c < res; c++) {
                  const pt = warp[r * res + c].projectedX;
                  const mx = toPxX(pt[0], iw, ih);
                  const my = toPxY(pt[1], iw, ih);
                  if (c === 0) ctx.moveTo(mx, my);
                  else ctx.lineTo(mx, my);
                }
                ctx.stroke();
              }

              // Draw vertical mesh lines
              for (let c = 0; c < res; c++) {
                ctx.beginPath();
                for (let r = 0; r < res; r++) {
                  const pt = warp[r * res + c].projectedX;
                  const mx = toPxX(pt[0], iw, ih);
                  const my = toPxY(pt[1], iw, ih);
                  if (r === 0) ctx.moveTo(mx, my);
                  else ctx.lineTo(mx, my);
                }
                ctx.stroke();
              }
              ctx.restore();
            }
          }
        }
      }
    };

    return subscribeDirtyRender(dirtyManager, render);
  }, [
    dirtyManager,
    isCarbon,
    isMLP,
    isBottleneck,
    isGAN,
    isRegressionDataset,
    is1DRegression,
    predictValueAtPoint,
    getPointColor,
    gridValues,
    discretize,
    neuralArtifact,
    latentArtifact,
    ganArtifact,
    magnifiedNeuron,
    probeZ,
    trainData,
    testData,
    effectiveShowTestData,
    latentViewport,
    toLatentPxX,
    toLatentPxY,
  ]);

  // Topology Update Handlers for Mode A (MLP)
  const numLayers = Number(hyperparams.numLayers ?? 2);
  const handleAddLayer = () => {
    if (numLayers < 4) {
      onUpdateHyperparam?.('numLayers', numLayers + 1);
    }
  };
  const handleRemoveLayer = () => {
    if (numLayers > 1) {
      onUpdateHyperparam?.('numLayers', numLayers - 1);
    }
  };

  const handleUpdateLayerNeurons = (layerIdx: number, delta: number) => {
    const key = `layer${layerIdx}`;
    const cur = Number(hyperparams[key] ?? (layerIdx === 1 ? 4 : 2));
    const next = Math.max(1, Math.min(8, cur + delta));
    onUpdateHyperparam?.(key, next);
  };

  const handleToggleFeature = (feat: BasisFeature) => {
    const curFeats = (hyperparams.features ? String(hyperparams.features).split(',') : ['x1', 'x2'])
      .map((s) => s.trim().toLowerCase());
    let next: string[];
    if (curFeats.includes(feat)) {
      if (curFeats.length > 1) {
        next = curFeats.filter((f) => f !== feat);
      } else {
        return;
      }
    } else {
      next = [...curFeats, feat];
    }
    onUpdateHyperparam?.('features', next.join(','));
  };

  const handleToggleSynapseLesion = (synId: string) => {
    const curLesioned = new Set(
      (hyperparams.lesioned ? String(hyperparams.lesioned).split(',') : []).map((s) => s.trim())
    );
    if (curLesioned.has(synId)) {
      curLesioned.delete(synId);
    } else {
      curLesioned.add(synId);
    }
    onUpdateHyperparam?.('lesioned', Array.from(curLesioned).join(','));
  };

  // Main Canvas Point Interaction
  const handleMainCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!mainCanvasRef.current) return;
    const rect = mainCanvasRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const x1 = toCoordX(px, rect.width, rect.height);
    const x2 = toCoordY(py, rect.width, rect.height);

    const visiblePoints = getVisibleDataPoints(trainData, testData, effectiveShowTestData);
    const hit = hitTestDataPoints(visiblePoints, {
      px,
      py,
      width: rect.width,
      height: rect.height,
      hitRadiusPx: 10,
      toPxX: (x, width) => toPxX(x, width, rect.height),
      toPxY: (y, height) => toPxY(y, rect.width, height),
      strictLessThan: true,
    });

    if (hit) {
      if (pointerMode === 'delete') {
        onDeletePoint(hit.id);
      } else if (pointerMode === 'flip') {
        onUpdatePoint({ ...hit, y: hit.y > 0 ? -1 : 1 });
      } else if (pointerMode === 'add-drag') {
        setDraggingPointId(hit.id);
        onDragStart?.();
      }
    } else {
      if (pointerMode === 'add-drag') {
        if (is1DRegression) {
          onAddPoint({ x1, x2, y: x2 });
        } else {
          onAddPoint({ x1, x2, y: activeClass });
        }
      }
    }
  };

  const handleMainCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (draggingPointId === null || !mainCanvasRef.current) return;
    const rect = mainCanvasRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const x1 = Math.max(-6, Math.min(6, toCoordX(px, rect.width, rect.height)));
    const x2 = Math.max(-6, Math.min(6, toCoordY(py, rect.width, rect.height)));

    const visiblePoints = getVisibleDataPoints(trainData, testData, effectiveShowTestData);
    const pt = visiblePoints.find((p) => p.id === draggingPointId);
    if (pt) {
      if (is1DRegression) {
        onUpdatePoint({ ...pt, x1, x2, y: x2 });
      } else {
        onUpdatePoint({ ...pt, x1, x2 });
      }
    }
  };

  const handleMainCanvasMouseUp = () => {
    if (draggingPointId !== null) {
      setDraggingPointId(null);
      onDragEnd?.();
    }
  };

  // Latent Probe Drag Handlers (Mode B)
  const handleInsetMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isBottleneck || !insetCanvasRef.current) return;
    const rect = insetCanvasRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const z1 = toLatentCoordX(px, rect.width, rect.height);
    const z2 = isAutoencoder && hyperparams.latentDim === 1 ? 0 : toLatentCoordY(py, rect.width, rect.height);

    setProbeZ([z1, z2]);
    setIsDraggingProbe(true);
    onUpdateHyperparam?.('probeZ1', z1);
    onUpdateHyperparam?.('probeZ2', z2);
    dirtyManager.markDirty();
  };

  const handleInsetMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isDraggingProbe || !insetCanvasRef.current) return;
    const rect = insetCanvasRef.current.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const z1 = Math.max(latentViewport.minZ, Math.min(latentViewport.maxZ, toLatentCoordX(px, rect.width, rect.height)));
    const z2 =
      isAutoencoder && hyperparams.latentDim === 1
        ? 0
        : Math.max(latentViewport.minZ, Math.min(latentViewport.maxZ, toLatentCoordY(py, rect.width, rect.height)));

    setProbeZ([z1, z2]);
    onUpdateHyperparam?.('probeZ1', z1);
    onUpdateHyperparam?.('probeZ2', z2);
    dirtyManager.markDirty();
  };

  const handleInsetMouseUp = () => {
    setIsDraggingProbe(false);
  };

  // Geometric Coordinate Layout for MLP Glass-Box Graph
  // ViewBox: 0 0 1000 1000
  const numHidden = neuralArtifact ? Math.max(1, neuralArtifact.layers.length - 2) : numLayers;
  const totalCols = numHidden + 2; // Col 0: Features, Col 1..numHidden: Hidden, Col totalCols-1: Output

  // Compute column X coordinate (0 to 1000)
  const getColX = useCallback(
    (colIdx: number) => {
      const leftMargin = 110;
      const rightMargin = 910;
      if (totalCols <= 1) return 500;
      return leftMargin + (colIdx / (totalCols - 1)) * (rightMargin - leftMargin);
    },
    [totalCols]
  );

  // Compute Y coordinate for the 7 basis features in Col 0 (0 to 1000)
  const getFeatureY = (featIdx: number) => {
    const topPad = 50;
    const usableHeight = 840;
    return topPad + ((featIdx + 0.5) / BASIS_FEATURES.length) * usableHeight;
  };

  // Compute Y coordinate for a neuron in layer l (0 to 1000)
  // Reserves clear top and bottom margin so neurons NEVER overlap ||∇W|| at bottom: 8px
  const getNeuronY = (neuronIdx: number, count: number) => {
    if (count <= 1) return 460;
    const topPad = 60;
    const usableHeight = 770;
    return topPad + ((neuronIdx + 0.5) / count) * usableHeight;
  };

  // Look up source position for a synapse
  const getSynapseCoords = useCallback(
    (syn: Extract<VisualArtifact, { type: 'neural-graph-overlay' }>['synapses'][0]) => {
      let x1 = 0;
      let y1 = 0;
      let x2 = 0;
      let y2 = 0;

      if (!neuralArtifact) return { x1, y1, x2, y2 };

      // Source endpoint
      if (syn.sourceLayer === 0) {
        x1 = getColX(0);
        // Find which feature this neuron corresponds to
        const featLabel = neuralArtifact.layers[0]?.neurons[syn.sourceNeuron]?.label?.toLowerCase() || '';
        const featIdx = BASIS_FEATURES.findIndex((f) => f.toLowerCase() === featLabel);
        y1 = getFeatureY(featIdx >= 0 ? featIdx : syn.sourceNeuron);
      } else {
        x1 = getColX(syn.sourceLayer);
        const count = neuralArtifact.layers[syn.sourceLayer]?.neuronCount || 1;
        y1 = getNeuronY(syn.sourceNeuron, count);
      }

      // Target endpoint
      x2 = getColX(syn.targetLayer);
      const tgtCount = neuralArtifact.layers[syn.targetLayer]?.neuronCount || 1;
      y2 = getNeuronY(syn.targetNeuron, tgtCount);

      return { x1, y1, x2, y2 };
    },
    [neuralArtifact, getColX]
  );

  const batchSize = String(hyperparams.batchSize ?? '10');

  return (
    <div
      ref={containerRef}
      className="stage04-container flex flex-col w-full h-full min-w-0 min-h-0 bg-[var(--bg-primary)] overflow-hidden select-none font-mono"
    >
      {/* 1. Top Header Bar for Mode A (MLP) */}
      {isMLP && (
        <div className="flex flex-wrap items-center justify-between px-4 py-2 border-b border-[var(--border-hairline)] bg-[var(--bg-elevated)] gap-2.5 text-[11px] min-w-0">
          {/* Hidden Layers +/- Count */}
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="text-[10px] text-[var(--text-muted)] uppercase">Hidden Layers:</span>
            <div className="flex items-center border border-[var(--border-hairline)] bg-[var(--bg-subtle)]">
              <button
                className="p-1 hover:bg-[var(--bg-active)] text-[var(--text-primary)] disabled:opacity-30"
                disabled={numLayers <= 1}
                onClick={handleRemoveLayer}
                title="Remove Hidden Layer"
              >
                <Minus size={12} />
              </button>
              <span className="px-2 font-bold text-[11px] text-[var(--text-primary)]">
                {numLayers}
              </span>
              <button
                className="p-1 hover:bg-[var(--bg-active)] text-[var(--text-primary)] disabled:opacity-30"
                disabled={numLayers >= 4}
                onClick={handleAddLayer}
                title="Add Hidden Layer"
              >
                <Plus size={12} />
              </button>
            </div>
          </div>

          {/* Batch Size Selector */}
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="text-[10px] text-[var(--text-muted)] uppercase">Batch Size:</span>
            <select
              value={batchSize}
              onChange={(e) => onUpdateHyperparam?.('batchSize', e.target.value === 'full' ? 'full' : Number(e.target.value))}
              className="bg-[var(--bg-subtle)] border border-[var(--border-hairline)] px-1.5 py-0.5 text-[11px] text-[var(--text-primary)] focus:outline-hidden"
            >
              <option value="10">10</option>
              <option value="20">20</option>
              <option value="32">32</option>
              <option value="50">50</option>
              <option value="full">Full Dataset</option>
            </select>
          </div>

          <div className="text-[10px] text-[var(--text-muted)] hidden xl:block shrink-0">
            STAGE 04 · DENSE & GENERATIVE LAB
          </div>
        </div>
      )}

      {/* 2. Main Stage Center Area */}
      <div className="flex flex-1 min-h-0 min-w-0">
        {/* MODE A: MLP Neural Graph (58% width) + Output Manifold (42% width) */}
        {isMLP && (
          <>
            {/* Interactive Neural Graph (58%) */}
            <div className="relative flex-[58] flex flex-col border-r border-[var(--border-hairline)] bg-[var(--bg-canvas)] overflow-hidden min-w-0">
              {/* Layer Headers Bar */}
              <div className="flex items-center justify-between px-3 py-1.5 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] min-w-0">
                {/* Features Header */}
                <div className="w-[95px] text-center font-semibold text-[var(--text-primary)] shrink-0">
                  FEATURES
                </div>

                {/* Hidden Layers Headers */}
                {Array.from({ length: numHidden }).map((_, hIdx) => {
                  const layerNum = hIdx + 1;
                  const key = `layer${layerNum}`;
                  const count = Number(hyperparams[key] ?? (layerNum === 1 ? 4 : 2));
                  return (
                    <div key={hIdx} className="flex flex-col items-center gap-0.5 flex-1 min-w-0 text-center">
                      <span className="text-[9px] uppercase text-[var(--text-muted)] truncate max-w-full">
                        Hidden {layerNum}
                      </span>
                      <div className="flex items-center gap-0.5 shrink-0 whitespace-nowrap">
                        <button
                          className="p-0.5 border border-[var(--border-hairline)] hover:bg-[var(--bg-active)] text-[var(--text-primary)] disabled:opacity-30 shrink-0"
                          disabled={count <= 1}
                          onClick={() => handleUpdateLayerNeurons(layerNum, -1)}
                          title="Decrease neurons"
                        >
                          <Minus size={9} />
                        </button>
                        <span className="font-bold text-[10px] px-1 min-w-[14px] text-center">{count}</span>
                        <button
                          className="p-0.5 border border-[var(--border-hairline)] hover:bg-[var(--bg-active)] text-[var(--text-primary)] disabled:opacity-30 shrink-0"
                          disabled={count >= 8}
                          onClick={() => handleUpdateLayerNeurons(layerNum, 1)}
                          title="Increase neurons"
                        >
                          <Plus size={9} />
                        </button>
                      </div>
                    </div>
                  );
                })}

                {/* Output Header */}
                <div className="w-[65px] text-center font-semibold text-[var(--text-primary)] shrink-0">
                  OUTPUT
                </div>
              </div>

              {/* Neural Graph Stage Canvas & Overlay Area */}
              <div className="relative flex-1 min-h-0 w-full h-full">
                {/* SVG Synapse Bezier Wires Layer */}
                <svg
                  viewBox="0 0 1000 1000"
                  preserveAspectRatio="none"
                  className="absolute inset-0 w-full h-full pointer-events-none"
                  style={{ zIndex: 1 }}
                >
                  {neuralArtifact?.synapses.map((syn) => {
                    const { x1, y1, x2, y2 } = getSynapseCoords(syn);
                    const dx = x2 - x1;
                    const cx1 = x1 + dx * 0.45;
                    const cx2 = x2 - dx * 0.45;

                    const isPos = syn.weight >= 0;
                    const absW = Math.abs(syn.weight);
                    const strokeWidth = syn.isLesioned ? 1.5 : Math.max(1, Math.min(6, 1 + absW * 2.2));
                    const strokeColor = syn.isLesioned
                      ? isCarbon ? '#575653' : '#D1CFC7'
                      : isPos
                      ? isCarbon ? '#E58344' : '#D94E34'
                      : isCarbon ? '#4E89B8' : '#1E40AF';

                    // TF Playground: dashed for negative weights, solid for positive weights
                    const strokeDash = syn.isLesioned ? '3,3' : !isPos ? '4,3' : undefined;

                    return (
                      <path
                        key={syn.id}
                        d={`M ${x1.toFixed(1)} ${y1.toFixed(1)} C ${cx1.toFixed(1)} ${y1.toFixed(1)}, ${cx2.toFixed(1)} ${y2.toFixed(1)}, ${x2.toFixed(1)} ${y2.toFixed(1)}`}
                        fill="none"
                        stroke={strokeColor}
                        strokeWidth={strokeWidth}
                        strokeDasharray={strokeDash}
                        opacity={syn.isLesioned ? 0.25 : Math.max(0.45, Math.min(0.95, 0.45 + absW * 0.3))}
                        className="pointer-events-auto cursor-pointer hover:opacity-100 transition-opacity"
                        onMouseEnter={(e) => setHoveredSynapse({ syn, x: e.clientX, y: e.clientY })}
                        onMouseMove={(e) => setHoveredSynapse({ syn, x: e.clientX, y: e.clientY })}
                        onMouseLeave={() => setHoveredSynapse(null)}
                        onClick={() => handleToggleSynapseLesion(syn.id)}
                      />
                    );
                  })}
                </svg>

                {/* HTML Nodes & Columns Layer */}
                <div className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 2 }}>
                  {/* Column 0: Vertical Features Column (All 7 Basis Features) */}
                  {BASIS_FEATURES.map((feat, fIdx) => {
                    const isDisabledFeature = is1DRegression && X2_DEPENDENT_FEATURES.includes(feat);
                    const isActive = !isDisabledFeature && activeFeatures.includes(feat);
                    const x = getColX(0);
                    const y = getFeatureY(fIdx);

                    return (
                      <div
                        key={feat}
                        aria-disabled={isDisabledFeature ? 'true' : undefined}
                        className={`absolute flex items-center gap-1.5 transition-all ${
                          isDisabledFeature
                            ? 'pointer-events-none opacity-25 cursor-not-allowed'
                            : 'pointer-events-auto cursor-pointer'
                        }`}
                        style={{
                          left: `${(x / 10).toFixed(2)}%`,
                          top: `${(y / 10).toFixed(2)}%`,
                          transform: 'translate(-50%, -50%)',
                        }}
                        onClick={() => {
                          if (!isDisabledFeature) handleToggleFeature(feat);
                        }}
                        title={
                          isDisabledFeature
                            ? `${feat} disabled on 1D regression datasets (x₂ = y)`
                            : `Click to ${isActive ? 'remove' : 'add'} ${feat} feature`
                        }
                      >
                        {/* Formula label */}
                        <div
                          className={`w-12 text-right text-[11px] font-semibold transition-opacity ${
                            isActive ? 'text-[var(--text-primary)]' : 'text-[var(--text-muted)] opacity-50'
                          }`}
                        >
                          {getFeatureMathLabel(feat)}
                        </div>

                        {/* 2D Thumbnail Box */}
                        <div
                          className={`w-7 h-7 border rounded-xs overflow-hidden transition-all shadow-xs ${
                            isActive
                              ? 'border-[var(--accent-amber)] ring-1 ring-[var(--accent-amber)]/50'
                              : 'border-[var(--border-hairline)] opacity-40 grayscale hover:opacity-75'
                          }`}
                        >
                          <FeatureThumbnail feat={feat} isCarbon={isCarbon} />
                        </div>
                      </div>
                    );
                  })}

                  {/* Hidden Layer Nodes */}
                  {neuralArtifact?.layers.map((layer) => {
                    if (layer.isInput || layer.isOutput) return null;
                    const colX = getColX(layer.index);

                    return (
                      <React.Fragment key={layer.index}>
                        {layer.neurons.map((neuron, nIdx) => {
                          const nY = getNeuronY(nIdx, layer.neuronCount);
                          const isMagnified =
                            magnifiedNeuron?.layer === layer.index && magnifiedNeuron?.neuron === nIdx;

                          return (
                            <div
                              key={neuron.id}
                              className="absolute pointer-events-auto cursor-pointer flex flex-col items-center gap-0.5 group"
                              style={{
                                left: `${(colX / 10).toFixed(2)}%`,
                                top: `${(nY / 10).toFixed(2)}%`,
                                transform: 'translate(-50%, -50%)',
                              }}
                              onMouseEnter={() => {
                                setMagnifiedNeuron({ layer: layer.index, neuron: nIdx });
                                dirtyManager.markDirty();
                              }}
                              onMouseLeave={() => {
                                setMagnifiedNeuron(null);
                                dirtyManager.markDirty();
                              }}
                              title={`Neuron N${nIdx + 1} (Layer ${layer.index}) | Bias: ${neuron.bias.toFixed(3)} | Hover to magnify activation`}
                            >
                              {/* Square Preview Node */}
                              <div
                                className={`w-8 h-8 border rounded-xs flex items-center justify-center font-mono text-[9px] overflow-hidden transition-transform shadow-xs ${
                                  isMagnified
                                    ? 'border-[var(--accent-amber)] scale-110 ring-2 ring-[var(--accent-amber)]/40 bg-[var(--bg-elevated)]'
                                    : 'border-[var(--border-hairline)] bg-[var(--bg-elevated)] hover:border-[var(--text-secondary)]'
                                }`}
                              >
                                {neuron.activationPreview ? (
                                  <NeuronMiniPreview
                                    values={neuron.activationPreview}
                                    isCarbon={isCarbon}
                                  />
                                ) : (
                                  <span className="text-[var(--text-secondary)] font-bold">
                                    N{nIdx + 1}
                                  </span>
                                )}
                              </div>
                              <span className="text-[8px] text-[var(--text-muted)]">
                                N{nIdx + 1}
                              </span>
                            </div>
                          );
                        })}

                        {/* Layer Gradient Norm Indicator at bottom */}
                        <div
                          className="absolute pointer-events-none flex flex-col items-center gap-0.5"
                          style={{
                            left: `${(colX / 10).toFixed(2)}%`,
                            bottom: '8px',
                            transform: 'translateX(-50%)',
                          }}
                        >
                          <span className="text-[7.5px] text-[var(--text-muted)] uppercase">
                            ||∇W||
                          </span>
                          <div className="w-9 h-1 bg-[var(--bg-subtle)] border border-[var(--border-hairline)] overflow-hidden">
                            <div
                              className="h-full bg-[var(--pigment-class-a)]"
                              style={{
                                width: `${Math.min(
                                  100,
                                  ((neuralArtifact.layerGradientNorms[layer.index - 1] ?? 0) / 0.1) * 100
                                )}%`,
                              }}
                            />
                          </div>
                          <span className="text-[7.5px] text-[var(--text-secondary)]">
                            {(neuralArtifact.layerGradientNorms[layer.index - 1] ?? 0).toFixed(4)}
                          </span>
                        </div>
                      </React.Fragment>
                    );
                  })}

                  {/* Output Node (Column totalCols - 1) */}
                  <div
                    className="absolute pointer-events-auto flex flex-col items-center gap-0.5"
                    style={{
                      left: `${(getColX(totalCols - 1) / 10).toFixed(2)}%`,
                      top: '50%',
                      transform: 'translate(-50%, -50%)',
                    }}
                    title="Output Neuron (Ŷ)"
                  >
                    <div className="w-8 h-8 border border-[var(--border-hairline)] rounded-xs bg-[var(--bg-elevated)] flex items-center justify-center font-bold text-[11px] text-[var(--text-primary)] shadow-xs">
                      Ŷ
                    </div>
                    <span className="text-[8px] text-[var(--text-muted)]">OUTPUT</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Output Manifold Canvas Area (42%) */}
            <div className="relative flex-[42] flex flex-col bg-[var(--bg-canvas)] overflow-hidden">
              {/* Output Canvas */}
              <div className="relative flex-1 min-h-0">
                <canvas
                  ref={mainCanvasRef}
                  className="canvas-main w-full h-full block cursor-crosshair"
                  onMouseDown={handleMainCanvasMouseDown}
                  onMouseMove={handleMainCanvasMouseMove}
                  onMouseUp={handleMainCanvasMouseUp}
                  onMouseLeave={handleMainCanvasMouseUp}
                />
              </div>

              {/* TensorFlow Playground Output Controls Bar */}
              <div className="flex flex-col px-3 py-2 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] gap-2 select-none">
                {/* Color Legend Bar */}
                <div className="flex items-center justify-between text-[var(--text-muted)]">
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
                      <span className="text-[9px]">Colors show data, neuron and weight values:</span>
                      <div className="flex items-center gap-1.5 font-mono font-bold text-[9px]">
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

                {/* Checkboxes: Show test data & Discretize output */}
                <div className="flex items-center justify-between pt-0.5 border-t border-[var(--border-hairline)]/50">
                  <label className="flex items-center gap-1.5 cursor-pointer text-[var(--text-primary)] hover:text-[var(--text-primary)]">
                    <input
                      type="checkbox"
                      checked={effectiveShowTestData}
                      onChange={handleToggleShowTestData}
                      className="rounded-xs border-[var(--border-hairline)] text-[var(--accent-amber)] focus:ring-0 cursor-pointer"
                    />
                    <span>Show test data</span>
                  </label>

                  {!is1DRegression && (
                    <label className="flex items-center gap-1.5 cursor-pointer text-[var(--text-primary)] hover:text-[var(--text-primary)]">
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
          </>
        )}

        {/* MODE B: Latent Bottleneck Lab (60% Main Canvas + 40% Latent Bottleneck Pad) */}
        {isBottleneck && (
          <div className="flex flex-col flex-1 min-h-0 min-w-0">
            <div className="flex flex-1 min-h-0 min-w-0">
              <div className="relative flex-[60] flex flex-col border-r border-[var(--border-hairline)] bg-[var(--bg-canvas)] overflow-hidden min-w-0 min-h-0">
                <div className="absolute top-3 left-4 z-10 flex items-center gap-2 bg-[var(--bg-canvas)]/90 backdrop-blur-sm px-2 py-1 border border-[var(--border-hairline)] font-mono text-[10px]">
                  <Compass size={12} className="text-[var(--text-primary)]" />
                  <span>Input & Reconstructed Data Space X</span>
                </div>
                <div className="relative flex-1 min-h-0 min-w-0 overflow-hidden">
                  <canvas
                    ref={mainCanvasRef}
                    className="canvas-main w-full h-full block cursor-crosshair"
                    onMouseDown={handleMainCanvasMouseDown}
                    onMouseMove={handleMainCanvasMouseMove}
                    onMouseUp={handleMainCanvasMouseUp}
                    onMouseLeave={handleMainCanvasMouseUp}
                  />
                </div>
              </div>

              <div className="relative flex-[40] flex flex-col bg-[var(--bg-elevated)] overflow-hidden min-w-0 min-h-0">
                <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border-hairline)] font-mono text-[11px] bg-[var(--bg-subtle)]">
                  <span className="font-semibold text-[var(--text-primary)]">
                    {isVAE ? '2D Latent Space Z ~ N(μ, σ²)' : '2D Latent Bottleneck Z'}
                  </span>
                  <span className="text-[10px] text-[var(--text-muted)] flex items-center gap-1">
                    <Move size={10} /> Drag Cursor (✦)
                  </span>
                </div>
                <div className="relative flex-1 min-h-0 min-w-0 overflow-hidden">
                  <canvas
                    ref={insetCanvasRef}
                    className="w-full h-full block cursor-crosshair"
                    onMouseDown={handleInsetMouseDown}
                    onMouseMove={handleInsetMouseMove}
                    onMouseUp={handleInsetMouseUp}
                    onMouseLeave={handleInsetMouseUp}
                  />
                </div>
              </div>
            </div>

            {/* Mode B Bottom Swiss Bar */}
            <div className="flex flex-wrap items-center justify-between px-3 py-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] gap-2 select-none shrink-0">
              <div className="flex items-center gap-3 text-[var(--text-muted)] font-mono text-[9px]">
                <div className="flex items-center gap-1">
                  <span className="w-2.5 h-0.5 bg-[var(--text-muted)] inline-block" />
                  <span>Reconstruction (X → X̂)</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-[var(--accent-amber)] font-bold">✦</span>
                  <span>Decoded Probe D(z)</span>
                </div>
                {latentArtifact?.pca1Axis && (
                  <div className="flex items-center gap-1">
                    <span className="w-2.5 h-0.5 border-t border-dashed border-[#D97706] inline-block" />
                    <span className="text-[#D97706]">PCA PC1 Subspace</span>
                  </div>
                )}
                {isVAE && (
                  <div className="flex items-center gap-1">
                    <span className="w-2.5 h-1.5 rounded-full border border-[var(--text-secondary)] inline-block" />
                    <span>1σ Posterior Ellipses q(z|x)</span>
                  </div>
                )}
              </div>

              <div className="flex items-center gap-3">
                <label className="flex items-center gap-1.5 cursor-pointer text-[var(--text-primary)] hover:text-[var(--text-primary)] text-[10px]">
                  <input
                    type="checkbox"
                    checked={effectiveShowTestData}
                    onChange={handleToggleShowTestData}
                    className="rounded-xs border-[var(--border-hairline)] text-[var(--accent-amber)] focus:ring-0 cursor-pointer"
                  />
                  <span>Show test data</span>
                </label>
              </div>
            </div>
          </div>
        )}

        {/* MODE C: Adversarial Minimax Arena (65% Canvas + 35% Latent Warp Inset) */}
        {isGAN && (
          <div className="flex flex-col flex-1 min-h-0 min-w-0">
            <div className="flex flex-1 min-h-0 min-w-0">
              <div className="relative flex-[65] flex flex-col border-r border-[var(--border-hairline)] bg-[var(--bg-canvas)] overflow-hidden min-w-0 min-h-0">
                <div className="absolute top-3 left-4 z-10 flex items-center gap-2 bg-[var(--bg-canvas)]/90 backdrop-blur-sm px-2 py-1 border border-[var(--border-hairline)] font-mono text-[10px]">
                  <Zap size={12} className="text-[var(--pigment-class-a)]" />
                  <span>Minimax Arena: D(x) Landscape & Gradient Pull Vectors ∇_x log D(x)</span>
                </div>
                <div className="relative flex-1 min-h-0 min-w-0 overflow-hidden">
                  <canvas
                    ref={mainCanvasRef}
                    className="canvas-main w-full h-full block cursor-crosshair"
                    onMouseDown={handleMainCanvasMouseDown}
                    onMouseMove={handleMainCanvasMouseMove}
                    onMouseUp={handleMainCanvasMouseUp}
                    onMouseLeave={handleMainCanvasMouseUp}
                  />
                </div>
              </div>

              <div className="relative flex-[35] flex flex-col bg-[var(--bg-elevated)] overflow-hidden min-w-0 min-h-0">
                <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--border-hairline)] font-mono text-[11px] bg-[var(--bg-subtle)]">
                  <span className="font-semibold text-[var(--text-primary)]">
                    Latent Noise Mesh Warp G(z)
                  </span>
                  {' '}
                  <span className="text-[10px] text-[var(--text-muted)]">z ~ N(0, I)</span>
                </div>
                <div className="relative flex-1 min-h-0 min-w-0 overflow-hidden">
                  <canvas ref={insetCanvasRef} className="w-full h-full block" />
                </div>
              </div>
            </div>

            {/* Mode C Bottom Swiss Bar */}
            <div className="flex flex-wrap items-center justify-between px-3 py-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] gap-2 select-none shrink-0">
              <div className="flex items-center gap-3 text-[var(--text-muted)] font-mono text-[9px]">
                <div className="flex items-center gap-1.5">
                  <span>D(x) Probability:</span>
                  <span className={isCarbon ? 'text-[#4E89B8]' : 'text-[#1E40AF]'}>Fake (0.0)</span>
                  <div
                    className="w-14 h-2 rounded-xs border border-[var(--border-hairline)]"
                    style={{
                      background: isCarbon
                        ? 'linear-gradient(to right, rgb(78, 137, 184), rgb(24, 24, 23), rgb(229, 131, 68))'
                        : 'linear-gradient(to right, rgb(43, 91, 132), rgb(250, 249, 245), rgb(200, 90, 23))',
                    }}
                  />
                  <span className={isCarbon ? 'text-[#E58344]' : 'text-[#D94E34]'}>Real (1.0)</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-[#D97706]">◆</span>
                  <span>Fake Samples G(z)</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="text-[#10B981]">→</span>
                  <span>Pull Vectors ∇_x log D(x)</span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                <label className="flex items-center gap-1.5 cursor-pointer text-[var(--text-primary)] hover:text-[var(--text-primary)] text-[10px]">
                  <input
                    type="checkbox"
                    checked={effectiveShowTestData}
                    onChange={handleToggleShowTestData}
                    className="rounded-xs border-[var(--border-hairline)] text-[var(--accent-amber)] focus:ring-0 cursor-pointer"
                  />
                  <span>Show test data</span>
                </label>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Floating Synapse Tooltip */}
      {hoveredSynapse && (
        <div
          className="fixed z-50 pointer-events-none px-2.5 py-1.5 bg-[var(--bg-elevated)] border border-[var(--border-hairline)] shadow-lg rounded-xs font-mono text-[10px] text-[var(--text-primary)]"
          style={{ left: hoveredSynapse.x + 12, top: hoveredSynapse.y - 24 }}
        >
          <div className="flex items-center gap-2">
            <span className="font-semibold text-[var(--text-secondary)]">Weight (w):</span>
            <span
              className={
                hoveredSynapse.syn.weight >= 0
                  ? isCarbon ? 'text-[#E58344] font-bold' : 'text-[#D94E34] font-bold'
                  : isCarbon ? 'text-[#4E89B8] font-bold' : 'text-[#1E40AF] font-bold'
              }
            >
              {hoveredSynapse.syn.weight > 0 ? '+' : ''}
              {hoveredSynapse.syn.weight.toFixed(4)}
            </span>
          </div>
          <div className="flex items-center gap-2 text-[var(--text-muted)]">
            <span>|∇w|:</span>
            <span className="text-[var(--text-secondary)]">
              {hoveredSynapse.syn.gradientMagnitude.toFixed(4)}
            </span>
          </div>
          <div className="text-[9px] text-[var(--text-muted)] mt-1 border-t border-[var(--border-hairline)] pt-0.5">
            {hoveredSynapse.syn.isLesioned ? '✂ Lesioned (click to restore)' : 'Click to prune / lesion'}
          </div>
        </div>
      )}
    </div>
  );
};

// Formatted mathematical feature labels
function getFeatureMathLabel(feat: BasisFeature): React.ReactNode {
  switch (feat) {
    case 'x1':
      return (
        <span>
          X<sub>1</sub>
        </span>
      );
    case 'x2':
      return (
        <span>
          X<sub>2</sub>
        </span>
      );
    case 'x1_sq':
      return (
        <span>
          X<sub>1</sub><sup>2</sup>
        </span>
      );
    case 'x2_sq':
      return (
        <span>
          X<sub>2</sub><sup>2</sup>
        </span>
      );
    case 'x1_x2':
      return (
        <span>
          X<sub>1</sub>X<sub>2</sub>
        </span>
      );
    case 'sin_x1':
      return <span>sin(X<sub>1</sub>)</span>;
    case 'sin_x2':
      return <span>sin(X<sub>2</sub>)</span>;
    default:
      return <span>{feat}</span>;
  }
}

// 2D Thumbnail Component for the 7 basis features
const FeatureThumbnail: React.FC<{ feat: BasisFeature; isCarbon: boolean }> = React.memo(
  ({ feat, isCarbon }) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;
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

          let v = 0;
          switch (feat) {
            case 'x1':
              v = x1 / 6.0;
              break;
            case 'x2':
              v = x2 / 6.0;
              break;
            case 'x1_sq':
              v = (x1 * x1) / 18.0 - 1.0;
              break;
            case 'x2_sq':
              v = (x2 * x2) / 18.0 - 1.0;
              break;
            case 'x1_x2':
              v = (x1 * x2) / 18.0;
              break;
            case 'sin_x1':
              v = Math.sin(x1);
              break;
            case 'sin_x2':
              v = Math.sin(x2);
              break;
          }

          const t = Math.max(-1, Math.min(1, v));
          const isPos = t > 0;
          const alpha = Math.min(0.85, Math.abs(t) * 0.85);

          const rCol = isPos ? (isCarbon ? 229 : 200) : isCarbon ? 78 : 43;
          const gCol = isPos ? (isCarbon ? 131 : 90) : isCarbon ? 137 : 91;
          const bCol = isPos ? (isCarbon ? 68 : 23) : isCarbon ? 184 : 132;

          const offset = idx * 4;
          data[offset] = rCol;
          data[offset + 1] = gCol;
          data[offset + 2] = bCol;
          data[offset + 3] = Math.floor(alpha * 255);
          idx++;
        }
      }

      ctx.putImageData(imgData, 0, 0);
    }, [feat, isCarbon]);

    return <canvas ref={canvasRef} className="w-full h-full block" />;
  }
);

// Mini 24x24 Preview Canvas for each hidden neuron node
interface NeuronMiniPreviewProps {
  values: number[];
  isCarbon: boolean;
}

const NeuronMiniPreview: React.FC<NeuronMiniPreviewProps> = ({ values, isCarbon }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const res = 24;
    canvas.width = res;
    canvas.height = res;

    const imgData = ctx.createImageData(res, res);
    const data = imgData.data;

    for (let i = 0; i < values.length; i++) {
      const v = values[i];
      const t = Math.tanh(v);
      const isPos = t > 0;
      const alpha = Math.min(0.85, Math.abs(t) * 0.85);

      const r = isPos ? (isCarbon ? 229 : 200) : isCarbon ? 78 : 43;
      const g = isPos ? (isCarbon ? 131 : 90) : isCarbon ? 137 : 91;
      const b = isPos ? (isCarbon ? 68 : 23) : isCarbon ? 184 : 132;

      const off = i * 4;
      data[off] = r;
      data[off + 1] = g;
      data[off + 2] = b;
      data[off + 3] = Math.floor(alpha * 255);
    }

    ctx.putImageData(imgData, 0, 0);
  }, [values, isCarbon]);

  return <canvas ref={canvasRef} className="w-full h-full block" />;
};
