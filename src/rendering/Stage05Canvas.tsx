import React, { useState, useMemo, useCallback } from 'react';
import { AlgorithmPlugin, VisualArtifact, SpatialVisionPipelineArtifact, VisionLayerArtifact } from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { DirtyManager } from './dirty-manager';
import { applyBrush, shiftGridRight } from '../datasets/visionGrid';
import { mapOutputToInputBoundingBox } from '../algorithms/stage05/visionMath';
import { LatexSpan } from '../utils/katex-render';
import { Zap } from 'lucide-react';

export interface Stage05CanvasProps {
  theme: ThemeMode;
  artifacts: VisualArtifact[];
  dirtyManager: DirtyManager;
  algorithm: AlgorithmPlugin;
  hyperparams?: Record<string, number | string>;
  inputGrid?: Float32Array;
  onUpdateInputGrid?: (grid: Float32Array) => void;
  onUpdateHyperparam?: (key: string, value: number | string) => void;
}

export const Stage05Canvas: React.FC<Stage05CanvasProps> = ({
  theme,
  artifacts,
  dirtyManager,
  algorithm,
  hyperparams = {},
  inputGrid: externalInputGrid,
  onUpdateInputGrid,
  onUpdateHyperparam,
}) => {
  const isCarbon = theme === 'carbon';

  // Find active spatial vision pipeline artifact
  const pipelineArtifact = useMemo(() => {
    return artifacts.find((a): a is SpatialVisionPipelineArtifact => a.type === 'spatial-vision-pipeline');
  }, [artifacts]);

  // Local fallback grid if not provided externally
  const [localGrid, setLocalGrid] = useState<Float32Array>(() => new Float32Array(256));
  const activeInputGrid = externalInputGrid || pipelineArtifact?.layers[0]?.channels[0]?.data || localGrid;

  // Brush tool state
  const [brushMode, setBrushMode] = useState<'draw' | 'erase'>('draw');
  const [brushSize, setBrushSize] = useState<1 | 2>(1);
  const [isDrawing, setIsDrawing] = useState(false);

  // Sliding window / hovered pixel state
  const [hoveredFeaturePixel, setHoveredFeaturePixel] = useState<{
    layerIndex: number;
    channelIndex: number;
    row: number;
    col: number;
    val: number;
  } | null>(null);

  // ViT selected query patch & hovered attention cell
  const [selectedPatchId, setSelectedPatchId] = useState<number>(5);
  const [hoveredAttentionCell, setHoveredAttentionCell] = useState<{
    query: number;
    key: number;
    weight: number;
  } | null>(null);

  const isViT = algorithm.id === 'vit';
  const isResNet = algorithm.id === 'cnn-resnet';

  // Handle drawing on the 16x16 input pad
  const handlePadPixelAction = useCallback(
    (row: number, col: number) => {
      const val = brushMode === 'draw' ? 1.0 : 0.0;
      const next = applyBrush(activeInputGrid, col, row, val, brushSize);
      if (onUpdateInputGrid) {
        onUpdateInputGrid(next);
      } else {
        setLocalGrid(next);
      }
      dirtyManager.markDirty();
    },
    [activeInputGrid, brushMode, brushSize, onUpdateInputGrid, dirtyManager]
  );

  const handlePadMouseDown = (r: number, c: number) => {
    setIsDrawing(true);
    handlePadPixelAction(r, c);
  };

  const handlePadMouseEnter = (r: number, c: number) => {
    if (isDrawing) {
      handlePadPixelAction(r, c);
    }
  };

  const handlePadMouseUp = () => {
    setIsDrawing(false);
  };

  // Active sliding window coords
  const slidingWindow = pipelineArtifact?.slidingWindow;
  const winRow = slidingWindow?.row ?? 0;
  const winCol = slidingWindow?.col ?? 0;
  const kSize = slidingWindow?.kernelSize ?? 3;
  const winStride = slidingWindow?.stride ?? 1;
  const winPadding = slidingWindow?.padding ?? 0;

  // Active Receptive Field bounding box on Input (16x16)
  const activeRfBoundingBox = useMemo((): [number, number, number, number] | null => {
    if (hoveredFeaturePixel && pipelineArtifact) {
      const layer = pipelineArtifact.layers[hoveredFeaturePixel.layerIndex];
      const rf = layer?.effectiveReceptiveField;
      if (rf) {
        return mapOutputToInputBoundingBox(
          hoveredFeaturePixel.row,
          hoveredFeaturePixel.col,
          rf.size,
          rf.stride,
          rf.padding ?? 0,
          16
        );
      }
    }
    // Default to sliding window on first layer if not hovering feature map
    if (!isViT && slidingWindow) {
      const minR = Math.max(0, winRow * winStride - winPadding);
      const minC = Math.max(0, winCol * winStride - winPadding);
      const maxR = Math.min(15, minR + kSize - 1);
      const maxC = Math.min(15, minC + kSize - 1);
      return [minR, minC, maxR, maxC];
    }
    return null;
  }, [hoveredFeaturePixel, pipelineArtifact, isViT, slidingWindow, winRow, winCol, winStride, winPadding, kSize]);

  // ViT Tokens & Attention
  const vitTokens = pipelineArtifact?.vitTokens;
  const activeQueryPatch = vitTokens
    ? (hyperparams.selectedQueryPatch !== undefined ? Number(hyperparams.selectedQueryPatch) : selectedPatchId)
    : 5;
  const attentionWeights = vitTokens?.attentionWeights;
  const numPatches = vitTokens?.numPatches ?? 16;
  const patchDim = vitTokens?.patchSize ?? 4;

  // Extract query attention weights for routing beams
  const queryAttentionScores = useMemo(() => {
    if (!attentionWeights || !vitTokens) return [];
    const totalTokens = numPatches + 1;
    const queryIdx = activeQueryPatch < numPatches ? activeQueryPatch + 1 : 0; // offset by [CLS]
    const scores: { targetPatch: number; weight: number }[] = [];

    for (let p = 0; p < numPatches; p++) {
      const targetIdx = p + 1;
      const w = attentionWeights[queryIdx * totalTokens + targetIdx];
      scores.push({ targetPatch: p, weight: w });
    }
    return scores;
  }, [attentionWeights, vitTokens, activeQueryPatch, numPatches]);

  return (
    <div
      className="flex-1 w-full h-full flex flex-col overflow-hidden select-none bg-[var(--bg-primary)] font-sans"
      onMouseUp={handlePadMouseUp}
      onMouseLeave={handlePadMouseUp}
    >
      {/* Upper 3-Zone Workspace */}
      <div className="flex-1 w-full min-h-0 flex flex-row overflow-hidden">
        {/* ========================================================= */}
        {/* ZONE A: Interactive 16x16 Input Pixel Pad & Sliding Window */}
        {/* ========================================================= */}
        <section className="w-[30%] min-w-[280px] max-w-[360px] border-r border-[var(--border-hairline)] flex flex-col p-4 bg-[var(--bg-elevated)] overflow-y-auto">
          {/* Header Strip */}
          <div className="flex items-center justify-between pb-2 mb-2 border-b border-[var(--border-hairline)]">
            <div className="flex items-center gap-1.5">
              <span className="text-[12px] font-semibold text-[var(--text-primary)]">Input Pixel Pad</span>
              <span className="font-mono text-[9px] px-1 py-0.5 rounded bg-[var(--bg-subtle)] text-[var(--text-secondary)] border border-[var(--border-hairline)]">
                16 × 16 px
              </span>
            </div>
            <span className="text-[9px] font-mono text-[var(--text-muted)]">Draw / Edit</span>
          </div>

          {/* Quick Brush / Invariance Actions Toolbar */}
          <div className="flex flex-wrap items-center justify-between gap-1 mb-3">
            <div className="flex items-center gap-1">
              <button
                className={`text-[9px] font-mono px-2 py-0.5 border rounded-xs transition-colors ${
                  brushMode === 'draw'
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] border-[var(--text-primary)]'
                    : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] border-[var(--border-hairline)]'
                }`}
                onClick={() => setBrushMode('draw')}
                title="Draw Brush [1.0]"
              >
                Draw
              </button>
              <button
                className={`text-[9px] font-mono px-2 py-0.5 border rounded-xs transition-colors ${
                  brushMode === 'erase'
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] border-[var(--text-primary)]'
                    : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] border-[var(--border-hairline)]'
                }`}
                onClick={() => setBrushMode('erase')}
                title="Erase Brush [0.0]"
              >
                Erase
              </button>
              <button
                className={`text-[9px] font-mono px-1.5 py-0.5 border rounded-xs transition-colors ${
                  brushSize === 2
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] border-[var(--text-primary)]'
                    : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] border-[var(--border-hairline)]'
                }`}
                onClick={() => setBrushSize(brushSize === 1 ? 2 : 1)}
                title="Toggle Brush Size (1px or 2px)"
              >
                {brushSize}px
              </button>
            </div>
            <button
              className="text-[9px] font-mono px-2 py-0.5 border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] rounded-xs shrink-0"
              onClick={() => {
                const shifted = shiftGridRight(activeInputGrid, 1);
                if (onUpdateInputGrid) onUpdateInputGrid(shifted);
                else setLocalGrid(shifted);
              }}
              title="Shift Pixel +1px Right (Tests Translation Invariance)"
            >
              +1px Right
            </button>
          </div>

          {/* 16x16 Pixel Grid Container */}
          <div className="relative mx-auto my-auto w-[256px] h-[256px] border border-[var(--border-hairline)] bg-[var(--bg-primary)] shadow-sm shrink-0">
            {/* Base 16x16 Pixels */}
            <div
              className="grid w-full h-full"
              style={{
                gridTemplateColumns: 'repeat(16, minmax(0, 1fr))',
                gridTemplateRows: 'repeat(16, minmax(0, 1fr))',
              }}
            >
              {Array.from({ length: 256 }).map((_, idx) => {
                const r = Math.floor(idx / 16);
                const c = idx % 16;
                const intensity = activeInputGrid[idx] ?? 0;

                // Check if pixel falls inside active Effective Receptive Field
                const isInsideRf =
                  activeRfBoundingBox &&
                  r >= activeRfBoundingBox[0] &&
                  r <= activeRfBoundingBox[2] &&
                  c >= activeRfBoundingBox[1] &&
                  c <= activeRfBoundingBox[3];

                // ViT: Check which patch this pixel belongs to
                const patchRow = Math.floor(r / patchDim);
                const patchCol = Math.floor(c / patchDim);
                const patchId = patchRow * (16 / patchDim) + patchCol;
                const isQueryPatch = isViT && patchId === activeQueryPatch;
                const isAttentionKey = isViT && hoveredAttentionCell !== null && hoveredAttentionCell.key === (patchId + 1);
                const isAttentionQuery = isViT && hoveredAttentionCell !== null && hoveredAttentionCell.query === (patchId + 1);

                const bgIntensity = isCarbon
                  ? Math.round(intensity * 230 + 20)
                  : Math.round(250 - intensity * 230);
                const bgColor = `rgb(${bgIntensity}, ${bgIntensity}, ${bgIntensity})`;

                return (
                  <div
                    key={idx}
                    className={`relative border-[0.5px] cursor-crosshair transition-colors duration-75 ${
                      isCarbon ? 'border-[#222220]' : 'border-[#EAE8E2]'
                    } ${
                      isAttentionQuery || isQueryPatch
                        ? 'ring-2 ring-[var(--accent-amber)] z-10'
                        : isAttentionKey
                        ? 'ring-2 ring-[var(--pigment-class-a)] z-10'
                        : isInsideRf
                        ? 'ring-1 ring-[var(--pigment-class-b)] z-10'
                        : ''
                    }`}
                    style={{ backgroundColor: bgColor }}
                    onClick={() => {
                      if (isViT) {
                        setSelectedPatchId(patchId);
                        onUpdateHyperparam?.('selectedQueryPatch', patchId);
                      }
                    }}
                    onMouseDown={() => handlePadMouseDown(r, c)}
                    onMouseEnter={() => handlePadMouseEnter(r, c)}
                  >
                    {/* Subtle ViT Patch Token Badges */}
                    {isViT && r % patchDim === 0 && c % patchDim === 0 && (
                      <span className="absolute top-0 left-0 text-[7px] font-mono text-[var(--accent-amber)] opacity-70 px-0.5 leading-none pointer-events-none">
                        #{patchId}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>

            {/* SVG Overlay: Sliding KxK Window / Receptive Field Ray-Tracing */}
            <svg
              className="absolute inset-0 w-full h-full pointer-events-none z-10"
              viewBox="0 0 256 256"
              preserveAspectRatio="none"
            >
              {/* CNN Sliding Window Box */}
              {!isViT && activeRfBoundingBox && (
                <rect
                  x={activeRfBoundingBox[1] * 16}
                  y={activeRfBoundingBox[0] * 16}
                  width={(activeRfBoundingBox[3] - activeRfBoundingBox[1] + 1) * 16}
                  height={(activeRfBoundingBox[2] - activeRfBoundingBox[0] + 1) * 16}
                  fill="rgba(82, 118, 184, 0.15)"
                  stroke="var(--pigment-class-b)"
                  strokeWidth={1.5}
                  strokeDasharray="2,2"
                />
              )}

            {/* ViT Attention Routing Beams from Query Patch */}
            {isViT &&
              queryAttentionScores.map(({ targetPatch, weight }) => {
                if (weight < 0.03 || targetPatch === activeQueryPatch) return null;
                const gridPatchesPerRow = 16 / patchDim;
                const srcR = (Math.floor(activeQueryPatch / gridPatchesPerRow) + 0.5) * patchDim * 16;
                const srcC = ((activeQueryPatch % gridPatchesPerRow) + 0.5) * patchDim * 16;
                const tgtR = (Math.floor(targetPatch / gridPatchesPerRow) + 0.5) * patchDim * 16;
                const tgtC = ((targetPatch % gridPatchesPerRow) + 0.5) * patchDim * 16;

                return (
                  <line
                    key={targetPatch}
                    x1={srcC}
                    y1={srcR}
                    x2={tgtC}
                    y2={tgtR}
                    stroke="var(--accent-amber)"
                    strokeWidth={Math.max(1, weight * 7)}
                    strokeOpacity={Math.min(0.9, weight * 3)}
                  />
                );
              })}
          </svg>
        </div>

        {/* Live Window / Patch Inspector Footnote */}
        <div className="mt-4 pt-3 border-t border-[var(--border-hairline)] text-[10px] font-mono text-[var(--text-secondary)] space-y-1">
          {isViT ? (
            <div className="flex items-center justify-between">
              <span>Query Patch Token:</span>
              <span className="font-semibold text-[var(--accent-amber)]">
                Patch #{activeQueryPatch} ({patchDim}x{patchDim} px)
              </span>
            </div>
          ) : (
            <>
              <div className="flex items-center justify-between">
                <span>Effective Receptive Field:</span>
                <span className="font-semibold text-[var(--pigment-class-b)]">
                  {activeRfBoundingBox
                    ? `${activeRfBoundingBox[2] - activeRfBoundingBox[0] + 1} × ${
                        activeRfBoundingBox[3] - activeRfBoundingBox[1] + 1
                      } px`
                    : `${kSize} × ${kSize} px`}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span>Kernel Stride / Padding:</span>
                <span>
                  S={winStride}, P={winPadding}
                </span>
              </div>
            </>
          )}
        </div>
      </section>

      {/* ========================================================= */}
      {/* ZONE B: Multi-Stage Architecture & Feature Map Inspector  */}
      {/* ========================================================= */}
      <section className="flex-1 min-w-0 flex flex-col p-4 overflow-y-auto bg-[var(--bg-primary)]">
        {/* Stage Header */}
        <div className="flex items-center justify-between pb-2 mb-3 border-b border-[var(--border-hairline)]">
          <div className="flex items-center gap-2">
            <span className="text-[12px] font-semibold text-[var(--text-primary)]">
              Glass-Box Feature Map Pipeline
            </span>
            <span className="text-[10px] font-mono text-[var(--text-muted)]">
              ({pipelineArtifact?.layers.length ?? 0} Sequential Stages)
            </span>
          </div>

          {/* Stepping controls for raster scanning */}
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono text-[var(--text-muted)]">
              Sliding Cursor: ({winRow}, {winCol})
            </span>
          </div>
        </div>

        {/* Sequential Architecture Pipeline Stages */}
        <div className="flex flex-col gap-5 pb-6">
          {pipelineArtifact?.layers.map((layer: VisionLayerArtifact, lIdx: number) => {
            const [cCount, h, w] = layer.shape;
            const isSkip = layer.type === 'skip-add';
            const isAttention = layer.type === 'self-attention';
            const isDenseHead = layer.type === 'dense-head';
            const skipConn = isResNet && isSkip
              ? pipelineArtifact?.resnetSkipConnections?.find((sc) => sc.targetLayerIndex === lIdx)
              : null;

            return (
              <div
                key={layer.id}
                className="flex flex-col p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-elevated)] shadow-sm"
              >
                {/* Stage Meta Header */}
                <div className="flex items-center justify-between pb-1.5 mb-2 border-b border-[var(--border-hairline)]">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] font-bold text-[var(--text-primary)]">
                      {layer.name}
                    </span>
                    <span className="font-mono text-[9px] px-1 py-0.2 rounded bg-[var(--bg-subtle)] text-[var(--text-secondary)]">
                      {cCount} ch × {h} × {w}
                    </span>
                  </div>

                  {layer.effectiveReceptiveField && (
                    <span className="font-mono text-[9px] text-[var(--pigment-class-b)]">
                      Receptive Field: {layer.effectiveReceptiveField.size}×{layer.effectiveReceptiveField.size} px
                    </span>
                  )}
                </div>

                {/* ResNet Identity Skip Highway Proof Arc */}
                {isResNet && isSkip && (
                  <div className="mb-2 p-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] flex items-center justify-between text-[10px] font-mono">
                    <span className="flex items-center gap-1.5 text-[var(--pigment-class-a)] font-medium">
                      <Zap size={11} /> Identity Skip Highway: H(x) = F(x) ⊕ x
                    </span>
                    <span className="text-[var(--text-muted)]">
                      {hyperparams.skipConnection !== 'disabled' ? 'Gradient Preserved: 100%' : 'SEVERED (Signal Decaying)'}
                    </span>
                  </div>
                )}

                {/* ResNet Residual Decomposition View: x ⊕ F(x) = H(x) */}
                {isResNet && isSkip && skipConn && skipConn.xMap && skipConn.fxMap && (
                  <div className="flex flex-wrap items-center gap-3 py-1">
                    {/* 1. Identity Input x */}
                    <div className="flex flex-col items-center gap-1">
                      <div className="relative border border-[var(--border-hairline)] bg-[var(--bg-primary)] shadow-inner w-[64px] h-[64px]">
                        <div
                          className="grid w-full h-full"
                          style={{
                            gridTemplateColumns: 'repeat(16, minmax(0, 1fr))',
                            gridTemplateRows: 'repeat(16, minmax(0, 1fr))',
                          }}
                        >
                          {Array.from({ length: 256 }).map((_, pIdx) => {
                            const val = Math.max(0, Math.min(1, Math.abs(skipConn.xMap![pIdx] ?? 0)));
                            const bgI = isCarbon ? Math.round(val * 230 + 20) : Math.round(250 - val * 230);
                            return (
                              <div
                                key={pIdx}
                                className={`border-[0.5px] ${isCarbon ? 'border-[#222220]' : 'border-[#EAE8E2]'}`}
                                style={{ backgroundColor: `rgb(${bgI}, ${bgI}, ${bgI})` }}
                              />
                            );
                          })}
                        </div>
                      </div>
                      <span className="font-mono text-[9px] text-[var(--text-secondary)] font-semibold">x (Identity)</span>
                    </div>

                    <span className="text-[14px] font-mono font-bold text-[var(--pigment-class-a)]">⊕</span>

                    {/* 2. Residual Delta F(x) */}
                    <div className="flex flex-col items-center gap-1">
                      <div className="relative border border-[var(--border-hairline)] bg-[var(--bg-primary)] shadow-inner w-[64px] h-[64px]">
                        <div
                          className="grid w-full h-full"
                          style={{
                            gridTemplateColumns: 'repeat(16, minmax(0, 1fr))',
                            gridTemplateRows: 'repeat(16, minmax(0, 1fr))',
                          }}
                        >
                          {Array.from({ length: 256 }).map((_, pIdx) => {
                            const val = Math.max(0, Math.min(1, Math.abs(skipConn.fxMap![pIdx] ?? 0)));
                            const bgI = isCarbon ? Math.round(val * 230 + 20) : Math.round(250 - val * 230);
                            return (
                              <div
                                key={pIdx}
                                className={`border-[0.5px] ${isCarbon ? 'border-[#222220]' : 'border-[#EAE8E2]'}`}
                                style={{ backgroundColor: `rgb(${bgI}, ${bgI}, ${bgI})` }}
                              />
                            );
                          })}
                        </div>
                      </div>
                      <span className="font-mono text-[9px] text-[var(--pigment-class-b)] font-semibold">F(x) (Conv Δ)</span>
                    </div>

                    <span className="text-[14px] font-mono font-bold text-[var(--text-muted)]">=</span>

                    {/* 3. Output H(x) */}
                    <div className="flex flex-col items-center gap-1">
                      <div className="relative border border-[var(--border-hairline)] bg-[var(--bg-primary)] shadow-inner w-[64px] h-[64px]">
                        <div
                          className="grid w-full h-full"
                          style={{
                            gridTemplateColumns: 'repeat(16, minmax(0, 1fr))',
                            gridTemplateRows: 'repeat(16, minmax(0, 1fr))',
                          }}
                        >
                          {Array.from({ length: 256 }).map((_, pIdx) => {
                            const val = Math.max(0, Math.min(1, Math.abs(layer.channels[0]?.data[pIdx] ?? 0)));
                            const bgI = isCarbon ? Math.round(val * 230 + 20) : Math.round(250 - val * 230);
                            return (
                              <div
                                key={pIdx}
                                className={`border-[0.5px] ${isCarbon ? 'border-[#222220]' : 'border-[#EAE8E2]'}`}
                                style={{ backgroundColor: `rgb(${bgI}, ${bgI}, ${bgI})` }}
                              />
                            );
                          })}
                        </div>
                      </div>
                      <span className="font-mono text-[9px] text-[var(--text-primary)] font-semibold">H(x) (Output)</span>
                    </div>
                  </div>
                )}

                {/* ViT Interactive Self-Attention Matrix */}
                {isViT && isAttention && (
                  <div className="flex flex-col gap-2 py-1">
                    <div className="flex items-center justify-between text-[10px] font-mono text-[var(--text-muted)]">
                      <span>Attention Heatmap (Rows: Query, Cols: Key)</span>
                      <span>Hover cell to highlight tokens</span>
                    </div>
                    <div className="relative border border-[var(--border-hairline)] bg-[var(--bg-primary)] p-1 shadow-inner max-w-[240px]">
                      <div
                        className="grid w-full aspect-square"
                        style={{
                          gridTemplateColumns: `repeat(${w}, minmax(0, 1fr))`,
                          gridTemplateRows: `repeat(${h}, minmax(0, 1fr))`,
                        }}
                      >
                        {Array.from({ length: h * w }).map((_, pIdx) => {
                          const r = Math.floor(pIdx / w);
                          const c = pIdx % w;
                          const weight = layer.channels[0]?.data[pIdx] ?? 0;
                          const isRowActive = r === (activeQueryPatch < (w - 1) ? activeQueryPatch + 1 : 0);
                          const isHovered =
                            hoveredAttentionCell?.query === r && hoveredAttentionCell?.key === c;

                          const intensity = Math.min(1.0, Math.max(0, weight * 4.5));
                          const bgI = isCarbon ? Math.round(intensity * 230 + 20) : Math.round(250 - intensity * 230);
                          const bg = isCarbon
                            ? `rgb(${Math.round(bgI * 1.05)}, ${Math.round(bgI * 0.9)}, ${bgI})`
                            : `rgb(${bgI}, ${Math.round(bgI * 0.95)}, ${Math.round(bgI * 0.85)})`;

                          return (
                            <div
                              key={pIdx}
                              className={`border-[0.5px] cursor-pointer transition-colors duration-75 ${
                                isCarbon ? 'border-[#222220]' : 'border-[#EAE8E2]'
                              } ${
                                isHovered
                                  ? 'ring-1 ring-[var(--accent-amber)] z-20'
                                  : isRowActive
                                  ? 'outline outline-1 outline-[var(--accent-amber)]/40'
                                  : ''
                              }`}
                              style={{ backgroundColor: bg }}
                              onMouseEnter={() => setHoveredAttentionCell({ query: r, key: c, weight })}
                              onMouseLeave={() => setHoveredAttentionCell(null)}
                              onClick={() => {
                                if (r > 0) {
                                  setSelectedPatchId(r - 1);
                                  onUpdateHyperparam?.('selectedQueryPatch', r - 1);
                                }
                              }}
                              title={`Query #${r === 0 ? 'CLS' : r - 1} → Key #${c === 0 ? 'CLS' : c - 1}: ${(weight * 100).toFixed(1)}%`}
                            />
                          );
                        })}
                      </div>
                    </div>
                    {hoveredAttentionCell && (
                      <span className="font-mono text-[9px] text-[var(--accent-amber)]">
                        Query #{hoveredAttentionCell.query === 0 ? 'CLS' : hoveredAttentionCell.query - 1} → Key #{hoveredAttentionCell.key === 0 ? 'CLS' : hoveredAttentionCell.key - 1}: {(hoveredAttentionCell.weight * 100).toFixed(2)}% attention weight
                      </span>
                    )}
                  </div>
                )}

                {/* Dense FC Node Array View (AlexNet) */}
                {isDenseHead && (
                  <div className="flex flex-wrap items-center gap-1.5 p-2 rounded bg-[var(--bg-primary)] border border-[var(--border-hairline)]">
                    {Array.from(layer.channels[0]?.data || []).map((val, nIdx) => {
                      const isDropped = val === 0;
                      return (
                        <div
                          key={nIdx}
                          className={`w-8 h-8 rounded-xs flex flex-col items-center justify-center font-mono text-[8px] border transition-colors ${
                            isDropped
                              ? 'border-dashed border-red-400/50 bg-red-500/10 text-red-400 line-through'
                              : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[var(--text-primary)] font-semibold'
                          }`}
                          title={`Dense Node #${nIdx}: ${isDropped ? 'Dropped (p=0.5)' : `Active (${val.toFixed(2)})`}`}
                        >
                          <span>#{nIdx}</span>
                          <span className="text-[7px] opacity-75">{isDropped ? '∅' : val.toFixed(1)}</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                {/* Standard Feature Maps Grid (When not ResNet decomposed or ViT Attention) */}
                {(!isResNet || !isSkip) && !isAttention && !isDenseHead && (
                  <div className="flex flex-wrap items-center gap-4">
                    {layer.channels.map((chan, cIdx) => {
                      const canvasSize = Math.max(48, Math.min(96, Math.floor(192 / Math.max(h, 1))));

                      return (
                        <div key={chan.id} className="flex flex-col items-center gap-1">
                          {/* Interactive Feature Map Canvas / Pixel Matrix */}
                          <div
                            className="relative border border-[var(--border-hairline)] bg-[var(--bg-primary)] shadow-inner"
                            style={{ width: `${canvasSize}px`, height: `${canvasSize}px` }}
                          >
                            <div
                              className="grid w-full h-full"
                              style={{
                                gridTemplateColumns: `repeat(${w}, minmax(0, 1fr))`,
                                gridTemplateRows: `repeat(${h}, minmax(0, 1fr))`,
                              }}
                            >
                              {Array.from({ length: h * w }).map((_, pIdx) => {
                                const r = Math.floor(pIdx / w);
                                const c = pIdx % w;
                                const val = chan.data[pIdx] ?? 0;

                                const isHovered =
                                  hoveredFeaturePixel?.layerIndex === lIdx &&
                                  hoveredFeaturePixel?.channelIndex === cIdx &&
                                  hoveredFeaturePixel?.row === r &&
                                  hoveredFeaturePixel?.col === c;

                                const normVal = Math.max(0, Math.min(1, Math.abs(val)));
                                const bgIntensity = isCarbon
                                  ? Math.round(normVal * 230 + 20)
                                  : Math.round(250 - normVal * 230);
                                const bgColor = `rgb(${bgIntensity}, ${bgIntensity}, ${bgIntensity})`;

                                return (
                                  <div
                                    key={pIdx}
                                    className={`relative border-[0.5px] cursor-pointer transition-colors duration-75 ${
                                      isCarbon ? 'border-[#222220]' : 'border-[#EAE8E2]'
                                    } ${isHovered ? 'ring-2 ring-[var(--pigment-class-b)] z-20' : ''}`}
                                    style={{ backgroundColor: bgColor }}
                                    onMouseEnter={() =>
                                      setHoveredFeaturePixel({
                                        layerIndex: lIdx,
                                        channelIndex: cIdx,
                                        row: r,
                                        col: c,
                                        val,
                                      })
                                    }
                                    onMouseLeave={() => setHoveredFeaturePixel(null)}
                                    title={`Output (${r}, ${c}): ${val.toFixed(3)}`}
                                  />
                                );
                              })}
                            </div>
                          </div>

                          <span className="font-mono text-[9px] text-[var(--text-muted)] truncate max-w-[80px]">
                            {chan.label}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* ========================================================= */}
      {/* ZONE C: Live Spatial Dimension & Class Logits Strip       */}
      {/* ========================================================= */}
      <section className="w-[25%] min-w-[240px] max-w-[300px] border-l border-[var(--border-hairline)] flex flex-col p-4 bg-[var(--bg-elevated)]">
        {/* Header */}
        <div className="flex items-center justify-between pb-2 mb-3 border-b border-[var(--border-hairline)]">
          <span className="text-[12px] font-semibold text-[var(--text-primary)]">Spatial Telemetry</span>
          <span className="font-mono text-[9px] text-[var(--text-muted)]">Live Flow</span>
        </div>

        {/* 1. Spatial Dimension Formula Card */}
        <div className="p-3 mb-4 rounded border border-[var(--border-hairline)] bg-[var(--bg-primary)] shadow-sm space-y-2">
          <span className="text-[10px] font-mono font-semibold uppercase text-[var(--text-muted)]">
            Spatial Dimension Formula
          </span>
          {isViT ? (
            <>
              <div className="font-mono text-[12px] text-[var(--text-primary)] leading-relaxed">
                <LatexSpan expression="N = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right)" displayMode={true} />
              </div>
              <div className="text-[10px] font-mono text-[var(--text-secondary)] pt-1 border-t border-[var(--border-hairline)]">
                <LatexSpan
                  expression={`N = \\left(\\frac{16}{${patchDim}}\\right) \\times \\left(\\frac{16}{${patchDim}}\\right) = ${16 / patchDim} \\times ${16 / patchDim} = ${numPatches}\\text{ tokens } (+1\\text{ [CLS]} = ${numPatches + 1})`}
                  displayMode={false}
                />
              </div>
            </>
          ) : (
            <>
              <div className="font-mono text-[12px] text-[var(--text-primary)] leading-relaxed">
                <LatexSpan expression="O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1" displayMode={true} />
              </div>
              <div className="text-[10px] font-mono text-[var(--text-secondary)] pt-1 border-t border-[var(--border-hairline)]">
                <LatexSpan
                  expression={`16 \\to \\left\\lfloor \\frac{16 - ${kSize} + 2(${winPadding})}{${winStride}} \\right\\rfloor + 1 = ${
                    Math.floor((16 - kSize + 2 * winPadding) / winStride) + 1
                  }`}
                  displayMode={false}
                />
              </div>
            </>
          )}
        </div>

        {/* 2. 4-Class Output Probability Bar Chart */}
        <div className="p-3 mb-4 rounded border border-[var(--border-hairline)] bg-[var(--bg-primary)] shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono font-semibold uppercase text-[var(--text-muted)]">
              Class Probabilities
            </span>
            <span className="font-mono text-[9px] text-[var(--text-secondary)]">Softmax Logits</span>
          </div>

          <div className="flex flex-col gap-2">
            {pipelineArtifact?.logits.map((item, idx) => {
              const isWinner = idx === pipelineArtifact.predictedClass;
              return (
                <div key={item.className} className="flex flex-col text-[10px] font-mono">
                  <div className="flex justify-between items-center mb-0.5">
                    <span className={isWinner ? 'font-bold text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}>
                      {item.className}
                    </span>
                    <span className={isWinner ? 'font-bold text-[var(--pigment-class-a)]' : 'text-[var(--text-muted)]'}>
                      {(item.probability * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full h-1.5 bg-[var(--bg-subtle)] rounded-full overflow-hidden border border-[var(--border-hairline)]">
                    <div
                      className={`h-full transition-all duration-150 ${
                        isWinner ? 'bg-[var(--pigment-class-a)]' : 'bg-[var(--text-muted)]'
                      }`}
                      style={{ width: `${Math.max(2, item.probability * 100)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 3. Per-Layer Backward Gradient Norm Bars */}
        <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-primary)] shadow-sm">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-mono font-semibold uppercase text-[var(--text-muted)]">
              Gradient Norm Flow (||∇W||)
            </span>
            <span className="font-mono text-[8px] text-[var(--pigment-class-b)]">Backward Survival</span>
          </div>

          <div className="flex flex-col gap-2">
            {pipelineArtifact?.gradientNorms.map((gn) => (
              <div key={gn.layerName} className="flex flex-col text-[10px] font-mono">
                <div className="flex justify-between items-center mb-0.5">
                  <span className="text-[var(--text-secondary)] truncate max-w-[140px]">{gn.layerName}</span>
                  <span className="font-semibold text-[var(--text-primary)]">{gn.norm.toFixed(4)}</span>
                </div>
                <div className="w-full h-1.5 bg-[var(--bg-subtle)] rounded-full overflow-hidden border border-[var(--border-hairline)]">
                  <div
                    className="h-full bg-[var(--pigment-class-b)] transition-all duration-150"
                    style={{ width: `${Math.min(100, Math.max(2, gn.norm * 500))}%` }}
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Translation Invariance Readout */}
          {pipelineArtifact?.translationDelta1px !== undefined && (
            <div className="mt-3 pt-2 border-t border-[var(--border-hairline)] flex items-center justify-between text-[10px] font-mono">
              <span className="text-[var(--text-muted)]">+1px Shift Delta:</span>
              <span className="font-semibold text-[var(--text-primary)]">
                {pipelineArtifact.translationDelta1px.toFixed(4)}
              </span>
            </div>
          )}
        </div>
      </section>
      </div>

      {/* ========================================================= */}
      {/* ZONE D: Swiss Bottom Status & Legend Bar                  */}
      {/* ========================================================= */}
      <footer className="w-full flex flex-wrap items-center justify-between px-3 py-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] gap-2 select-none shrink-0">
        <div className="flex items-center gap-3 text-[var(--text-muted)] font-mono text-[9px]">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 border border-dashed border-[var(--pigment-class-b)] bg-[var(--pigment-class-b)]/20 inline-block" />
            <span>Effective Receptive Field (R_L)</span>
          </div>
          {isViT ? (
            <div className="flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-[var(--accent-amber)] inline-block" />
              <span>Self-Attention Routing Beams</span>
            </div>
          ) : isResNet ? (
            <div className="flex items-center gap-1.5">
              <Zap size={11} className="text-[var(--pigment-class-a)]" />
              <span>Identity Highway: H(x) = F(x) ⊕ x</span>
            </div>
          ) : (
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[var(--text-primary)] inline-block" />
              <span>Sliding Kernel Window (K×K)</span>
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <div className="w-12 h-1.5 rounded-full overflow-hidden bg-gradient-to-r from-neutral-200 to-neutral-900 dark:from-neutral-800 dark:to-neutral-100" />
            <span>Activation [0 → 1]</span>
          </div>
        </div>

        <div className="flex items-center gap-3 text-[var(--text-muted)] font-mono text-[9px]">
          <span>{algorithm.name}</span>
          <span className="text-[var(--border-hairline)]">|</span>
          <span>16 × 16 px (256 dims)</span>
          <span className="text-[var(--border-hairline)]">|</span>
          <span className="font-semibold text-[var(--text-primary)]">
            {hyperparams.mode === 'micro-training' ? 'Live SGD Micro-Training' : 'Architecture Inspector'}
          </span>
        </div>
      </footer>
    </div>
  );
};
