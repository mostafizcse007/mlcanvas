import React, { useEffect, useRef, useState } from 'react';
import {
  DatasetKind,
  DatasetOptions,
  PointerToolMode,
  FeatureScalingMode,
  generateTabular2dDataset,
  getDatasetTaskType,
} from '../datasets/tabular2d';
import {
  HyperparameterDef,
  StageKind,
  SpatialVisionDataset,
  SequenceDataset,
  SequencePresetKind,
  RLGridworldDataset,
  GridworldPresetKind,
  GridworldTileKind,
} from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { HyperparameterLabel } from './HyperparameterLabel';
import { getPresetGrid, VisionPresetKind } from '../datasets/visionGrid';
import { generatePresetTiles } from '../datasets/rlGridworld';
import {
  Lock,
  Unlock,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Shuffle,
} from 'lucide-react';

interface DataSidebarProps {
  options: DatasetOptions;
  theme: ThemeMode;
  pointerMode: PointerToolMode;
  activeClass: number;
  hyperparams?: Record<string, number | string>;
  advancedParams?: HyperparameterDef[];
  stage?: StageKind;
  algorithmId?: string;
  spatialVision?: SpatialVisionDataset;
  sequenceDataset?: SequenceDataset;
  onUpdateOptions: (partial: Partial<DatasetOptions>) => void;
  onSetPointerMode: (mode: PointerToolMode) => void;
  onSetActiveClass: (cls: number) => void;
  onRegenerate: () => void;
  onUpdateHyperparam?: (keyOrBatch: string | Record<string, number | string>, value?: number | string) => void;
  onSelectVisionPreset?: (preset: VisionPresetKind, variant?: 'default' | 'digit0' | 'digit7') => void;
  onUpdateSpatialVision?: (updater: (prev: SpatialVisionDataset) => SpatialVisionDataset) => void;
  onShiftInputRight?: () => void;
  onSelectSequencePreset?: (preset: SequencePresetKind) => void;
  onUpdateSequenceDataset?: (updater: (prev: SequenceDataset) => SequenceDataset) => void;
  onToggleTokenMask?: (index: number) => void;
  onSelectFocusToken?: (index: number) => void;
  rlGridworld?: RLGridworldDataset;
  activeBrush?: GridworldTileKind;
  onSelectGridworldPreset?: (preset: GridworldPresetKind) => void;
  onUpdateRLGridworld?: (updater: (prev: RLGridworldDataset) => RLGridworldDataset) => void;
  onSelectBrush?: (brush: GridworldTileKind) => void;
}

const GridworldThumbnail: React.FC<{ preset: GridworldPresetKind; isCarbon: boolean }> = ({ preset, isCarbon }) => {
  const { tiles } = generatePresetTiles(preset);
  return (
    <svg viewBox="0 0 36 36" className="w-[32px] h-[32px] block border border-[var(--border-hairline)] bg-[var(--bg-canvas)]">
      {Array.from({ length: 6 }).map((_, r) =>
        Array.from({ length: 6 }).map((_, c) => {
          const idx = r * 6 + c;
          const t = tiles[idx];
          let fill = isCarbon ? '#181817' : '#FAF9F5';
          if (t === 1) fill = isCarbon ? '#383835' : '#BDBAB0'; // wall
          else if (t === 2) fill = isCarbon ? '#C26752' : '#D94E34'; // lava
          else if (t === 3) fill = isCarbon ? '#4D8F68' : '#15803D'; // goal
          else if (t === 4) fill = isCarbon ? '#D97706' : '#B45309'; // bait
          else if (t === 5) fill = isCarbon ? '#5276B8' : '#1E40AF'; // start

          return (
            <rect
              key={`${r}-${c}`}
              x={c * 6}
              y={r * 6}
              width={6}
              height={6}
              fill={fill}
              stroke={isCarbon ? '#252523' : '#ECEAE3'}
              strokeWidth={0.5}
            />
          );
        })
      )}
    </svg>
  );
};

const VisionThumbnail: React.FC<{ grid: Float32Array; isCarbon: boolean }> = ({ grid, isCarbon }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const imgData = ctx.createImageData(16, 16);
    const data = imgData.data;
    for (let i = 0; i < 256; i++) {
      const v = grid[i];
      const val = isCarbon ? (v > 0.5 ? 255 : 17) : (v > 0.5 ? 17 : 255);
      const offset = i * 4;
      data[offset] = val;
      data[offset + 1] = val;
      data[offset + 2] = val;
      data[offset + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }, [grid, isCarbon]);

  return (
    <canvas
      ref={canvasRef}
      width={16}
      height={16}
      className="w-[32px] h-[32px] block border border-[var(--border-hairline)] [image-rendering:pixelated]"
    />
  );
};

export const DataSidebar: React.FC<DataSidebarProps> = ({
  options,
  theme,
  pointerMode,
  activeClass,
  hyperparams = {},
  advancedParams = [],
  stage,
  algorithmId,
  spatialVision,
  sequenceDataset,
  onUpdateOptions,
  onSetPointerMode,
  onSetActiveClass,
  onRegenerate,
  onUpdateHyperparam,
  onSelectVisionPreset,
  onUpdateSpatialVision,
  onShiftInputRight,
  onSelectSequencePreset,
  onUpdateSequenceDataset,
  onToggleTokenMask,
  onSelectFocusToken,
  rlGridworld,
  activeBrush = 'wall',
  onSelectGridworldPreset,
  onUpdateRLGridworld,
  onSelectBrush,
}) => {
  const isCarbon = theme === 'carbon';
  const [isAdvancedOpen, setIsAdvancedOpen] = useState(false);
  const [isSeedLocked, setIsSeedLocked] = useState(true);
  const [activeTab, setActiveTab] = useState<'classification' | 'regression'>(
    getDatasetTaskType(options.kind)
  );

  // Sync tab when dataset kind changes
  useEffect(() => {
    const task = getDatasetTaskType(options.kind);
    if (task !== activeTab) {
      setActiveTab(task);
    }
  }, [options.kind]);

  const classificationKinds: { kind: DatasetKind; label: string }[] = [
    { kind: 'linear-blobs', label: 'Linear Blobs' },
    { kind: 'concentric-circles', label: 'Concentric' },
    { kind: 'xor-quadrants', label: 'XOR Quadrants' },
    { kind: 'twin-spirals', label: 'Twin Spirals' },
  ];

  const regressionKinds: { kind: DatasetKind; label: string }[] = [
    { kind: 'sine-wave', label: '1D Sine Wave' },
    { kind: 'polynomial-cubic', label: '1D Poly Curve' },
    { kind: 'linear-trend', label: '1D Linear Trend' },
    { kind: 'linear-surface', label: '2D Surface' },
  ];

  const currentKinds = activeTab === 'classification' ? classificationKinds : regressionKinds;

  const thumbCanvasesRef = useRef<Map<DatasetKind, HTMLCanvasElement>>(new Map());

  // Render Mini Dataset Thumbnails
  useEffect(() => {
    currentKinds.forEach((k) => {
      const canvas = thumbCanvasesRef.current.get(k.kind);
      if (!canvas) return;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = isCarbon ? '#131312' : '#FAF9F5';
      ctx.fillRect(0, 0, w, h);

      ctx.strokeStyle = isCarbon ? '#2E2E2B' : '#DCDAD3';
      ctx.lineWidth = 1;
      ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

      // Generate small thumbnail sample
      const sample = generateTabular2dDataset({
        kind: k.kind,
        sampleCount: 40,
        noise: 0.05,
        testSplit: 0,
        seed: options.seed,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const pigmentA = isCarbon ? '#C26752' : '#D94E34';
      const pigmentB = isCarbon ? '#5276B8' : '#1E40AF';
      const scale = (w * 0.42) / 6.0;
      const cx = w / 2;
      const cy = h / 2;

      for (const p of sample.train) {
        const px = cx + p.x1 * scale;
        const py = cy - p.x2 * scale;
        ctx.fillStyle = activeTab === 'regression' ? pigmentA : (p.y >= 0 ? pigmentA : pigmentB);
        ctx.beginPath();
        ctx.arc(px, py, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }, [options.seed, isCarbon, activeTab]);

  const handleToggleSeedLock = () => {
    if (isSeedLocked) {
      setIsSeedLocked(false);
    } else {
      setIsSeedLocked(true);
    }
  };

  const handleNextSeed = () => {
    const nextSeed = Math.floor(Math.random() * 9000) + 1000;
    onUpdateOptions({ seed: nextSeed });
  };

  return (
    <aside className="sidebar-data" data-node-id="3:18" role="complementary" aria-label="Dataset and Parameter Controls">
      {stage === 'vision-grid' ? (
        <div className="panel-section">
          {/* Header */}
          <div className="panel-header">
            <span className="panel-title font-mono tracking-wider">VISION DATA (16×16)</span>
            <span className="font-mono text-[9px] px-1 py-0.5 rounded bg-[var(--bg-subtle)] text-[var(--text-secondary)] border border-[var(--border-hairline)]">
              4-Class Toy
            </span>
          </div>

          {/* 2x2 Spatial Vision Presets */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            {[
              { id: 'digit', label: "Digit / Symbol ('7' vs '0')", variant: 'digit7' as const },
              { id: 'edges', label: 'Edges (Vert vs Horiz)', variant: 'default' as const },
              { id: 'corner-cross', label: 'Corner & Cross', variant: 'default' as const },
              { id: 'checkerboard', label: 'Checkerboard (Hi-Freq)', variant: 'default' as const },
            ].map((p) => {
              const isSelected = (spatialVision?.preset || 'digit') === p.id;
              const previewGrid = getPresetGrid(p.id as VisionPresetKind, p.variant);

              return (
                <button
                  key={p.id}
                  className={`p-2 flex flex-col items-center gap-1.5 border text-left transition-all ${
                    isSelected
                      ? 'border-[var(--text-primary)] bg-[var(--bg-active)] shadow-sm'
                      : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--border-strong)]'
                  }`}
                  onClick={() => onSelectVisionPreset?.(p.id as VisionPresetKind, p.variant)}
                  title={p.label}
                  aria-label={`Select vision preset: ${p.label}`}
                >
                  {/* 16x16 Mini Raster Preview */}
                  <VisionThumbnail grid={previewGrid} isCarbon={isCarbon} />
                  <span className="font-mono text-[9px] text-[var(--text-primary)] text-center leading-tight truncate w-full">
                    {p.id === 'digit' ? 'Digit 7 vs 0' : p.id === 'edges' ? 'Edge Filters' : p.id === 'corner-cross' ? 'Corner & Cross' : 'Checkerboard'}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Interactive Translation Invariance & Image Controls */}
          <div className="flex flex-col gap-2 pt-2 border-t border-[var(--border-hairline)]">
            <button
              className="w-full py-1.5 px-2 flex items-center justify-center gap-1.5 border border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:bg-[var(--bg-elevated)] text-[10px] font-mono text-[var(--text-primary)] transition-colors"
              onClick={() => onShiftInputRight?.()}
              title="Shifts drawn pattern by 1px right to test Pooling Translation Invariance"
              aria-label="Shift drawn pattern 1px right to test translation invariance"
            >
              <span>Shift Input +1px Right</span>
              <span className="text-[9px] text-[var(--pigment-class-b)]">(Invariance Test)</span>
            </button>

            {/* Noise Slider */}
            <div className="slider-row">
              <div className="slider-label-row">
                <label htmlFor="vision-noise-slider" className="slider-label text-[10px]">Gaussian Noise (σ)</label>
                <span className="slider-value font-mono text-[10px]">
                  {(spatialVision?.noise ?? 0.0).toFixed(2)}
                </span>
              </div>
              <input
                id="vision-noise-slider"
                type="range"
                className="scientific-slider"
                aria-label="Gaussian noise level"
                min={0}
                max={0.4}
                step={0.02}
                value={spatialVision?.noise ?? 0.0}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  onUpdateSpatialVision?.((prev) => ({ ...prev, noise: val }));
                }}
              />
            </div>

            {/* Contrast Slider */}
            <div className="slider-row">
              <div className="slider-label-row">
                <label htmlFor="vision-contrast-slider" className="slider-label text-[10px]">Contrast Scale</label>
                <span className="slider-value font-mono text-[10px]">
                  {(spatialVision?.contrast ?? 1.0).toFixed(1)}x
                </span>
              </div>
              <input
                id="vision-contrast-slider"
                type="range"
                className="scientific-slider"
                aria-label="Contrast scale"
                min={0.5}
                max={2.0}
                step={0.1}
                value={spatialVision?.contrast ?? 1.0}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  onUpdateSpatialVision?.((prev) => ({ ...prev, contrast: val }));
                }}
              />
            </div>

            {/* Invert Toggle */}
            <div className="flex items-center justify-between text-[10px] font-mono pt-1">
              <span className="text-[var(--text-secondary)]">Invert Pixels (1 - x)</span>
              <button
                className={`px-2 py-0.5 border text-[9px] ${
                  spatialVision?.inverted
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] border-[var(--text-primary)]'
                    : 'bg-[var(--bg-subtle)] text-[var(--text-secondary)] border-[var(--border-hairline)]'
                }`}
                onClick={() =>
                  onUpdateSpatialVision?.((prev) => ({ ...prev, inverted: !prev.inverted }))
                }
                aria-label={`Invert pixel colors, currently ${spatialVision?.inverted ? 'Inverted' : 'Normal'}`}
              >
                {spatialVision?.inverted ? 'Inverted' : 'Normal'}
              </button>
            </div>
          </div>

          {/* Break It Lab Card */}
          <div className="mt-4 p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-elevated)] space-y-2">
            <div className="flex items-center gap-1.5 text-[10px] font-mono font-semibold uppercase text-[var(--pigment-class-a)]">
              <AlertTriangle size={11} />
              <span>Break It Lab</span>
            </div>

            {algorithmId === 'cnn-lenet' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Stride Blindness: Window jumps over thin lines!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => {
                      onUpdateHyperparam?.({ stride: 3, pooling: 'none' });
                    }}
                    aria-label="Break model: Set stride to 3"
                  >
                    Break: S=3
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => {
                      onUpdateHyperparam?.({ stride: 1, pooling: 'max-pool' });
                    }}
                    aria-label="Fix model: Set stride to 1 with max pool"
                  >
                    Fix: S=1+Pool
                  </button>
                </div>
              </div>
            ) : algorithmId === 'cnn-alexnet' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Dropout Starvation: Zeroes 85% of dense units!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('dropoutRate', 0.85)}
                    aria-label="Break model: Set dropout rate to 0.85"
                  >
                    Break: p=0.85
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('dropoutRate', 0.3)}
                    aria-label="Fix model: Set dropout rate to 0.30"
                  >
                    Fix: p=0.30
                  </button>
                </div>
              </div>
            ) : algorithmId === 'cnn-vgg' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Spatial Collapse (P=0): 16→14→12→10→8→6→4 px!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => {
                      onUpdateHyperparam?.({ stackedDepth: 6, padding: 'valid-0' });
                    }}
                    aria-label="Break model: Set valid padding without same padding"
                  >
                    Break: P=0
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('padding', 'same-1')}
                    aria-label="Fix model: Set same padding P=1"
                  >
                    Fix: P=1 (Same)
                  </button>
                </div>
              </div>
            ) : algorithmId === 'cnn-resnet' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Sever Skip Highway: Gradients vanish on 8 blocks!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => {
                      onUpdateHyperparam?.({ residualBlocks: 8, skipConnection: 'disabled' });
                    }}
                    aria-label="Break model: Disable residual skip connections"
                  >
                    Break: Sever Skip
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('skipConnection', 'enabled')}
                    aria-label="Fix model: Enable residual skip connection"
                  >
                    Fix: H=F(x)+x
                  </button>
                </div>
              </div>
            ) : algorithmId === 'vit' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Bag-of-Patches: Scrambling patches without 2D pos!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('positionalEmbedding', 'scrambled-off')}
                    aria-label="Break model: Disable positional embeddings"
                  >
                    Break: Off
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('positionalEmbedding', 'learned-2d')}
                    aria-label="Fix model: Enable learned 2D grid positional embeddings"
                  >
                    Fix: 2D Grid
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : stage === 'sequence-attn' ? (
        <div className="panel-section">
          {/* Header */}
          <div className="panel-header">
            <span className="panel-title font-mono tracking-wider">SEQUENCE & TOKEN DATA</span>
            <span className="font-mono text-[9px] px-1 py-0.5 rounded bg-[var(--bg-subtle)] text-[var(--text-secondary)] border border-[var(--border-hairline)]">
              T = {sequenceDataset?.length ?? 10}
            </span>
          </div>

          {/* 2x2 Interactive Sequence Presets */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            {[
              { id: 'coreference' as const, label: 'Long-Range Coreference', desc: '"trophy" vs "suitcase"' },
              { id: 'bracket' as const, label: 'Bracket / Parity Memory', desc: '[ A B ( C ) D ]' },
              { id: 'sentiment' as const, label: 'Sentiment Negation Flip', desc: '"Not only was the plot slow..."' },
              { id: 'seq2seq' as const, label: 'Seq2Seq Translation Pair', desc: 'Bengali ↔ English' },
            ].map((p) => {
              const isSelected = (sequenceDataset?.preset ?? 'coreference') === p.id;

              return (
                <button
                  key={p.id}
                  onClick={() => onSelectSequencePreset?.(p.id)}
                  className={`flex flex-col p-2 text-left rounded border transition-all ${
                    isSelected
                      ? 'border-[var(--pigment-class-b)] bg-[var(--bg-elevated)] ring-1 ring-[var(--pigment-class-b)] shadow-xs'
                      : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--border-strong)]'
                  }`}
                >
                  <span className="font-semibold text-[11px] text-[var(--text-primary)] leading-tight">
                    {p.label}
                  </span>
                  <span className="text-[9px] text-[var(--text-muted)] font-mono truncate mt-1">
                    {p.desc}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Active Token Sequence Pills */}
          <div className="mb-3">
            <div className="flex items-center justify-between text-[11px] font-medium text-[var(--text-secondary)] mb-1.5">
              <span>Sequence Tokens (Click to Focus t*):</span>
            </div>
            <div className="flex flex-wrap gap-1 p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
              {sequenceDataset?.sequence.map((tok, idx) => {
                const isFocused = idx === sequenceDataset.selectedTokenIndex;
                const isMasked = tok.isMasked;

                return (
                  <button
                    key={tok.id}
                    onClick={() => onSelectFocusToken?.(idx)}
                    className={`px-2 py-1 rounded font-mono text-[11px] transition-all border ${
                      isMasked
                        ? 'bg-amber-950/20 border-amber-500/50 text-amber-400'
                        : isFocused
                        ? 'border-[var(--pigment-class-b)] bg-[var(--bg-elevated)] text-[var(--pigment-class-b)] font-bold shadow-xs ring-1 ring-[var(--pigment-class-b)]'
                        : 'border-[var(--border-hairline)] bg-[var(--bg-elevated)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                    }`}
                  >
                    {tok.text}
                  </button>
                );
              })}
            </div>

            {/* Mask/Perturb Action Toolbar for Selected Token */}
            <div className="flex items-center gap-1.5 mt-2">
              <button
                className="flex-1 py-1 px-2 rounded bg-[var(--bg-subtle)] hover:bg-[var(--bg-elevated)] border border-[var(--border-hairline)] font-mono text-[10px] text-[var(--text-secondary)] flex items-center justify-center gap-1"
                onClick={() => {
                  if (sequenceDataset) {
                    onToggleTokenMask?.(sequenceDataset.selectedTokenIndex);
                  }
                }}
              >
                <Shuffle size={11} />
                <span>Toggle [MASK] on Token</span>
              </button>
            </div>
          </div>

          {/* Interactive Sliders: Sequence Length & Noise */}
          <div className="space-y-3 mb-4">
            <div className="slider-row">
              <div className="slider-label-row">
                <span className="slider-label">Sequence Length (T)</span>
                <span className="slider-value font-mono">{sequenceDataset?.length ?? 10}</span>
              </div>
              <input
                type="range"
                className="scientific-slider"
                min={4}
                max={10}
                step={1}
                value={sequenceDataset?.length ?? 10}
                onChange={(e) => {
                  const len = parseInt(e.target.value, 10);
                  onUpdateSequenceDataset?.((prev) => ({ ...prev, length: len }));
                }}
              />
            </div>

            <div className="slider-row">
              <div className="slider-label-row">
                <span className="slider-label">Embedding Noise (σ)</span>
                <span className="slider-value font-mono">{(sequenceDataset?.noise ?? 0).toFixed(2)}</span>
              </div>
              <input
                type="range"
                className="scientific-slider"
                min={0}
                max={0.5}
                step={0.05}
                value={sequenceDataset?.noise ?? 0}
                onChange={(e) => {
                  const noise = parseFloat(e.target.value);
                  onUpdateSequenceDataset?.((prev) => ({ ...prev, noise }));
                }}
              />
            </div>
          </div>

          {/* Break It Lab Panel for Stage 06 */}
          <div className="p-2.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] mb-3">
            <div className="flex items-center gap-1.5 text-[11px] font-semibold text-[var(--pigment-class-a)] mb-1.5 font-mono">
              <AlertTriangle size={13} />
              <span>BREAK IT LAB · STAGE 06</span>
            </div>

            {algorithmId === 'rnn' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Long-Range Vanishing: Scale ||W_hh|| = 0.65 collapses step-1 memory!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.({ recurrentScale: 0.65, sequenceLength: 10 })}
                  >
                    Break: Vanish
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.({ recurrentScale: 1.0, sequenceLength: 10 })}
                  >
                    Fix: Scale 1.0
                  </button>
                </div>

                <span className="text-[var(--text-secondary)] leading-tight mt-1">
                  [2] Exploding Gradient: Scale ||W_hh|| = 1.65 blows up gradients!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.({ recurrentScale: 1.65, gradientClip: 'off' })}
                  >
                    Break: Explode
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.({ gradientClip: 'clip-1.0' })}
                  >
                    Fix: Clip 1.0
                  </button>
                </div>
              </div>
            ) : algorithmId === 'lstm' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Close Forget Gate: b_f = -3.0 flushes cell state conveyor belt!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('forgetGateBias', -3.0)}
                  >
                    Break: Close Gate
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('forgetGateBias', 1.5)}
                  >
                    Fix: Open Gate
                  </button>
                </div>
              </div>
            ) : algorithmId === 'gru' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Force Overwrite: b_z = 2.0 forces z_t ≈ 1.0, overwriting hidden state!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('updateGateBias', 2.0)}
                  >
                    Break: Overwrite
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('updateGateBias', 0.0)}
                  >
                    Fix: Balanced
                  </button>
                </div>
              </div>
            ) : algorithmId === 'attention' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Unscaled High-Dim: d_k = 128 without scaling saturates Softmax!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.({ dk: 128, scaleFactor: 'unscaled-1.0' })}
                  >
                    Break: Saturate
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.({ scaleFactor: 'scaled-sqrt-dk' })}
                  >
                    Fix: 1/√d_k
                  </button>
                </div>
              </div>
            ) : algorithmId === 'transformer-encoder' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Strip Positional Encoding: "Dog bites man" == "Man bites dog"
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('posEncoding', 'off-bag-of-words')}
                  >
                    Break: Strip PE
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('posEncoding', 'sinusoidal')}
                  >
                    Fix: Sinusoidal
                  </button>
                </div>
              </div>
            ) : algorithmId === 'transformer-decoder' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] Disable Causal Mask: Future token data leakage trap!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('causalMask', 'unmasked-leak-trap')}
                  >
                    Break: Unmask Future
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('causalMask', 'masked-causal')}
                  >
                    Fix: Causal Mask
                  </button>
                </div>
              </div>
            ) : algorithmId === 'transformer-enc-dec' ? (
              <div className="flex flex-col gap-1.5 text-[10px] font-mono">
                <span className="text-[var(--text-secondary)] leading-tight">
                  [1] High τ Softmax Blur: High temperature dilutes cross-attention routing!
                </span>
                <div className="flex gap-1.5">
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('temperature', 2.0)}
                  >
                    Break: High τ Blur
                  </button>
                  <button
                    className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                    onClick={() => onUpdateHyperparam?.('temperature', 1.0)}
                  >
                    Fix: Sharp τ (1.0)
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : stage === 'rl-gridworld' ? (
        <div className="panel-section">
          {/* Header */}
          <div className="panel-header">
            <span className="panel-title font-mono tracking-wider">GRIDWORLD MDP (6×6)</span>
            <span className="font-mono text-[9px] px-1 py-0.5 rounded bg-[var(--bg-subtle)] text-[var(--text-secondary)] border border-[var(--border-hairline)]">
              36 States
            </span>
          </div>

          {/* 2x2 Visual Environment Thumbnails */}
          <div className="grid grid-cols-2 gap-2 mb-3">
            {[
              { id: 'cliff-walking' as const, label: 'Cliff Walking', desc: 'Risky vs safe edge' },
              { id: 'frozen-lake' as const, label: 'Frozen Lake', desc: 'Slippery stochastic ice' },
              { id: 'twin-treasure' as const, label: 'Twin Treasure', desc: 'Myopic bait vs goal (γ)' },
              { id: 'four-rooms' as const, label: 'Four-Rooms', desc: 'Doorway bottlenecks' },
            ].map((p) => {
              const isSelected = (rlGridworld?.preset || 'cliff-walking') === p.id;

              return (
                <button
                  key={p.id}
                  className={`p-2 flex flex-col items-center gap-1.5 border text-left transition-all ${
                    isSelected
                      ? 'border-[var(--text-primary)] bg-[var(--bg-active)] shadow-sm'
                      : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--border-strong)]'
                  }`}
                  onClick={() => onSelectGridworldPreset?.(p.id)}
                  title={p.desc}
                >
                  <GridworldThumbnail preset={p.id} isCarbon={isCarbon} />
                  <span className="font-mono text-[9px] text-[var(--text-primary)] text-center leading-tight truncate w-full">
                    {p.label}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Interactive Grid Tile Brush */}
          <div className="flex flex-col gap-2 pt-2 border-t border-[var(--border-hairline)]">
            <div className="flex items-center justify-between text-[10px] font-mono">
              <span className="text-[var(--text-muted)] uppercase">Grid Tile Brush:</span>
              <span className="text-[9px] text-[var(--pigment-class-b)] font-semibold uppercase">
                Active: {activeBrush}
              </span>
            </div>

            <div className="grid grid-cols-3 gap-1">
              {[
                { id: 'wall' as const, label: 'Wall #', color: isCarbon ? '#383835' : '#BDBAB0' },
                { id: 'lava' as const, label: 'Lava -10', color: isCarbon ? '#C26752' : '#D94E34' },
                { id: 'goal' as const, label: 'Goal +10', color: isCarbon ? '#4D8F68' : '#15803D' },
                { id: 'bait' as const, label: 'Bait +2', color: isCarbon ? '#D97706' : '#B45309' },
                { id: 'start' as const, label: 'Start S', color: isCarbon ? '#5276B8' : '#1E40AF' },
                { id: 'empty' as const, label: 'Clear .', color: isCarbon ? '#20201E' : '#FAF9F5' },
              ].map((b) => (
                <button
                  key={b.id}
                  className={`py-1 px-1.5 text-[9px] font-mono border rounded transition-colors flex items-center justify-center gap-1 min-w-0 ${
                    activeBrush === b.id
                      ? 'border-[var(--text-primary)] bg-[var(--bg-active)] font-bold text-[var(--text-primary)]'
                      : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                  }`}
                  onClick={() => onSelectBrush?.(b.id)}
                >
                  <span
                    className="w-1.5 h-1.5 rounded-full inline-block shrink-0"
                    style={{ backgroundColor: b.color }}
                  />
                  <span className="truncate">{b.label}</span>
                </button>
              ))}
            </div>

            {/* Step Living Penalty Slider */}
            <div className="slider-row mt-1">
              <div className="slider-label-row">
                <span className="slider-label text-[10px]">Living Step Penalty (r_step)</span>
                <span className="slider-value font-mono text-[10px]">
                  {(rlGridworld?.livingPenalty ?? -0.04).toFixed(2)}
                </span>
              </div>
              <input
                type="range"
                className="scientific-slider"
                min={-1.0}
                max={0.1}
                step={0.02}
                value={rlGridworld?.livingPenalty ?? -0.04}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  onUpdateRLGridworld?.((prev) => ({ ...prev, livingPenalty: val }));
                }}
              />
            </div>

            {/* Stochastic Slip Probability Slider */}
            <div className="slider-row">
              <div className="slider-label-row">
                <span className="slider-label text-[10px]">Stochastic Slip P(slip)</span>
                <span className="slider-value font-mono text-[10px]">
                  {((rlGridworld?.slipProbability ?? 0.0) * 100).toFixed(0)}%
                </span>
              </div>
              <input
                type="range"
                className="scientific-slider"
                min={0.0}
                max={0.4}
                step={0.05}
                value={rlGridworld?.slipProbability ?? 0.0}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  onUpdateRLGridworld?.((prev) => ({ ...prev, slipProbability: val }));
                }}
              />
            </div>
          </div>

          {/* Break It Lab Card */}
          <div className="mt-4 p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-elevated)] space-y-2">
            <div className="flex items-center gap-1.5 text-[10px] font-mono font-semibold uppercase text-[var(--pigment-class-a)]">
              <AlertTriangle size={13} />
              <span>BREAK IT LAB · STAGE 07</span>
            </div>

            {algorithmId === 'q-learning' ? (
              <div className="flex flex-col gap-2 text-[10px] font-mono">
                {/* [1] Zero Exploration */}
                <div className="flex flex-col gap-1">
                  <span className="text-[var(--text-secondary)] leading-tight">
                    [1] Zero Exploration: Trapped on +2 bait forever!
                  </span>
                  <div className="flex gap-1.5">
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => {
                        onSelectGridworldPreset?.('twin-treasure');
                        onUpdateHyperparam?.('explorationEpsilon', 0.0);
                      }}
                      aria-label="Break model: Set exploration epsilon to 0.0"
                    >
                      Break: ε=0.0
                    </button>
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.('explorationEpsilon', 0.25)}
                      aria-label="Fix model: Set exploration epsilon to 0.25"
                    >
                      Fix: ε=0.25
                    </button>
                  </div>
                </div>

                {/* [2] Myopic Zero Discount */}
                <div className="flex flex-col gap-1">
                  <span className="text-[var(--text-secondary)] leading-tight">
                    [2] Myopic Zero Discount: Q-values blind beyond 1 tile!
                  </span>
                  <div className="flex gap-1.5">
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.('discountFactor', 0.0)}
                      aria-label="Break model: Set discount factor to 0.0"
                    >
                      Break: γ=0.0
                    </button>
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.('discountFactor', 0.95)}
                      aria-label="Fix model: Set discount factor to 0.95"
                    >
                      Fix: γ=0.95
                    </button>
                  </div>
                </div>

                {/* [3] Positive Living Reward Loitering */}
                <div className="flex flex-col gap-1">
                  <span className="text-[var(--text-secondary)] leading-tight">
                    [3] Positive Living Reward: Infinite loitering loop!
                  </span>
                  <div className="flex gap-1.5">
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateRLGridworld?.((prev) => ({ ...prev, livingPenalty: 0.2 }))}
                    >
                      Break: r=+0.20
                    </button>
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateRLGridworld?.((prev) => ({ ...prev, livingPenalty: -0.04 }))}
                    >
                      Fix: r=-0.04
                    </button>
                  </div>
                </div>
              </div>
            ) : algorithmId === 'dqn' ? (
              <div className="flex flex-col gap-2 text-[10px] font-mono">
                <div className="flex flex-col gap-1">
                  <span className="text-[var(--text-secondary)] leading-tight">
                    [1] Disable Replay & Target: Correlated stream collapse!
                  </span>
                  <div className="flex gap-1.5">
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => {
                        onUpdateHyperparam?.({
                          replayBufferMode: 'disabled-online-stream',
                          targetNetworkSync: 1,
                        });
                      }}
                    >
                      Break: Stream+C=1
                    </button>
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => {
                        onUpdateHyperparam?.({
                          replayBufferMode: 'enabled-random-batch',
                          targetNetworkSync: 20,
                        });
                      }}
                    >
                      Fix: Replay+C=20
                    </button>
                  </div>
                </div>

                <div className="flex flex-col gap-1 pt-1 border-t border-[var(--border-hairline)]">
                  <span className="text-[var(--text-secondary)] leading-tight">
                    [2] Moving Target (C=1): Non-stationary Bellman divergence!
                  </span>
                  <div className="flex gap-1.5">
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.('targetNetworkSync', 1)}
                    >
                      Break: C=1
                    </button>
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.('targetNetworkSync', 20)}
                    >
                      Fix: C=20
                    </button>
                  </div>
                </div>
              </div>
            ) : algorithmId === 'policy-gradient' ? (
              <div className="flex flex-col gap-2 text-[10px] font-mono">
                <div className="flex flex-col gap-1">
                  <span className="text-[var(--text-secondary)] leading-tight">
                    [1] No Baseline: High variance policy swings!
                  </span>
                  <div className="flex gap-1.5">
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.('baselineMode', 'no-baseline-raw')}
                    >
                      Break: b=0
                    </button>
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.('baselineMode', 'mean-return-baseline')}
                    >
                      Fix: Baseline
                    </button>
                  </div>
                </div>

                <div className="flex flex-col gap-1">
                  <span className="text-[var(--text-secondary)] leading-tight">
                    [2] Premature Collapse: Deterministic lock-in (β=0)!
                  </span>
                  <div className="flex gap-1.5">
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-a)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.({ entropyBonus: 0.0, learningRate: 0.1 })}
                    >
                      Break: β=0, α=0.1
                    </button>
                    <button
                      className="flex-1 py-1 px-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--pigment-class-b)] hover:bg-[var(--bg-active)] truncate text-center whitespace-nowrap text-[9px]"
                      onClick={() => onUpdateHyperparam?.({ entropyBonus: 0.05, learningRate: 0.02 })}
                    >
                      Fix: β=0.05
                    </button>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      ) : (
        <div className="panel-section">
          <div className="panel-header">
            <h2 className="panel-title font-mono tracking-wider font-semibold text-[11px]">DATA</h2>
            <div className="flex items-center gap-1.5 font-mono text-[10px]">
              {/* Deterministic Seed Lock */}
              <button
                className="flex items-center gap-1 text-[var(--text-secondary)] hover:text-[var(--text-primary)] border border-[var(--border-hairline)] px-1.5 py-0.5 bg-[var(--bg-subtle)] min-h-[26px]"
                onClick={handleToggleSeedLock}
                title={isSeedLocked ? 'Deterministic Mulberry32 Seed Locked' : 'Seed Unlocked'}
                aria-label={`SEED: [${options.seed.toString().padStart(4, '0')}] ${isSeedLocked ? 'Locked' : 'Unlocked'}`}
              >
                {isSeedLocked ? <Lock size={10} /> : <Unlock size={10} />}
                <span>SEED: [{options.seed.toString().padStart(4, '0')}]</span>
              </button>

              <button
                onClick={() => {
                  handleNextSeed();
                  onRegenerate();
                }}
                className="text-[var(--text-secondary)] hover:text-[var(--text-primary)] p-0.5 border border-[var(--border-hairline)] bg-[var(--bg-subtle)] min-w-[26px] min-h-[26px] flex items-center justify-center"
                title="Generate New Seed & Reseed Dataset"
                aria-label="Generate New Seed & Reseed Dataset"
              >
                <RefreshCw size={10} />
              </button>
            </div>
          </div>

        {/* Segmented Tabs: Classification vs Regression */}
        <div role="tablist" aria-label="Dataset task type" className="flex border border-[var(--border-hairline)] bg-[var(--bg-subtle)] mb-2 p-0.5">
          <button
            role="tab"
            aria-selected={activeTab === 'classification'}
            aria-label="Classification"
            className={`flex-1 py-1.5 min-h-[28px] text-[10px] font-mono font-medium transition-colors ${
              activeTab === 'classification'
                ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm'
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
            }`}
            onClick={() => {
              setActiveTab('classification');
              if (getDatasetTaskType(options.kind) !== 'classification') {
                onUpdateOptions({ kind: 'linear-blobs' });
              }
            }}
          >
            Classification
          </button>
          <button
            role="tab"
            aria-selected={activeTab === 'regression'}
            aria-label="Regression"
            className={`flex-1 py-1.5 min-h-[28px] text-[10px] font-mono font-medium transition-colors ${
              activeTab === 'regression'
                ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] shadow-sm'
                : 'text-[var(--text-muted)] hover:text-[var(--text-primary)]'
            }`}
            onClick={() => {
              setActiveTab('regression');
              if (getDatasetTaskType(options.kind) !== 'regression') {
                onUpdateOptions({ kind: 'sine-wave', numClasses: 2 });
              }
            }}
          >
            Regression
          </button>
        </div>

        {/* 2x2 Visual Dataset Thumbnails Grid */}
        <div className="grid grid-cols-2 gap-2 mb-3">
          {currentKinds.map((k) => {
            const isSelected = options.kind === k.kind;
            return (
              <button
                type="button"
                key={k.kind}
                className={`dataset-card text-left w-full p-1.5 flex flex-col items-center cursor-pointer border focus:outline-none focus-visible:ring-2 focus-visible:ring-[var(--border-strong)] ${
                  isSelected ? 'border-[var(--border-strong)] bg-[var(--bg-active)]' : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)]'
                }`}
                onClick={() => onUpdateOptions({ kind: k.kind })}
                title={k.label}
                aria-pressed={isSelected}
                aria-label={`Select dataset: ${k.label}`}
              >
                <canvas
                  ref={(el) => {
                    if (el) thumbCanvasesRef.current.set(k.kind, el);
                  }}
                  width={96}
                  height={64}
                  className="w-full h-[54px] block mb-1 border border-[var(--border-hairline)]"
                />
                <span className="dataset-name text-[10px] font-medium text-[var(--text-primary)]">
                  {k.label}
                </span>
              </button>
            );
          })}
        </div>

        {/* Pointer Tool Mode Switcher */}
        <div className="mb-3">
          <h3 className="text-[10px] font-mono uppercase text-[var(--text-muted)] mb-1">
            Pointer Tool Mode
          </h3>
          <div role="toolbar" aria-label="Pointer tool mode" className="flex border border-[var(--border-hairline)] bg-[var(--bg-subtle)]">
            <button
              className={`flex-1 py-1.5 min-h-[28px] text-[11px] font-medium text-center ${
                pointerMode === 'add-drag' ? 'bg-[var(--text-primary)] text-[var(--text-inverse)]' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              onClick={() => onSetPointerMode('add-drag')}
              title="Click to Add, Drag to Move (Ctrl/Cmd+Click to add on top of existing points)"
              aria-label="Add / Drag pointer mode"
            >
              Add / Drag
            </button>
            <button
              className={`flex-1 py-1.5 min-h-[28px] text-[11px] font-medium text-center border-l border-r border-[var(--border-hairline)] ${
                pointerMode === 'flip' ? 'bg-[var(--text-primary)] text-[var(--text-inverse)]' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              onClick={() => onSetPointerMode('flip')}
              title="Click point to flip label (or Alt+Click)"
              aria-label="Flip (Alt) pointer mode"
            >
              Flip (Alt)
            </button>
            <button
              className={`flex-1 py-1.5 min-h-[28px] text-[11px] font-medium text-center ${
                pointerMode === 'delete' ? 'bg-[var(--text-primary)] text-[var(--text-inverse)]' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
              onClick={() => onSetPointerMode('delete')}
              title="Click point to delete (or Shift+Click)"
              aria-label="Del (Shift) pointer mode"
            >
              Del (Shift)
            </button>
          </div>
        </div>

        {/* Class Selector & 2 | 3 Classes Toggle (Classification only) vs Single Target (Regression) */}
        {activeTab === 'classification' ? (
          <div className="mb-3 flex flex-wrap items-center justify-between gap-1.5">
            <div className="flex flex-wrap items-center gap-1.5 shrink-0" role="toolbar" aria-label="Target class selector">
              <button
                className={`px-2 py-1 min-h-[26px] text-[10px] font-mono border shrink-0 ${
                  activeClass === 1 ? 'border-[var(--border-strong)] font-bold' : 'border-[var(--border-hairline)]'
                }`}
                style={{ color: isCarbon ? '#F87171' : '#B91C1C' }}
                onClick={() => onSetActiveClass(1)}
                title="Class +1 (Positive Target / Crimson)"
                aria-label="● +1"
              >
                ● +1
              </button>
              <button
                className={`px-2 py-1 min-h-[26px] text-[10px] font-mono border shrink-0 ${
                  activeClass === -1 ? 'border-[var(--border-strong)] font-bold' : 'border-[var(--border-hairline)]'
                }`}
                style={{ color: isCarbon ? '#93C5FD' : '#1E40AF' }}
                onClick={() => onSetActiveClass(-1)}
                title="Class -1 (Negative Target / Blue)"
                aria-label="■ -1"
              >
                ■ -1
              </button>
              {options.numClasses === 3 && (
                <button
                  className={`px-2 py-1 min-h-[26px] text-[10px] font-mono border shrink-0 ${
                    activeClass === 2 ? 'border-[var(--border-strong)] font-bold' : 'border-[var(--border-hairline)]'
                  }`}
                  style={{ color: isCarbon ? '#86EFAC' : '#15803D' }}
                  onClick={() => onSetActiveClass(2)}
                  title="Class +2"
                  aria-label="▲ +2"
                >
                  ▲ +2
                </button>
              )}
            </div>

            <button
              className="text-[10px] font-mono text-[var(--text-secondary)] hover:text-[var(--text-primary)] border border-[var(--border-hairline)] px-2 py-1 min-h-[26px] bg-[var(--bg-subtle)] shrink-0"
              onClick={() => onUpdateOptions({ numClasses: options.numClasses === 2 ? 3 : 2 })}
              title="Toggle 2 or 3 Classes"
              aria-label={`${options.numClasses} Classes`}
            >
              {options.numClasses} Classes
            </button>
          </div>
        ) : (
          <div className="mb-3 flex items-center justify-between gap-1.5 p-1.5 border border-[var(--border-hairline)] bg-[var(--bg-subtle)] font-mono text-[10px]">
            <span className="flex items-center gap-1.5 font-semibold" style={{ color: isCarbon ? '#F87171' : '#B91C1C' }}>
              <span className="w-2 h-2 rounded-full inline-block" style={{ backgroundColor: isCarbon ? '#C26752' : '#D94E34' }} />
              <span>Target y ∈ ℝ (Continuous)</span>
            </span>
            <span className="text-[var(--text-muted)] text-[9px]">Single Target</span>
          </div>
        )}

        {/* Noise Slider */}
        <div className="slider-row mb-2">
          <div className="slider-label-row">
            <label htmlFor="dataset-noise-slider" className="slider-label text-[11px]">Noise</label>
            <span className="slider-value font-mono text-[11px]">{Math.round(options.noise * 100)}%</span>
          </div>
          <input
            id="dataset-noise-slider"
            type="range"
            className="scientific-slider"
            aria-label="Dataset noise level"
            min={0}
            max={0.5}
            step={0.02}
            value={options.noise}
            onChange={(e) => onUpdateOptions({ noise: parseFloat(e.target.value) })}
          />
        </div>

        {/* Train / Test Split Slider */}
        <div className="slider-row mb-2">
          <div className="slider-label-row">
            <label htmlFor="dataset-split-slider" className="slider-label text-[11px]">Train / Test Split</label>
            <span className="slider-value font-mono text-[11px]">{Math.round((1 - options.testSplit) * 100)} / {Math.round(options.testSplit * 100)}</span>
          </div>
          <input
            id="dataset-split-slider"
            type="range"
            className="scientific-slider"
            aria-label="Train and test data split"
            min={0.1}
            max={0.4}
            step={0.05}
            value={options.testSplit}
            onChange={(e) => onUpdateOptions({ testSplit: parseFloat(e.target.value) })}
          />
        </div>

        {/* Feature Scaling Switch */}
        <div className="mt-3 pt-3 border-t border-[var(--border-hairline)]">
          <div className="flex flex-wrap items-center justify-between gap-1.5 mb-1.5">
            <h3 className="text-[10px] font-mono uppercase text-[var(--text-muted)]">Feature Scaling</h3>
            <button
              className={`text-[9px] font-mono px-2 py-1 min-h-[26px] border shrink-0 ${
                options.stretchX2 ? 'bg-[var(--accent-amber)] text-white border-[var(--accent-amber)]' : 'border-[var(--border-hairline)] text-[var(--text-secondary)]'
              }`}
              onClick={() => onUpdateOptions({ stretchX2: !options.stretchX2 })}
              title="10x Anisotropic Aspect Distortion along X2"
              aria-label={options.stretchX2 ? '10x Stretch ON' : '10x Stretch'}
            >
              {options.stretchX2 ? '10x Stretch ON' : '10x Stretch'}
            </button>
          </div>

          <div className="flex border border-[var(--border-hairline)] bg-[var(--bg-subtle)] font-mono text-[10px]">
            {(['raw', 'zscore', 'minmax'] as FeatureScalingMode[]).map((mode) => (
              <button
                key={mode}
                className={`flex-1 py-1.5 min-h-[28px] text-center capitalize ${
                  options.scaling === mode
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] font-medium'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
                onClick={() => onUpdateOptions({ scaling: mode })}
                aria-label={`Select ${mode === 'zscore' ? 'Z-Score' : mode === 'minmax' ? 'Min-Max' : 'Raw'} scaling`}
              >
                {mode === 'zscore' ? 'Z-Score' : mode === 'minmax' ? 'Min-Max' : 'Raw'}
              </button>
            ))}
          </div>
        </div>
      </div>
      )}

      {/* Collapsible [+ Advanced Parameters] Drawer */}
      <div className="border-t border-[var(--border-hairline)] bg-[var(--bg-elevated)]">
        <button
          className="w-full px-4 py-2 flex items-center justify-between text-[11px] font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
          onClick={() => setIsAdvancedOpen(!isAdvancedOpen)}
          aria-expanded={isAdvancedOpen}
        >
          <span>Advanced Parameters</span>
          {isAdvancedOpen ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
        </button>

        {isAdvancedOpen && (
          <div className="p-4 pt-1 flex flex-col gap-2.5 border-t border-[var(--border-hairline)] text-[11px]">
            {/* Algorithm Advanced Parameters */}
            {advancedParams.map((hp) => (
              <div key={hp.id} className="slider-row" title={hp.description}>
                <div className="slider-label-row">
                  <HyperparameterLabel
                    as="span"
                    name={hp.name}
                    symbol={hp.symbol}
                    className="slider-label text-[11px]"
                  />
                  <span className="slider-value font-mono text-[11px]">
                    {hyperparams[hp.id] ?? hp.default}
                  </span>
                </div>
                {hp.type === 'select' && hp.options ? (
                  <select
                    className="param-select w-full mt-1 text-[11px] font-mono bg-[var(--bg-subtle)] border border-[var(--border-hairline)] px-2 py-1"
                    value={hyperparams[hp.id] ?? hp.default}
                    onChange={(e) => onUpdateHyperparam?.(hp.id, e.target.value)}
                  >
                    {hp.options.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                ) : hp.type === 'slider' ? (
                  <input
                    type="range"
                    className="scientific-slider mt-1"
                    min={hp.min ?? 0}
                    max={hp.max ?? 10}
                    step={hp.step ?? 1}
                    value={Number(hyperparams[hp.id] ?? hp.default)}
                    onChange={(e) => onUpdateHyperparam?.(hp.id, parseFloat(e.target.value))}
                  />
                ) : null}
              </div>
            ))}

            {/* Sub-Epochs per Step: only show if not already defined in advancedParams and not vision-grid / sequence-attn */}
            {onUpdateHyperparam && stage !== 'vision-grid' && stage !== 'sequence-attn' && !advancedParams.some(p => p.id === 'batchPasses' || p.id === 'subEpochs' || p.name.toLowerCase().includes('sub-epoch')) && (
              <div className="slider-row">
                <div className="slider-label-row">
                  <span className="slider-label text-[11px]">Sub-Epochs per Step</span>
                  <span className="slider-value font-mono text-[11px]">
                    {hyperparams.batchPasses ?? hyperparams.cyclesPerStep ?? 5} passes
                  </span>
                </div>
                <select
                  className="param-select w-full mt-1 text-[11px] font-mono bg-[var(--bg-subtle)] border border-[var(--border-hairline)] px-2 py-1"
                  value={Number(hyperparams.batchPasses ?? hyperparams.cyclesPerStep ?? 5)}
                  onChange={(e) => {
                    const val = parseInt(e.target.value, 10);
                    onUpdateHyperparam?.({ batchPasses: val, cyclesPerStep: val });
                  }}
                >
                  <option value={1}>1 pass (Granular / Slow)</option>
                  <option value={5}>5 passes (Balanced / Standard)</option>
                  <option value={10}>10 passes (Fast)</option>
                  <option value={20}>20 passes (Accelerated)</option>
                </select>
              </div>
            )}

            {stage !== 'vision-grid' && stage !== 'sequence-attn' && (
              <>
                <div className="slider-row">
                  <div className="slider-label-row">
                    <span className="slider-label">Sample Count (N)</span>
                    <span className="slider-value font-mono">{options.sampleCount}</span>
                  </div>
                  <input
                    type="range"
                    className="scientific-slider"
                    min={60}
                    max={400}
                    step={20}
                    value={options.sampleCount}
                    onChange={(e) => onUpdateOptions({ sampleCount: parseInt(e.target.value, 10) })}
                  />
                </div>

                <div className="text-[10px] text-[var(--text-muted)] font-mono leading-relaxed bg-[var(--bg-subtle)] p-2 border border-[var(--border-hairline)]">
                  Data generator uses deterministic 32-bit Mulberry32 PRNG. Solid shapes = Training set, hollow rings = Test set.
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </aside>
  );
};
