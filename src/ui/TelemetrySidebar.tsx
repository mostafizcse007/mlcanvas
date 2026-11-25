import React, { useRef, useEffect, useState, useCallback } from 'react';
import { AlgorithmPlugin, TelemetryMetrics, ExplanationPayload } from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { HistoryRingBuffer } from '../utils/ringBuffer';
import { observeElementSize, syncCanvasBufferSize } from '../rendering/canvasUtils';
import { LatexSpan } from '../utils/katex-render';
import { Copy, Check, Clock, RotateCcw } from 'lucide-react';

interface TelemetrySidebarProps {
  algorithm: AlgorithmPlugin;
  datasetTaskType?: 'classification' | 'regression';
  metrics: TelemetryMetrics | null;
  historyBuffer: HistoryRingBuffer;
  theme: ThemeMode;
  decisionThreshold: number;
  onUpdateThreshold: (val: number) => void;
  isConverged?: boolean;
  explanation?: ExplanationPayload | null;
  currentEpoch?: number;
  rlEpisodeCount?: number;
  scrubIndex?: number | null;
  onScrubIndexChange?: (idx: number | null) => void;
  onReturnToLive?: () => void;
  onPause?: () => void;
  isRunning?: boolean;
}

interface LossHoverInfo {
  x: number;
  y: number;
  epoch: number;
  trainLoss: number;
  testLoss: number;
  trainY: number;
  testY: number;
}

function getFourthTelemetryMetric(
  algorithmId: string,
  isRegression: boolean,
  metrics: TelemetryMetrics | null
): { label: React.ReactNode; value: string } {
  if (algorithmId === 'knn' && !isRegression) {
    const radius = metrics?.custom?.probeRadius ?? metrics?.marginWidth;
    return {
      label: 'Probe Radius',
      value: radius !== undefined ? Number(radius).toFixed(3) : '—',
    };
  }
  if (algorithmId === 'gaussian-process' && !isRegression) {
    const meanStd = metrics?.custom?.meanStd;
    return {
      label: (
        <>
          Mean Std <span className="normal-case">σ</span>
        </>
      ),
      value: meanStd !== undefined ? Number(meanStd).toFixed(3) : '—',
    };
  }
  if (metrics?.sparsity) {
    return {
      label: 'Sparsity',
      value: `${metrics.sparsity.zeroCount}/${metrics.sparsity.totalCount}`,
    };
  }
  if (isRegression) {
    return {
      label: 'RMSE',
      value: metrics ? Math.sqrt(Math.max(0, metrics.trainLoss)).toFixed(3) : '—',
    };
  }
  if (algorithmId === 'linear-svm' && metrics?.marginWidth !== undefined) {
    return {
      label: (
        <>
          Margin <span className="normal-case">2/||w||</span>
        </>
      ),
      value: metrics.marginWidth.toFixed(3),
    };
  }
  return {
    label: 'Sparsity',
    value: '—',
  };
}

export const TelemetrySidebar: React.FC<TelemetrySidebarProps> = ({
  algorithm,
  datasetTaskType,
  metrics,
  historyBuffer,
  theme,
  decisionThreshold,
  onUpdateThreshold,
  isConverged = false,
  explanation,
  currentEpoch = 0,
  rlEpisodeCount,
  scrubIndex = null,
  onScrubIndexChange,
  onReturnToLive,
  onPause,
  isRunning = false,
}) => {
  const lossCanvasRef = useRef<HTMLCanvasElement>(null);
  const lossContainerRef = useRef<HTMLDivElement>(null);
  const lossChartWidthRef = useRef<number>(280);
  const [hoverLoss, setHoverLoss] = useState<LossHoverInfo | null>(null);
  const [activeLensTab, setActiveLensTab] = useState<'intuition' | 'math' | 'code'>('intuition');
  const [copiedSnippet, setCopiedSnippet] = useState<string | null>(null);

  const handleCopy = (key: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSnippet(key);
    setTimeout(() => setCopiedSnippet(null), 2000);
  };

  const isCarbon = theme === 'carbon';
  const isClustering = ['k-means', 'hierarchical-clustering', 'dbscan', 'gmm'].includes(algorithm.id);
  const isManifold = ['pca', 'lda', 'ica', 't-sne', 'umap'].includes(algorithm.id);
  const isMLP = algorithm.id === 'mlp';
  const isBottleneck = algorithm.id === 'autoencoder' || algorithm.id === 'vae';
  const isGAN = algorithm.id === 'gan';
  const isVision = algorithm.stage === 'vision-grid';
  const isSequence = algorithm.stage === 'sequence-attn';
  const isRL = algorithm.stage === 'rl-gridworld';
  const isRecurrent = isSequence && ['rnn', 'lstm', 'gru'].includes(algorithm.id);
  const isCrossTaskClassifier =
    algorithm.stage === 'continuous-2d' &&
    algorithm.taskType === 'classification' &&
    algorithm.id !== 'knn' &&
    datasetTaskType === 'regression';
  const isRegression =
    !isCrossTaskClassifier &&
    (datasetTaskType !== undefined
      ? datasetTaskType === 'regression'
      : algorithm.taskType === 'regression');
  const fourthTelemetryMetric = getFourthTelemetryMetric(algorithm.id, isRegression, metrics);

  // 1. Render Dedicated Train vs Test Loss Sparkline
  const drawLossChart = useCallback(() => {
    const canvas = lossCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = lossChartWidthRef.current;
    const h = 105;
    syncCanvasBufferSize(canvas, w, h, 2);

    // Background & Boundary
    ctx.fillStyle = isCarbon ? '#131312' : '#FAF9F5';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = isCarbon ? '#2E2E2B' : '#DCDAD3';
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    let epochs: number[];
    let trainLosses: number[];
    let testLosses: number[];

    if (isRL) {
      const rlSeries = historyBuffer.getRLReturnSeries();
      epochs = rlSeries.epochs;
      trainLosses = rlSeries.episodeReturns;
      testLosses = rlSeries.movingAvgReturns;

      if (epochs.length === 0 && metrics) {
        epochs = [0];
        trainLosses = [Number(metrics.custom?.episodeReturn ?? 0)];
        testLosses = [Number(metrics.custom?.movingAvgReturn ?? 0)];
      }
    } else {
      const series = historyBuffer.getLossSeries();
      epochs = series.epochs;
      trainLosses = series.trainLosses;
      testLosses = series.testLosses;

      // Fallback: If ring buffer is empty but we have latest metrics (e.g. at initial load before run)
      if (epochs.length === 0 && metrics) {
        epochs = [0];
        trainLosses = [metrics.trainLoss];
        testLosses = [metrics.testLoss];
      }
    }

    const count = epochs.length;
    if (count === 0) {
      ctx.fillStyle = isCarbon ? '#686763' : '#8A8884';
      ctx.font = '10px "JetBrains Mono"';
      ctx.textAlign = 'center';
      ctx.fillText('Press Play [Space] or Step [->] to begin', w / 2, h / 2);
      return;
    }

    let minYVal = 0;
    let maxYVal = 0.01;
    if (isRL) {
      minYVal = Infinity;
      maxYVal = -Infinity;
      for (let i = 0; i < count; i++) {
        if (trainLosses[i] < minYVal) minYVal = trainLosses[i];
        if (testLosses[i] < minYVal) minYVal = testLosses[i];
        if (trainLosses[i] > maxYVal) maxYVal = trainLosses[i];
        if (testLosses[i] > maxYVal) maxYVal = testLosses[i];
      }
      if (!Number.isFinite(minYVal) || !Number.isFinite(maxYVal)) {
        minYVal = -1;
        maxYVal = 10;
      }
      if (Math.abs(maxYVal - minYVal) < 0.5) {
        minYVal -= 1;
        maxYVal += 1;
      } else {
        const pad = (maxYVal - minYVal) * 0.12;
        minYVal -= pad;
        maxYVal += pad;
      }
    } else {
      for (let i = 0; i < count; i++) {
        if (trainLosses[i] > maxYVal) maxYVal = trainLosses[i];
        if (testLosses[i] > maxYVal) maxYVal = testLosses[i];
      }
      maxYVal *= 1.15;
    }

    const padLeft = 32;
    const padRight = 12;
    const padTop = 10;
    const padBottom = 16;
    const cw = w - padLeft - padRight;
    const ch = h - padTop - padBottom;
    const ySpan = Math.max(1e-6, maxYVal - minYVal);

    const toX = (i: number) => padLeft + (count <= 1 ? cw / 2 : (i / (count - 1)) * cw);
    const toY = (val: number) => padTop + (1.0 - Math.min(1.0, Math.max(0, (val - minYVal) / ySpan))) * ch;

    // Subtle Hairline Gridlines
    ctx.save();
    ctx.strokeStyle = isCarbon ? '#232320' : '#EAE8E1';
    ctx.lineWidth = 1;

    // Top gridline (maxYVal)
    ctx.beginPath();
    ctx.moveTo(padLeft, padTop);
    ctx.lineTo(padLeft + cw, padTop);
    ctx.stroke();

    // Mid gridline
    const midY = padTop + ch / 2;
    ctx.beginPath();
    ctx.moveTo(padLeft, midY);
    ctx.lineTo(padLeft + cw, midY);
    ctx.stroke();

    // Baseline gridline (minYVal)
    const zeroY = padTop + ch;
    ctx.beginPath();
    ctx.moveTo(padLeft, zeroY);
    ctx.lineTo(padLeft + cw, zeroY);
    ctx.stroke();

    const fmtY = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

    // Y-Axis numeric labels
    ctx.fillStyle = isCarbon ? '#75736E' : '#99978F';
    ctx.font = '8px "JetBrains Mono"';
    ctx.textAlign = 'right';
    ctx.fillText(fmtY(maxYVal), padLeft - 4, padTop + 3);
    ctx.fillText(fmtY((minYVal + maxYVal) / 2), padLeft - 4, midY + 3);
    ctx.fillText(fmtY(minYVal), padLeft - 4, zeroY + 1);

    // X-Axis epoch labels
    ctx.textAlign = 'left';
    ctx.fillText('0', padLeft, h - 4);
    ctx.textAlign = 'right';
    const latestEpoch = epochs[count - 1] ?? 0;
    ctx.fillText(isRL ? `ep ${rlEpisodeCount ?? latestEpoch}` : `ep ${latestEpoch}`, padLeft + cw, h - 4);
    ctx.restore();

    const trainColor = isCarbon ? '#C26752' : '#B91C1C';
    const testColor = isCarbon ? '#5276B8' : '#1E40AF';

    if (count === 1) {
      // Single Epoch 0 point
      const trVal = trainLosses[0];
      const teVal = testLosses[0];
      const ptX = toX(0);

      // Train baseline dot
      ctx.fillStyle = trainColor;
      ctx.beginPath();
      ctx.arc(ptX - 16, toY(trVal), 3.5, 0, Math.PI * 2);
      ctx.fill();

      // Test baseline dot (hollow with center pip)
      ctx.strokeStyle = testColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(ptX + 16, toY(teVal), 3.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = testColor;
      ctx.beginPath();
      ctx.arc(ptX + 16, toY(teVal), 1.2, 0, Math.PI * 2);
      ctx.fill();

      // Text note
      ctx.fillStyle = isCarbon ? '#A5A39D' : '#575653';
      ctx.font = '9px "JetBrains Mono"';
      ctx.textAlign = 'center';
      ctx.fillText(
        isRL
          ? `Ep ${rlEpisodeCount ?? 0} Baseline · Ret ${trVal.toFixed(2)} | Avg ${teVal.toFixed(2)}`
          : `Epoch 000 Baseline · Tr ${trVal.toFixed(3)} | Te ${teVal.toFixed(3)}`,
        w / 2,
        padTop + ch / 2 + 14
      );
    } else {
      // Solid Train Loss Curve
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = trainColor;
      ctx.lineWidth = 1.8;
      for (let i = 0; i < count; i++) {
        const x = toX(i);
        const y = toY(trainLosses[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();

      // Dashed Test Loss Curve
      ctx.save();
      ctx.beginPath();
      ctx.strokeStyle = testColor;
      ctx.lineWidth = 1.8;
      ctx.setLineDash([3, 3]);
      for (let i = 0; i < count; i++) {
        const x = toX(i);
        const y = toY(testLosses[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Hover Crosshair & Anchors
    if (hoverLoss) {
      ctx.save();
      ctx.strokeStyle = isCarbon ? '#75736E' : '#99978F';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(hoverLoss.x, padTop);
      ctx.lineTo(hoverLoss.x, padTop + ch);
      ctx.stroke();
      ctx.restore();

      // Dot on Train curve
      ctx.fillStyle = trainColor;
      ctx.beginPath();
      ctx.arc(hoverLoss.x, hoverLoss.trainY, 3.5, 0, Math.PI * 2);
      ctx.fill();

      // Dot on Test curve
      ctx.strokeStyle = testColor;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(hoverLoss.x, hoverLoss.testY, 3.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = testColor;
      ctx.beginPath();
      ctx.arc(hoverLoss.x, hoverLoss.testY, 1.2, 0, Math.PI * 2);
      ctx.fill();
    } else if (scrubIndex !== null && count > 0) {
      // Scrubber Position Marker on Loss Canvas
      const targetEp = historyBuffer.get(scrubIndex)?.epoch ?? scrubIndex;
      const scrubPlotIdx = epochs.findIndex((ep) => ep === targetEp);
      if (scrubPlotIdx >= 0) {
        const scrubX = toX(scrubPlotIdx);
        const scrubTrY = toY(trainLosses[scrubPlotIdx]);
        const scrubTeY = toY(testLosses[scrubPlotIdx]);

        ctx.save();
        ctx.strokeStyle = '#D97706';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(scrubX, padTop);
        ctx.lineTo(scrubX, padTop + ch);
        ctx.stroke();
        ctx.restore();

        ctx.fillStyle = '#D97706';
        ctx.beginPath();
        ctx.arc(scrubX, scrubTrY, 3.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.strokeStyle = '#D97706';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(scrubX, scrubTeY, 3.5, 0, Math.PI * 2);
        ctx.stroke();
      }
    }
  }, [historyBuffer, metrics, isCarbon, hoverLoss, scrubIndex, isRL, rlEpisodeCount]);

  useEffect(() => {
    drawLossChart();
  }, [drawLossChart]);

  // Track sparkline container width without forced synchronous reflow or extra React re-renders
  useEffect(() => {
    return observeElementSize(lossContainerRef.current, ({ width }) => {
      const nextWidth = Math.round(width);
      if (nextWidth > 0 && nextWidth !== lossChartWidthRef.current) {
        lossChartWidthRef.current = nextWidth;
        drawLossChart();
      }
    });
  }, [drawLossChart]);

  // Handle Loss Canvas Mouse Move (Tooltip Inspector)
  const handleLossMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const canvas = lossCanvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      let epochs: number[];
      let trainLosses: number[];
      let testLosses: number[];

      if (isRL) {
        const rlSeries = historyBuffer.getRLReturnSeries();
        epochs = rlSeries.epochs;
        trainLosses = rlSeries.episodeReturns;
        testLosses = rlSeries.movingAvgReturns;

        if (epochs.length === 0 && metrics) {
          epochs = [0];
          trainLosses = [Number(metrics.custom?.episodeReturn ?? 0)];
          testLosses = [Number(metrics.custom?.movingAvgReturn ?? 0)];
        }
      } else {
        const series = historyBuffer.getLossSeries();
        epochs = series.epochs;
        trainLosses = series.trainLosses;
        testLosses = series.testLosses;

        if (epochs.length === 0 && metrics) {
          epochs = [0];
          trainLosses = [metrics.trainLoss];
          testLosses = [metrics.testLoss];
        }
      }

      const count = epochs.length;
      if (count === 0) return;

      const w = rect.width;
      const padLeft = 32;
      const padRight = 12;
      const padTop = 10;
      const padBottom = 16;
      const cw = w - padLeft - padRight;
      const ch = rect.height - padTop - padBottom;

      let minYVal = 0;
      let maxYVal = 0.01;
      if (isRL) {
        minYVal = Infinity;
        maxYVal = -Infinity;
        for (let i = 0; i < count; i++) {
          if (trainLosses[i] < minYVal) minYVal = trainLosses[i];
          if (testLosses[i] < minYVal) minYVal = testLosses[i];
          if (trainLosses[i] > maxYVal) maxYVal = trainLosses[i];
          if (testLosses[i] > maxYVal) maxYVal = testLosses[i];
        }
        if (!Number.isFinite(minYVal) || !Number.isFinite(maxYVal)) {
          minYVal = -1;
          maxYVal = 10;
        }
        if (Math.abs(maxYVal - minYVal) < 0.5) {
          minYVal -= 1;
          maxYVal += 1;
        } else {
          const pad = (maxYVal - minYVal) * 0.12;
          minYVal -= pad;
          maxYVal += pad;
        }
      } else {
        for (let i = 0; i < count; i++) {
          if (trainLosses[i] > maxYVal) maxYVal = trainLosses[i];
          if (testLosses[i] > maxYVal) maxYVal = testLosses[i];
        }
        maxYVal *= 1.15;
      }

      const ySpan = Math.max(1e-6, maxYVal - minYVal);
      const toX = (i: number) => padLeft + (count <= 1 ? cw / 2 : (i / (count - 1)) * cw);
      const toY = (val: number) => padTop + (1.0 - Math.min(1.0, Math.max(0, (val - minYVal) / ySpan))) * ch;

      if (count === 1) {
        setHoverLoss({
          x: toX(0),
          y: mouseY,
          epoch: epochs[0],
          trainLoss: trainLosses[0],
          testLoss: testLosses[0],
          trainY: toY(trainLosses[0]),
          testY: toY(testLosses[0]),
        });
        return;
      }

      const ratio = Math.max(0, Math.min(1, (mouseX - padLeft) / Math.max(1, cw)));
      const idx = Math.max(0, Math.min(count - 1, Math.round(ratio * (count - 1))));

      setHoverLoss({
        x: toX(idx),
        y: mouseY,
        epoch: epochs[idx],
        trainLoss: trainLosses[idx],
        testLoss: testLosses[idx],
        trainY: toY(trainLosses[idx]),
        testY: toY(testLosses[idx]),
      });
    },
    [historyBuffer, metrics, isRL]
  );

  const handleLossMouseLeave = useCallback(() => {
    setHoverLoss(null);
  }, []);

  return (
    <aside className="sidebar-telemetry flex flex-col h-full bg-[var(--bg-elevated)] border-l border-[var(--border-hairline)] overflow-y-auto" data-node-id="3:22" role="complementary" aria-label="Output and Telemetry Diagnostics">
      {/* 1. Quiet Section Header: OUTPUT & TELEMETRY */}
      <div className="panel-section">
        <div className="panel-header">
          <div className="flex flex-col">
            <div className="flex items-center gap-1.5">
              <h2 className="panel-title font-mono tracking-wider font-semibold text-[11px]">OUTPUT & TELEMETRY</h2>
              {isConverged && (
                <span className="font-mono text-[8px] uppercase tracking-wide px-1.5 py-0.5 rounded border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[var(--pigment-class-a)]">
                  Converged
                </span>
              )}
            </div>
            <span className="text-[10px] text-[var(--text-muted)] font-mono truncate max-w-[140px]" title={algorithm.name}>
              {algorithm.shortCode}
            </span>
          </div>
          <div className="flex items-center gap-3 text-[10px] font-mono">
            <span className="flex items-center gap-1.5 whitespace-nowrap" style={{ color: isCarbon ? '#C26752' : '#B91C1C' }}>
              <span className="w-2.5 h-0.5 bg-current inline-block"></span> {isRL ? 'Episode Return' : isGAN ? 'D_Loss' : 'Train'}
            </span>
            <span className="flex items-center gap-1.5 whitespace-nowrap" style={{ color: isCarbon ? '#5276B8' : '#1E40AF' }}>
              <span className="w-2.5 h-0.5 border-t border-dashed border-current inline-block"></span> {isRL ? '10-Ep Moving Avg' : isGAN ? 'G_Loss' : 'Test'}
            </span>
          </div>
        </div>

        {/* Dedicated Dual Loss Sparkline */}
        <div
          ref={lossContainerRef}
          className="telemetry-chart-container h-[105px] mb-2 relative cursor-crosshair"
          onMouseMove={handleLossMouseMove}
          onMouseLeave={handleLossMouseLeave}
        >
          <canvas ref={lossCanvasRef} className="telemetry-chart-canvas block w-full h-full" />

          {/* Crosshair Hover Tooltip */}
          {hoverLoss && (
            <div
              className="absolute pointer-events-none z-10 font-mono text-[9px] bg-[var(--bg-canvas)]/95 backdrop-blur-sm border border-[var(--border-hairline)] shadow-sm px-2 py-1 flex flex-col gap-0.5"
              style={{
                top: 8,
                left: hoverLoss.x > 150 ? undefined : hoverLoss.x + 8,
                right: hoverLoss.x > 150 ? lossChartWidthRef.current - hoverLoss.x + 8 : undefined,
              }}
            >
              <span className="font-semibold text-[var(--text-primary)]">
                {isRL ? `Tick ${hoverLoss.epoch}` : `Epoch ${hoverLoss.epoch}`}
              </span>
              <span style={{ color: isCarbon ? '#C26752' : '#B91C1C' }}>
                {isRL ? 'Ep Return' : isGAN ? 'D_Loss' : 'Train'}: {hoverLoss.trainLoss.toFixed(isRL ? 2 : 4)}
              </span>
              <span style={{ color: isCarbon ? '#5276B8' : '#1E40AF' }}>
                {isRL ? '10-Ep Avg' : isGAN ? 'G_Loss' : 'Test'}: {hoverLoss.testLoss.toFixed(isRL ? 2 : 4)}
              </span>
            </div>
          )}
        </div>

        {/* Timeline Scrubber */}
        <div className="timeline-scrubber-section mb-3 pt-2 border-t border-[var(--border-hairline)]">
          <div className="flex items-center justify-between mb-1 text-[10px] font-mono">
            <span className="flex items-center gap-1 text-[var(--text-muted)] uppercase tracking-wider">
              <Clock size={11} />
              <span>Timeline Scrubber</span>
            </span>
            {scrubIndex !== null ? (
              <button
                onClick={() => onReturnToLive?.()}
                className="flex items-center gap-1 text-[9px] px-1.5 py-0.5 rounded bg-[var(--accent-amber)]/10 text-[var(--accent-amber)] hover:bg-[var(--accent-amber)]/20 transition-colors cursor-pointer"
                title="Return to Live Execution"
              >
                <RotateCcw size={10} />
                <span>Return to Live</span>
              </button>
            ) : (
              <span className="flex items-center gap-1 text-[9px] text-[var(--pigment-class-b)] font-semibold">
                <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse"></span>
                <span>
                  Live (Ep {String(isRL && rlEpisodeCount !== undefined ? rlEpisodeCount : currentEpoch).padStart(3, '0')})
                </span>
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="font-mono text-[9px] text-[var(--text-muted)] shrink-0">
              {historyBuffer.getStats().earliestEpoch}
            </span>
            <input
              type="range"
              aria-label="Timeline Scrubber"
              className="scientific-slider flex-1"
              min={0}
              max={Math.max(0, historyBuffer.getStats().count - 1)}
              value={scrubIndex !== null ? scrubIndex : Math.max(0, historyBuffer.getStats().count - 1)}
              disabled={algorithm.executionMode === 'closed-form' || historyBuffer.getStats().count <= 1}
              onPointerDown={() => {
                if (isRunning && onPause) onPause();
              }}
              onChange={(e) => {
                const idx = parseInt(e.target.value, 10);
                onScrubIndexChange?.(idx);
              }}
            />
            <span className="font-mono text-[9px] text-[var(--text-primary)] font-semibold shrink-0">
              Ep {scrubIndex !== null ? historyBuffer.get(scrubIndex)?.epoch ?? scrubIndex : currentEpoch}
            </span>
          </div>
        </div>

        {/* Metric Readouts */}
        {isClustering ? (
          <>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {algorithm.id === 'gmm'
                    ? 'Log-Likelihood'
                    : algorithm.id === 'dbscan'
                    ? 'Core Points'
                    : 'Inertia (WCSS)'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {algorithm.id === 'dbscan'
                    ? (metrics?.custom?.corePoints ?? '—')
                    : metrics?.custom?.inertia !== undefined
                    ? Number(metrics.custom.inertia).toFixed(2)
                    : metrics?.custom?.logLikelihood !== undefined
                    ? Number(metrics.custom.logLikelihood).toFixed(2)
                    : metrics
                    ? metrics.trainLoss.toFixed(2)
                    : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {algorithm.id === 'dbscan' ? 'Border Points' : 'Silhouette Score'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {algorithm.id === 'dbscan'
                    ? (metrics?.custom?.borderPoints ?? '—')
                    : metrics?.custom?.silhouette !== undefined
                    ? Number(metrics.custom.silhouette).toFixed(3)
                    : '—'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Active Clusters</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.activeClusters ?? metrics?.custom?.activeComponents ?? '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {algorithm.id === 'dbscan' ? 'Noise Points' : 'Current Sub-Step'}
                </span>
                <span className="font-semibold text-[11px] text-[var(--text-primary)] truncate">
                  {metrics?.custom?.noisePoints !== undefined
                    ? `${metrics.custom.noisePoints} rejected`
                    : metrics?.custom?.subStep
                    ? String(metrics.custom.subStep)
                    : '—'}
                </span>
              </div>
            </div>
          </>
        ) : isManifold ? (
          <>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {algorithm.id === 'pca'
                    ? 'Explained Var PC1'
                    : algorithm.id === 'lda'
                    ? (
                      <>
                        Fisher Ratio <span className="normal-case">J(w)</span>
                      </>
                    )
                    : algorithm.id === 'ica'
                    ? 'Kurtosis IC1'
                    : algorithm.id === 't-sne'
                    ? 'KL Divergence'
                    : 'Cross-Entropy'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.explainedVariancePct !== undefined
                    ? String(metrics.custom.explainedVariancePct)
                    : metrics?.custom?.fisherRatio !== undefined
                    ? Number(metrics.custom.fisherRatio).toFixed(3)
                    : metrics?.custom?.kurtosis1 !== undefined
                    ? Number(metrics.custom.kurtosis1).toFixed(3)
                    : metrics?.custom?.klDivergence !== undefined
                    ? Number(metrics.custom.klDivergence).toFixed(4)
                    : metrics?.custom?.crossEntropyLoss !== undefined
                    ? Number(metrics.custom.crossEntropyLoss).toFixed(3)
                    : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {algorithm.id === 'pca'
                    ? 'Reconstruction MSE'
                    : algorithm.id === 'lda'
                    ? 'Classes Found'
                    : algorithm.id === 'ica'
                    ? 'Kurtosis IC2'
                    : algorithm.id === 't-sne'
                    ? 'Perplexity'
                    : 'Graph Edges'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.reconstructionMSE !== undefined
                    ? Number(metrics.custom.reconstructionMSE).toFixed(4)
                    : metrics?.custom?.classesFound !== undefined
                    ? String(metrics.custom.classesFound)
                    : metrics?.custom?.kurtosis2 !== undefined
                    ? Number(metrics.custom.kurtosis2).toFixed(3)
                    : metrics?.custom?.perplexity !== undefined
                    ? String(metrics.custom.perplexity)
                    : metrics?.custom?.graphEdges !== undefined
                    ? String(metrics.custom.graphEdges)
                    : '—'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Optimization Loss</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics ? metrics.trainLoss.toFixed(4) : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Embedding Target</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {algorithm.id === 't-sne' || algorithm.id === 'umap' ? '2D Manifold' : '1D Linear Axis'}
                </span>
              </div>
            </div>
          </>
        ) : isMLP ? (
          <>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Train Loss (MSE)</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics ? metrics.trainLoss.toFixed(4) : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Test Loss</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics ? metrics.testLoss.toFixed(4) : '—'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {isRegression ? 'R² Score' : 'Accuracy / R²'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {isRegression
                    ? metrics?.trainR2 !== undefined
                      ? metrics.trainR2.toFixed(4)
                      : metrics?.custom?.trainR2 !== undefined
                      ? String(metrics.custom.trainR2)
                      : '—'
                    : metrics?.trainAccuracy !== undefined
                    ? `${metrics.trainAccuracy.toFixed(1)}%`
                    : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Dead Neurons (ReLU)</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.deadNeurons !== undefined
                    ? String(metrics.custom.deadNeurons)
                    : '0'}
                </span>
              </div>
            </div>

            <div className="flex flex-col font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[10px] text-[var(--text-muted)] uppercase">Layer Gradient Norms</span>
              <span className="font-semibold text-[11px] text-[var(--text-primary)] truncate mt-0.5">
                {metrics?.custom?.layerGradNorms
                  ? String(metrics.custom.layerGradNorms)
                  : (metrics?.gradientNorm ?? 0).toFixed(4)}
              </span>
            </div>

            {/* Decision Threshold (τ) slider: only show for classification! */}
            {!isRegression && (
              <div className="slider-row mt-3 pt-2 border-t border-[var(--border-hairline)]">
                <div className="slider-label-row">
                  <label htmlFor="decision-threshold-slider-1" className="slider-label text-[11px]">Decision Threshold (τ)</label>
                  <span className="slider-value font-mono text-[11px]">{decisionThreshold.toFixed(2)}</span>
                </div>
                <input
                  id="decision-threshold-slider-1"
                  aria-label="Decision Threshold"
                  type="range"
                  className="scientific-slider"
                  min={-2.0}
                  max={2.0}
                  step={0.05}
                  value={decisionThreshold}
                  onChange={(e) => onUpdateThreshold(parseFloat(e.target.value))}
                />
              </div>
            )}
          </>
        ) : isBottleneck ? (
          <>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Reconstruction MSE</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.reconstructionMSE !== undefined
                    ? Number(metrics.custom.reconstructionMSE).toFixed(4)
                    : metrics?.custom?.reconstructionLoss !== undefined
                    ? String(metrics.custom.reconstructionLoss)
                    : metrics
                    ? metrics.trainLoss.toFixed(4)
                    : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {algorithm.id === 'vae' ? 'KL Divergence D_KL' : 'Latent Dimension'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.klDivergence !== undefined
                    ? String(metrics.custom.klDivergence)
                    : metrics?.custom?.latentDim !== undefined
                    ? String(metrics.custom.latentDim)
                    : '—'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {algorithm.id === 'vae' ? 'Total β-ELBO Loss' : 'Optimization Loss'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.betaELBO !== undefined
                    ? String(metrics.custom.betaELBO)
                    : metrics
                    ? metrics.trainLoss.toFixed(4)
                    : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  Latent Probe <span className="normal-case">(z)</span>
                </span>
                <span className="font-semibold text-[11px] text-[var(--text-primary)]">
                  {metrics?.custom?.probeCoords ?? metrics?.custom?.probeZ ?? '—'}
                </span>
              </div>
            </div>

            <div className="flex flex-col font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[10px] text-[var(--text-muted)] uppercase">
                Decoded Position <span className="normal-case">D(z)</span>
              </span>
              <span className="font-semibold text-[11px] text-[var(--accent-amber)] mt-0.5">
                {metrics?.custom?.decodedCoords ?? metrics?.custom?.decodedHat ?? '—'}
              </span>
            </div>
          </>
        ) : isGAN ? (
          <>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Discriminator Loss (L_D)</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.dLoss ?? (metrics ? metrics.trainLoss.toFixed(4) : '—')}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Generator Loss (L_G)</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.gLoss ?? (metrics ? metrics.testLoss.toFixed(4) : '—')}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Mean D(Real) [→ 50%]</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.meanDReal ?? '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Mean D(Fake) [→ 50%]</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.meanDFake ?? '—'}
                </span>
              </div>
            </div>

            <div className="flex flex-col font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[10px] text-[var(--text-muted)] uppercase">Arena Status</span>
              <span className="font-semibold text-[11px] text-[var(--text-secondary)] mt-0.5">
                Minimax Equilibrium: L_D* = 2·ln(2) ≈ 1.386
              </span>
            </div>
          </>
        ) : isSequence ? (
          isRecurrent ? (
            <>
              <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase">Seq Loss / Perplexity</span>
                  <span className="font-semibold text-[13px] text-[var(--pigment-class-b)] truncate">
                    {metrics?.custom?.perplexity ? `PPL ${metrics.custom.perplexity}` : metrics?.trainLoss.toFixed(4) ?? '—'}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase">Step-1 Survival</span>
                  <span className="font-semibold text-[13px] text-[var(--pigment-class-a)] truncate">
                    {metrics?.custom?.survivalRatio !== undefined ? `${(Number(metrics.custom.survivalRatio) * 100).toFixed(1)}%` : '—'}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase">Selected Token (t*)</span>
                  <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                    "{metrics?.custom?.selectedToken ?? '—'}"
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase">Gate Activations</span>
                  <span className="font-semibold text-[11px] text-[var(--text-primary)] truncate">
                    {metrics?.custom?.forgetGateAvg !== undefined
                      ? `f:${metrics.custom.forgetGateAvg} i:${metrics.custom.inputGateAvg}`
                      : metrics?.custom?.updateGateZ !== undefined
                      ? `z:${metrics.custom.updateGateZ} r:${metrics.custom.resetGateR}`
                      : `||∇h1||: ${metrics?.custom?.step1GradNorm ?? '—'}`}
                  </span>
                </div>
              </div>

              <div className="flex flex-col font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Architecture Telemetry</span>
                <span className="font-semibold text-[10px] text-[var(--text-secondary)] mt-0.5 truncate">
                  {algorithm.id === 'lstm' ? 'LSTM Additive Conveyor Highway' : algorithm.id === 'gru' ? 'GRU Coupled Gate Balance' : 'Vanilla RNN Recurrent Bottleneck'}
                </span>
              </div>
            </>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase">Token Loss</span>
                  <span className="font-semibold text-[13px] text-[var(--pigment-class-b)] truncate">
                    {metrics ? metrics.trainLoss.toFixed(4) : '—'}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase">Attention Entropy</span>
                  <span className="font-semibold text-[13px] text-[var(--pigment-class-a)] truncate">
                    {metrics?.custom?.entropy ?? metrics?.custom?.combinedEntropy ?? '—'} nats
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase">Max Attention</span>
                  <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                    {metrics?.custom?.maxSoftmaxScore ?? metrics?.custom?.topPrediction ?? '—'}
                  </span>
                </div>
                <div className="flex flex-col">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase">Routing Horizon</span>
                  <span className="font-semibold text-[12px] text-[var(--text-primary)] truncate">
                    {metrics?.custom?.causalMask ?? metrics?.custom?.positionalEncoding ?? metrics?.custom?.pathLength ?? 'O(1) Direct'}
                  </span>
                </div>
              </div>

              <div className="flex flex-col font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Attention Telemetry</span>
                <span className="font-semibold text-[10px] text-[var(--text-secondary)] mt-0.5 truncate">
                  {metrics?.custom?.topPrediction ? `Top-1: ${metrics.custom.topPrediction}` : metrics?.custom?.activePhase ?? 'Multi-Head Attention Engine'}
                </span>
              </div>
            </>
          )
        ) : isVision ? (
          <>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Receptive Field</span>
                <span className="font-semibold text-[13px] leading-tight text-[var(--pigment-class-b)]">
                  {metrics?.custom?.effectiveReceptiveField ?? metrics?.custom?.receptiveField ?? '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Accuracy / Loss</span>
                <span className="font-semibold text-[13px] text-[var(--pigment-class-a)] truncate">
                  {metrics?.trainAccuracy !== undefined ? `${metrics.trainAccuracy.toFixed(1)}%` : metrics?.trainLoss.toFixed(3) ?? '—'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Gradient Norm</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.gradientNorm !== undefined ? metrics.gradientNorm.toFixed(4) : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">+1px Shift Delta</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.translationDelta1px !== undefined
                    ? Number(metrics.custom.translationDelta1px).toFixed(4)
                    : '0.0000'}
                </span>
              </div>
            </div>

            <div className="flex flex-col font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[10px] text-[var(--text-muted)] uppercase">Architecture Telemetry</span>
              <span className="font-semibold text-[10px] text-[var(--text-secondary)] mt-0.5 truncate">
                {metrics?.custom?.parameterCount ?? metrics?.custom?.patchTokensCount ?? metrics?.custom?.skipConnection ?? 'Deterministic Vision Pipeline'}
              </span>
            </div>
          </>
        ) : isRL ? (
          <>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Episode Return</span>
                <span
                  className={`font-semibold text-[13px] ${
                    (Number(metrics?.custom?.episodeReturn) || 0) >= 0
                      ? 'text-[var(--pigment-class-b)]'
                      : 'text-[var(--pigment-class-a)]'
                  }`}
                >
                  {metrics?.custom?.episodeReturn !== undefined
                    ? `${Number(metrics.custom.episodeReturn) >= 0 ? '+' : ''}${metrics.custom.episodeReturn}`
                    : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">10-Ep Moving Avg</span>
                <span
                  className={`font-semibold text-[13px] ${
                    (Number(metrics?.custom?.movingAvgReturn) || 0) >= 0
                      ? 'text-[var(--pigment-class-b)]'
                      : 'text-[var(--pigment-class-a)]'
                  }`}
                >
                  {metrics?.custom?.movingAvgReturn !== undefined
                    ? `${Number(metrics.custom.movingAvgReturn) >= 0 ? '+' : ''}${metrics.custom.movingAvgReturn}`
                    : '—'}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {algorithm.id === 'policy-gradient' ? 'Policy Loss / ||∇||' : 'Mean TD Error |δ|'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {algorithm.id === 'policy-gradient'
                    ? (metrics?.custom?.gradientNorm ?? (metrics ? metrics.trainLoss.toFixed(4) : '—'))
                    : (metrics?.custom?.meanTdError ?? (metrics ? metrics.trainLoss.toFixed(4) : '—'))}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Episode Length</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics?.custom?.episodeLength !== undefined
                    ? `${metrics.custom.episodeLength} steps`
                    : '—'}
                </span>
              </div>
            </div>

            <div className="flex flex-col font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <span className="text-[10px] text-[var(--text-muted)] uppercase">RL Telemetry</span>
              <span className="font-semibold text-[10px] text-[var(--text-secondary)] mt-0.5 truncate">
                {algorithm.id === 'q-learning'
                  ? `Q-Learning Tabular Control · Ep #${metrics?.custom?.totalEpisodes ?? 0}`
                  : algorithm.id === 'dqn'
                  ? `DQN Memory: ${metrics?.custom?.replayCount ?? '—'} · Sync: ${metrics?.custom?.targetSyncCounter ?? '—'}`
                  : `Entropy: ${metrics?.custom?.policyEntropy ?? '—'} · b: ${metrics?.custom?.baselineValue ?? 0} · Ep #${metrics?.custom?.totalEpisodes ?? 0}`}
              </span>
            </div>
          </>
        ) : (
          <>
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] pt-1 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Train Loss</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics ? metrics.trainLoss.toFixed(4) : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">Test Loss</span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {metrics ? metrics.testLoss.toFixed(4) : '—'}
                </span>
              </div>
            </div>

            {/* Accuracy & Margin metrics */}
            <div className="grid grid-cols-2 gap-2 font-mono text-[11px] mt-2 pt-2 border-t border-[var(--border-hairline)]">
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {isRegression ? 'R² Score' : 'Accuracy'}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {isRegression
                    ? metrics?.trainR2 !== undefined
                      ? `R² ${metrics.trainR2.toFixed(3)}`
                      : '—'
                    : metrics?.trainAccuracy !== undefined
                    ? `${metrics.trainAccuracy.toFixed(1)}%`
                    : '—'}
                </span>
              </div>
              <div className="flex flex-col">
                <span className="text-[10px] text-[var(--text-muted)] uppercase">
                  {fourthTelemetryMetric.label}
                </span>
                <span className="font-semibold text-[13px] text-[var(--text-primary)]">
                  {fourthTelemetryMetric.value}
                </span>
              </div>
            </div>

            {/* Decision Threshold (τ) slider: only show for classification! */}
            {!isRegression && (
              <div className="slider-row mt-3 pt-2 border-t border-[var(--border-hairline)]">
                <div className="slider-label-row">
                  <label htmlFor="decision-threshold-slider-2" className="slider-label text-[11px]">Decision Threshold (τ)</label>
                  <span className="slider-value font-mono text-[11px]">{decisionThreshold.toFixed(2)}</span>
                </div>
                <input
                  id="decision-threshold-slider-2"
                  aria-label="Decision Threshold"
                  type="range"
                  className="scientific-slider"
                  min={algorithm.id === 'logistic-regression' ? 0.01 : algorithm.id === 'decision-tree' ? 0.05 : -2.0}
                  max={algorithm.id === 'logistic-regression' ? 0.99 : algorithm.id === 'decision-tree' ? 0.95 : 2.0}
                  step={algorithm.id === 'logistic-regression' ? 0.01 : 0.05}
                  value={decisionThreshold}
                  onChange={(e) => onUpdateThreshold(parseFloat(e.target.value))}
                />
              </div>
            )}
          </>
        )}

        {/* 3-Lens Inspector (Intuition, Math, Code) */}
        <div className="three-lens-inspector mt-3 pt-3 border-t border-[var(--border-hairline)]">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-mono text-[10.5px] uppercase tracking-wider font-semibold text-[var(--text-primary)]">
              3-LENS INSPECTOR
            </h3>
            <div className="flex items-center gap-1 bg-[var(--bg-subtle)] p-0.5 rounded-md border border-[var(--border-hairline)] text-[10px] font-mono">
              <button
                className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
                  activeLensTab === 'intuition'
                    ? 'bg-[var(--bg-elevated)] font-semibold text-[var(--text-primary)] shadow-xs border border-[var(--border-hairline)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent'
                }`}
                onClick={() => setActiveLensTab('intuition')}
              >
                Intuition
              </button>
              <button
                className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
                  activeLensTab === 'math'
                    ? 'bg-[var(--bg-elevated)] font-semibold text-[var(--text-primary)] shadow-xs border border-[var(--border-hairline)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent'
                }`}
                onClick={() => setActiveLensTab('math')}
              >
                Math
              </button>
              <button
                className={`px-2 py-0.5 rounded transition-all cursor-pointer ${
                  activeLensTab === 'code'
                    ? 'bg-[var(--bg-elevated)] font-semibold text-[var(--text-primary)] shadow-xs border border-[var(--border-hairline)]'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] border border-transparent'
                }`}
                onClick={() => setActiveLensTab('code')}
              >
                Code
              </button>
            </div>
          </div>

          {/* Lens Tab Content */}
          {activeLensTab === 'intuition' && (
            <div className="space-y-2 text-[11px] font-sans">
              <div className="p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                <span className="font-mono text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-0.5 font-semibold">
                  Core Summary
                </span>
                <p className="text-[var(--text-primary)] leading-relaxed break-words">
                  {explanation?.summary || algorithm.description}
                </p>
              </div>

              {explanation?.coreAssumption && (
                <div className="p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                  <span className="font-mono text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-0.5 font-semibold">
                    Core Assumption
                  </span>
                  <p className="text-[var(--text-secondary)] leading-relaxed break-words">
                    {explanation.coreAssumption}
                  </p>
                </div>
              )}

              {explanation?.whenItFails && (
                <div className="p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                  <span className="font-mono text-[9px] uppercase tracking-wider text-[var(--accent-amber)] block mb-0.5 font-semibold">
                    Failure Boundary
                  </span>
                  <p className="text-[var(--text-secondary)] leading-relaxed break-words">
                    {explanation.whenItFails}
                  </p>
                </div>
              )}

              {explanation?.microLog && explanation.microLog.length > 0 && (
                <div className="p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                  <span className="font-mono text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1 font-semibold">
                    Step Micro-Log
                  </span>
                  <ul className="space-y-1 font-mono text-[10px] text-[var(--text-secondary)]">
                    {explanation.microLog.map((log, idx) => (
                      <li key={idx} className="flex items-start gap-1.5 break-words">
                        <span className="text-[var(--accent-amber)] shrink-0">•</span>
                        <span>{log}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {activeLensTab === 'math' && (
            <div className="space-y-2.5 font-mono text-[10px]">
              {explanation?.mathLatex?.objective && (
                <div className="p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] overflow-hidden">
                  <span className="text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1 font-semibold">
                    Optimization Objective
                  </span>
                  <div className="overflow-x-auto py-1 text-[var(--text-primary)] max-w-full">
                    <LatexSpan expression={explanation.mathLatex.objective} displayMode />
                  </div>
                </div>
              )}

              {explanation?.mathLatex?.lossOrUpdate && (
                <div className="p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] overflow-hidden">
                  <span className="text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1 font-semibold">
                    Parameter Update Rule
                  </span>
                  <div className="overflow-x-auto py-1 text-[var(--text-primary)] max-w-full">
                    <LatexSpan expression={explanation.mathLatex.lossOrUpdate} displayMode />
                  </div>
                </div>
              )}

              {explanation?.mathLatex?.dualOrConstraint && (
                <div className="p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] overflow-hidden">
                  <span className="text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1 font-semibold">
                    Dual / Manifold Constraint
                  </span>
                  <div className="overflow-x-auto py-1 text-[var(--text-primary)] max-w-full">
                    <LatexSpan expression={explanation.mathLatex.dualOrConstraint} displayMode />
                  </div>
                </div>
              )}
            </div>
          )}

          {activeLensTab === 'code' && (
            <div className="space-y-2.5 font-mono text-[10px]">
              {explanation?.codeSnippet?.sklearn && (
                <div className="rounded border border-[var(--border-hairline)] overflow-hidden bg-[var(--bg-canvas)]">
                  <div className="flex items-center justify-between px-2.5 py-1 bg-[var(--bg-subtle)] border-b border-[var(--border-hairline)]">
                    <span className="text-[9px] font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                      scikit-learn
                    </span>
                    <button
                      onClick={() => handleCopy('sklearn', explanation.codeSnippet.sklearn)}
                      className="flex items-center gap-1 text-[9px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
                    >
                      {copiedSnippet === 'sklearn' ? <Check size={11} className="text-[var(--pigment-class-b)]" /> : <Copy size={11} />}
                      <span>{copiedSnippet === 'sklearn' ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>
                  <pre className="p-2 overflow-x-auto text-[9.5px] leading-relaxed text-[var(--text-primary)] font-mono max-w-full">
                    <code>{explanation.codeSnippet.sklearn}</code>
                  </pre>
                </div>
              )}

              {explanation?.codeSnippet?.pytorch && (
                <div className="rounded border border-[var(--border-hairline)] overflow-hidden bg-[var(--bg-canvas)]">
                  <div className="flex items-center justify-between px-2.5 py-1 bg-[var(--bg-subtle)] border-b border-[var(--border-hairline)]">
                    <span className="text-[9px] font-semibold text-[var(--text-muted)] uppercase tracking-wider">
                      PyTorch
                    </span>
                    <button
                      onClick={() => handleCopy('pytorch', explanation.codeSnippet.pytorch)}
                      className="flex items-center gap-1 text-[9px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer"
                    >
                      {copiedSnippet === 'pytorch' ? <Check size={11} className="text-[var(--pigment-class-b)]" /> : <Copy size={11} />}
                      <span>{copiedSnippet === 'pytorch' ? 'Copied' : 'Copy'}</span>
                    </button>
                  </div>
                  <pre className="p-2 overflow-x-auto text-[9.5px] leading-relaxed text-[var(--text-primary)] font-mono max-w-full">
                    <code>{explanation.codeSnippet.pytorch}</code>
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
};
