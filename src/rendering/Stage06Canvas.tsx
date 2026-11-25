import React, { useState, useMemo } from 'react';
import { AlgorithmPlugin, VisualArtifact, SequenceGateAttnArtifact, SequenceDataset } from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { DirtyManager } from './dirty-manager';
import { renderLatex } from '../utils/katex-render';
import { Layers, Activity, Zap, Lock, Unlock } from 'lucide-react';

export interface Stage06CanvasProps {
  theme: ThemeMode;
  artifacts: VisualArtifact[];
  dirtyManager: DirtyManager;
  algorithm: AlgorithmPlugin;
  hyperparams?: Record<string, number | string>;
  sequenceDataset?: SequenceDataset;
  onSelectToken?: (index: number) => void;
  onPinGate?: (stepIndex: number, gateId: string, value?: number) => void;
  onUpdateHyperparam?: (key: string, value: number | string) => void;
}

export const Stage06Canvas: React.FC<Stage06CanvasProps> = ({
  theme: _theme,
  artifacts,
  dirtyManager,
  algorithm,
  hyperparams = {},
  sequenceDataset,
  onSelectToken,
  onPinGate,
  onUpdateHyperparam,
}) => {

  // Find active Stage 06 artifact
  const seqArtifact = useMemo(() => {
    return artifacts.find((a): a is SequenceGateAttnArtifact => a.type === 'sequence-gate-attn');
  }, [artifacts]);

  const [hoveredQueryIndex, setHoveredQueryIndex] = useState<number | null>(null);
  const [selectedHeadTab, setSelectedHeadTab] = useState<number>(-1); // -1 = Combined
  const [matrixDisplayMode, setMatrixDisplayMode] = useState<'weights' | 'scores'>('weights');

  const modelFamily = seqArtifact?.modelFamily ?? (['rnn', 'lstm', 'gru'].includes(algorithm.id) ? 'recurrent' : 'attention');
  const tokens = seqArtifact?.tokens ?? sequenceDataset?.sequence.map((s) => s.text) ?? ['Token_1', 'Token_2', 'Token_3'];
  const T = tokens.length;
  const activeStep = seqArtifact?.selectedStep ?? sequenceDataset?.selectedTokenIndex ?? 0;

  const recurrent = seqArtifact?.recurrent;
  const attention = seqArtifact?.attention;

  const handleTokenClick = (idx: number) => {
    if (onSelectToken) {
      onSelectToken(idx);
    }
    dirtyManager.markDirty();
  };

  const handleHeadSelect = (headId: number) => {
    setSelectedHeadTab(headId);
    dirtyManager.markDirty();
  };

  // Determine active query index for bipartite beam isolation
  const queryFocusIndex = hoveredQueryIndex !== null ? hoveredQueryIndex : activeStep;

  // -------------------------------------------------------------
  // LAYOUT A: Recurrent Highway (RNN, LSTM, GRU)
  // -------------------------------------------------------------
  if (modelFamily === 'recurrent') {
    const isLSTM = algorithm.id === 'lstm';
    const isGRU = algorithm.id === 'gru';
    const isRNN = algorithm.id === 'rnn';

    const timeSteps = recurrent?.timeSteps ?? [];
    const mag = recurrent?.magnifiedStep;

    return (
      <div className="w-full h-full flex flex-col overflow-hidden bg-[var(--bg-primary)] p-4 select-none font-sans">
        {/* Top Control & Diagnostic Sub-Bar */}
        <div className="flex flex-wrap items-center justify-between gap-2.5 pb-3 border-b border-[var(--border-hairline)] mb-3 min-w-0">
          <div className="flex items-center gap-3">
            <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)] font-semibold flex items-center gap-1.5">
              <Activity size={13} className="text-[var(--pigment-class-b)]" />
              Unrolled Temporal Ribbon ({T} Time-Steps)
            </span>
            {seqArtifact?.phaseName && (
              <span className="font-mono text-[10px] px-2 py-0.5 rounded bg-[var(--bg-subtle)] text-[var(--text-secondary)] border border-[var(--border-hairline)]">
                {seqArtifact.phaseName}
              </span>
            )}
          </div>

          <div className="flex items-center gap-4 font-mono text-[11px] shrink-0">
            <div className="flex items-center gap-1.5">
              <span className="text-[var(--text-muted)]">Step-1 Survival Ratio:</span>
              <span className="font-semibold text-[var(--pigment-class-a)]">
                {recurrent?.conveyorBelt?.gradientSurvivalRatio !== undefined
                  ? `${(recurrent.conveyorBelt.gradientSurvivalRatio * 100).toFixed(2)}%`
                  : '—'}
              </span>
            </div>
          </div>
        </div>

        {/* 1. TOP UNROLLED TIME-STEP RIBBON */}
        <div className="relative rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-secondary)] p-3 mb-3 shadow-inner shrink-0">
          {/* LSTM Additive Cell State Conveyor Belt Visual Rail */}
          {isLSTM && (
            <div className="mb-2.5 flex flex-col gap-1 px-1">
              <div className="flex items-center justify-between font-mono text-[9px] uppercase tracking-wider text-emerald-500 font-semibold">
                <span className="flex items-center gap-1">
                  <Zap size={10} /> Top Cell-State Conveyor Belt (C_t-1 → C_t)
                </span>
                <span className="text-[8px] text-[var(--text-muted)]">Additive Highway</span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-gradient-to-r from-emerald-500/30 via-emerald-400 to-emerald-500/80 shadow-[0_0_8px_rgba(16,185,129,0.25)]" />
            </div>
          )}

          <div className="flex items-stretch justify-between gap-1.5 overflow-x-auto pb-1">
            {timeSteps.map((step, idx) => {
              const isSelected = idx === activeStep;
              const gradRatio = Math.min(1.0, step.gradNorm);

              return (
                <div
                  key={idx}
                  onClick={() => handleTokenClick(idx)}
                  className={`flex-1 min-w-[52px] max-w-[110px] flex flex-col items-center p-1.5 rounded-md cursor-pointer transition-all border ${
                    isSelected
                      ? 'border-[var(--pigment-class-b)] bg-[var(--bg-elevated)] shadow-md ring-1 ring-[var(--pigment-class-b)]'
                      : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--border-strong)]'
                  }`}
                >
                  {/* Step Index & Token Label */}
                  <span className="font-mono text-[9px] text-[var(--text-muted)] uppercase mb-1">
                    t = {idx + 1}
                  </span>
                  <span className="font-semibold text-[13px] text-[var(--text-primary)] truncate max-w-full mb-2">
                    "{step.token}"
                  </span>

                  {/* Hidden State Mini Vector Bars (Full 8 Dimensions) */}
                  <div className="w-full flex items-end justify-center gap-0.5 h-10 bg-[var(--bg-primary)] p-1 rounded border border-[var(--border-hairline)] mb-2">
                    {Array.from(step.hiddenState).map((val, d) => {
                      const hPct = Math.min(100, Math.max(5, Math.abs(val) * 75));
                      const isPos = val >= 0;
                      return (
                        <div
                          key={d}
                          style={{ height: `${hPct}%` }}
                          className={`w-1 rounded-sm ${
                            isPos ? 'bg-[var(--pigment-class-b)]' : 'bg-[var(--pigment-class-a)] opacity-80'
                          }`}
                          title={`h_${idx}[${d}] = ${val.toFixed(3)}`}
                        />
                      );
                    })}
                  </div>

                  {/* Cell State Vector Indicator (LSTM only, Euclidean Norm) */}
                  {isLSTM && step.cellState && (() => {
                    let cellNormSq = 0;
                    for (let d = 0; d < step.cellState.length; d++) {
                      cellNormSq += step.cellState[d] * step.cellState[d];
                    }
                    const cNorm = Math.sqrt(cellNormSq);
                    return (
                      <div className="w-full mb-1.5">
                        <div className="flex items-center justify-between font-mono text-[8px] text-emerald-400">
                          <span>||C_{idx + 1}|| = {cNorm.toFixed(2)}</span>
                          <span>{step.gates?.f !== undefined ? `f:${step.gates.f.toFixed(2)}` : ''}</span>
                        </div>
                        <div className="w-full h-1 bg-emerald-950/40 rounded overflow-hidden">
                          <div
                            style={{ width: `${Math.min(100, cNorm * 35)}%` }}
                            className="h-full bg-emerald-400"
                          />
                        </div>
                      </div>
                    );
                  })()}

                  {/* Backward Gradient Norm Gauge ||dL/dh_t|| */}
                  <div className="w-full flex flex-col gap-0.5 mt-auto pt-1 border-t border-[var(--border-hairline)]">
                    <div className="flex items-center justify-between font-mono text-[8px]">
                      <span className="text-[var(--text-muted)]">||∇||</span>
                      <span
                        className={`font-semibold ${
                          step.gradNorm < 0.05
                            ? 'text-red-400'
                            : step.gradNorm > 1.2
                            ? 'text-amber-400'
                            : 'text-[var(--text-primary)]'
                        }`}
                      >
                        {step.gradNorm.toFixed(step.gradNorm < 0.01 ? 5 : 3)}
                      </span>
                    </div>
                    <div className="w-full h-1 bg-[var(--bg-primary)] rounded overflow-hidden">
                      <div
                        style={{ width: `${Math.max(2, Math.min(100, gradRatio * 100))}%` }}
                        className={`h-full ${
                          step.gradNorm < 0.05
                            ? 'bg-red-400'
                            : step.gradNorm > 1.2
                            ? 'bg-amber-400'
                            : 'bg-[var(--pigment-class-b)]'
                        }`}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* 2. BOTTOM MAGNIFIED INTERNAL CELL SCHEMATIC */}
        <div className="flex-1 min-h-0 flex flex-col rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-secondary)] p-4 shadow-sm overflow-y-auto">
          <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-[var(--border-hairline)] mb-3 min-w-0 shrink-0">
            <div className="flex items-center gap-2">
              <span className="font-mono text-[12px] font-semibold text-[var(--text-primary)]">
                Magnified Step t* = {activeStep + 1} ("{tokens[activeStep]}")
              </span>
              <span className="font-mono text-[10px] text-[var(--text-muted)]">
                · {isLSTM ? 'LSTM Internal Valve Array' : isGRU ? 'GRU Coupled Gate Balance' : 'Vanilla RNN Recurrent Mixing'}
              </span>
            </div>

            <div className="flex items-center gap-2 text-[10px] font-mono text-[var(--text-secondary)] shrink-0">
              <span>Click step pill above to scrub time</span>
            </div>
          </div>

          {/* Live SVG Cell Flow Schematic */}
          <div className="w-full mb-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-primary)] px-3 py-2 shrink-0">
            <svg viewBox="0 0 640 96" className="w-full h-20 block" aria-label="Recurrent cell flow diagram">
              <defs>
                <marker id="stage06-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                  <path d="M 0 1 L 10 5 L 0 9 z" fill="var(--text-muted)" />
                </marker>
              </defs>
              {/* Conveyor belt for LSTM */}
              {isLSTM && (
                <g>
                  <line x1="40" y1="20" x2="600" y2="20" stroke="#10B981" strokeWidth="3" strokeOpacity="0.85" markerEnd="url(#stage06-arrow)" />
                  <circle cx="230" cy="20" r="10" fill="var(--bg-elevated)" stroke="#10B981" strokeWidth="1.5" />
                  <text x="230" y="24" textAnchor="middle" fontSize="10" fill="#10B981" fontFamily="monospace">×</text>
                  <circle cx="380" cy="20" r="10" fill="var(--bg-elevated)" stroke="#10B981" strokeWidth="1.5" />
                  <text x="380" y="24" textAnchor="middle" fontSize="10" fill="#10B981" fontFamily="monospace">+</text>
                  <text x="28" y="23" textAnchor="end" fontSize="10" fill="#10B981" fontFamily="monospace">C_{activeStep}</text>
                  <text x="610" y="23" textAnchor="start" fontSize="10" fill="#10B981" fontFamily="monospace">C_{activeStep + 1}</text>
                </g>
              )}
              {/* Hidden state rail */}
              <line x1="40" y1="68" x2="600" y2="68" stroke="var(--pigment-class-b, #1E40AF)" strokeWidth="2.5" strokeOpacity="0.85" markerEnd="url(#stage06-arrow)" />
              <text x="28" y="72" textAnchor="end" fontSize="10" fill="var(--text-secondary)" fontFamily="monospace">h_{activeStep}</text>
              <text x="610" y="72" textAnchor="start" fontSize="10" fill="var(--pigment-class-b, #1E40AF)" fontFamily="monospace">h_{activeStep + 1}</text>
              {/* Input injection */}
              <line x1="150" y1="92" x2="150" y2="72" stroke="var(--pigment-class-a, #C83D24)" strokeWidth="2" markerEnd="url(#stage06-arrow)" />
              <text x="150" y="94" textAnchor="middle" fontSize="9" fill="var(--pigment-class-a, #C83D24)" fontFamily="monospace">x_{activeStep + 1}</text>
              {/* Central junction */}
              <rect x="210" y="46" width="220" height="36" rx="4" fill="var(--bg-elevated)" stroke="var(--border-strong)" strokeWidth="1.2" />
              <text x="320" y="68" textAnchor="middle" fontSize="11" fill="var(--text-primary)" fontFamily="monospace" fontWeight="600">
                {isLSTM
                  ? `f=${(mag?.f ?? 0.85).toFixed(2)}  i=${(mag?.i ?? 0.5).toFixed(2)}  o=${(mag?.o ?? 0.7).toFixed(2)}`
                  : isGRU
                  ? `z=${(mag?.z ?? 0.45).toFixed(2)}  r=${(mag?.r ?? 0.6).toFixed(2)}  (1-z)h + z h̃`
                  : 'tanh(W_hh h_{t-1} + W_xh x_t)'}
              </text>
            </svg>
          </div>

          <div className="flex-1 min-h-0 grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
            {/* Left: Input & Prior Context */}
            <div className="flex flex-col gap-3 p-3 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                1. Context & Injection
              </span>

              <div className="flex flex-col gap-1">
                <span className="text-[11px] font-medium text-[var(--text-secondary)]">
                  Previous Hidden State (h_{activeStep}):
                </span>
                <span className="font-mono text-[12px] text-[var(--text-primary)] font-semibold">
                  ||h_{activeStep}|| = {mag?.prevHiddenNorm.toFixed(3) ?? '0.000'}
                </span>
              </div>

              <div className="flex flex-col gap-1 pt-2 border-t border-[var(--border-hairline)]">
                <span className="text-[11px] font-medium text-[var(--text-secondary)]">
                  Input Vector x_{activeStep + 1} ("{tokens[activeStep]}"):
                </span>
                <div className="flex flex-col gap-1">
                  <div className="w-full flex items-center justify-between gap-0.5 h-3 bg-[var(--bg-primary)] p-0.5 rounded border border-[var(--border-hairline)]">
                    {Array.from(timeSteps[activeStep]?.inputVec ?? []).map((val, d) => (
                      <div
                        key={d}
                        style={{ opacity: Math.max(0.2, Math.min(1.0, Math.abs(val))) }}
                        className={`h-full flex-1 rounded-xs ${val >= 0 ? 'bg-[var(--pigment-class-b)]' : 'bg-[var(--pigment-class-a)]'}`}
                        title={`x[${d}] = ${val.toFixed(3)}`}
                      />
                    ))}
                  </div>
                  <span className="font-mono text-[9px] text-[var(--text-muted)]">8D Semantic Coordinates</span>
                </div>
              </div>
            </div>

            {/* Center: Live Gate Valves & Pipes */}
            <div className="flex flex-col gap-3 p-3 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] shadow-inner">
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold flex items-center justify-between">
                <span>2. Gate Valve Array</span>
                <span className="text-[9px] text-[var(--pigment-class-a)]">Numeric [0.00 .. 1.00]</span>
              </span>

              {isLSTM && (
                <div className="flex flex-col gap-2.5">
                  {/* Forget Gate Valve */}
                  <div className="flex items-center justify-between p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                    <div className="flex flex-col">
                      <span className="font-mono text-[11px] font-semibold text-emerald-400">Forget Gate (f_t)</span>
                      <span className="text-[9px] text-[var(--text-muted)]">Controls C_{activeStep} retention</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[13px] font-bold text-[var(--text-primary)]">
                        {mag?.f !== undefined ? mag.f.toFixed(2) : '0.85'}
                      </span>
                      {onPinGate && (
                        <button
                          onClick={() => onPinGate(activeStep, 'f', mag?.pinnedGates?.f !== undefined ? undefined : 0.0)}
                          className={`p-1 rounded text-[10px] border ${
                            mag?.pinnedGates?.f !== undefined
                              ? 'bg-red-500/20 border-red-500 text-red-400'
                              : 'bg-[var(--bg-elevated)] border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                          }`}
                          title="Pin/Clamp Forget Gate to 0.0 (flush memory) or release"
                        >
                          {mag?.pinnedGates?.f !== undefined ? <Lock size={11} /> : <Unlock size={11} />}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Input Gate Valve */}
                  <div className="flex items-center justify-between p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                    <div className="flex flex-col">
                      <span className="font-mono text-[11px] font-semibold text-[var(--pigment-class-b)]">Input Gate (i_t)</span>
                      <span className="text-[9px] text-[var(--text-muted)]">Candidate write intensity</span>
                    </div>
                    <span className="font-mono text-[13px] font-bold text-[var(--text-primary)]">
                      {mag?.i !== undefined ? mag.i.toFixed(2) : '0.50'}
                    </span>
                  </div>

                  {/* Output Gate Valve */}
                  <div className="flex items-center justify-between p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                    <div className="flex flex-col">
                      <span className="font-mono text-[11px] font-semibold text-purple-400">Output Gate (o_t)</span>
                      <span className="text-[9px] text-[var(--text-muted)]">Hidden state readout valve</span>
                    </div>
                    <span className="font-mono text-[13px] font-bold text-[var(--text-primary)]">
                      {mag?.o !== undefined ? mag.o.toFixed(2) : '0.70'}
                    </span>
                  </div>
                </div>
              )}

              {isGRU && (
                <div className="flex flex-col gap-2.5">
                  {/* Update Gate */}
                  <div className="flex items-center justify-between p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                    <div className="flex flex-col">
                      <span className="font-mono text-[11px] font-semibold text-blue-400">Update Gate (z_t)</span>
                      <span className="text-[9px] text-[var(--text-muted)]">Coupled interpolation ratio</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[13px] font-bold text-[var(--text-primary)]">
                        {mag?.z !== undefined ? mag.z.toFixed(2) : '0.45'}
                      </span>
                      {onPinGate && (
                        <button
                          onClick={() => onPinGate(activeStep, 'z', mag?.pinnedGates?.z !== undefined ? undefined : 1.0)}
                          className={`p-1 rounded text-[10px] border ${
                            mag?.pinnedGates?.z !== undefined
                              ? 'bg-red-500/20 border-red-500 text-red-400'
                              : 'bg-[var(--bg-elevated)] border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                          }`}
                          title="Pin/Clamp Update Gate to 1.0 (overwrite history) or release"
                        >
                          {mag?.pinnedGates?.z !== undefined ? <Lock size={11} /> : <Unlock size={11} />}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Reset Gate */}
                  <div className="flex items-center justify-between p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                    <div className="flex flex-col">
                      <span className="font-mono text-[11px] font-semibold text-amber-400">Reset Gate (r_t)</span>
                      <span className="text-[9px] text-[var(--text-muted)]">Past history filter</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-[13px] font-bold text-[var(--text-primary)]">
                        {mag?.r !== undefined ? mag.r.toFixed(2) : '0.60'}
                      </span>
                      {onPinGate && (
                        <button
                          onClick={() => onPinGate(activeStep, 'r', mag?.pinnedGates?.r !== undefined ? undefined : 0.0)}
                          className={`p-1 rounded text-[10px] border ${
                            mag?.pinnedGates?.r !== undefined
                              ? 'bg-red-500/20 border-red-500 text-red-400'
                              : 'bg-[var(--bg-elevated)] border-[var(--border-hairline)] text-[var(--text-muted)] hover:text-[var(--text-primary)]'
                          }`}
                          title="Pin/Clamp Reset Gate to 0.0 (sever context) or release"
                        >
                          {mag?.pinnedGates?.r !== undefined ? <Lock size={11} /> : <Unlock size={11} />}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Coupled Balance Indicator */}
                  <div className="p-2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
                    <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                      <span className="text-emerald-400">{( (1 - (mag?.z ?? 0.45)) * 100).toFixed(0)}% Keep</span>
                      <span className="text-blue-400">{( ((mag?.z ?? 0.45)) * 100).toFixed(0)}% Write</span>
                    </div>
                    <div className="w-full h-1.5 bg-blue-500 rounded-full overflow-hidden flex">
                      <div style={{ width: `${(1 - (mag?.z ?? 0.45)) * 100}%` }} className="h-full bg-emerald-400" />
                    </div>
                  </div>
                </div>
              )}

              {isRNN && (
                <div className="flex flex-col items-center justify-center p-4 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-center">
                  <span className="font-mono text-[11px] font-semibold text-red-400 mb-1">
                    No Protective Gate Valves!
                  </span>
                  <span className="text-[10px] text-[var(--text-secondary)] leading-relaxed">
                    Vanilla RNN passes state directly through a saturating tanh junction without gated additive isolation:
                  </span>
                  <div
                    className="font-mono text-[11px] mt-2 text-[var(--pigment-class-b)]"
                    dangerouslySetInnerHTML={{
                      __html: renderLatex('h_t = \\tanh(W_{hh} h_{t-1} + W_{xh} x_t + b)'),
                    }}
                  />
                </div>
              )}
            </div>

            {/* Right: Output State & Equations */}
            <div className="flex flex-col gap-3 p-3 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)]">
              <span className="font-mono text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                3. Emitted State
              </span>

              <div className="flex flex-col gap-1">
                <span className="text-[11px] font-medium text-[var(--text-secondary)]">
                  Updated Hidden State (h_{activeStep + 1}):
                </span>
                <span className="font-mono text-[12px] text-[var(--pigment-class-b)] font-semibold">
                  ||h_{activeStep + 1}|| = {mag?.nextHiddenNorm.toFixed(3) ?? '0.000'}
                </span>
              </div>

              <div className="p-2.5 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] mt-1">
                <span className="font-mono text-[9px] uppercase tracking-wider text-[var(--text-muted)] block mb-1">
                  Temporal BPTT Formulation:
                </span>
                <div
                  className="font-mono text-[10px] overflow-x-auto text-[var(--text-secondary)]"
                  dangerouslySetInnerHTML={{
                    __html: isLSTM
                      ? renderLatex('C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t')
                      : isGRU
                      ? renderLatex('h_t = (1 - z_t) \\odot h_{t-1} + z_t \\odot \\tilde{h}_t')
                      : renderLatex('\\partial h_t / \\partial h_1 = \\prod_{k=2}^t W_{hh}^T \\text{diag}(1 - h_k^2)'),
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Swiss Bottom Status & Legend Bar for Layout A */}
        <footer className="w-full flex flex-wrap items-center justify-between px-3 py-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] gap-2 select-none shrink-0 mt-auto">
          <div className="flex items-center gap-3 text-[var(--text-muted)] font-mono text-[9px]">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 border border-dashed border-[var(--pigment-class-b)] bg-[var(--pigment-class-b)]/20 inline-block" />
              <span>Temporal Ribbon ({T} Steps)</span>
            </div>
            {isLSTM && (
              <div className="flex items-center gap-1.5">
                <Zap size={11} className="text-emerald-400" />
                <span>Additive Cell Highway: C_t = f_t ⊙ C_(t-1) + i_t ⊙ C̃_t</span>
              </div>
            )}
            {isGRU && (
              <div className="flex items-center gap-1.5">
                <Zap size={11} className="text-blue-400" />
                <span>Coupled Balance: (1 - z_t) Keep vs z_t Write</span>
              </div>
            )}
            {isRNN && (
              <div className="flex items-center gap-1.5 text-red-400">
                <span>⚠ Ungated Recurrent Chain (BPTT Bottleneck)</span>
              </div>
            )}
          </div>
          <div className="flex items-center gap-3 text-[var(--text-muted)] font-mono text-[9px]">
            <span>{algorithm.name}</span>
            <span className="text-[var(--border-hairline)]">|</span>
            <span>8D Hidden States</span>
            <span className="text-[var(--border-hairline)]">|</span>
            <span className="font-semibold text-[var(--text-primary)]">Phase Walkthrough</span>
          </div>
        </footer>
      </div>
    );
  }

  // -------------------------------------------------------------
  // LAYOUT B: Attention & Transformer Arena
  // -------------------------------------------------------------
  const isDecoder = algorithm.id === 'transformer-decoder';
  const isEncoder = algorithm.id === 'transformer-encoder';
  const isEncDec = algorithm.id === 'transformer-enc-dec';

  const activeSubLayer = attention?.activeSubLayer ?? (hyperparams.activeSubLayer as string) ?? 'cross-attention';
  const isCrossAttention = isEncDec && activeSubLayer === 'cross-attention';

  const sourceTokens = attention?.crossAttention?.sourceTokens ?? ['Ami', 'bhat', 'khai'];
  const targetTokens = attention?.crossAttention?.targetTokens ?? ['<bos>', 'I', 'eat', 'rice'];

  const keyTokens = isCrossAttention ? sourceTokens : tokens;
  const queryTokens = isCrossAttention ? targetTokens : tokens;
  const numKeys = keyTokens.length;
  const numQueries = queryTokens.length;

  const rawScores = attention?.rawScores ?? new Float32Array(numQueries * numKeys);
  const attentionWeights = isCrossAttention
    ? (attention?.crossAttention?.weights ?? new Float32Array(numQueries * numKeys))
    : (attention?.attentionWeights ?? new Float32Array(numQueries * numKeys));
  const heads = attention?.heads ?? [];
  const nextTokens = attention?.nextTokens ?? [];
  const isCausal = attention?.isCausal ?? false;

  const displayMatrix = matrixDisplayMode === 'scores' && !isCrossAttention ? rawScores : attentionWeights;

  return (
    <div className="w-full h-full flex flex-col overflow-hidden bg-[var(--bg-primary)] p-4 select-none font-sans">
      {/* Top Transformer Sub-Header & Multi-Head / Matrix Mode Tabs */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 pb-3 border-b border-[var(--border-hairline)] mb-3 min-w-0">
        <div className="flex flex-wrap items-center gap-2.5 min-w-0">
          <span className="font-mono text-[11px] uppercase tracking-wider text-[var(--text-muted)] font-semibold flex items-center gap-1.5 shrink-0">
            <Layers size={13} className="text-[var(--pigment-class-b)]" />
            Attention Routing & Representation Space
          </span>

          {/* Phase Badge */}
          {seqArtifact?.phaseName && (
            <span className="px-2 py-0.5 rounded font-mono text-[10px] bg-blue-500/10 text-blue-400 border border-blue-500/30 font-semibold shrink-0">
              {seqArtifact.phaseName}
            </span>
          )}

          {/* Multi-Head Selector Tabs (for transformer-encoder) */}
          {isEncoder && heads.length > 0 && (
            <div className="flex flex-wrap items-center gap-1 bg-[var(--bg-subtle)] p-0.5 rounded border border-[var(--border-hairline)]">
              <button
                onClick={() => handleHeadSelect(-1)}
                className={`px-2 py-0.5 rounded font-mono text-[10px] transition-all shrink-0 ${
                  selectedHeadTab === -1
                    ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] font-semibold shadow-xs'
                    : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                }`}
              >
                Combined
              </button>
              {heads.map((h) => (
                <button
                  key={h.id}
                  onClick={() => handleHeadSelect(h.id)}
                  style={{ color: selectedHeadTab === h.id ? h.color : undefined }}
                  className={`px-2 py-0.5 rounded font-mono text-[10px] transition-all flex items-center gap-1 shrink-0 ${
                    selectedHeadTab === h.id
                      ? 'bg-[var(--bg-elevated)] font-semibold shadow-xs border border-[var(--border-hairline)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                  }`}
                >
                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: h.color }} />
                  {h.name.split(':')[0]}
                </button>
              ))}
            </div>
          )}

          {/* Sub-Layer Selector Tabs (for transformer-enc-dec) */}
          {isEncDec && onUpdateHyperparam && (
            <div className="flex flex-wrap items-center gap-1 bg-[var(--bg-subtle)] p-0.5 rounded border border-[var(--border-hairline)]">
              {[
                { id: 'cross-attention', label: 'Cross-Attention' },
                { id: 'enc-self', label: 'Source Encoder Self' },
                { id: 'dec-causal-self', label: 'Target Decoder Self' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => onUpdateHyperparam('activeSubLayer', tab.id)}
                  className={`px-2 py-0.5 rounded font-mono text-[10px] transition-all shrink-0 ${
                    activeSubLayer === tab.id
                      ? 'bg-[var(--bg-elevated)] text-[var(--pigment-class-b)] font-semibold shadow-xs border border-[var(--border-hairline)]'
                      : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Matrix Display Mode Toggle */}
        <div className="flex items-center gap-2 shrink-0">
          <div className="flex items-center gap-1 bg-[var(--bg-subtle)] p-0.5 rounded border border-[var(--border-hairline)] font-mono text-[10px]">
            <button
              onClick={() => setMatrixDisplayMode('weights')}
              className={`px-2 py-0.5 rounded transition-all ${
                matrixDisplayMode === 'weights'
                  ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] font-semibold shadow-xs'
                  : 'text-[var(--text-muted)]'
              }`}
            >
              Softmax Weights (α)
            </button>
            <button
              onClick={() => setMatrixDisplayMode('scores')}
              className={`px-2 py-0.5 rounded transition-all ${
                matrixDisplayMode === 'scores'
                  ? 'bg-[var(--bg-elevated)] text-[var(--text-primary)] font-semibold shadow-xs'
                  : 'text-[var(--text-muted)]'
              }`}
            >
              Raw Scores (Q·K^T / √d_k)
            </button>
          </div>
        </div>
      </div>

      {/* Main Dual-Column Split: Left = Bipartite Attention Arcs, Right = T x T Heatmap Matrix */}
      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* LEFT COLUMN (7 Cols): Bipartite Attention Routing Arcs */}
        <div className="lg:col-span-7 flex flex-col rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-secondary)] p-3 overflow-hidden shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-1.5 pb-2 border-b border-[var(--border-hairline)] mb-2 font-mono text-[11px] min-w-0">
            <span className="text-[var(--text-muted)] uppercase tracking-wider font-semibold truncate">
              Bipartite Token Attention Beams
            </span>
            <span className="text-[var(--pigment-class-b)] font-semibold shrink-0">
              Focused Query: "{queryTokens[queryFocusIndex] ?? '—'}" (t = {queryFocusIndex + 1})
            </span>
          </div>

          {/* SVG Bipartite Routing Canvas */}
          <div className="flex-1 min-h-0 relative flex flex-col justify-between py-2">
            {/* Top Row: Keys / Source Tokens */}
            <div className="flex items-center justify-between px-1 z-10 gap-1">
              {keyTokens.map((tok, j) => (
                <div
                  key={j}
                  className="flex-1 min-w-0 flex flex-col items-center gap-0.5 px-1 py-1 rounded bg-[var(--bg-elevated)] border border-[var(--border-hairline)] text-center shadow-xs"
                >
                  <span className="font-mono text-[8px] text-[var(--text-muted)] truncate max-w-full">
                    {isCrossAttention ? `Src K_${j + 1}` : `K_${j + 1}`}
                  </span>
                  <span className="font-semibold text-[11px] text-[var(--text-primary)] truncate max-w-full" title={tok}>
                    {tok}
                  </span>
                </div>
              ))}
            </div>

            {/* SVG Connecting Bezier Beams */}
            <svg viewBox="0 0 1000 120" preserveAspectRatio="none" className="w-full h-36 min-h-[120px] my-1 pointer-events-none">
              {queryTokens.map((_, i) => {
                if (queryFocusIndex !== null && queryFocusIndex !== i) return null;

                const qX = ((i + 0.5) / numQueries) * 1000;
                return keyTokens.map((_, j) => {
                  const kX = ((j + 0.5) / numKeys) * 1000;
                  const weight = isCrossAttention
                    ? (attention?.crossAttention?.weights[i * numKeys + j] ?? 0)
                    : (attentionWeights[i * numKeys + j] ?? 0);
                  if (weight < 0.01) return null;

                  const strokeW = Math.max(1.2, weight * 8);
                  const opacity = Math.max(0.2, Math.min(0.95, weight * 1.25));
                  const strokeColor =
                    selectedHeadTab >= 0 && heads[selectedHeadTab]
                      ? heads[selectedHeadTab].color
                      : 'var(--pigment-class-b, #1E40AF)';

                  return (
                    <path
                      key={`${i}-${j}`}
                      d={`M ${qX} 115 C ${qX} 60, ${kX} 60, ${kX} 5`}
                      fill="none"
                      stroke={strokeColor}
                      strokeWidth={strokeW}
                      strokeOpacity={opacity}
                      strokeLinecap="round"
                    />
                  );
                });
              })}
            </svg>

            {/* Bottom Row: Queries / Target Tokens */}
            <div className="flex items-center justify-between px-1 z-10 gap-1">
              {queryTokens.map((tok, i) => {
                const isSelected = i === queryFocusIndex;
                return (
                  <div
                    key={i}
                    onClick={() => handleTokenClick(i)}
                    onMouseEnter={() => setHoveredQueryIndex(i)}
                    onMouseLeave={() => setHoveredQueryIndex(null)}
                    className={`flex-1 min-w-0 flex flex-col items-center gap-0.5 px-1 py-1 rounded cursor-pointer transition-all border shadow-xs ${
                      isSelected
                        ? 'border-[var(--pigment-class-b)] bg-[var(--bg-elevated)] ring-1 ring-[var(--pigment-class-b)]'
                        : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] hover:border-[var(--border-strong)]'
                    }`}
                  >
                    <span className="font-mono text-[8px] text-[var(--text-muted)] truncate max-w-full">
                      {isCrossAttention ? `Tgt Q_${i + 1}` : `Q_${i + 1}`}
                    </span>
                    <span className="font-semibold text-[11px] text-[var(--text-primary)] truncate max-w-full" title={tok}>
                      {tok}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Bottom Diagnostic: Positional Encoding or Next-Token Prediction */}
          {isEncoder && attention?.positionalEncoding?.enabled && (
            <div className="mt-2 pt-2 border-t border-[var(--border-hairline)] flex flex-col gap-1 font-mono text-[10px]">
              <div className="flex items-center justify-between text-[var(--text-muted)]">
                <span>Sinusoidal Positional Encoding (PE Wave Added to 8D Embeddings)</span>
                <span className="text-emerald-400 font-semibold">PE Active</span>
              </div>
              <div className="grid grid-cols-8 gap-1 h-3 bg-[var(--bg-primary)] p-0.5 rounded border border-[var(--border-hairline)]">
                {Array.from(attention.positionalEncoding.matrix[queryFocusIndex] ?? []).map((val, d) => (
                  <div
                    key={d}
                    style={{ opacity: Math.max(0.2, Math.abs(val)) }}
                    className="h-full bg-emerald-400 rounded-xs"
                    title={`PE[${queryFocusIndex}][${d}] = ${val.toFixed(3)}`}
                  />
                ))}
              </div>
            </div>
          )}

          {(isDecoder || isEncoder) && nextTokens.length > 0 && (
            <div className="mt-2 pt-2 border-t border-[var(--border-hairline)] flex flex-col gap-1.5 font-mono text-[10px]">
              <span className="text-[var(--text-muted)] uppercase tracking-wider font-semibold">
                {isEncoder
                  ? 'Masked Language Model ([MASK]) Predictions:'
                  : `Next-Token Softmax Logit Probabilities (from Query t=${queryFocusIndex + 1}):`}
              </span>
              <div className="grid grid-cols-3 gap-2">
                {nextTokens.slice(0, 3).map((tok) => (
                  <div
                    key={tok.token}
                    className="flex flex-col p-1.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] min-w-0"
                  >
                    <div className="flex items-center justify-between gap-1 min-w-0">
                      <span className="font-semibold text-[11px] text-[var(--text-primary)] truncate" title={tok.token}>
                        "{tok.token}"
                      </span>
                      <span className="font-bold text-[var(--pigment-class-b)] shrink-0">{(tok.prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full h-1 bg-[var(--bg-primary)] rounded overflow-hidden mt-1">
                      <div style={{ width: `${tok.prob * 100}%` }} className="h-full bg-[var(--pigment-class-b)]" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* RIGHT COLUMN (5 Cols): Interactive T x T / Rectangular Attention Heatmap Matrix */}
        <div className="lg:col-span-5 flex flex-col rounded-lg border border-[var(--border-hairline)] bg-[var(--bg-secondary)] p-3 overflow-hidden shadow-sm">
          <div className="flex items-center justify-between pb-2 border-b border-[var(--border-hairline)] mb-2 font-mono text-[11px]">
            <span className="text-[var(--text-muted)] uppercase tracking-wider font-semibold">
              {numQueries}×{numKeys} Routing Matrix {isCrossAttention ? '(Target × Source)' : ''}
            </span>
            <span className="text-[10px] text-[var(--text-muted)]">
              {matrixDisplayMode === 'weights' || isCrossAttention ? 'Row Sum = 1.0000' : 'Raw Dot Product'}
            </span>
          </div>

          {/* Matrix Grid */}
          <div className="flex-1 min-h-0 flex flex-col justify-center overflow-auto p-1">
            <div className="flex flex-col gap-1 w-full max-w-[340px] mx-auto">
              {/* Header Columns (Keys) */}
              <div className="flex items-center gap-1 font-mono text-[8px] text-[var(--text-muted)] text-center pl-8">
                {keyTokens.map((tok, j) => (
                  <span key={j} className="flex-1 truncate" title={`Key ${j + 1}: ${tok}`}>
                    {tok.slice(0, 3)}
                  </span>
                ))}
              </div>

              {/* Matrix Rows (Queries) */}
              {queryTokens.map((qTok, i) => {
                const isSelectedRow = i === queryFocusIndex;

                return (
                  <div key={i} className="flex items-center gap-1">
                    <span
                      className={`font-mono text-[8px] w-7 truncate text-right ${
                        isSelectedRow ? 'font-bold text-[var(--pigment-class-b)]' : 'text-[var(--text-muted)]'
                      }`}
                      title={`Query ${i + 1}: ${qTok}`}
                    >
                      {qTok.slice(0, 3)}
                    </span>

                    <div className="flex-1 flex items-center gap-1">
                      {keyTokens.map((kTok, j) => {
                        const isMaskedCausal = isCausal && !isCrossAttention && j > i;
                        const idx = i * numKeys + j;
                        const val = isCrossAttention
                          ? (attention?.crossAttention?.weights[idx] ?? 0)
                          : (displayMatrix[idx] ?? 0);
                        const weight = isCrossAttention
                          ? (attention?.crossAttention?.weights[idx] ?? 0)
                          : (attentionWeights[idx] ?? 0);

                        return (
                          <div
                            key={j}
                            onClick={() => handleTokenClick(i)}
                            className={`flex-1 aspect-square rounded flex items-center justify-center font-mono text-[8px] transition-all relative border cursor-pointer ${
                              isMaskedCausal
                                ? 'bg-red-950/20 border-red-900/40 text-red-400 [background-image:repeating-linear-gradient(45deg,transparent,transparent_2px,rgba(239,68,68,0.15)_2px,rgba(239,68,68,0.15)_4px)]'
                                : isSelectedRow
                                ? 'border-[var(--pigment-class-b)]/70 text-[var(--text-primary)]'
                                : 'border-[var(--border-hairline)] text-[var(--text-secondary)]'
                            }`}
                            style={{
                              backgroundColor: !isMaskedCausal
                                ? `rgba(59, 130, 246, ${Math.max(0.04, weight * 0.85)})`
                                : undefined,
                            }}
                            title={
                              isMaskedCausal
                                ? `-∞ (CAUSAL MASKED: α = 0.00)`
                                : `Q[${i}] ("${qTok}") -> K[${j}] ("${kTok}"): ${val.toFixed(3)}`
                            }
                          >
                            {isMaskedCausal ? (
                              <span className="text-[7px] opacity-70">-∞</span>
                            ) : matrixDisplayMode === 'weights' || isCrossAttention ? (
                              weight.toFixed(2)
                            ) : (
                              val.toFixed(1)
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Matrix Footer Legend */}
          <div className="mt-2 pt-2 border-t border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-2 font-mono text-[9px] text-[var(--text-muted)]">
            <div className="flex items-center gap-1.5 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-blue-500/60 border border-blue-400" />
              <span>High Attention (α ≈ 1.0)</span>
            </div>
            {isCausal && (
              <div className="flex items-center gap-1 text-red-400 shrink-0">
                <span className="w-2.5 h-2.5 rounded bg-red-900/30 border border-red-500/50 [background-image:repeating-linear-gradient(45deg,transparent,transparent_2px,rgba(239,68,68,0.3)_2px,rgba(239,68,68,0.3)_4px)]" />
                <span>Causal Mask (-∞)</span>
              </div>
            )}
            {isCrossAttention && (
              <div className="flex items-center gap-1 text-purple-400 shrink-0">
                <span className="w-2.5 h-2.5 rounded bg-purple-500/50 border border-purple-400" />
                <span>Rectangular Cross Matrix</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Swiss Bottom Status & Legend Bar (Zone D) */}
      <footer className="mt-3 pt-2.5 border-t border-[var(--border-hairline)] flex flex-wrap items-center justify-between gap-3 text-[10px] shrink-0 font-mono">
        <div className="flex flex-wrap items-center gap-4 text-[var(--text-secondary)]">
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2.5 h-2.5 rounded bg-blue-500/60 border border-blue-400" />
            <span>High Attention (α ≈ 1.0)</span>
          </div>
          {isCausal && (
            <div className="flex items-center gap-1 text-red-400 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-red-900/30 border border-red-500/50 [background-image:repeating-linear-gradient(45deg,transparent,transparent_2px,rgba(239,68,68,0.3)_2px,rgba(239,68,68,0.3)_4px)]" />
              <span>Causal Autoregressive Mask (-∞)</span>
            </div>
          )}
          {isCrossAttention && (
            <div className="flex items-center gap-1 text-purple-400 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-purple-500/50 border border-purple-400" />
              <span>Seq2Seq Cross-Layer: Target Q ({numQueries}) × Source K ({numKeys})</span>
            </div>
          )}
          {isEncoder && attention?.positionalEncoding?.enabled && (
            <div className="flex items-center gap-1 text-emerald-400 shrink-0">
              <span className="w-2.5 h-2.5 rounded bg-emerald-500/50 border border-emerald-400" />
              <span>Sinusoidal Positional Encoding Active</span>
            </div>
          )}
        </div>
        <div className="flex items-center gap-3 text-[var(--text-muted)] font-mono text-[9px]">
          <span>{algorithm.name}</span>
          <span className="text-[var(--border-hairline)]">|</span>
          <span>d_k = {attention?.d_k ?? hyperparams.dk ?? 8}</span>
          <span className="text-[var(--border-hairline)]">|</span>
          <span className="font-semibold text-[var(--text-primary)]">
            {isCrossAttention ? 'Dual-Tower Cross-Attention' : isDecoder ? 'Autoregressive Causal' : 'Self-Attention Space'}
          </span>
        </div>
      </footer>
    </div>
  );
};
