import React, { useState } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  StepForward,
  Sun,
  Moon,
  Command,
  Share2,
  Camera,
  Activity,
  Check,
} from 'lucide-react';
import { ThemeMode } from '../types/theme';
import { AlgorithmPlugin } from '../types/algorithm';
import {
  HyperparameterLabel,
  getAccessibleHyperparameterLabel,
} from './HyperparameterLabel';

interface TopControlBarProps {
  isRunning: boolean;
  epoch: number;
  rlEpisodeCount?: number;
  isScrubbing?: boolean;
  isConverged?: boolean;
  currentAlgorithm: AlgorithmPlugin;
  datasetTaskType?: 'classification' | 'regression';
  hyperparams: Record<string, number | string>;
  theme: ThemeMode;
  reducedMotion: boolean;
  onTogglePlay: () => void;
  onReset: () => void;
  onStep: () => void;
  onOpenAlgorithmModal: () => void;
  onUpdateHyperparam: (key: string, value: number | string) => void;
  onToggleTheme: () => void;
  onToggleReducedMotion: () => void;
  onCopyShareLink: () => void;
  onExportPng: () => void;
}

export const TopControlBar: React.FC<TopControlBarProps> = ({
  isRunning,
  epoch,
  rlEpisodeCount,
  isScrubbing = false,
  isConverged = false,
  currentAlgorithm,
  datasetTaskType,
  hyperparams,
  theme,
  reducedMotion,
  onTogglePlay,
  onReset,
  onStep,
  onOpenAlgorithmModal,
  onUpdateHyperparam,
  onToggleTheme,
  onToggleReducedMotion,
  onCopyShareLink,
  onExportPng,
}) => {
  const [copiedLink, setCopiedLink] = useState(false);

  const handleCopy = () => {
    onCopyShareLink();
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 1800);
  };

  const primaryParams = (currentAlgorithm.primaryParams ?? [])
    .filter((hp) => hp.id !== 'batchPasses' && hp.id !== 'cyclesPerStep' && hp.id !== 'subEpochs')
    .slice(0, 3);

  const isClosedForm = currentAlgorithm.executionMode === 'closed-form';

  return (
    <header className="top-transport-bar" role="banner">
      {/* Left: Brand Mark & Transport Cluster */}
      <div className="flex items-center gap-2 shrink-0">
        <div className="top-brand">
          <span className="brand-title">ML Canvas</span>
        </div>

        <nav aria-label="Solver controls" className="transport-cluster ml-1 pl-2 border-l border-[var(--border-hairline)] flex items-center gap-1.5">
          <button
            className="transport-btn shrink-0"
            onClick={onReset}
            title="Reset Solver State [R]"
          >
            <RotateCcw size={12} />
            <span>Reset</span>
            <kbd>R</kbd>
          </button>

          <button
            className={`transport-btn shrink-0 ${isClosedForm ? 'opacity-45 cursor-not-allowed text-[var(--text-muted)]' : isRunning ? 'btn-primary' : ''}`}
            onClick={isClosedForm ? undefined : onTogglePlay}
            disabled={isClosedForm}
            title={isClosedForm ? 'Closed-form analytical solution computed in 1 pass; no gradient descent needed' : 'Start / Pause Solver [Space]'}
          >
            {isRunning ? <Pause size={12} /> : <Play size={12} />}
            <span>{isClosedForm ? 'Fitted' : isRunning ? 'Pause' : 'Play'}</span>
            {!isClosedForm && <kbd>Space</kbd>}
          </button>

          <button
            className={`transport-btn shrink-0 ${isClosedForm ? 'opacity-45 cursor-not-allowed text-[var(--text-muted)]' : ''}`}
            onClick={isClosedForm ? undefined : onStep}
            disabled={isClosedForm || isRunning}
            title={isClosedForm ? 'Closed-form analytical solution computed in 1 pass' : 'Step +1 Iteration [->]'}
          >
            <StepForward size={12} />
            <span>Step</span>
            {!isClosedForm && <kbd>→</kbd>}
          </button>

          <div className="epoch-readout ml-2 pl-2.5 border-l border-[var(--border-hairline)] flex flex-col justify-center min-w-[75px] shrink-0">
            <div className="flex items-center gap-1.5 leading-none">
              <span className="epoch-label text-[9px] uppercase tracking-wider text-[var(--text-muted)] font-medium">
                {isClosedForm ? 'Pass' : currentAlgorithm.stage === 'rl-gridworld' ? 'Episode' : 'Epoch'}
              </span>
              {isScrubbing ? (
                <span className="font-mono text-[8px] uppercase tracking-wide px-1 py-0.5 rounded border border-[var(--accent-amber)]/40 bg-[var(--accent-amber)]/10 text-[var(--accent-amber)] font-semibold">
                  Scrubbed
                </span>
              ) : isClosedForm ? (
                <span
                  className="font-mono text-[8px] uppercase tracking-wide px-1 py-0.5 rounded border border-emerald-500/30 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-semibold"
                  title="Exact analytical Maximum Likelihood Estimation; solved in 1 statistical pass"
                >
                  1-Pass Closed-Form
                </span>
              ) : isConverged ? (
                <span className="font-mono text-[8px] uppercase tracking-wide px-1 py-0.5 rounded border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[var(--pigment-class-a)]">
                  Converged
                </span>
              ) : null}
            </div>
            <span className="epoch-value font-mono text-[14px] font-semibold text-[var(--text-primary)] tracking-tight leading-tight mt-0.5">
              {isClosedForm
                ? '000,001 (Fitted)'
                : String(
                    currentAlgorithm.stage === 'rl-gridworld' && rlEpisodeCount !== undefined
                      ? rlEpisodeCount
                      : epoch
                  )
                    .padStart(6, '0')
                    .replace(/\B(?=(\d{3})+(?!\d))/g, ',')}
            </span>
          </div>
        </nav>
      </div>

      {/* Center: Algorithm Selector & Primary Hyperparameters */}
      <div className="flex items-center gap-2 min-w-0 flex-1 overflow-x-auto top-params-scroll">
        {/* Unified Algorithm Selector Button */}
        <button
          className="transport-btn flex items-center gap-1.5 border-[var(--border-strong)] shrink-0 cursor-pointer"
          onClick={onOpenAlgorithmModal}
          title="Switch Algorithm (Cmd+K / Ctrl+K)"
        >
          <span className="font-semibold text-[12px]">{currentAlgorithm.shortCode}</span>
          <span aria-hidden="true" className="text-[10px] text-[var(--text-secondary)]">▾</span>
          <kbd aria-hidden="true" className="flex items-center gap-0.5 ml-1 text-[9px] bg-[var(--bg-elevated)] px-1 py-0.5 border border-[var(--border-hairline)] font-mono">
            <Command size={9} />K
          </kbd>
        </button>

        {/* Inline Primary Hyperparameter Selects / Sliders */}
        <div className="top-params-cluster shrink-0">
          {primaryParams.map((hp) => {
            const isRegMode = datasetTaskType === 'regression' || hyperparams.task === 'regression';
            if (currentAlgorithm.id === 'voting-classifier' && isRegMode && hp.id === 'votingRule') {
              return null;
            }
            let effectiveOptions = hp.options;
            if (currentAlgorithm.id === 'decision-tree' && hp.id === 'criterion' && hp.options) {
              effectiveOptions = isRegMode
                ? hp.options.filter((opt) => opt.value === 'mse')
                : hp.options.filter((opt) => opt.value !== 'mse');
            } else if (currentAlgorithm.id === 'stacking' && hp.id === 'metaLearner' && isRegMode) {
              effectiveOptions = [{ label: 'Ridge Regression', value: 'ridge' }];
            }
            const accessibleLabel = getAccessibleHyperparameterLabel(hp.name, hp.symbol);
            const controlId = `top-param-${hp.id}`;
            return (
              <div key={hp.id} className="param-group shrink-0" title={hp.description}>
                <HyperparameterLabel
                  name={hp.name}
                  symbol={hp.symbol}
                  htmlFor={controlId}
                  className="param-label whitespace-nowrap shrink-0"
                />
                {hp.type === 'select' && effectiveOptions ? (
                  <select
                    id={controlId}
                    className="param-select"
                    aria-label={accessibleLabel}
                    value={
                      currentAlgorithm.id === 'decision-tree' && hp.id === 'criterion' && isRegMode
                        ? 'mse'
                        : currentAlgorithm.id === 'stacking' && hp.id === 'metaLearner' && isRegMode
                        ? 'ridge'
                        : (hyperparams[hp.id] ?? hp.default)
                    }
                    onChange={(e) =>
                      onUpdateHyperparam(hp.id, Number(e.target.value) || e.target.value)
                    }
                  >
                    {effectiveOptions.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                ) : hp.type === 'slider' ? (
                  <div className="flex items-center gap-1.5 shrink-0">
                    <input
                      id={controlId}
                      type="range"
                      className="w-16 h-1 accent-[var(--text-primary)] cursor-pointer"
                      aria-label={accessibleLabel}
                      min={hp.min ?? 1}
                      max={hp.max ?? 10}
                      step={hp.step ?? 1}
                      value={Number(hyperparams[hp.id] ?? hp.default)}
                      onChange={(e) => onUpdateHyperparam(hp.id, parseFloat(e.target.value))}
                    />
                    <span className="font-mono text-[10px] text-[var(--text-secondary)] min-w-[20px]">
                      {hp.id === 'threshold'
                        ? Number(hyperparams[hp.id] ?? hp.default).toFixed(2)
                        : (hyperparams[hp.id] ?? hp.default)}
                    </span>
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>

      {/* Right: Share, Export, Motion, Theme */}
      <div className="flex items-center gap-1 relative shrink-0">
        {/* Copy Share Link */}
        <button
          className="transport-btn px-2 text-[11px] shrink-0"
          onClick={handleCopy}
          title="Serialize and copy reproducible URL experiment state"
          aria-label={copiedLink ? 'Copied link' : 'Link share URL'}
        >
          {copiedLink ? <Check size={12} className="text-emerald-600" /> : <Share2 size={12} />}
          <span>{copiedLink ? 'Copied' : 'Link'}</span>
        </button>

        {/* Export PNG */}
        <button
          className="transport-btn px-2 text-[11px] shrink-0"
          onClick={onExportPng}
          title="Export high-resolution canvas screenshot"
          aria-label="Export canvas as high-resolution PNG"
        >
          <Camera size={12} />
          <span>Export</span>
        </button>

        {/* Reduced Motion Toggle */}
        <button
          className={`transport-btn px-1.5 text-[10px] font-mono shrink-0 ${reducedMotion ? 'is-active' : ''}`}
          onClick={onToggleReducedMotion}
          title={`Reduced Motion: ${reducedMotion ? 'ON' : 'OFF'}`}
          aria-label={`Toggle reduced animation motion (currently ${reducedMotion ? 'On' : 'Off'})`}
        >
          <Activity size={12} />
        </button>

        {/* Theme Switcher (Paper / Carbon) */}
        <button
          className="transport-btn px-2 shrink-0"
          onClick={onToggleTheme}
          title={`Switch to ${theme === 'paper' ? 'Softened Carbon Dark' : 'Archival Paper Light'} theme`}
          aria-label={`Switch to ${theme === 'paper' ? 'Softened Carbon Dark' : 'Archival Paper Light'} theme`}
        >
          {theme === 'paper' ? <Moon size={12} /> : <Sun size={12} />}
        </button>
      </div>
    </header>
  );
};
