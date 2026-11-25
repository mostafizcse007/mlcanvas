import React, { useState } from 'react';
import { X, Play, Sliders, MousePointer } from 'lucide-react';

export const CoachmarkStrip: React.FC = () => {
  const [isVisible, setIsVisible] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return !localStorage.getItem('mlcanvas-coachmarks-dismissed');
  });

  const handleDismiss = () => {
    setIsVisible(false);
    localStorage.setItem('mlcanvas-coachmarks-dismissed', 'true');
  };

  if (!isVisible) return null;

  return (
    <div className="coachmark-strip hidden sm:flex flex-wrap items-center justify-between px-4 py-1.5 bg-[var(--bg-subtle)] border-b border-[var(--border-hairline)] text-[11px] font-sans text-[var(--text-secondary)] gap-2">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 min-w-0">
        <span className="font-mono text-[10px] uppercase font-semibold text-[var(--text-primary)] tracking-wider shrink-0">
          Calibration Guide
        </span>

        <div className="flex items-center gap-1.5 shrink-0">
          <span className="font-mono text-[10px] border border-[var(--border-hairline)] px-1 bg-[var(--bg-elevated)]">1</span>
          <MousePointer size={11} className="text-[var(--text-primary)]" />
          <span>Select a manifold or click to add / drag points on the canvas.</span>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <span className="font-mono text-[10px] border border-[var(--border-hairline)] px-1 bg-[var(--bg-elevated)]">2</span>
          <Play size={11} className="text-[var(--text-primary)]" />
          <span>Press <kbd className="font-mono text-[10px] bg-[var(--bg-elevated)] px-1 border border-[var(--border-hairline)]">Space</kbd> to start the off-thread solver loop.</span>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <span className="font-mono text-[10px] border border-[var(--border-hairline)] px-1 bg-[var(--bg-elevated)]">3</span>
          <Sliders size={11} className="text-[var(--text-primary)]" />
          <span>Tune hyperparameters or switch models with <kbd className="font-mono text-[10px] bg-[var(--bg-elevated)] px-1 border border-[var(--border-hairline)]">Cmd+K</kbd>.</span>
        </div>
      </div>

      <button
        onClick={handleDismiss}
        className="flex items-center gap-1 text-[10px] font-mono hover:text-[var(--text-primary)] cursor-pointer shrink-0 ml-auto"
        title="Dismiss calibration guide"
        aria-label="Dismiss calibration guide"
      >
        <span>Dismiss</span>
        <X size={12} />
      </button>
    </div>
  );
};

