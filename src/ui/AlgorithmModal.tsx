import React, { useState, useEffect, useMemo, useRef } from 'react';
import { ALGORITHM_CATALOG, STAGES } from '../algorithms/registry';
import { Search, X, Check, ArrowRight } from 'lucide-react';

interface AlgorithmModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedAlgorithmId: string;
  onSelectAlgorithm: (algorithmId: string) => void;
}

export const AlgorithmModal: React.FC<AlgorithmModalProps> = ({
  isOpen,
  onClose,
  selectedAlgorithmId,
  onSelectAlgorithm,
}) => {
  const [search, setSearch] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const searchInputRef = useRef<HTMLInputElement>(null);

  // Identify the stage of the currently selected algorithm
  const currentAlgoEntry = useMemo(() => {
    return ALGORITHM_CATALOG.find((a) => a.id === selectedAlgorithmId);
  }, [selectedAlgorithmId]);
  const currentStageId = currentAlgoEntry?.stageId ?? STAGES[0].id;

  // Active hovered/selected category in Split-Pane
  const [activeStageId, setActiveStageId] = useState<string>(currentStageId);

  useEffect(() => {
    if (isOpen) {
      setSearch('');
      setSelectedIndex(0);
      setActiveStageId(currentStageId);
      setTimeout(() => searchInputRef.current?.focus(), 50);
    }
  }, [isOpen, currentStageId]);

  // Global search filtering across all 51 models
  const filtered = useMemo(() => {
    const q = search.toLowerCase().trim();
    if (!q) return ALGORITHM_CATALOG;
    return ALGORITHM_CATALOG.filter(
      (algo) =>
        algo.name.toLowerCase().includes(q) ||
        algo.shortCode.toLowerCase().includes(q) ||
        algo.description.toLowerCase().includes(q)
    );
  }, [search]);

  // Group by Stage for stage counters
  const stageModelCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const stage of STAGES) {
      counts.set(stage.id, 0);
    }
    for (const item of ALGORITHM_CATALOG) {
      counts.set(item.stageId, (counts.get(item.stageId) ?? 0) + 1);
    }
    return counts;
  }, []);

  const isSearching = search.trim().length > 0;

  // Algorithms in active category for Split-Pane browsing
  const activeCategoryAlgorithms = useMemo(() => {
    return ALGORITHM_CATALOG.filter((a) => a.stageId === activeStageId);
  }, [activeStageId]);

  const visibleAlgorithms = isSearching ? filtered : activeCategoryAlgorithms;

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'Escape') {
        if (isSearching) {
          setSearch('');
          setSelectedIndex(0);
        } else {
          onClose();
        }
      } else if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.min(prev + 1, visibleAlgorithms.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((prev) => Math.max(prev - 1, 0));
      } else if (!isSearching && e.key === 'ArrowLeft') {
        e.preventDefault();
        const stageIdx = STAGES.findIndex((s) => s.id === activeStageId);
        if (stageIdx > 0) {
          setActiveStageId(STAGES[stageIdx - 1].id);
          setSelectedIndex(0);
        }
      } else if (!isSearching && e.key === 'ArrowRight') {
        e.preventDefault();
        const stageIdx = STAGES.findIndex((s) => s.id === activeStageId);
        if (stageIdx < STAGES.length - 1) {
          setActiveStageId(STAGES[stageIdx + 1].id);
          setSelectedIndex(0);
        }
      } else if (e.key === 'Enter') {
        e.preventDefault();
        const active = visibleAlgorithms[selectedIndex];
        if (active && active.isImplemented) {
          onSelectAlgorithm(active.id);
          onClose();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, selectedIndex, visibleAlgorithms, isSearching, activeStageId, onSelectAlgorithm, onClose]);

  if (!isOpen) return null;

  const currentStageInfo = STAGES.find((s) => s.id === activeStageId);

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div className="modal-header">
          <div className="flex items-center gap-2">
            <span className="font-serif text-lg font-normal">Algorithm Switcher</span>
            <span className="font-mono text-[10px] text-[var(--text-muted)] border border-[var(--border-hairline)] px-1.5 py-0.5">
              51 Models · 7 Stages
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-[var(--text-muted)] hover:text-[var(--text-primary)]"
          >
            <X size={16} />
          </button>
        </div>

        {/* Search Input with Guaranteed Spacing */}
        <div className="relative flex items-center">
          <input
            ref={searchInputRef}
            type="text"
            className="modal-search"
            style={{ paddingLeft: '38px' }}
            placeholder="Search algorithms, penalties, methods (e.g. SVM, Lasso, CART, Attention)..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setSelectedIndex(0);
            }}
          />
          <Search size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[var(--text-muted)] pointer-events-none shrink-0" />
        </div>

        {/* Main Body: Split-Pane or Global Search Results */}
        {!isSearching ? (
          <div className="modal-split-layout">
            {/* Left Category Sidebar (Hoverable Stages) */}
            <div className="modal-category-sidebar">
              <div className="px-3 py-1.5 text-[9px] font-mono uppercase tracking-wider text-[var(--text-muted)] border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] shrink-0">
                Categories ({STAGES.length})
              </div>
              <div className="flex-1 overflow-y-auto py-1">
                {STAGES.map((stage) => {
                  const isActive = stage.id === activeStageId;
                  const isCurrentStage = stage.id === currentStageId;
                  const count = stageModelCounts.get(stage.id) ?? 0;

                  return (
                    <div
                      key={stage.id}
                      onMouseEnter={() => {
                        setActiveStageId(stage.id);
                        setSelectedIndex(0);
                      }}
                      onClick={() => {
                        setActiveStageId(stage.id);
                        setSelectedIndex(0);
                      }}
                      className={`modal-category-item ${isActive ? 'is-active' : ''}`}
                    >
                      <div className="flex items-center gap-1.5 min-w-0 pr-1">
                        {isCurrentStage ? (
                          <span className="w-1.5 h-1.5 rounded-full bg-[var(--signal-pos)] shrink-0" title="Contains active algorithm" />
                        ) : (
                          <span className="w-1.5 h-1.5 rounded-full bg-transparent shrink-0" />
                        )}
                        <span className="text-[11px] text-[var(--text-primary)] truncate">
                          {String(stage.index).padStart(2, '0')} · {stage.shortName}
                        </span>
                      </div>
                      <span className="font-mono text-[9px] px-1.5 py-0.2 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] text-[var(--text-muted)] shrink-0">
                        {count}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right Algorithm Panel */}
            <div className="modal-algorithm-panel flex flex-col">
              <div className="px-3.5 py-1.5 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] flex items-center justify-between font-mono text-[10px] shrink-0">
                <span className="font-semibold text-[var(--text-primary)] uppercase tracking-wider truncate">
                  {currentStageInfo?.name}
                </span>
                <span className="text-[var(--text-muted)] shrink-0">
                  {activeCategoryAlgorithms.length} models
                </span>
              </div>

              <div className="flex-1 overflow-y-auto p-1 space-y-0.5">
                {activeCategoryAlgorithms.map((algo, idx) => {
                  const isHighlighted = idx === selectedIndex;
                  const isCurrent = algo.id === selectedAlgorithmId;

                  return (
                    <div
                      key={algo.id}
                      className={`algo-item ${isHighlighted ? 'is-selected' : ''} ${
                        !algo.isImplemented ? 'opacity-60 cursor-not-allowed' : ''
                      }`}
                      onMouseEnter={() => setSelectedIndex(idx)}
                      onClick={() => {
                        if (algo.isImplemented) {
                          onSelectAlgorithm(algo.id);
                          onClose();
                        }
                      }}
                    >
                      <div className="flex-1 min-w-0 pr-3 flex flex-col gap-0.5">
                        <div className="flex flex-wrap items-center gap-1.5 min-w-0">
                          <span className="algo-item-title truncate">{algo.name}</span>
                          <span className="algo-item-badge shrink-0">{algo.shortCode}</span>
                        </div>
                        <p className="algo-item-desc text-[10.5px] leading-snug break-normal line-clamp-2">{algo.description}</p>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        {isCurrent ? (
                          <span className="flex items-center gap-1 font-mono text-[10px] text-[var(--signal-pos)] font-medium">
                            <Check size={13} /> Active
                          </span>
                        ) : algo.isImplemented ? (
                          <span className="text-[var(--text-muted)] hover:text-[var(--text-primary)]">
                            <ArrowRight size={13} />
                          </span>
                        ) : (
                          <span className="font-mono text-[9px] text-[var(--text-muted)]">
                            Stage {stageModelCounts.get(algo.stageId)}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        ) : (
          /* Global Search Mode (When user types in search bar) */
          <div className="modal-content flex flex-col" style={{ height: '420px', maxHeight: '55vh' }}>
            <div className="px-3.5 py-1.5 border-b border-[var(--border-hairline)] bg-[var(--bg-subtle)] flex items-center justify-between font-mono text-[10px] shrink-0">
              <span className="text-[var(--text-muted)] uppercase tracking-wider">
                Search Results for "{search}"
              </span>
              <span className="text-[var(--text-primary)] font-semibold">
                {filtered.length} found
              </span>
            </div>

            {filtered.length === 0 ? (
              <div className="flex-1 flex flex-col items-center justify-center p-8 text-center text-[var(--text-muted)] font-mono text-[11px]">
                <span>No algorithms matching "{search}"</span>
                <span className="text-[9px] mt-1 opacity-70">Press Esc or clear input to browse by category</span>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto p-1 space-y-0.5">
                {filtered.map((algo, idx) => {
                  const isHighlighted = idx === selectedIndex;
                  const isCurrent = algo.id === selectedAlgorithmId;
                  const stageInfo = STAGES.find((s) => s.id === algo.stageId);

                  return (
                    <div
                      key={algo.id}
                      className={`algo-item ${isHighlighted ? 'is-selected' : ''} ${
                        !algo.isImplemented ? 'opacity-60 cursor-not-allowed' : ''
                      }`}
                      onMouseEnter={() => setSelectedIndex(idx)}
                      onClick={() => {
                        if (algo.isImplemented) {
                          onSelectAlgorithm(algo.id);
                          onClose();
                        }
                      }}
                    >
                      <div className="flex-1 min-w-0 pr-3 flex flex-col gap-0.5">
                        <div className="flex flex-wrap items-center gap-1.5 min-w-0">
                          <span className="algo-item-title truncate">{algo.name}</span>
                          <span className="algo-item-badge shrink-0">{algo.shortCode}</span>
                          <span className="font-mono text-[9px] px-1 py-0.2 rounded bg-[var(--bg-subtle)] text-[var(--text-muted)] border border-[var(--border-hairline)] shrink-0">
                            {stageInfo?.shortName ?? algo.stageId}
                          </span>
                        </div>
                        <p className="algo-item-desc text-[10.5px] leading-snug break-normal line-clamp-2">{algo.description}</p>
                      </div>

                      <div className="flex items-center gap-2 shrink-0">
                        {isCurrent ? (
                          <span className="flex items-center gap-1 font-mono text-[10px] text-[var(--signal-pos)] font-medium">
                            <Check size={13} /> Active
                          </span>
                        ) : algo.isImplemented ? (
                          <span className="text-[var(--text-muted)] hover:text-[var(--text-primary)]">
                            <ArrowRight size={13} />
                          </span>
                        ) : (
                          <span className="font-mono text-[9px] text-[var(--text-muted)]">
                            Stage {stageInfo?.index}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {/* Footer */}
        <div className="p-2 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] flex items-center justify-between font-mono text-[10px] text-[var(--text-secondary)]">
          <div>
            {!isSearching ? (
              <>
                Use <kbd className="border border-[var(--border-hairline)] px-1 py-0.5">↑</kbd>{' '}
                <kbd className="border border-[var(--border-hairline)] px-1 py-0.5">↓</kbd> to navigate,{' '}
                <kbd className="border border-[var(--border-hairline)] px-1 py-0.5">←</kbd>{' '}
                <kbd className="border border-[var(--border-hairline)] px-1 py-0.5">→</kbd> for stages,{' '}
                <kbd className="border border-[var(--border-hairline)] px-1 py-0.5">Enter</kbd> to select
              </>
            ) : (
              <>
                Use <kbd className="border border-[var(--border-hairline)] px-1 py-0.5">↑</kbd>{' '}
                <kbd className="border border-[var(--border-hairline)] px-1 py-0.5">↓</kbd> to navigate,{' '}
                <kbd className="border border-[var(--border-hairline)] px-1 py-0.5">Esc</kbd> to clear
              </>
            )}
          </div>
          <div>Active: {currentAlgoEntry?.name}</div>
        </div>
      </div>
    </div>
  );
};
