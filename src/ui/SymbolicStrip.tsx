import React, { useId, useState } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import { SymbolicMathLink } from '../types/algorithm';
import { LatexSpan } from '../utils/katex-render';

export interface SymbolicStripProps {
  links: SymbolicMathLink[];
  activeHighlight: string | null;
  onHighlightChange: (target: string | null) => void;
  defaultExpanded?: boolean;
}

/**
 * Collapsible header strip rendering hoverable KaTeX math symbols linked to
 * visual canvas artifacts using semantic Light Paper and Dark Carbon tokens.
 */
export const SymbolicStrip: React.FC<SymbolicStripProps> = ({
  links,
  activeHighlight,
  onHighlightChange,
  defaultExpanded = true,
}) => {
  const [isExpanded, setIsExpanded] = useState<boolean>(defaultExpanded);
  const contentId = useId();

  if (!links || links.length === 0) {
    return null;
  }

  return (
    <div className="symbolic-strip flex items-center gap-2 bg-[var(--bg-subtle)] px-2 py-1 border border-[var(--border-hairline)] font-mono text-[11px] text-[var(--text-primary)] shrink-0 max-w-full">
      <button
        type="button"
        aria-expanded={isExpanded}
        aria-controls={contentId}
        onClick={() => setIsExpanded((prev) => !prev)}
        className="flex items-center gap-1 px-2 py-1 min-h-[24px] border border-[var(--border-hairline)] bg-[var(--bg-elevated)] hover:bg-[var(--bg-active)] text-[var(--text-primary)] text-[9px] uppercase tracking-wider font-medium shrink-0 cursor-pointer transition-colors"
        title={isExpanded ? 'Collapse Symbolic Strip' : 'Expand Symbolic Strip'}
      >
        {isExpanded ? (
          <ChevronDown size={11} className="shrink-0 text-[var(--text-primary)]" aria-hidden="true" />
        ) : (
          <ChevronRight size={11} className="shrink-0 text-[var(--text-primary)]" aria-hidden="true" />
        )}
        <span>Symbolic Strip</span>
      </button>

      {isExpanded && (
        <div
          id={contentId}
          className="flex items-center gap-2 overflow-x-auto no-scrollbar min-w-0"
        >
          {links.map((link) => {
            const isHighlighted = activeHighlight === link.highlightTarget;
            return (
              <div
                key={link.id}
                className={`cursor-pointer px-1.5 py-0.5 border transition-colors shrink-0 ${
                  isHighlighted
                    ? 'border-[var(--accent-amber)] bg-[var(--bg-active)] text-[var(--text-primary)] font-bold'
                    : 'border-transparent text-[var(--text-primary)] hover:border-[var(--border-hairline)] hover:bg-[var(--bg-elevated)]'
                }`}
                onMouseEnter={() => onHighlightChange(link.highlightTarget)}
                onMouseLeave={() => onHighlightChange(null)}
                title={`${link.label}: ${link.description}`}
              >
                <LatexSpan expression={link.symbolLatex} />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
