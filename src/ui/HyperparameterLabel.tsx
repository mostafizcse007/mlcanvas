import React from 'react';

export interface LabelSegment {
  text: string;
  isSymbol: boolean;
}

const SYMBOL_TOKEN_RE =
  /(\([^()\s]+\)|[\u0370-\u03FF]+(?:_[A-Za-z0-9]+)?[²³₀-₉]?|(?<=\s)[a-z](?=\)))/g;

/**
 * Splits a hyperparameter display name (and optional explicit symbol) into prose
 * segments (subject to uppercase CSS transforms) and mathematical symbol segments
 * rendered with `normal-case` so lowercase Greek/variable symbols are preserved.
 */
export function splitHyperparameterLabel(name: string, symbol?: string): LabelSegment[] {
  const trimmedSymbol = symbol?.trim();
  if (trimmedSymbol) {
    const formattedSymbol = trimmedSymbol.startsWith('(')
      ? trimmedSymbol
      : `(${trimmedSymbol})`;
    const baseName = name.endsWith(` ${formattedSymbol}`)
      ? name.slice(0, -formattedSymbol.length).trimEnd()
      : name;
    return [
      { text: `${baseName} `, isSymbol: false },
      { text: formattedSymbol, isSymbol: true },
    ];
  }

  const segments: LabelSegment[] = [];
  let lastIndex = 0;

  for (const match of name.matchAll(SYMBOL_TOKEN_RE)) {
    const idx = match.index ?? 0;
    if (idx > lastIndex) {
      segments.push({ text: name.slice(lastIndex, idx), isSymbol: false });
    }
    segments.push({ text: match[0], isSymbol: true });
    lastIndex = idx + match[0].length;
  }

  if (lastIndex < name.length) {
    segments.push({ text: name.slice(lastIndex), isSymbol: false });
  }

  return segments.length > 0 ? segments : [{ text: name, isSymbol: false }];
}

/**
 * Builds an accessible label string that includes `symbol` without duplicating
 * it if already embedded inside `name`.
 */
export function getAccessibleHyperparameterLabel(name: string, symbol?: string): string {
  const trimmedSymbol = symbol?.trim();
  if (!trimmedSymbol) return name;
  const formattedSymbol = trimmedSymbol.startsWith('(')
    ? trimmedSymbol
    : `(${trimmedSymbol})`;
  return name.endsWith(` ${formattedSymbol}`) ? name : `${name} ${formattedSymbol}`;
}

export interface HyperparameterLabelProps {
  name: string;
  symbol?: string;
  className?: string;
  htmlFor?: string;
  as?: 'label' | 'span';
}

export const HyperparameterLabel: React.FC<HyperparameterLabelProps> = ({
  name,
  symbol,
  className = 'param-label whitespace-nowrap shrink-0',
  htmlFor,
  as = 'label',
}) => {
  const segments = splitHyperparameterLabel(name, symbol);
  const children = segments.map((seg, idx) =>
    seg.isSymbol ? (
      <span key={idx} className="normal-case">
        {seg.text}
      </span>
    ) : (
      <React.Fragment key={idx}>{seg.text}</React.Fragment>
    )
  );

  if (as === 'span') {
    return <span className={className}>{children}</span>;
  }

  return (
    <label htmlFor={htmlFor} className={className}>
      {children}
    </label>
  );
};
