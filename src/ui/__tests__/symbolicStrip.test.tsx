// @vitest-environment jsdom
import { describe, it, expect, beforeAll, afterAll, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
// @ts-expect-error Node built-in used by Vitest test runner without @types/node
import { readFileSync } from 'node:fs';
// @ts-expect-error Node built-in used by Vitest test runner without @types/node
import { dirname, resolve } from 'node:path';
// @ts-expect-error Node built-in used by Vitest test runner without @types/node
import { fileURLToPath } from 'node:url';
import { SymbolicStrip } from '../SymbolicStrip';
import { InteractiveStageContainer } from '../InteractiveStageContainer';
import { linearSvmPlugin } from '../../algorithms/stage01/linearSvm';
import { DirtyManager } from '../../rendering/dirty-manager';
import { DataPoint2D } from '../../datasets/tabular2d';
import { SymbolicMathLink } from '../../types/algorithm';

function hexToRgb(hex: string): [number, number, number] {
  const clean = hex.trim().replace('#', '');
  if (clean.length === 3) {
    return [
      parseInt(clean[0] + clean[0], 16),
      parseInt(clean[1] + clean[1], 16),
      parseInt(clean[2] + clean[2], 16),
    ];
  }
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16),
  ];
}

function relativeLuminance(hex: string): number {
  const [r, g, b] = hexToRgb(hex).map((channel) => {
    const srgb = channel / 255;
    return srgb <= 0.03928 ? srgb / 12.92 : Math.pow((srgb + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

function contrastRatio(fgHex: string, bgHex: string): number {
  const l1 = relativeLuminance(fgHex);
  const l2 = relativeLuminance(bgHex);
  const lighter = Math.max(l1, l2);
  const darker = Math.min(l1, l2);
  return (lighter + 0.05) / (darker + 0.05);
}

function extractCssBlockVars(css: string, headerPattern: RegExp): Record<string, string> {
  const match = css.match(headerPattern);
  expect(match, `Expected CSS block matching ${headerPattern}`).not.toBeNull();
  const body = match![1];
  const vars: Record<string, string> = {};
  const declRegex = /([\w-]+)\s*:\s*([^;]+);/g;
  let decl: RegExpExecArray | null;
  while ((decl = declRegex.exec(body)) !== null) {
    vars[decl[1].trim()] = decl[2].trim();
  }
  return vars;
}

const sampleTrain2D: DataPoint2D[] = [
  { id: 1, x1: -2, x2: -2, y: -1, isTest: false },
  { id: 2, x1: 2, x2: 2, y: 1, isTest: false },
];

const sampleTest2D: DataPoint2D[] = [
  { id: 101, x1: -3, x2: 3, y: -1, isTest: true },
];

const noop = () => {};
const predictZero = () => 0;

let prevResizeObserver: typeof globalThis.ResizeObserver | undefined;
let prevGetContext: typeof HTMLCanvasElement.prototype.getContext;

beforeAll(() => {
  prevResizeObserver = globalThis.ResizeObserver;
  globalThis.ResizeObserver = class MockResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;

  prevGetContext = HTMLCanvasElement.prototype.getContext;
  const noopFn = () => {};
  const mockCtx = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    font: '',
    textAlign: 'left',
    textBaseline: 'alphabetic',
    beginPath: noopFn,
    closePath: noopFn,
    moveTo: noopFn,
    lineTo: noopFn,
    arc: noopFn,
    arcTo: noopFn,
    quadraticCurveTo: noopFn,
    rect: noopFn,
    fill: noopFn,
    stroke: noopFn,
    save: noopFn,
    restore: noopFn,
    clip: noopFn,
    scale: noopFn,
    translate: noopFn,
    setTransform: noopFn,
    setLineDash: noopFn,
    fillText: noopFn,
    clearRect: noopFn,
    fillRect: noopFn,
    strokeRect: noopFn,
    measureText: () => ({ width: 0 }),
    createLinearGradient: () => ({ addColorStop: noopFn }),
  };
  HTMLCanvasElement.prototype.getContext = (() => mockCtx) as unknown as typeof HTMLCanvasElement.prototype.getContext;
});

afterEach(() => {
  cleanup();
});

afterAll(() => {
  if (prevResizeObserver === undefined) {
    delete (globalThis as { ResizeObserver?: unknown }).ResizeObserver;
  } else {
    globalThis.ResizeObserver = prevResizeObserver;
  }
  HTMLCanvasElement.prototype.getContext = prevGetContext;
});

describe('SymbolicStrip collapse/expand behavior and theme contrast', () => {
  const svmLinks: SymbolicMathLink[] = linearSvmPlugin.symbolicLinks ?? [];

  it('defaults to expanded and toggles aria-expanded and formula visibility on click', () => {
    const onHighlightChange = vi.fn();
    render(
      <SymbolicStrip
        links={svmLinks}
        activeHighlight={null}
        onHighlightChange={onHighlightChange}
      />
    );

    const toggle = screen.getByRole('button', { name: /symbolic strip/i });
    expect(toggle.getAttribute('aria-expanded')).toBe('true');

    const controlsId = toggle.getAttribute('aria-controls');
    expect(controlsId).toBeTruthy();

    const expandedPanel = document.getElementById(controlsId!);
    expect(expandedPanel).not.toBeNull();
    expect(expandedPanel!.children.length).toBe(svmLinks.length);
    expect(expandedPanel!.textContent).toContain('\\mathbf{w}');

    // Collapse the strip
    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(document.getElementById(controlsId!)).toBeNull();

    // Expand the strip again
    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    const reopenedPanel = document.getElementById(controlsId!);
    expect(reopenedPanel).not.toBeNull();
    expect(reopenedPanel!.children.length).toBe(svmLinks.length);
  });

  it('respects defaultExpanded={false} and emits highlight targets on chip hover', () => {
    const onHighlightChange = vi.fn();
    render(
      <SymbolicStrip
        links={svmLinks}
        activeHighlight={null}
        onHighlightChange={onHighlightChange}
        defaultExpanded={false}
      />
    );

    const toggle = screen.getByRole('button', { name: /symbolic strip/i });
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    const controlsId = toggle.getAttribute('aria-controls')!;
    expect(document.getElementById(controlsId)).toBeNull();

    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');

    const panel = document.getElementById(controlsId)!;
    const firstChip = panel.children[0] as HTMLElement;
    fireEvent.mouseEnter(firstChip);
    expect(onHighlightChange).toHaveBeenCalledWith(svmLinks[0].highlightTarget);

    fireEvent.mouseLeave(firstChip);
    expect(onHighlightChange).toHaveBeenCalledWith(null);
  });

  it('toggles inside InteractiveStageContainer and uses semantic theme tokens without hardcoded white/black', () => {
    const dirtyManager = new DirtyManager();
    const markDirtySpy = vi.spyOn(dirtyManager, 'markDirty');

    const { container } = render(
      <InteractiveStageContainer
        algorithm={linearSvmPlugin}
        theme="carbon"
        trainData={sampleTrain2D}
        testData={sampleTest2D}
        artifacts={[]}
        gridResolution={96}
        dirtyManager={dirtyManager}
        circuitBreakerTripped={true}
        pointerMode="add-drag"
        activeClass={1}
        showTestData={true}
        onToggleShowTestData={noop}
        onAddPoint={noop}
        onUpdatePoint={noop}
        onDeletePoint={noop}
        onClampAndRecover={noop}
        predictValueAtPoint={predictZero}
        scrubbedEpoch={5}
        onReturnToLive={noop}
      />
    );

    const toggle = screen.getByRole('button', { name: /symbolic strip/i });
    const controlsId = toggle.getAttribute('aria-controls')!;
    expect(toggle.getAttribute('aria-expanded')).toBe('true');
    expect(document.getElementById(controlsId)).not.toBeNull();

    // Ensure semantic theme tokens replaced bg-white/text-black/#fff/#000 across SymbolicStrip and action buttons
    const stripRoot = container.querySelector('.symbolic-strip') as HTMLElement;
    expect(stripRoot).not.toBeNull();
    expect(stripRoot.outerHTML).not.toMatch(/#000\b|#fff\b|\bbg-white\b|\btext-black\b/i);

    const returnBtn = screen.getByRole('button', { name: /return to live/i });
    expect(returnBtn.className).toContain('text-[var(--text-inverse)]');
    expect(returnBtn.className).not.toMatch(/\b(bg-white|text-black)\b/);

    const clampBtn = screen.getByRole('button', { name: /clamp & recover/i });
    expect(clampBtn.className).toContain('bg-[var(--bg-canvas)]');
    expect(clampBtn.className).toContain('text-[var(--text-primary)]');
    expect(clampBtn.className).not.toMatch(/\b(bg-white|text-black)\b/);

    // Toggle collapse / expand
    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('false');
    expect(document.getElementById(controlsId)).toBeNull();

    fireEvent.click(toggle);
    expect(toggle.getAttribute('aria-expanded')).toBe('true');

    const panel = document.getElementById(controlsId)!;
    markDirtySpy.mockClear();
    fireEvent.mouseEnter(panel.children[0] as HTMLElement);
    expect(markDirtySpy).toHaveBeenCalledTimes(1);
    fireEvent.mouseLeave(panel.children[0] as HTMLElement);
    expect(markDirtySpy).toHaveBeenCalledTimes(2);
  });

  it('meets WCAG AA (>= 4.5:1) contrast for Symbolic Strip text tokens in Paper and Carbon themes', () => {
    const testDir = dirname(fileURLToPath(import.meta.url));
    const tokensCss: string = readFileSync(resolve(testDir, '../../styles/tokens.css'), 'utf8');
    const paperVars = extractCssBlockVars(
      tokensCss,
      /:root,\s*\[data-theme="paper"\]\s*\{([\s\S]*?)\}/
    );
    const carbonVars = extractCssBlockVars(
      tokensCss,
      /\[data-theme="carbon"\]\s*\{([\s\S]*?)\}/
    );

    expect(paperVars['color-scheme']).toBe('light');
    expect(carbonVars['color-scheme']).toBe('dark');

    const surfaces = ['--bg-primary', '--bg-elevated', '--bg-subtle', '--bg-active', '--bg-canvas'];
    const textTokens = ['--text-primary', '--text-secondary'];

    for (const [themeName, vars] of [
      ['paper', paperVars],
      ['carbon', carbonVars],
    ] as const) {
      for (const fg of textTokens) {
        for (const bg of surfaces) {
          const ratio = contrastRatio(vars[fg], vars[bg]);
          expect(
            ratio,
            `${themeName}: ${fg} (${vars[fg]}) on ${bg} (${vars[bg]}) contrast ${ratio.toFixed(2)}:1 < 4.5:1`
          ).toBeGreaterThanOrEqual(4.5);
        }
      }
    }
  });

  it('defines .no-scrollbar and KaTeX inherit rules in workbench.css', () => {
    const testDir = dirname(fileURLToPath(import.meta.url));
    const workbenchCss: string = readFileSync(resolve(testDir, '../../styles/workbench.css'), 'utf8');
    expect(workbenchCss).toMatch(/\.no-scrollbar\s*\{[\s\S]*?scrollbar-width:\s*none/);
    expect(workbenchCss).toMatch(/\.no-scrollbar::-webkit-scrollbar\s*\{[\s\S]*?display:\s*none/);
    expect(workbenchCss).toMatch(/\.symbolic-strip\s+\.katex[\s\S]*?color:\s*inherit/);
  });
});
