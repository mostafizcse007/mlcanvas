import React, { useState, useEffect } from 'react';

let katexInstance: any = null;
let katexPromise: Promise<any> | null = null;

export function loadKatex(): Promise<any> {
  if (katexInstance) return Promise.resolve(katexInstance);
  if (!katexPromise) {
    katexPromise = Promise.all([
      import('katex'),
      // @ts-expect-error CSS dynamic import
      import('katex/dist/katex.min.css'),
    ]).then(([mod]) => {
      katexInstance = mod.default || mod;
      return katexInstance;
    });
  }
  return katexPromise;
}

export function renderLatex(expression: string, displayMode: boolean = false): string {
  if (katexInstance) {
    try {
      return katexInstance.renderToString(expression, {
        displayMode,
        throwOnError: false,
        output: 'html',
      });
    } catch {
      return `<span class="katex-fallback">${expression}</span>`;
    }
  }
  return `<span class="katex-fallback">${expression}</span>`;
}

export const LatexSpan: React.FC<{
  expression: string;
  displayMode?: boolean;
  className?: string;
  title?: string;
}> = ({ expression, displayMode = false, className, title }) => {
  const [rendered, setRendered] = useState<string>(() => renderLatex(expression, displayMode));

  useEffect(() => {
    if (katexInstance) {
      setRendered(renderLatex(expression, displayMode));
      return;
    }

    let isCancelled = false;

    // Load on user interaction or after page has fully settled (6 seconds)
    const events = ['pointerdown', 'mousemove', 'keydown', 'scroll'];
    const onInteract = () => {
      events.forEach((e) => window.removeEventListener(e, onInteract));
      loadKatex().then(() => {
        if (!isCancelled) {
          setRendered(renderLatex(expression, displayMode));
        }
      });
    };
    events.forEach((e) => window.addEventListener(e, onInteract, { passive: true, once: true }));

    const timer = setTimeout(() => {
      events.forEach((e) => window.removeEventListener(e, onInteract));
      loadKatex().then(() => {
        if (!isCancelled) {
          setRendered(renderLatex(expression, displayMode));
        }
      });
    }, 6000);

    return () => {
      isCancelled = true;
      clearTimeout(timer);
      events.forEach((e) => window.removeEventListener(e, onInteract));
    };
  }, [expression, displayMode]);

  return (
    <span
      className={className}
      title={title}
      dangerouslySetInnerHTML={{ __html: rendered }}
    />
  );
};
