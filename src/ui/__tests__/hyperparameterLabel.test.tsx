// @vitest-environment jsdom
import { describe, it, expect, afterEach } from 'vitest';
import { render, cleanup } from '@testing-library/react';
import {
  HyperparameterLabel,
  splitHyperparameterLabel,
  getAccessibleHyperparameterLabel,
} from '../HyperparameterLabel';
import { TopControlBar } from '../TopControlBar';
import { ALGORITHM_CATALOG } from '../../algorithms/registry';
import { AlgorithmPlugin } from '../../types/algorithm';
import { logisticRegressionPlugin } from '../../algorithms/stage01/logisticRegression';
import { xgboostPlugin } from '../../algorithms/stage02/xgboost';
import { dbscanPlugin } from '../../algorithms/stage03/dbscan';
import { pcaPlugin } from '../../algorithms/stage03/pca';
import { vaePlugin } from '../../algorithms/stage04/vae';
import { ganPlugin } from '../../algorithms/stage04/gan';
import { transformerEncDecPlugin } from '../../algorithms/stage06/transformerEncDec';
import { qLearningPlugin } from '../../algorithms/stage07/qLearning';
import { dqnPlugin } from '../../algorithms/stage07/dqn';
import { reinforcePlugin } from '../../algorithms/stage07/reinforce';

const noop = () => {};

afterEach(() => {
  cleanup();
});

describe('HyperparameterLabel Greek symbol preservation', () => {
  it('separates parenthesized Greek and lowercase math symbols into normal-case segments', () => {
    const cases: [string, string[]][] = [
      ['Learning Rate (η)', ['(η)']],
      ['Discount (γ)', ['(γ)']],
      ['Exploration (ε)', ['(ε)']],
      ['Learning Rate (α)', ['(α)']],
      ['Decision Threshold (τ)', ['(τ)']],
      ['Epsilon Radius (ε)', ['(ε)']],
      ['Ridge Stabilizer (λ)', ['(λ)']],
      ['Signal Var (σ_f²)', ['(σ_f²)']],
      ['KL Weight (β-VAE)', ['(β-VAE)']],
      ['Generator LR (η_G)', ['(η_G)']],
      ['Degree (d)', ['(d)']],
      ['Length-Scale (l)', ['(l)']],
      ['Whiten (Scale / √λ)', ['λ']],
      ['Perplexity (Effective k)', ['k']],
    ];

    for (const [label, expectedSymbols] of cases) {
      const parts = splitHyperparameterLabel(label);
      const symbolParts = parts.filter((p) => p.isSymbol).map((p) => p.text);
      expect(symbolParts, `Label "${label}" should exempt ${expectedSymbols}`).toEqual(
        expectedSymbols
      );
    }

    // Plain multi-word parens remain subject to uppercase prose styling
    const proseOnly = splitHyperparameterLabel('Min Dist (Tightly Packed)');
    expect(proseOnly.some((p) => p.isSymbol)).toBe(false);
  });

  it('supports explicit symbol prop without duplicating parenthesized symbols', () => {
    const { container, rerender } = render(
      <HyperparameterLabel name="Learning Rate" symbol="η" />
    );
    const label = container.querySelector('label.param-label')!;
    const spans = Array.from(label.querySelectorAll('span.normal-case'));
    expect(spans.map((s) => s.textContent)).toEqual(['(η)']);
    expect(label.textContent).toBe('Learning Rate (η)');
    expect(getAccessibleHyperparameterLabel('Learning Rate', 'η')).toBe('Learning Rate (η)');

    rerender(<HyperparameterLabel name="Learning Rate (η)" symbol="η" />);
    const dedupedLabel = container.querySelector('label.param-label')!;
    expect(dedupedLabel.textContent).toBe('Learning Rate (η)');
    expect(getAccessibleHyperparameterLabel('Learning Rate (η)', 'η')).toBe('Learning Rate (η)');
  });

  it('audits all Stage 01-07 parameter metadata for true lowercase Unicode Greek symbols (no Latin lookalikes)', async () => {
    const plugins: AlgorithmPlugin[] = [];
    for (const entry of ALGORITHM_CATALOG) {
      if (!entry.isImplemented) continue;
      if (entry.plugin) {
        plugins.push(entry.plugin);
      } else if (entry.loader) {
        plugins.push(await entry.loader());
      }
    }

    expect(plugins.length).toBeGreaterThanOrEqual(35);

    const greekTerms = [
      'Learning Rate',
      'Discount',
      'Exploration',
      'Decision Threshold',
      'Epsilon Radius',
      'Regularization (λ',
      'Ridge',
      'Penalty Strength',
      'L1 Ratio',
      'L1 Reg',
      'L2 Reg',
      'L2 Leaf Reg',
      'Min Split Loss',
      'RBF Gamma',
      'Insensitive Tube',
      'Sampling Noise',
      'Covariance Reg',
      'Residual Scale',
      'Entropy Bonus',
      'Laplace Smoothing',
      'Huber Threshold',
      'Temperature',
    ];

    for (const plugin of plugins) {
      const allParams = [...(plugin.primaryParams ?? []), ...(plugin.advancedParams ?? [])];
      for (const hp of allParams) {
        // Never use uppercase Greek capitals in parameter metadata
        expect(
          hp.name,
          `${plugin.id} param "${hp.name}" must not contain uppercase Greek capitals`
        ).not.toMatch(/[\u0391-\u03A9]/);

        // No Latin H/E/T/A lookalikes on Greek-symbol parameters
        const matchesGreekConcept = greekTerms.some((term) => hp.name.includes(term));
        if (matchesGreekConcept && hp.name.includes('(')) {
          expect(
            hp.name,
            `${plugin.id} param "${hp.name}" has Latin lookalike instead of Unicode Greek`
          ).not.toMatch(/\((H|E|T|A)\)/);
          expect(
            hp.name,
            `${plugin.id} param "${hp.name}" should contain lowercase Unicode Greek symbol`
          ).toMatch(/[\u03B1-\u03C9]/);
        }
      }
    }
  });

  it('renders normal-case spans inside TopControlBar param labels across Stages 01, 02, 03, 04, 06, and 07', () => {
    const expectedByAlgorithm: [AlgorithmPlugin, string[]][] = [
      [logisticRegressionPlugin, ['(η)', '(C)', '(τ)']],
      [xgboostPlugin, ['(M)', '(η)']],
      [dbscanPlugin, ['(ε)']],
      [pcaPlugin, ['λ']],
      [vaePlugin, ['(β-VAE)', '(η)', '(ε)']],
      [ganPlugin, ['(k)', '(η_G)', '(η_D)']],
      [transformerEncDecPlugin, ['(H)', '(τ)']],
      [qLearningPlugin, ['(γ)', '(ε)', '(α)']],
      [dqnPlugin, ['(C)', '(η)']],
      [reinforcePlugin, ['(α)', '(β)']],
    ];

    for (const [plugin, expectedSymbols] of expectedByAlgorithm) {
      const { container } = render(
        <TopControlBar
          isRunning={false}
          epoch={0}
          currentAlgorithm={plugin}
          hyperparams={{}}
          theme="carbon"
          reducedMotion={false}
          onTogglePlay={noop}
          onReset={noop}
          onStep={noop}
          onOpenAlgorithmModal={noop}
          onUpdateHyperparam={noop}
          onToggleTheme={noop}
          onToggleReducedMotion={noop}
          onCopyShareLink={noop}
          onExportPng={noop}
        />
      );

      const normalCaseTokens = Array.from(
        container.querySelectorAll('.top-params-cluster label.param-label span.normal-case')
      ).map((el) => el.textContent);

      expect(
        normalCaseTokens,
        `${plugin.id} top header should render normal-case symbols ${expectedSymbols.join(', ')}`
      ).toEqual(expectedSymbols);

      cleanup();
    }
  });
});
