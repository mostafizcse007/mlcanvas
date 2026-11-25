import { describe, it, expect } from 'vitest';
import { linearSvmPlugin } from '../stage01/linearSvm';
import { linearRegressionPlugin } from '../stage01/linearRegression';
import { generateTabular2dDataset, DatasetKind } from '../../datasets/tabular2d';
import {
  evaluateHyperplaneY,
  formatHyperplaneDiagnostic,
} from '../../rendering/Stage01Canvas';

function parseExplicitLineEquation(label: string): { slope: number; intercept: number } {
  // Matches either "(x₂ = <slope>x₁ <+|-> <intercept>)" or "Fit: y = <slope>x₁ <+|-> <intercept>"
  const match = label.match(/(?:x₂|y)\s*=\s*(-?\d+(?:\.\d+)?)x₁\s*([+-])\s*(\d+(?:\.\d+)?)/);
  if (!match) {
    throw new Error(`Could not parse explicit line equation from "${label}"`);
  }
  const slope = parseFloat(match[1]);
  const sign = match[2] === '-' ? -1 : 1;
  const intercept = sign * parseFloat(match[3]);
  return { slope, intercept };
}

describe('Stage 01 — Linear SVM Cross-Task Equation Sign & Coordinate Agreement', () => {
  it.each(['polynomial-cubic', 'linear-trend', 'sine-wave'] as DatasetKind[])(
    'displayed hyperplane equation matches plotted line coordinates at two distinct x₁ values on %s',
    (kind) => {
      const dataset = generateTabular2dDataset({
        kind,
        sampleCount: 160,
        noise: 0.1,
        testSplit: 0.2,
        seed: 42,
        numClasses: 2,
        scaling: 'raw',
        stretchX2: false,
      });

      const params = { cPenalty: 1.0, learningRate: 0.03 };
      let state = linearSvmPlugin.init(dataset, params, 42);
      for (let i = 0; i < 20; i++) {
        state = linearSvmPlugin.step(state, dataset, params).state;
      }

      const artifacts = linearSvmPlugin.getVisualArtifacts(state, dataset, params);
      const hp = artifacts.find((a) => a.type === 'hyperplane');
      expect(hp).toBeDefined();
      if (hp?.type !== 'hyperplane') return;

      const label = formatHyperplaneDiagnostic(hp.normal, hp.bias, {
        is1DRegression: false,
        isRegressionDataset: true,
        isCrossTaskClassification: true,
      });

      // Evaluate actual plotted line coordinates at two distinct x1 values
      const xA = -3.0;
      const xB = 3.0;
      const yPlotA = evaluateHyperplaneY(hp.normal, hp.bias, xA);
      const yPlotB = evaluateHyperplaneY(hp.normal, hp.bias, xB);
      const plottedSlope = (yPlotB - yPlotA) / (xB - xA);
      const plottedIntercept = evaluateHyperplaneY(hp.normal, hp.bias, 0);

      // Parse the displayed explicit equation from the badge label
      const parsed = parseExplicitLineEquation(label);

      // Displayed slope & intercept MUST match plotted coordinates within rounding tolerance (0.01)
      expect(Math.abs(parsed.slope - plottedSlope)).toBeLessThan(0.015);
      expect(Math.abs(parsed.intercept - plottedIntercept)).toBeLessThan(0.015);

      // Verify exact equality at both x1 points against parsed displayed formula
      const yLabelA = parsed.slope * xA + parsed.intercept;
      const yLabelB = parsed.slope * xB + parsed.intercept;
      expect(Math.abs(yLabelA - yPlotA)).toBeLessThan(0.06);
      expect(Math.abs(yLabelB - yPlotB)).toBeLessThan(0.06);

      // Slope signs must strictly match (never inverted!)
      if (Math.abs(plottedSlope) > 0.02) {
        expect(Math.sign(parsed.slope)).toBe(Math.sign(plottedSlope));
      }

      // Specifically check Linear Trend has negative Cross-Task boundary slope orthogonal to upward trend
      if (kind === 'linear-trend') {
        expect(plottedSlope).toBeLessThan(0);
        expect(parsed.slope).toBeLessThan(0);
      }

      // Implicit hyperplane coefficients must satisfy w0*x1 + w1*yPlot + b = 0
      const [w0, w1] = hp.normal;
      expect(w0 * xA + w1 * yPlotA + hp.bias).toBeCloseTo(0, 6);
      expect(w0 * xB + w1 * yPlotB + hp.bias).toBeCloseTo(0, 6);
    }
  );

  it('never inverts slope/intercept even if formatHyperplaneDiagnostic receives arbitrary w1 in 1D Fit mode', () => {
    // Test with Linear Regression (w1 = -1) AND with non-unit positive w1 (+0.5)
    const dataset = generateTabular2dDataset({
      kind: 'linear-trend',
      sampleCount: 80,
      noise: 0.05,
      testSplit: 0.2,
      seed: 42,
      numClasses: 2,
      scaling: 'raw',
      stretchX2: false,
    });
    const lrState = linearRegressionPlugin.step(
      linearRegressionPlugin.init(dataset, { solver: 'closed-form' }, 42),
      dataset,
      { solver: 'closed-form' }
    ).state;

    const linLabel = formatHyperplaneDiagnostic([lrState.weights[0], lrState.weights[1]], lrState.bias, {
      is1DRegression: true,
      isRegressionDataset: true,
      isCrossTaskClassification: false,
    });
    const linParsed = parseExplicitLineEquation(linLabel);
    expect(Math.abs(linParsed.slope - evaluateHyperplaneY([lrState.weights[0], lrState.weights[1]], lrState.bias, 1) + evaluateHyperplaneY([lrState.weights[0], lrState.weights[1]], lrState.bias, 0))).toBeLessThan(0.015);

    // Non-unit positive w1: -0.13*x1 + 0.50*x2 + 0.07 = 0 => x2 = +0.26*x1 - 0.14
    const labelPosW1 = formatHyperplaneDiagnostic([-0.13, 0.5], 0.07, {
      is1DRegression: true,
      isRegressionDataset: true,
      isCrossTaskClassification: false,
    });
    const parsedPosW1 = parseExplicitLineEquation(labelPosW1);
    expect(parsedPosW1.slope).toBeCloseTo(0.26, 2);
    expect(parsedPosW1.intercept).toBeCloseTo(-0.14, 2);
  });
});
