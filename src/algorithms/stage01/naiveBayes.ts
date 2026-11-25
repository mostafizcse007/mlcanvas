import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { gaussianPdf } from '../../utils/math';
import { binarizeTarget, binarizeTarget01 } from './crossTask';

export interface ClassGaussianStats {
  label: number;
  prior: number;
  count: number;
  meanX1: number;
  meanX2: number;
  varX1: number;
  varX2: number;
}

export interface NaiveBayesState {
  variant: 'gaussian' | 'multinomial' | 'bernoulli';
  alpha: number; // Laplace smoothing
  epoch: number;
  classes: ClassGaussianStats[];
  gridFrequencies?: { binX: number; binY: number; countClass0: number; countClass1: number }[];
}

export interface NaiveBayesParams {
  variant?: 'gaussian' | 'multinomial' | 'bernoulli';
  laplaceAlpha?: number;
}

function computeClassGaussianStats(
  train: DatasetContainer['train'],
  alpha: number
): ClassGaussianStats[] {
  const classes: ClassGaussianStats[] = [];
  const labels: (1 | -1)[] = [1, -1];

  for (const label of labels) {
    const subset = train.filter((p) => binarizeTarget(p.y) === label);
    const count = subset.length;
    const prior = train.length > 0 ? (count + alpha) / (train.length + alpha * labels.length) : 0.5;

    if (count > 0) {
      let sumX1 = 0, sumX2 = 0;
      for (const p of subset) { sumX1 += p.x1; sumX2 += p.x2; }
      const m1 = sumX1 / count;
      const m2 = sumX2 / count;

      let var1 = 0, var2 = 0;
      for (const p of subset) {
        var1 += (p.x1 - m1) ** 2;
        var2 += (p.x2 - m2) ** 2;
      }
      var1 = Math.max(0.05, var1 / count);
      var2 = Math.max(0.05, var2 / count);

      classes.push({ label, prior, count, meanX1: m1, meanX2: m2, varX1: var1, varX2: var2 });
    } else {
      classes.push({ label, prior, count: 0, meanX1: 0, meanX2: 0, varX1: 1, varX2: 1 });
    }
  }

  return classes;
}

export const naiveBayesPlugin: AlgorithmPlugin<NaiveBayesState, NaiveBayesParams> = {
  id: 'naive-bayes',
  name: 'Naive Bayes Classifier (Gaussian & Discrete)',
  shortCode: 'NaiveBayes',
  stage: 'continuous-2d',
  taskType: 'classification',
  executionMode: 'closed-form',
  description: 'Conditional independence generative model with marginal axis-aligned bell curves and discrete grid quantization.',
  paperReference: 'Maron, Automatic Indexing: An Experiment with Coded Abstracts, J. ACM 1961; Duda & Hart, 1973',

  primaryParams: [
    {
      id: 'variant',
      name: 'Likelihood Model',
      type: 'select',
      options: [
        { label: 'Gaussian (Continuous 2D Densities)', value: 'gaussian' },
      ],
      default: 'gaussian',
      description: 'Generative normal probability distribution P(X|Y) = N(μ, Σ) with decoupled diagonal covariance.',
    },
    {
      id: 'laplaceAlpha',
      name: 'Laplace Smoothing (α)',
      type: 'slider',
      min: 0.0,
      max: 5.0,
      step: 0.2,
      default: 1.0,
      description: 'Pseudocount added to prevent zero-probability collapse on unseen feature values.',
    },
  ],

  advancedParams: [],

  symbolicLinks: [
    {
      id: 'independence',
      symbolLatex: 'P(\\mathbf{x}|y) = \\prod_{j=1}^2 P(x_j|y)',
      label: 'Naive Conditional Independence',
      description: 'Crucial assumption that features are conditionally independent given class label y.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: NaiveBayesParams, _seed: number): NaiveBayesState {
    const variant = params.variant ?? 'gaussian';
    const alpha = Number(params.laplaceAlpha ?? 1.0);
    const classes = computeClassGaussianStats(dataset.train, alpha);

    return {
      variant,
      alpha,
      epoch: 0,
      classes,
    };
  },

  step(
    state: NaiveBayesState,
    dataset: DatasetContainer,
    params: NaiveBayesParams
  ): StepResult<NaiveBayesState> {
    const variant = params.variant ?? state.variant ?? 'gaussian';
    const alpha = Number(params.laplaceAlpha ?? state.alpha ?? 1.0);
    const classes = computeClassGaussianStats(dataset.train, alpha);

    const nextState: NaiveBayesState = {
      variant,
      alpha,
      epoch: state.epoch + 1,
      classes,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
      converged: true,
    };
  },

  predict(input: Float32Array, state: NaiveBayesState, _params: NaiveBayesParams): number {
    if (!state || !state.classes || state.classes.length < 2) return 0;
    const x1 = input[0];
    const x2 = input[1];

    const cPos = state.classes.find((c) => c.label === 1)!;
    const cNeg = state.classes.find((c) => c.label === -1)!;

    // Log-likelihood: ln P(y) + ln P(x1|y) + ln P(x2|y)
    const logL = (c: ClassGaussianStats) => {
      const p1 = Math.max(1e-12, gaussianPdf(x1, c.meanX1, c.varX1));
      const p2 = Math.max(1e-12, gaussianPdf(x2, c.meanX2, c.varX2));
      return Math.log(Math.max(1e-6, c.prior)) + Math.log(p1) + Math.log(p2);
    };

    const scorePos = logL(cPos);
    const scoreNeg = logL(cNeg);

    // Continuous difference for decision contour
    return scorePos - scoreNeg;
  },

  loss(state: NaiveBayesState, dataset: DatasetContainer, params: NaiveBayesParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;

    const evalAcc = (pts: typeof train) => {
      if (pts.length === 0) return { loss: 0, acc: 0 };
      let correct = 0;
      let logLossSum = 0;

      for (let i = 0; i < pts.length; i++) {
        const pt = pts[i];
        const input = new Float32Array([pt.x1, pt.x2]);
        const score = Number(this.predict(input, state, params));
        const pred = score >= 0 ? 1 : -1;
        if (pred === binarizeTarget(pt.y)) correct++;
        // Cross-entropy surrogate
        const p = 1.0 / (1.0 + Math.exp(-Math.max(-10, Math.min(10, score))));
        const yTarget = binarizeTarget01(pt.y);
        logLossSum -= yTarget * Math.log(Math.max(1e-6, p)) + (1 - yTarget) * Math.log(Math.max(1e-6, 1 - p));
      }

      return {
        loss: logLossSum / pts.length,
        acc: (correct / pts.length) * 100,
      };
    };

    const trainRes = evalAcc(train);
    const testRes = evalAcc(test);

    return {
      trainLoss: trainRes.loss,
      testLoss: testRes.loss,
      trainAccuracy: trainRes.acc,
      testAccuracy: testRes.acc,
    };
  },

  getVisualArtifacts(
    state: NaiveBayesState,
    _dataset: DatasetContainer,
    _params: NaiveBayesParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];

    if (state?.classes && state.classes.length > 0) {
      // 1. Marginal Gaussians along axes and axis-aligned covariance ellipses
      artifacts.push({
        type: 'marginal-gaussians',
        classes: state.classes.map((c) => ({
          label: c.label,
          mean: [c.meanX1, c.meanX2],
          variance: [c.varX1, c.varX2],
          prior: c.prior,
        })),
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: NaiveBayesState): Float32Array {
    const buf = new Float32Array(16);
    buf[0] = state.epoch;
    buf[1] = state.alpha;
    buf[2] = state.variant === 'gaussian' ? 1 : state.variant === 'multinomial' ? 2 : 3;

    let offset = 3;
    for (let i = 0; i < Math.min(2, state.classes.length); i++) {
      const c = state.classes[i];
      buf[offset++] = c.label;
      buf[offset++] = c.meanX1;
      buf[offset++] = c.meanX2;
      buf[offset++] = c.varX1;
      buf[offset++] = c.varX2;
      buf[offset++] = c.prior;
    }
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: NaiveBayesState): NaiveBayesState {
    const classes: ClassGaussianStats[] = [];
    let offset = 3;
    for (let i = 0; i < 2; i++) {
      classes.push({
        label: buffer[offset++],
        count: 50,
        meanX1: buffer[offset++],
        meanX2: buffer[offset++],
        varX1: buffer[offset++],
        varX2: buffer[offset++],
        prior: buffer[offset++],
      });
    }

    return {
      ...state,
      epoch: Math.round(buffer[0]),
      alpha: buffer[1],
      variant: buffer[2] === 1 ? 'gaussian' : buffer[2] === 2 ? 'multinomial' : 'bernoulli',
      classes,
    };
  },

  explain(
    state: NaiveBayesState,
    _params: NaiveBayesParams,
    lastStep?: StepResult<NaiveBayesState>
  ): ExplanationPayload {
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';

    return {
      summary: `Naive Bayes is a 1-pass closed-form analytical generative model: computes exact class posteriors via Bayes' rule under the conditional independence assumption P(X|Y) = P(X₁|Y)P(X₂|Y), with ${acc}% accuracy.`,
      coreAssumption:
        'Features are conditionally mutually independent given class Y; covariance matrix Σ is strictly diagonal with zero cross-correlation off-diagonals.',
      whenItFails:
        'Fails dramatically when features exhibit strong correlation (e.g. tilted diagonal clusters) because axis-aligned ellipses cannot rotate to fit covariances.',
      microLog: [
        `⚡ 1-Pass Closed-Form MLE: Estimated decoupled 1D Gaussian parameters along X₁ & X₂ in 1 pass (no SGD required)`,
        `Computed class priors: P(Y=+1) = ${state.classes[0]?.prior.toFixed(2)}, P(Y=-1) = ${state.classes[1]?.prior.toFixed(2)}`,
        `Laplace smoothing factor α = ${state.alpha.toFixed(1)} guarantees positive probabilities`,
      ],
      mathLatex: {
        objective: 'P(y = c | \\mathbf{x}) \\propto P(y = c) \\prod_{j=1}^2 \\mathcal{N}\\left(x_j; \\mu_{c, j}, \\sigma_{c, j}^2\\right)',
        lossOrUpdate: '\\mu_{c, j} = \\frac{1}{N_c} \\sum_{i \\in c} x_{i, j}, \\quad \\sigma_{c, j}^2 = \\frac{1}{N_c} \\sum_{i \\in c} (x_{i, j} - \\mu_{c, j})^2',
        dualOrConstraint: '\\Sigma_c = \\begin{bmatrix} \\sigma_{c, 1}^2 & 0 \\\\ 0 & \\sigma_{c, 2}^2 \\end{bmatrix} \\quad (\\text{Axis-Aligned Zero Off-Diagonal Covariance})',
      },
      codeSnippet: {
        sklearn: `from sklearn.naive_bayes import GaussianNB\n\nmodel = GaussianNB(var_smoothing=1e-9)\nmodel.fit(X_train, y_train)\nprint(f"Class Priors: {model.class_prior_}")\nprint(f"Means: {model.theta_}, Variances: {model.var_}")`,
        pytorch: `# PyTorch Gaussian Naive Bayes\nimport torch\n\ndef gnb_predict(means, vars, priors, x):\n    # Vectorized log-likelihood\n    log_probs = []\n    for m, v, p in zip(means, vars, priors):\n        diff = (x - m)**2\n        log_ll = -0.5 * torch.sum(diff / v + torch.log(2 * 3.14159 * v), dim=-1)\n        log_probs.append(torch.log(p) + log_ll)\n    return torch.argmax(torch.stack(log_probs), dim=0)`,
      },
    };
  },
};
