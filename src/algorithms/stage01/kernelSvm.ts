import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { rbfKernel, polyKernel } from '../../utils/math';
import { mulberry32 } from '../../utils/prng';
import { binarizeTarget } from './crossTask';

export interface SupportVectorItem {
  id: number;
  x1: number;
  x2: number;
  y: number;
  alpha: number;
}

export interface KernelSVMState {
  alphas: Float32Array; // Dual multipliers α_i for each training point
  bias: number;
  epoch: number;
  supportVectors: SupportVectorItem[];
  supportVectorIds: number[];
  supportVectorCount: number;
  gamma: number;
  kernel: 'rbf' | 'poly';
  degree: number;
  marginWidth: number;
}

export interface KernelSVMParams {
  kernel?: 'rbf' | 'poly';
  gamma?: number;
  degree?: number;
  cPenalty?: number;
  batchPasses?: number;
}

export const kernelSvmPlugin: AlgorithmPlugin<KernelSVMState, KernelSVMParams> = {
  id: 'kernel-svm',
  name: 'Support Vector Machine (Kernel RBF & Poly)',
  shortCode: 'Kernel SVM',
  stage: 'continuous-2d',
  taskType: 'classification',
  executionMode: 'epoch-loop',
  description: 'Dual representation in Reproducing Kernel Hilbert Space (RKHS) with active support vector isolation.',
  paperReference: 'Boser, Guyon & Vapnik, A Training Algorithm for Optimal Margin Classifiers, COLT 1992',

  primaryParams: [
    {
      id: 'kernel',
      name: 'Kernel Function',
      type: 'select',
      options: [
        { label: 'Gaussian RBF (Radial)', value: 'rbf' },
        { label: 'Polynomial Basis', value: 'poly' },
      ],
      default: 'rbf',
      description: 'Geometric similarity inner product mapping into infinite or polynomial RKHS.',
    },
    {
      id: 'gamma',
      name: 'RBF Gamma (γ)',
      type: 'select',
      options: [
        { label: '0.1 (Broad Margin)', value: 0.1 },
        { label: '0.5 (Balanced)', value: 0.5 },
        { label: '2.0 (Tight Contour)', value: 2.0 },
        { label: '8.0 (Complex Island)', value: 8.0 },
        { label: '25.0 (Extreme Overfit)', value: 25.0 },
      ],
      default: 0.5,
      description: 'Inverse kernel bandwidth γ = 1/(2σ²). High γ creates isolated shrink-wrap islands.',
    },
    {
      id: 'cPenalty',
      name: 'Soft Margin (C)',
      type: 'select',
      options: [
        { label: '0.1 (High Slack)', value: 0.1 },
        { label: '1.0 (Standard)', value: 1.0 },
        { label: '10.0 (Hard Margin)', value: 10.0 },
      ],
      default: 1.0,
      description: 'Box constraint upper bound 0 ≤ α_i ≤ C penalizing margin violations.',
    },
  ],

  advancedParams: [
    {
      id: 'degree',
      name: 'Poly Degree (d)',
      type: 'slider',
      min: 2,
      max: 5,
      step: 1,
      default: 3,
      description: 'Degree of polynomial kernel when Polynomial basis is selected.',
    },
  ],

  symbolicLinks: [
    {
      id: 'rbf',
      symbolLatex: 'K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp(-\\gamma \\|\\mathbf{x}_i - \\mathbf{x}_j\\|^2)',
      label: 'Radial Basis Function (RBF)',
      description: 'Infinite-dimensional Hilbert projection capturing arbitrary non-linear topologies.',
      highlightTarget: 'decision-hyperplane',
    },
    {
      id: 'sv',
      symbolLatex: '\\alpha_i > 0',
      label: 'Active Support Vectors',
      description: 'The sparse subset of training points lying exactly on or inside the ±1 margin street.',
      highlightTarget: 'support-vectors',
    },
  ],

  init(dataset: DatasetContainer, params: KernelSVMParams, seed: number): KernelSVMState {
    const N = dataset.train.length;
    const alphas = new Float32Array(Math.max(1, N));
    const gamma = Number(params.gamma ?? 0.5);
    const kernel = params.kernel ?? 'rbf';
    const degree = Number(params.degree ?? 3);
    const rand = mulberry32(seed);

    // Initialize sample points with non-zero alphas
    const svItems: SupportVectorItem[] = [];
    const svIds: number[] = [];
    const sampleCount = Math.min(6, Math.max(2, Math.floor(N * 0.15)));
    for (let k = 0; k < sampleCount && k < N; k++) {
      const idx = Math.floor(rand() * N);
      if (!alphas[idx]) {
        const pt = dataset.train[idx];
        const val = 0.4 + rand() * 0.4;
        alphas[idx] = val;
        svIds.push(pt.id);
        svItems.push({
          id: pt.id,
          x1: pt.x1,
          x2: pt.x2,
          y: binarizeTarget(pt.y),
          alpha: val,
        });
      }
    }

    return {
      alphas,
      bias: 0,
      epoch: 0,
      supportVectors: svItems,
      supportVectorIds: svIds,
      supportVectorCount: svIds.length,
      gamma,
      kernel,
      degree,
      marginWidth: 1.0,
    };
  },

  step(
    state: KernelSVMState,
    dataset: DatasetContainer,
    params: KernelSVMParams
  ): StepResult<KernelSVMState> {
    const kernel = params.kernel ?? state.kernel ?? 'rbf';
    const gamma = Number(params.gamma ?? state.gamma ?? 0.5);
    const degree = Number(params.degree ?? state.degree ?? 3);
    const C = Number(params.cPenalty ?? 1.0);

    const train = dataset.train;
    const N = train.length;
    if (N === 0) {
      return { state, epoch: state.epoch, metrics: this.loss(state, dataset, params) };
    }

    const alphas = new Float32Array(N);
    for (let i = 0; i < Math.min(alphas.length, state.alphas.length); i++) {
      alphas[i] = state.alphas[i];
    }
    let b = state.bias;

    // Kernel evaluation helper
    const kVal = (x1: number, y1: number, x2: number, y2: number) => {
      return kernel === 'rbf'
        ? rbfKernel(x1, y1, x2, y2, gamma)
        : polyKernel(x1, y1, x2, y2, degree, 1.0);
    };

    // Precompute diagonal K(x_i, x_i)
    const diagK = new Float32Array(N);
    for (let i = 0; i < N; i++) {
      diagK[i] = Math.max(1e-4, kVal(train[i].x1, train[i].x2, train[i].x1, train[i].x2));
    }

    // Exact Sequential Coordinate Descent passes
    for (let pass = 0; pass < 6; pass++) {
      for (let i = 0; i < N; i++) {
        const pi = train[i];
        const yi = binarizeTarget(pi.y);

        // Compute f_{-i}(x_i) excluding alpha_i
        let f_minus_i = b;
        for (let j = 0; j < N; j++) {
          if (j !== i && alphas[j] > 1e-5) {
            const pj = train[j];
            const yj = binarizeTarget(pj.y);
            f_minus_i += alphas[j] * yj * kVal(pj.x1, pj.x2, pi.x1, pi.x2);
          }
        }

        // Optimal coordinate step: alpha_i = (1 - yi * f_{-i}) / K(xi, xi)
        const unconstrainedAlpha = (1.0 - yi * f_minus_i) / diagK[i];
        const nextAlpha = Math.max(0.0, Math.min(C, unconstrainedAlpha));

        // Smooth update
        alphas[i] = alphas[i] * 0.3 + nextAlpha * 0.7;
        if (alphas[i] < 1e-4) alphas[i] = 0.0;
      }

      // Update bias b from free support vectors (0 < alpha < C)
      let biasSum = 0;
      let freeCount = 0;
      for (let i = 0; i < N; i++) {
        if (alphas[i] > 1e-3 && alphas[i] < C - 1e-3) {
          const pi = train[i];
          const yi = binarizeTarget(pi.y);
          let kSum = 0;
          for (let j = 0; j < N; j++) {
            if (alphas[j] > 1e-5) {
              const pj = train[j];
              const yj = binarizeTarget(pj.y);
              kSum += alphas[j] * yj * kVal(pj.x1, pj.x2, pi.x1, pi.x2);
            }
          }
          biasSum += yi - kSum;
          freeCount++;
        }
      }
      if (freeCount > 0) {
        b = biasSum / freeCount;
      }
    }

    // Identify active support vectors (alpha > 1e-3)
    const svItems: SupportVectorItem[] = [];
    const svIds: number[] = [];
    let normSq = 0;
    for (let i = 0; i < N; i++) {
      if (alphas[i] > 1e-3) {
        const pt = train[i];
        const yi = binarizeTarget(pt.y);
        svIds.push(pt.id);
        svItems.push({
          id: pt.id,
          x1: pt.x1,
          x2: pt.x2,
          y: yi,
          alpha: alphas[i],
        });
        for (let j = 0; j < N; j++) {
          if (alphas[j] > 1e-4) {
            const yj = binarizeTarget(train[j].y);
            normSq += alphas[i] * alphas[j] * yi * yj * kVal(train[i].x1, train[i].x2, train[j].x1, train[j].x2);
          }
        }
      }
    }

    const marginWidth = normSq > 1e-6 ? 2.0 / Math.sqrt(normSq) : 1.0;

    const nextState: KernelSVMState = {
      alphas,
      bias: b,
      epoch: state.epoch + 1,
      supportVectors: svItems,
      supportVectorIds: svIds,
      supportVectorCount: svIds.length,
      gamma,
      kernel,
      degree,
      marginWidth,
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
    };
  },

  predict(input: Float32Array, state: KernelSVMState, _params: KernelSVMParams): number {
    if (!state || !state.supportVectors) return 0;
    let f = state.bias;
    const isRBF = state.kernel === 'rbf';
    const gamma = state.gamma;
    const degree = state.degree;

    for (let i = 0; i < state.supportVectors.length; i++) {
      const sv = state.supportVectors[i];
      const k = isRBF
        ? rbfKernel(sv.x1, sv.x2, input[0], input[1], gamma)
        : polyKernel(sv.x1, sv.x2, input[0], input[1], degree, 1.0);
      f += sv.alpha * sv.y * k;
    }
    return f;
  },

  loss(state: KernelSVMState, dataset: DatasetContainer, params: KernelSVMParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;

    const evalAcc = (pts: typeof train) => {
      if (pts.length === 0) return { loss: 0, acc: 0 };
      let correct = 0;
      let hinge = 0;

      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        const y = binarizeTarget(p.y);
        const input = new Float32Array([p.x1, p.x2]);
        const f = Number(this.predict(input, state, params));
        const predClass = f >= 0 ? 1 : -1;
        if (predClass === y) correct++;
        const m = y * f;
        if (m < 1) hinge += 1.0 - m;
      }
      return {
        loss: hinge / pts.length,
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
      marginWidth: state.marginWidth,
      sparsity: {
        zeroCount: train.length - state.supportVectorCount,
        totalCount: train.length,
      },
    };
  },

  getVisualArtifacts(
    state: KernelSVMState,
    _dataset: DatasetContainer,
    _params: KernelSVMParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    if (!state) return artifacts;

    // Decision boundary f(x) = 0 and ±1 RKHS margin contours evaluated via gridValues
    artifacts.push({
      type: 'decision-contour',
      level: 0,
      showMargins: true,
      label: `Boundary (${state.kernel.toUpperCase()}): f(x) = 0`,
    });

    if (state.supportVectorIds && state.supportVectorIds.length > 0) {
      // Bold rings around active support vectors (alpha_i > 0)
      artifacts.push({
        type: 'support-vector-rings',
        pointIds: state.supportVectorIds,
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: KernelSVMState): Float32Array {
    const buf = new Float32Array(state.alphas.length + 5);
    buf[0] = state.bias;
    buf[1] = state.epoch;
    buf[2] = state.gamma;
    buf[3] = state.kernel === 'rbf' ? 1 : 2;
    buf[4] = state.marginWidth;
    for (let i = 0; i < state.alphas.length; i++) {
      buf[5 + i] = state.alphas[i];
    }
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: KernelSVMState): KernelSVMState {
    const alphas = new Float32Array(buffer.length - 5);
    for (let i = 0; i < alphas.length; i++) {
      alphas[i] = buffer[5 + i];
    }
    return {
      ...state,
      alphas,
      bias: buffer[0],
      epoch: Math.round(buffer[1]),
      gamma: buffer[2],
      kernel: buffer[3] === 1 ? 'rbf' : 'poly',
      marginWidth: buffer[4],
    };
  },

  explain(
    state: KernelSVMState,
    params: KernelSVMParams,
    lastStep?: StepResult<KernelSVMState>
  ): ExplanationPayload {
    const svCount = state.supportVectorCount;
    const gamma = Number(params.gamma ?? state.gamma ?? 0.5).toFixed(2);
    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';

    return {
      summary: `Kernel SVM maps 2D data into an infinite-dimensional RKHS via ${state.kernel.toUpperCase()} kernel, isolating ${svCount} active support vectors with ${acc}% accuracy.`,
      coreAssumption:
        'Maximum margin separation in RKHS controls structural risk; non-support vectors (outside ±1 margin) exert zero pull on the decision boundary.',
      whenItFails:
        'Excessively high gamma (γ ≥ 8.0) causes shrink-wrap overfitting into disconnected islands; high C on noisy datasets fits rogue label noise.',
      microLog: [
        `Kernel evaluation: ${state.kernel.toUpperCase()} with bandwidth parameter γ = ${gamma}`,
        `Dual sparsity: ${svCount} support vectors (α_i > 0) determine decision topology`,
        `Effective RKHS margin street width γ_margin = ${state.marginWidth.toFixed(3)}`,
      ],
      mathLatex: {
        objective: '\\max_{\\boldsymbol{\\alpha}} \\sum_{i=1}^N \\alpha_i - \\frac{1}{2} \\sum_{i,j=1}^N \\alpha_i \\alpha_j y_i y_j K(\\mathbf{x}_i, \\mathbf{x}_j) \\quad \\text{s.t.} \\; 0 \\le \\alpha_i \\le C',
        lossOrUpdate: 'f(\\mathbf{x}) = \\sum_{i=1}^N \\alpha_i y_i K(\\mathbf{x}_i, \\mathbf{x}) + b \\quad (\\text{Only } \\alpha_i > 0 \\text{ contribute})',
        dualOrConstraint: "K_{\\text{RBF}}(\\mathbf{x}, \\mathbf{x}') = \\exp(-\\gamma \\|\\mathbf{x} - \\mathbf{x}'\\|^2)",
      },
      codeSnippet: {
        sklearn: `from sklearn.svm import SVC\n\nmodel = SVC(kernel='${state.kernel}', gamma=${gamma}, C=${params.cPenalty ?? 1.0})\nmodel.fit(X_train, y_train)\nprint(f"Support Vector Count: {len(model.support_)}")`,
        pytorch: `# Kernel SVM via Gram Matrix Inversion / QP Solver\n# In deep learning, RBF layers approximate RKHS projections:\nimport torch\nimport torch.nn as nn\n\nclass RBFLayer(nn.Module):\n    def __init__(self, centers, gamma=${gamma}):\n        super().__init__()\n        self.centers = nn.Parameter(centers)\n        self.gamma = gamma\n    def forward(self, x):\n        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)\n        return torch.exp(-self.gamma * torch.sum(diff**2, dim=-1))`,
      },
    };
  },
};
