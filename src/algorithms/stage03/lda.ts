import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { eigen2x2Symmetric } from './clusteringUtils';
import { invertMatrix } from '../../utils/math';

export interface LDAState {
  shrinkage: number;
  classesCount: number;
  classMeans: { label: number; mean: [number, number]; count: number }[];
  overallMean: [number, number];
  wAxis: [number, number]; // Fisher discriminant axis
  pcaAxis: [number, number]; // Ghost PCA axis for comparison
  rayleighRatio: number;
  projected1D: { pointId: number; value: number; label: number }[];
  dropLines: { from: [number, number]; to: [number, number]; pointId: number; label: number }[];
  swEllipses: { center: [number, number]; radiusX: number; radiusY: number; angle: number; classLabel: number }[];
  sbConnector: { from: [number, number]; to: [number, number] };
  epoch: number;
  converged: boolean;
}

export interface LDAParams {
  shrinkage?: number;
}

function computeLDA(
  points: { id: number; x1: number; x2: number; y: number }[],
  shrinkage: number
): LDAState {
  const N = points.length;
  if (N < 2) {
    return {
      shrinkage,
      classesCount: 1,
      classMeans: [],
      overallMean: [0, 0],
      wAxis: [1, 0],
      pcaAxis: [1, 0],
      rayleighRatio: 0,
      projected1D: [],
      dropLines: [],
      swEllipses: [],
      sbConnector: { from: [0, 0], to: [0, 0] },
      epoch: 0,
      converged: true,
    };
  }

  // 1. Group points by class (or quantile slices for continuous regression)
  const isContinuous = points.some((p) => p.y !== 1 && p.y !== -1 && p.y !== 2);
  const classMap = new Map<number, { id: number; x1: number; x2: number; y: number }[]>();
  let sumAllX = 0;
  let sumAllY = 0;

  for (const p of points) {
    sumAllX += p.x1;
    sumAllY += p.x2;
  }

  if (isContinuous) {
    // Binarize continuous regression targets cleanly into 2 classes (y >= 0 vs y < 0)
    // matching the 2-color point rendering on 1D Sine Wave and other regression datasets.
    for (const p of points) {
      const binLabel = p.y >= 0 ? 1 : -1;
      const list = classMap.get(binLabel) || [];
      list.push(p);
      classMap.set(binLabel, list);
    }
  } else {
    for (const p of points) {
      const list = classMap.get(p.y) || [];
      list.push(p);
      classMap.set(p.y, list);
    }
  }

  const overallMean: [number, number] = [sumAllX / N, sumAllY / N];
  const uniqueLabels = Array.from(classMap.keys()).sort((a, b) => a - b);

  // Compute Class Means
  const classMeans: LDAState['classMeans'] = [];
  uniqueLabels.forEach((label) => {
    const pts = classMap.get(label)!;
    let sx = 0;
    let sy = 0;
    for (const p of pts) {
      sx += p.x1;
      sy += p.x2;
    }
    classMeans.push({
      label,
      mean: [sx / pts.length, sy / pts.length],
      count: pts.length,
    });
  });

  // 2. Compute Total Covariance & PCA Ghost Axis
  let tot00 = 0;
  let tot01 = 0;
  let tot11 = 0;
  for (const p of points) {
    const dx = p.x1 - overallMean[0];
    const dy = p.x2 - overallMean[1];
    tot00 += dx * dx;
    tot01 += dx * dy;
    tot11 += dy * dy;
  }
  const pcaEigen = eigen2x2Symmetric(tot00, tot01, tot11);
  const pcaAxis = pcaEigen.vectors[0];

  // 3. Compute Within-Class Scatter S_W
  let sw00 = 0;
  let sw01 = 0;
  let sw11 = 0;
  const swEllipses: LDAState['swEllipses'] = [];

  classMeans.forEach((cm) => {
    const pts = classMap.get(cm.label)!;
    let cSw00 = 0;
    let cSw01 = 0;
    let cSw11 = 0;

    for (const p of pts) {
      const dx = p.x1 - cm.mean[0];
      const dy = p.x2 - cm.mean[1];
      cSw00 += dx * dx;
      cSw01 += dx * dy;
      cSw11 += dy * dy;
    }

    sw00 += cSw00;
    sw01 += cSw01;
    sw11 += cSw11;

    // Class scatter ellipse
    const cEigen = eigen2x2Symmetric(
      cSw00 / Math.max(1, pts.length),
      cSw01 / Math.max(1, pts.length),
      cSw11 / Math.max(1, pts.length)
    );
    swEllipses.push({
      center: cm.mean,
      radiusX: Math.sqrt(Math.max(0.01, cEigen.values[0])) * 1.5,
      radiusY: Math.sqrt(Math.max(0.01, cEigen.values[1])) * 1.5,
      angle: cEigen.angle,
      classLabel: cm.label,
    });
  });

  // Apply Ledoit-Wolf Shrinkage: S_W_shrunk = (1 - lambda) * S_W + lambda * (tr(S_W)/2) * I
  const trSw = (sw00 + sw11) / 2;
  sw00 = (1 - shrinkage) * sw00 + shrinkage * trSw;
  sw11 = (1 - shrinkage) * sw11 + shrinkage * trSw;
  sw01 = (1 - shrinkage) * sw01;

  // Add numerical floor
  sw00 += 1e-4;
  sw11 += 1e-4;

  // Invert S_W
  const invSw = invertMatrix([
    [sw00, sw01],
    [sw01, sw11],
  ]);

  // Compute Between-Class Scatter S_B
  let sb00 = 0;
  let sb01 = 0;
  let sb11 = 0;
  classMeans.forEach((cm) => {
    const dx = cm.mean[0] - overallMean[0];
    const dy = cm.mean[1] - overallMean[1];
    sb00 += cm.count * dx * dx;
    sb01 += cm.count * dx * dy;
    sb11 += cm.count * dy * dy;
  });

  let wAxis: [number, number] = [1, 0];
  let sbConnector: LDAState['sbConnector'] = { from: [0, 0], to: [0, 0] };

  if (classMeans.length >= 2 && invSw) {
    if (classMeans.length === 2) {
      // Fisher 2-class optimal: w = S_W^(-1) * (mu_1 - mu_2)
      const m1 = classMeans[0].mean;
      const m2 = classMeans[1].mean;
      sbConnector = { from: m1, to: m2 };

      const diffX = m1[0] - m2[0];
      const diffY = m1[1] - m2[1];

      const wx = invSw[0][0] * diffX + invSw[0][1] * diffY;
      const wy = invSw[1][0] * diffX + invSw[1][1] * diffY;
      const normW = Math.hypot(wx, wy);

      if (normW > 1e-6) {
        wAxis = [wx / normW, wy / normW];
      }
    } else {
      // Multiclass (C >= 3) or Multi-Slice: Leading eigenvector of S_W^(-1) * S_B
      const m00 = invSw[0][0] * sb00 + invSw[0][1] * sb01;
      const m01 = invSw[0][0] * sb01 + invSw[0][1] * sb11;
      const m10 = invSw[1][0] * sb00 + invSw[1][1] * sb01;
      const m11 = invSw[1][0] * sb01 + invSw[1][1] * sb11;

      const trace = m00 + m11;
      const det = m00 * m11 - m01 * m10;
      const disc = Math.max(0, (trace * trace) / 4 - det);
      const lambda1 = trace / 2 + Math.sqrt(disc);

      let vx = m01;
      let vy = lambda1 - m00;
      if (Math.hypot(vx, vy) < 1e-6) {
        vx = lambda1 - m11;
        vy = m10;
      }
      const normV = Math.hypot(vx, vy);
      if (normV > 1e-6) {
        wAxis = [vx / normV, vy / normV];
      } else {
        wAxis = [1, 0];
      }

      // Connect the two furthest class/slice means
      let maxDist = -1;
      let pair: [[number, number], [number, number]] = [classMeans[0].mean, classMeans[1].mean];
      for (let i = 0; i < classMeans.length; i++) {
        for (let j = i + 1; j < classMeans.length; j++) {
          const d = Math.hypot(
            classMeans[i].mean[0] - classMeans[j].mean[0],
            classMeans[i].mean[1] - classMeans[j].mean[1]
          );
          if (d > maxDist) {
            maxDist = d;
            pair = [classMeans[i].mean, classMeans[j].mean];
          }
        }
      }
      sbConnector = { from: pair[0], to: pair[1] };
    }
  }

  // 4. Compute Rayleigh quotient J(w) = (w^T S_B w) / (w^T S_W w)
  const num = wAxis[0] * (sb00 * wAxis[0] + sb01 * wAxis[1]) + wAxis[1] * (sb01 * wAxis[0] + sb11 * wAxis[1]);
  const den = wAxis[0] * (sw00 * wAxis[0] + sw01 * wAxis[1]) + wAxis[1] * (sw01 * wAxis[0] + sw11 * wAxis[1]);
  const rayleighRatio = den > 1e-6 ? num / den : 0;

  // 5. Projected 1D Points & Drop-Lines
  const projected1D: LDAState['projected1D'] = [];
  const dropLines: LDAState['dropLines'] = [];

  for (const pt of points) {
    const dx = pt.x1 - overallMean[0];
    const dy = pt.x2 - overallMean[1];
    const z = dx * wAxis[0] + dy * wAxis[1];

    projected1D.push({
      pointId: pt.id,
      value: z,
      label: pt.y,
    });

    const reconX = overallMean[0] + z * wAxis[0];
    const reconY = overallMean[1] + z * wAxis[1];

    dropLines.push({
      from: [pt.x1, pt.x2],
      to: [reconX, reconY],
      pointId: pt.id,
      label: pt.y,
    });
  }

  return {
    shrinkage,
    classesCount: uniqueLabels.length,
    classMeans,
    overallMean,
    wAxis,
    pcaAxis,
    rayleighRatio,
    projected1D,
    dropLines,
    swEllipses,
    sbConnector,
    epoch: 0,
    converged: true,
  };
}

export const ldaPlugin: AlgorithmPlugin<LDAState, LDAParams> = {
  id: 'lda',
  name: "Linear Discriminant Analysis (Fisher's Criterion)",
  shortCode: 'LDA',
  stage: 'unsupervised-2d',
  taskType: 'dimensionality-reduction',
  executionMode: 'closed-form',
  description:
    'Supervised linear projection maximizing the Rayleigh quotient between-class scatter over within-class scatter.',

  primaryParams: [
    {
      id: 'shrinkage',
      name: 'Shrinkage (λ)',
      type: 'slider',
      min: 0.0,
      max: 1.0,
      step: 0.05,
      default: 0.0,
      description: 'Ledoit-Wolf covariance shrinkage towards spherical identity matrix.',
    },
  ],

  advancedParams: [],

  init(dataset: DatasetContainer, params: LDAParams, _seed: number = 42): LDAState {
    const shrinkage = Math.max(0, Math.min(1, Number(params.shrinkage ?? 0.0)));
    return computeLDA(dataset.train, shrinkage);
  },

  step(state: LDAState, dataset: DatasetContainer, params: LDAParams): StepResult<LDAState> {
    const shrinkage = Math.max(0, Math.min(1, Number(params.shrinkage ?? state.shrinkage)));
    const nextState = {
      ...computeLDA(dataset.train, shrinkage),
      epoch: state.epoch + 1,
    };
    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics: this.loss(nextState, dataset, params),
      converged: true,
    };
  },

  predict(input: Float32Array, state: LDAState, _params: LDAParams): number {
    const dx = input[0] - state.overallMean[0];
    const dy = input[1] - state.overallMean[1];
    return dx * state.wAxis[0] + dy * state.wAxis[1];
  },

  loss(state: LDAState, _dataset: DatasetContainer, _params: LDAParams): TelemetryMetrics {
    return {
      trainLoss: 1.0 / Math.max(0.01, state.rayleighRatio),
      testLoss: 1.0 / Math.max(0.01, state.rayleighRatio),
      trainAccuracy: Math.min(100, Number((state.rayleighRatio * 15).toFixed(1))),
      custom: {
        fisherRatio: Number(state.rayleighRatio.toFixed(3)),
        classesFound: state.classesCount,
        shrinkage: state.shrinkage,
      },
    };
  },

  getVisualArtifacts(state: LDAState, _dataset: DatasetContainer, _params: LDAParams): VisualArtifact[] {
    const axes = [
      {
        id: 'lda-w',
        direction: state.wAxis,
        length: 4.5,
        label: `Fisher Axis w (J = ${state.rayleighRatio.toFixed(2)})`,
        color: '#DC2626', // Crimson
      },
      {
        id: 'pca-ghost',
        direction: state.pcaAxis,
        length: 4.0,
        label: 'PCA Ghost (Max Variance)',
        color: '#9CA3AF', // Muted Gray
        isGhost: true,
      },
    ];

    return [
      {
        type: 'projection-overlay',
        method: 'lda',
        center: state.overallMean,
        axes,
        dropLines: state.dropLines,
        projected1D: state.projected1D,
        ldaScatter: {
          swEllipses: state.swEllipses,
          sbConnector: state.sbConnector,
          rayleighScore: state.rayleighRatio,
        },
      },
    ];
  },

  serializeSnapshot(state: LDAState): Float32Array {
    const buf = new Float32Array(8);
    buf[0] = state.epoch;
    buf[1] = state.wAxis[0];
    buf[2] = state.wAxis[1];
    buf[3] = state.pcaAxis[0];
    buf[4] = state.pcaAxis[1];
    buf[5] = state.rayleighRatio;
    buf[6] = state.shrinkage;
    buf[7] = state.classesCount;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: LDAState): LDAState {
    return {
      ...state,
      epoch: Math.round(buffer[0]),
      wAxis: [buffer[1], buffer[2]],
      pcaAxis: [buffer[3], buffer[4]],
      rayleighRatio: buffer[5],
      shrinkage: buffer[6],
      classesCount: Math.round(buffer[7]),
    };
  },

  explain(state: LDAState, _params: LDAParams, _lastStep?: StepResult<LDAState>): ExplanationPayload {
    return {
      summary: `Linear Discriminant Analysis finds the projection axis w that maximizes between-class separation while minimizing within-class scatter (Rayleigh ratio J(w) = ${state.rayleighRatio.toFixed(2)}).`,
      coreAssumption:
        'Classes are normally distributed with equal covariance matrices; linear boundaries suffice to separate class distributions.',
      whenItFails:
        'Nonlinear class boundaries (e.g. concentric circles or spirals) and severe covariance heterogeneity between classes (resolved via QDA).',
      microLog: [
        `Compute within-class scatter S_W and between-class scatter S_B for ${state.classesCount} classes`,
        `Solve Fisher Rayleigh quotient: w ∝ S_W^(-1) (μ₁ - μ₂)`,
        `Contrast Fisher discriminant axis against dashed PCA ghost axis to highlight variance vs separation angle`,
      ],
      mathLatex: {
        objective: `J(w) = \\frac{w^T S_B w}{w^T S_W w}, \\quad S_B = \\sum_c N_c (\\mu_c - \\mu)(\\mu_c - \\mu)^T`,
        lossOrUpdate: `S_W = \\sum_c \\sum_{x \\in C_c} (x - \\mu_c)(x - \\mu_c)^T, \\quad w^* \\propto S_W^{-1}(\\mu_1 - \\mu_2)`,
      },
      codeSnippet: {
        sklearn: `from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nlda = LinearDiscriminantAnalysis(shrinkage=${state.shrinkage})\nX_lda = lda.fit_transform(X, y)`,
        pytorch: `# Generalized Eigendecomposition S_B v = lambda S_W v\n# via torch.linalg.eigh(torch.linalg.solve(S_W, S_B))`,
      },
    };
  },
};

