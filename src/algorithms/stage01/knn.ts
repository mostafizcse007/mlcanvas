import {
  AlgorithmPlugin,
  DatasetContainer,
  StepResult,
  TelemetryMetrics,
  VisualArtifact,
  ExplanationPayload,
} from '../../types/algorithm';
import { DataPoint2D } from '../../datasets/tabular2d';
import { euclideanDist, manhattanDist } from '../../utils/math';
import { binarizeTarget, isContinuousRegressionTargets } from './crossTask';
import { is1DRegressionDataset } from './linearRegression';

export interface KNNState {
  k: number;
  metric: 'euclidean' | 'manhattan';
  weighting: 'uniform' | 'distance';
  phase: 'compute-distances' | 'rank-top-k' | 'vote';
  epoch: number;
  probePoint: [number, number];
  neighborRays: { id: number; point: [number, number]; distance: number; weight: number; label: number }[];
  boundingRadius: number;
  isRegression?: boolean;
  is1D?: boolean;
  curvePoints?: [number, number][];
  residuals?: { id: number; x1: number; x2: number; y: number; yHat: number; error: number }[];
  trainingPoints?: DataPoint2D[];
}

export interface KNNParams {
  neighbors?: number;
  metric?: 'euclidean' | 'manhattan';
  weighting?: 'uniform' | 'distance';
  probeX1?: number;
  probeX2?: number;
}

function computeKnnDistance(
  x1: number,
  x2: number,
  p: DataPoint2D,
  metric: 'euclidean' | 'manhattan',
  is1D: boolean
): number {
  if (is1D) {
    return Math.abs(x1 - p.x1);
  }
  return metric === 'euclidean'
    ? euclideanDist(x1, x2, p.x1, p.x2)
    : manhattanDist(x1, x2, p.x1, p.x2);
}

function evaluateKnnAtPoint(
  x1: number,
  x2: number,
  train: DataPoint2D[],
  k: number,
  metric: 'euclidean' | 'manhattan',
  weighting: 'uniform' | 'distance',
  isRegression: boolean,
  is1D: boolean,
  excludeId?: number
): {
  value: number;
  neighbors: KNNState['neighborRays'];
  boundingRadius: number;
} {
  const candidates: { id: number; point: [number, number]; distance: number; y: number }[] = [];
  for (let i = 0; i < train.length; i++) {
    const p = train[i];
    if (excludeId !== undefined && p.id === excludeId && train.length > 1) {
      continue;
    }
    const d = computeKnnDistance(x1, x2, p, metric, is1D);
    candidates.push({ id: p.id, point: [p.x1, p.x2], distance: d, y: p.y });
  }

  if (candidates.length === 0) {
    return { value: 0, neighbors: [], boundingRadius: 1.0 };
  }

  candidates.sort((a, b) => a.distance - b.distance);
  const actualK = Math.min(Math.max(1, k), candidates.length);

  let weightedSum = 0;
  let weightTotal = 0;
  const neighbors: KNNState['neighborRays'] = [];

  for (let i = 0; i < actualK; i++) {
    const c = candidates[i];
    const w = weighting === 'distance' ? 1.0 / Math.max(1e-4, c.distance) : 1.0;
    const targetVal = isRegression ? c.y : binarizeTarget(c.y);
    weightedSum += w * targetVal;
    weightTotal += w;
    neighbors.push({
      id: c.id,
      point: c.point,
      distance: c.distance,
      weight: w,
      label: c.y,
    });
  }

  const value = weightTotal > 0 ? weightedSum / weightTotal : 0;
  const boundingRadius = neighbors.length > 0 ? neighbors[neighbors.length - 1].distance : 1.0;
  return { value, neighbors, boundingRadius };
}

function build1DKnnCurveAndResiduals(
  train: DataPoint2D[],
  k: number,
  metric: 'euclidean' | 'manhattan',
  weighting: 'uniform' | 'distance'
): {
  curvePoints: [number, number][];
  residuals: NonNullable<KNNState['residuals']>;
} {
  const curvePoints: [number, number][] = [];
  if (train.length > 0) {
    const steps = 160;
    for (let i = 0; i <= steps; i++) {
      const x1 = -6.0 + (i / steps) * 12.0;
      const { value } = evaluateKnnAtPoint(x1, 0, train, k, metric, weighting, true, true);
      curvePoints.push([x1, Math.max(-20, Math.min(20, value))]);
    }
  }

  const residuals = train.map((pt) => {
    const { value: yHat } = evaluateKnnAtPoint(pt.x1, pt.x2, train, k, metric, weighting, true, true);
    return {
      id: pt.id,
      x1: pt.x1,
      x2: pt.x2,
      y: pt.y,
      yHat,
      error: pt.y - yHat,
    };
  });

  return { curvePoints, residuals };
}

export function knnPredictWithDataset(
  input: Float32Array,
  state: KNNState,
  dataset: DatasetContainer
): number {
  const train = dataset?.train?.length ? dataset.train : (state?.trainingPoints ?? []);
  if (!train || train.length === 0) return 0;

  const isRegression =
    state.isRegression ?? isContinuousRegressionTargets({ train, test: dataset?.test ?? [] });
  const is1D =
    state.is1D ?? (isRegression && is1DRegressionDataset({ train, test: dataset?.test ?? [] }));

  const { value } = evaluateKnnAtPoint(
    input[0],
    input[1],
    train,
    state.k,
    state.metric,
    state.weighting,
    isRegression,
    is1D
  );
  return value;
}

export const knnPlugin: AlgorithmPlugin<KNNState, KNNParams> = {
  id: 'knn',
  name: 'k-Nearest Neighbors (Lazy Instance Learner)',
  shortCode: 'k-NN',
  stage: 'continuous-2d',
  taskType: 'classification',
  executionMode: 'phase-walkthrough',
  description: 'Non-parametric local metric voting and mean regression with interactive probe rays, distance weighting, and metric distortion.',
  paperReference: 'Fix & Hodges, Discriminatory Analysis: Nonparametric Discrimination, 1951; Cover & Hart, 1967',

  primaryParams: [
    {
      id: 'neighbors',
      name: 'Neighbors (K)',
      type: 'slider',
      min: 1,
      max: 25,
      step: 2, // Prefer odd numbers to break ties
      default: 5,
      description: 'Number of nearest prototype instances consulted for local voting or mean regression.',
    },
    {
      id: 'metric',
      name: 'Distance Metric',
      type: 'select',
      options: [
        { label: 'Euclidean (L2 Round Ball)', value: 'euclidean' },
        { label: 'Manhattan (L1 Diamond Metric)', value: 'manhattan' },
      ],
      default: 'euclidean',
      description: 'Norm metric measuring geometric distance in the feature manifold.',
    },
    {
      id: 'weighting',
      name: 'Voting Weight',
      type: 'select',
      options: [
        { label: 'Uniform (1 vote per neighbor)', value: 'uniform' },
        { label: 'Distance-Weighted (1 / d)', value: 'distance' },
      ],
      default: 'uniform',
      description: 'Weighting scheme applied to neighbor votes or target values.',
    },
  ],

  advancedParams: [],

  symbolicLinks: [
    {
      id: 'knn',
      symbolLatex: '\\hat{y}(\\mathbf{x}) = \\text{mode} \\{ y_i : \\mathbf{x}_i \\in \\mathcal{N}_K(\\mathbf{x}) \\}',
      label: 'Local Majority Voting',
      description: 'Assigns class label or local weighted mean within the K-neighborhood ball.',
      highlightTarget: 'decision-hyperplane',
    },
  ],

  init(dataset: DatasetContainer, params: KNNParams, _seed: number): KNNState {
    const k = Number(params.neighbors ?? 5);
    const metric = params.metric ?? 'euclidean';
    const weighting = params.weighting ?? 'uniform';
    const probeX1 = Number(params.probeX1 ?? 0.0);
    const probeX2 = Number(params.probeX2 ?? 0.0);

    const train = dataset.train;
    const isRegression = isContinuousRegressionTargets(dataset);
    const is1D = isRegression && is1DRegressionDataset(dataset);

    const { neighbors, boundingRadius } = evaluateKnnAtPoint(
      probeX1,
      probeX2,
      train,
      k,
      metric,
      weighting,
      isRegression,
      is1D
    );

    const oneD = isRegression && is1D
      ? build1DKnnCurveAndResiduals(train, k, metric, weighting)
      : { curvePoints: undefined, residuals: undefined };

    return {
      k,
      metric,
      weighting,
      phase: 'vote',
      epoch: 0,
      probePoint: [probeX1, probeX2],
      neighborRays: neighbors,
      boundingRadius,
      isRegression,
      is1D,
      curvePoints: oneD.curvePoints,
      residuals: oneD.residuals,
      trainingPoints: train.map((p) => ({ ...p })),
    };
  },

  step(
    state: KNNState,
    dataset: DatasetContainer,
    params: KNNParams
  ): StepResult<KNNState> {
    const k = Number(params.neighbors ?? state.k ?? 5);
    const metric = params.metric ?? state.metric ?? 'euclidean';
    const weighting = params.weighting ?? state.weighting ?? 'uniform';
    const probeX1 = Number(params.probeX1 ?? state.probePoint[0]);
    const probeX2 = Number(params.probeX2 ?? state.probePoint[1]);

    // Phase cycle: compute-distances -> rank-top-k -> vote
    const nextPhase: KNNState['phase'] =
      state.phase === 'compute-distances'
        ? 'rank-top-k'
        : state.phase === 'rank-top-k'
        ? 'vote'
        : 'compute-distances';

    const train = dataset.train;
    const isRegression = isContinuousRegressionTargets(dataset);
    const is1D = isRegression && is1DRegressionDataset(dataset);

    const { neighbors, boundingRadius } = evaluateKnnAtPoint(
      probeX1,
      probeX2,
      train,
      k,
      metric,
      weighting,
      isRegression,
      is1D
    );

    const oneD = isRegression && is1D
      ? build1DKnnCurveAndResiduals(train, k, metric, weighting)
      : { curvePoints: undefined, residuals: undefined };

    const nextState: KNNState = {
      k,
      metric,
      weighting,
      phase: nextPhase,
      epoch: state.epoch + 1,
      probePoint: [probeX1, probeX2],
      neighborRays: neighbors,
      boundingRadius,
      isRegression,
      is1D,
      curvePoints: oneD.curvePoints,
      residuals: oneD.residuals,
      trainingPoints: train.map((p) => ({ ...p })),
    };

    const metrics = this.loss(nextState, dataset, params);

    return {
      state: nextState,
      epoch: nextState.epoch,
      metrics,
    };
  },

  predict(input: Float32Array, state: KNNState, _params: KNNParams): number {
    if (!state) return 0;
    if (state.trainingPoints && state.trainingPoints.length > 0) {
      return knnPredictWithDataset(input, state, { train: state.trainingPoints, test: [] });
    }
    if (!state.neighborRays || state.neighborRays.length === 0) return 0;

    const isRegression = Boolean(state.isRegression);
    let weightedSum = 0;
    let weightTotal = 0;
    for (const n of state.neighborRays) {
      const w = n.weight || 1.0;
      weightedSum += w * (isRegression ? n.label : binarizeTarget(n.label));
      weightTotal += w;
    }
    return weightTotal > 0 ? weightedSum / weightTotal : 0;
  },

  loss(state: KNNState, dataset: DatasetContainer, _params: KNNParams): TelemetryMetrics {
    const train = dataset.train;
    const test = dataset.test;
    const isRegression = state.isRegression ?? isContinuousRegressionTargets(dataset);
    const is1D = state.is1D ?? (isRegression && is1DRegressionDataset(dataset));

    if (isRegression) {
      const evalReg = (evalSet: DataPoint2D[], isTrainSet: boolean) => {
        if (evalSet.length === 0 || train.length === 0) {
          return { mse: 0, r2: 1.0, acc: 100.0 };
        }
        let sse = 0;
        let sumY = 0;
        for (let i = 0; i < evalSet.length; i++) sumY += evalSet[i].y;
        const meanY = sumY / evalSet.length;
        let sst = 0;
        let correct = 0;

        for (let i = 0; i < evalSet.length; i++) {
          const pt = evalSet[i];
          const { value: yHat } = evaluateKnnAtPoint(
            pt.x1,
            pt.x2,
            train,
            state.k,
            state.metric,
            state.weighting,
            true,
            is1D,
            isTrainSet ? pt.id : undefined
          );
          const err = pt.y - yHat;
          sse += err * err;
          sst += (pt.y - meanY) ** 2;
          if (binarizeTarget(yHat) === binarizeTarget(pt.y)) correct++;
        }

        const mse = sse / evalSet.length;
        const r2 = sst > 1e-8 ? Math.max(-1.0, 1.0 - sse / sst) : (sse < 1e-8 ? 1.0 : 0.0);
        const acc = (correct / evalSet.length) * 100;
        return { mse, r2, acc };
      };

      const trainMetrics = evalReg(train, true);
      const testMetrics = evalReg(test, false);

      return {
        trainLoss: trainMetrics.mse,
        testLoss: testMetrics.mse,
        trainR2: trainMetrics.r2,
        testR2: testMetrics.r2,
        trainAccuracy: trainMetrics.acc,
        testAccuracy: testMetrics.acc,
        custom: {
          probeRadius: state.boundingRadius,
        },
      };
    }

    const evalAcc = (evalSet: DataPoint2D[], isTrainSet: boolean) => {
      if (evalSet.length === 0 || train.length === 0) return 0;
      let correct = 0;

      for (let i = 0; i < evalSet.length; i++) {
        const pt = evalSet[i];
        const { value: vote } = evaluateKnnAtPoint(
          pt.x1,
          pt.x2,
          train,
          state.k,
          state.metric,
          state.weighting,
          false,
          false,
          isTrainSet ? pt.id : undefined
        );

        if (binarizeTarget(vote) === binarizeTarget(pt.y)) correct++;
      }
      return (correct / evalSet.length) * 100;
    };

    const trainAcc = evalAcc(train, true);
    const testAcc = evalAcc(test, false);

    return {
      trainLoss: 100 - trainAcc,
      testLoss: 100 - testAcc,
      trainAccuracy: trainAcc,
      testAccuracy: testAcc,
      custom: {
        probeRadius: state.boundingRadius,
      },
    };
  },

  getVisualArtifacts(
    state: KNNState,
    _dataset: DatasetContainer,
    _params: KNNParams
  ): VisualArtifact[] {
    const artifacts: VisualArtifact[] = [];
    if (!state) return artifacts;

    if (state.isRegression && state.is1D) {
      if (state.curvePoints && state.curvePoints.length > 1) {
        artifacts.push({
          type: 'curve-fit',
          points: state.curvePoints,
          color: '#141413',
          style: 'solid',
          label: `Fit: k-NN Mean Regression (K=${state.k}, ${state.weighting})`,
        });
      }
      if (state.residuals && state.residuals.length > 0) {
        artifacts.push({
          type: 'residual-bars',
          residuals: state.residuals,
        });
      }
    } else if (state.isRegression && !state.is1D) {
      artifacts.push({
        type: 'curve-fit',
        points: [],
        color: '#141413',
        style: 'solid',
        label: `Fit Surface: k-NN (K=${state.k}, ${state.weighting})`,
      });
    }

    if (state.neighborRays) {
      // Interactive Cursor Probe Rays with distance labels and boundary ring
      artifacts.push({
        type: 'knn-probe-rays',
        probePoint: state.probePoint,
        neighbors: state.neighborRays,
        metric: state.metric,
        radius: state.boundingRadius,
      });
    }

    return artifacts;
  },

  serializeSnapshot(state: KNNState): Float32Array {
    const buf = new Float32Array(7);
    buf[0] = state.k;
    buf[1] = state.metric === 'euclidean' ? 1 : 2;
    buf[2] = state.weighting === 'uniform' ? 1 : 2;
    buf[3] = state.epoch;
    buf[4] = state.boundingRadius;
    buf[5] = state.isRegression ? 1 : 0;
    buf[6] = state.is1D ? 1 : 0;
    return buf;
  },

  restoreSnapshot(buffer: Float32Array, state: KNNState): KNNState {
    return {
      ...state,
      k: Math.round(buffer[0]),
      metric: buffer[1] === 1 ? 'euclidean' : 'manhattan',
      weighting: buffer[2] === 1 ? 'uniform' : 'distance',
      epoch: Math.round(buffer[3]),
      boundingRadius: buffer[4],
      isRegression: buffer.length > 5 ? buffer[5] === 1 : state.isRegression,
      is1D: buffer.length > 6 ? buffer[6] === 1 : state.is1D,
    };
  },

  explain(
    state: KNNState,
    _params: KNNParams,
    lastStep?: StepResult<KNNState>
  ): ExplanationPayload {
    const radius = state.boundingRadius.toFixed(3);
    if (state.isRegression) {
      const r2 = lastStep?.metrics.trainR2 !== undefined ? lastStep.metrics.trainR2.toFixed(3) : '0.000';
      const mse = lastStep?.metrics.trainLoss !== undefined ? lastStep.metrics.trainLoss.toFixed(4) : '0.0000';
      return {
        summary: `k-Nearest Neighbors Regression estimates ŷ(x) as the ${state.weighting === 'distance' ? 'inverse-distance weighted' : 'uniform'} mean of the K = ${state.k} closest observations${state.is1D ? ' along X₁' : ''}, achieving R² = ${r2} (MSE = ${mse}).`,
        coreAssumption:
          'Local smoothness: target response y varies smoothly over small neighborhoods measured by the chosen feature distance metric.',
        whenItFails:
          'Cannot extrapolate beyond the observed min/max training range at domain boundaries; high K oversmooths peaks while K = 1 memorizes observational noise.',
        microLog: [
          `Phase: [${state.phase.toUpperCase()}] - Querying ${state.is1D ? `1D coordinate X₁ = ${state.probePoint[0].toFixed(2)}` : `2D coordinate (${state.probePoint[0].toFixed(2)}, ${state.probePoint[1].toFixed(2)})`}`,
          `Identified ${state.neighborRays.length} nearest targets within ${state.is1D ? '1D interval' : state.metric} radius R = ${radius}`,
          `Target aggregation: ${state.weighting === 'distance' ? '1/d weighted local mean' : 'Uniform 1/K neighbor mean'}`,
        ],
        mathLatex: {
          objective: state.is1D
            ? 'd(x_1, z_1) = |x_1 - z_1|, \\quad \\mathcal{N}_K(x_1) = \\text{top-}K\\text{ nearest observations}'
            : 'd_{\\text{Euclidean}}(\\mathbf{x}, \\mathbf{z}) = \\sqrt{\\sum_{j=1}^2 (x_j - z_j)^2}, \\quad d_{\\text{Manhattan}}(\\mathbf{x}, \\mathbf{z}) = \\sum_{j=1}^2 |x_j - z_j|',
          lossOrUpdate: '\\hat{y}(\\mathbf{x}) = \\frac{\\sum_{i \\in \\mathcal{N}_K(\\mathbf{x})} w_i y_i}{\\sum_{i \\in \\mathcal{N}_K(\\mathbf{x})} w_i}',
          dualOrConstraint: 'w_i = \\begin{cases} 1 & (\\text{Uniform}) \\\\ 1 / d(\\mathbf{x}, \\mathbf{x}_i) & (\\text{Distance-Weighted}) \\end{cases}',
        },
        codeSnippet: {
          sklearn: `from sklearn.neighbors import KNeighborsRegressor\n\nmodel = KNeighborsRegressor(n_neighbors=${state.k}, metric='${state.metric}', weights='${state.weighting}')\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`,
          pytorch: `# PyTorch Vectorized KNN Regression\nimport torch\n\ndef knn_regress(X_train, y_train, x_query, k=${state.k}):\n    dists = torch.cdist(x_query, X_train, p=${state.metric === 'euclidean' ? 2 : 1})\n    _, top_indices = torch.topk(dists, k=k, largest=False)\n    return y_train[top_indices].mean(dim=-1)`,
        },
      };
    }

    const acc = lastStep?.metrics.trainAccuracy?.toFixed(1) ?? '0.0';

    return {
      summary: `k-Nearest Neighbors is a lazy, instance-based non-parametric classifier consulting K = ${state.k} closest samples, currently achieving ${acc}% accuracy.`,
      coreAssumption:
        'Smooth metric locality: instances close in feature space under chosen norm (Euclidean/Manhattan) share identical class conditional labels.',
      whenItFails:
        'Susceptible to the curse of dimensionality and anisotropic feature scaling: unscaled features with large variance dominate distances and flip rankings.',
      microLog: [
        `Phase: [${state.phase.toUpperCase()}] - Querying probe coordinate (${state.probePoint[0].toFixed(2)}, ${state.probePoint[1].toFixed(2)})`,
        `Identified ${state.neighborRays.length} neighbors within ${state.metric} radius R = ${radius}`,
        `Class vote: ${state.weighting === 'distance' ? 'Inverse distance weighted' : 'Uniform majority consensus'}`,
      ],
      mathLatex: {
        objective: 'd_{\\text{Euclidean}}(\\mathbf{x}, \\mathbf{z}) = \\sqrt{\\sum_{j=1}^2 (x_j - z_j)^2}, \\quad d_{\\text{Manhattan}}(\\mathbf{x}, \\mathbf{z}) = \\sum_{j=1}^2 |x_j - z_j|',
        lossOrUpdate: 'P(y = c | \\mathbf{x}) = \\frac{\\sum_{i \\in \\mathcal{N}_K(\\mathbf{x})} w_i \\cdot \\mathbb{I}(y_i = c)}{\\sum_{i \\in \\mathcal{N}_K(\\mathbf{x})} w_i}',
        dualOrConstraint: 'w_i = \\begin{cases} 1 & (\\text{Uniform}) \\\\ 1 / d(\\mathbf{x}, \\mathbf{x}_i) & (\\text{Distance-Weighted}) \\end{cases}',
      },
      codeSnippet: {
        sklearn: `from sklearn.neighbors import KNeighborsClassifier\n\nmodel = KNeighborsClassifier(n_neighbors=${state.k}, metric='${state.metric}', weights='${state.weighting}')\nmodel.fit(X_train, y_train)\ny_pred = model.predict(X_test)`,
        pytorch: `# PyTorch Vectorized KNN Inference\nimport torch\n\ndef knn_predict(X_train, y_train, x_query, k=${state.k}):\n    dists = torch.cdist(x_query, X_train, p=${state.metric === 'euclidean' ? 2 : 1})\n    _, top_indices = torch.topk(dists, k=k, largest=False)\n    votes = y_train[top_indices]\n    return torch.mode(votes, dim=-1).values`,
      },
    };
  },
};

(knnPlugin as any).predictWithDataset = knnPredictWithDataset;
