import React, { useState, lazy, Suspense } from 'react';
import { Stage01Canvas, is1DRegressionData } from '../rendering/Stage01Canvas';

// Code-split stages 02-07 to optimize initial First Contentful Paint & bundle size
const Stage02Canvas = lazy(() => import('../rendering/Stage02Canvas').then((m) => ({ default: m.Stage02Canvas })));
const Stage03Canvas = lazy(() => import('../rendering/Stage03Canvas').then((m) => ({ default: m.Stage03Canvas })));
const Stage04Canvas = lazy(() => import('../rendering/Stage04Canvas').then((m) => ({ default: m.Stage04Canvas })));
const Stage05Canvas = lazy(() => import('../rendering/Stage05Canvas').then((m) => ({ default: m.Stage05Canvas })));
const Stage06Canvas = lazy(() => import('../rendering/Stage06Canvas').then((m) => ({ default: m.Stage06Canvas })));
const Stage07Canvas = lazy(() => import('../rendering/Stage07Canvas').then((m) => ({ default: m.Stage07Canvas })));
import { DataPoint2D, PointerToolMode } from '../datasets/tabular2d';
import { AlgorithmPlugin, VisualArtifact, SequenceDataset, RLGridworldDataset, GridworldTileKind, TaskType } from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { DirtyManager } from '../rendering/dirty-manager';
import { SymbolicStrip } from './SymbolicStrip';
import { binarizeTarget } from '../algorithms/stage01/crossTask';
import { AlertCircle, AlertTriangle, RefreshCw, Clock, RotateCcw } from 'lucide-react';

interface InteractiveStageContainerProps {
  algorithm: AlgorithmPlugin;
  theme: ThemeMode;
  trainData: DataPoint2D[];
  testData: DataPoint2D[];
  artifacts: VisualArtifact[];
  gridValues?: Float32Array | null;
  gridResolution: number;
  dirtyManager: DirtyManager;
  circuitBreakerTripped: boolean;
  pointerMode: PointerToolMode;
  activeClass: number;
  showTestData?: boolean;
  onToggleShowTestData?: () => void;
  datasetTaskType?: TaskType;
  isCrossTaskMismatch?: boolean;
  crossTaskMessage?: string;
  hyperparams?: Record<string, number | string>;
  onAddPoint: (pt: { x1: number; x2: number; y: number }) => void;
  onUpdatePoint: (pt: DataPoint2D) => void;
  onDeletePoint: (id: number) => void;
  onClampAndRecover: () => void;
  onDragStart?: () => void;
  onDragEnd?: () => void;
  predictValueAtPoint: (x1: number, x2: number) => number;
  onUpdateHyperparam?: (keyOrBatch: string | Record<string, number | string>, value?: number | string) => void;
  onSyncCentroids?: (centroids: [number, number][]) => void;
  inputGrid?: Float32Array;
  sequenceDataset?: SequenceDataset;
  rlGridworld?: RLGridworldDataset;
  activeBrush?: GridworldTileKind;
  onUpdateInputGrid?: (grid: Float32Array) => void;
  onPaintTile?: (row: number, col: number, tile: GridworldTileKind) => void;
  onSelectToken?: (index: number) => void;
  onPinGate?: (stepIndex: number, gateId: string, value?: number) => void;
  historyBuffer?: any;
  currentEpoch?: number;
  scrubbedEpoch?: number | null;
  onScrubToEpoch?: (epoch: number) => void;
  onReturnToLive?: () => void;
}

export const InteractiveStageContainer: React.FC<InteractiveStageContainerProps> = ({
  algorithm,
  theme,
  trainData,
  testData,
  artifacts,
  gridValues,
  gridResolution,
  dirtyManager,
  circuitBreakerTripped,
  pointerMode,
  activeClass,
  showTestData = true,
  onToggleShowTestData,
  datasetTaskType,
  isCrossTaskMismatch,
  crossTaskMessage,
  hyperparams,
  onAddPoint,
  onUpdatePoint,
  onDeletePoint,
  onClampAndRecover,
  onDragStart,
  onDragEnd,
  predictValueAtPoint,
  onUpdateHyperparam,
  onSyncCentroids,
  inputGrid,
  sequenceDataset,
  rlGridworld,
  activeBrush,
  onUpdateInputGrid,
  onSelectToken,
  onPinGate,
  onPaintTile,
  scrubbedEpoch,
  onReturnToLive,
}) => {
  const [activeHighlight, setActiveHighlight] = useState<string | null>(null);

  // Pre-flight Data Validator
  const isNativeRegressor =
    (algorithm.id === 'knn' || algorithm.stage === 'tree-partition') && datasetTaskType === 'regression';
  const isClassification = algorithm.taskType === 'classification' && !isNativeRegressor;
  const uniqueTrainClasses = new Set(
    trainData.map((p) => (datasetTaskType === 'regression' ? binarizeTarget(p.y) : p.y))
  );
  const hasInsufficientClasses = isClassification && uniqueTrainClasses.size < 2;

  // Stage 03 Pre-flight Validators
  const kVal = Number(hyperparams?.k ?? 3);
  const isClusteringAlgo = ['k-means', 'gmm', 'hierarchical-clustering'].includes(algorithm.id);
  const hasInsufficientPointsForClusters = isClusteringAlgo && trainData.length < kVal;

  const isLDA = algorithm.id === 'lda';
  const hasInsufficientClassesForLDA = isLDA && uniqueTrainClasses.size < 2;

  return (
    <main className="stage-center flex flex-col flex-1 h-full min-w-0 overflow-hidden bg-[var(--bg-primary)]" data-node-id="3:20" role="main" aria-label="Interactive Algorithm Canvas Area">
      {/* 1. Header: Instrument Serif Algorithm Title + Hoverable KaTeX Formula Strip */}
      <div className="flex flex-wrap items-center justify-between px-6 py-2 border-b border-[var(--border-hairline)] bg-[var(--bg-primary)] gap-2 min-w-0">
        <div className="flex items-baseline gap-2.5 min-w-0 flex-wrap">
          <h1 className="font-serif text-2xl font-normal text-[var(--text-primary)] tracking-tight">
            {algorithm.name}
          </h1>
          <span className="font-mono text-[10px] text-[var(--text-secondary)] uppercase shrink-0">
            {algorithm.stage === 'tree-partition'
              ? 'Stage 02 · Partition, Ensemble & Boosting Arena'
              : algorithm.stage === 'unsupervised-2d'
              ? 'Stage 03 · Unsupervised Clustering & Manifold Projection Lab'
              : algorithm.stage === 'neural-graph'
              ? 'Stage 04 · Dense & Generative Neural Graph Lab'
              : algorithm.stage === 'vision-grid'
              ? 'Stage 05 · Spatial Vision & Patch Architecture Lab'
              : algorithm.stage === 'sequence-attn'
              ? 'Stage 06 · Sequence Gates & Attention Arena'
              : algorithm.stage === 'rl-gridworld'
              ? 'Stage 07 · Gridworld Policy & Value Arena'
              : datasetTaskType === 'regression' &&
                (algorithm.taskType !== 'classification' || algorithm.id === 'knn') &&
                (trainData.length >= 2 ? is1DRegressionData(trainData) : is1DRegressionData(testData))
              ? 'Stage 01 · Continuous 1D Regression Curve'
              : 'Stage 01 · Continuous 2D Manifold'}
          </span>
        </div>

        {/* Hoverable KaTeX Symbolic Formula Strip */}
        {algorithm.symbolicLinks && algorithm.symbolicLinks.length > 0 && (() => {
          const isRegMode = datasetTaskType === 'regression' || hyperparams?.task === 'regression';
          let resolvedLinks = algorithm.symbolicLinks;
          if (algorithm.id === 'decision-tree') {
            const crit = isRegMode ? 'mse' : (hyperparams?.criterion ?? 'gini');
            if (crit === 'mse') {
              resolvedLinks = [
                {
                  id: 'mse',
                  symbolLatex: '\\text{MSE}(t) = \\frac{1}{N_t}\\sum_{i \\in t} (y_i - \\bar{y}_t)^2',
                  label: 'Mean Squared Error',
                  description: 'Node variance around the continuous leaf mean prediction.',
                  highlightTarget: 'decision-hyperplane' as const,
                },
                {
                  id: 'delta-mse',
                  symbolLatex: '\\Delta \\text{MSE} = \\text{MSE}(t) - \\frac{N_L}{N}\\text{MSE}(t_L) - \\frac{N_R}{N}\\text{MSE}(t_R)',
                  label: 'Variance Reduction',
                  description: 'Exact MSE reduction achieved by partitioning feature space along X₁ or X₂.',
                  highlightTarget: 'normal-vector' as const,
                },
                algorithm.symbolicLinks[2],
              ].filter(Boolean);
            } else if (crit === 'entropy') {
              resolvedLinks = [
                {
                  id: 'entropy',
                  symbolLatex: 'H(t) = -\\sum_{k=1}^K p_k \\log_2 p_k',
                  label: 'Shannon Entropy',
                  description: 'Information-theoretic impurity measuring class uncertainty at node t.',
                  highlightTarget: 'decision-hyperplane' as const,
                },
                {
                  id: 'delta-ig',
                  symbolLatex: '\\text{IG}(s, t) = H(t) - \\frac{N_L}{N} H(t_L) - \\frac{N_R}{N} H(t_R)',
                  label: 'Information Gain',
                  description: 'Entropy reduction achieved by partitioning feature space.',
                  highlightTarget: 'normal-vector' as const,
                },
                algorithm.symbolicLinks[2],
              ].filter(Boolean);
            }
          } else if (algorithm.id === 'gradient-boosting' && isRegMode) {
            resolvedLinks = [
              {
                id: 'gbm-residual-reg',
                symbolLatex: 'r_{im} = -\\left[\\frac{\\partial L(y_i, F(\\mathbf{x}_i))}{\\partial F(\\mathbf{x}_i)}\\right]_{F_{m-1}} = y_i - F_{m-1}(\\mathbf{x}_i)',
                label: 'Squared-Error Residual',
                description: 'Negative gradient of squared error loss for continuous regression.',
                highlightTarget: 'normal-vector' as const,
              },
              algorithm.symbolicLinks[1],
            ].filter(Boolean);
          } else if (algorithm.id === 'voting-classifier' && isRegMode) {
            resolvedLinks = [
              {
                id: 'vote-reg-avg',
                symbolLatex: '\\hat{y}(\\mathbf{x}) = \\frac{\\sum_{m=1}^M w_m \\hat{y}_m(\\mathbf{x})}{\\sum_{m=1}^M w_m}',
                label: 'Weighted Average Consensus',
                description: 'Continuous weighted average across OLS, CART, and k-NN regressors.',
                highlightTarget: 'decision-hyperplane' as const,
              },
            ];
          } else if (algorithm.id === 'stacking' && isRegMode) {
            resolvedLinks = [
              {
                id: 'stacking-meta-feature-reg',
                symbolLatex: '\\mathbf{z}_i = \\left[ \\hat{y}_{\\text{tree}}(\\mathbf{x}_i), \\; \\hat{y}_{\\text{knn}}(\\mathbf{x}_i) \\right]^T',
                label: 'Meta-Feature Vector',
                description: 'Level-0 base regressor predictions mapped into meta-feature space.',
                highlightTarget: 'weights-inset' as const,
              },
              {
                id: 'stacking-meta-learner-reg',
                symbolLatex: '\\hat{y}(\\mathbf{x}) = w_1 z_1 + w_2 z_2 + b_\\text{meta}',
                label: 'Level-1 Meta-Regressor',
                description: 'Linear Ridge meta-regressor trained on out-of-fold base predictions.',
                highlightTarget: 'decision-hyperplane' as const,
              },
            ];
          }
          return (
            <SymbolicStrip
              links={resolvedLinks}
              activeHighlight={activeHighlight}
              onHighlightChange={(target) => {
                setActiveHighlight(target);
                dirtyManager.markDirty();
              }}
            />
          );
        })()}
      </div>

      {/* Scrubbed Historical Snapshot Indicator Banner */}
      {scrubbedEpoch !== null && scrubbedEpoch !== undefined && (
        <div className="px-6 py-1.5 bg-[var(--accent-amber)]/10 border-b border-[var(--accent-amber)]/30 flex items-center justify-between text-[11px] text-[var(--accent-amber)] font-mono">
          <div className="flex items-center gap-1.5">
            <Clock size={12} className="shrink-0" />
            <span>
              Inspecting Historical Snapshot: <strong>Epoch {scrubbedEpoch}</strong>
            </span>
          </div>
          {onReturnToLive && (
            <button
              onClick={onReturnToLive}
              className="px-2 py-0.5 rounded bg-[var(--accent-amber)] text-[var(--text-inverse)] font-semibold text-[10px] hover:opacity-90 transition-opacity cursor-pointer flex items-center gap-1 shrink-0"
              title="Return to Live State"
            >
              <RotateCcw size={10} />
              <span>Return to Live</span>
            </button>
          )}
        </div>
      )}

      {/* Cross-Task Interpretation Banner */}
      {isCrossTaskMismatch && (
        <div className="px-6 py-2 bg-[var(--bg-subtle)] border-b border-[var(--border-hairline)] flex flex-wrap items-center justify-between text-[11px] text-[var(--text-secondary)] gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-mono text-[9px] uppercase px-1.5 py-0.5 bg-[var(--text-secondary)] text-[var(--text-inverse)] font-medium shrink-0">
              Cross-Task Mode
            </span>
            <span className="truncate">
              {crossTaskMessage || 'Regression model is fitting binary class labels y ∈ {-1, +1} as continuous numerical targets.'}
            </span>
          </div>
        </div>
      )}

      {/* 2. Pre-Flight Data Validator Banner */}
      {hasInsufficientClasses && (
        <div className="px-6 py-2 bg-[var(--bg-subtle)] border-b border-[var(--accent-amber)] flex flex-wrap items-center justify-between text-[11px] text-[var(--text-primary)] gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <AlertCircle size={14} className="text-[var(--accent-amber)] shrink-0" />
            <span>
              <strong>Pre-Flight Validator:</strong> Less than 2 classes present on canvas ({uniqueTrainClasses.size} found). Add opposite class points or reseed dataset.
            </span>
          </div>
        </div>
      )}

      {hasInsufficientPointsForClusters && (
        <div className="px-6 py-2 bg-[var(--bg-subtle)] border-b border-[var(--accent-amber)] flex flex-wrap items-center justify-between text-[11px] text-[var(--text-primary)] gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <AlertCircle size={14} className="text-[var(--accent-amber)] shrink-0" />
            <span>
              <strong>Pre-Flight Validator:</strong> Total points ({trainData.length}) &lt; K ({kVal}) clusters. Add at least {kVal} points to initialize {kVal} clusters.
            </span>
          </div>
        </div>
      )}

      {hasInsufficientClassesForLDA && (
        <div className="px-6 py-2 bg-[var(--bg-subtle)] border-b border-[var(--accent-amber)] flex flex-wrap items-center justify-between text-[11px] text-[var(--text-primary)] gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <AlertCircle size={14} className="text-[var(--accent-amber)] shrink-0" />
            <span>
              <strong>Pre-Flight Validator:</strong> LDA requires at least 2 distinct classes to compute a discriminant projection axis (found {uniqueTrainClasses.size}). Toggle 3 Classes in the Data sidebar to test 2D discriminant projections.
            </span>
          </div>
        </div>
      )}

      {/* 3. Numerical Overflow Circuit Breaker Banner */}
      {circuitBreakerTripped && (
        <div className="px-6 py-2.5 bg-[var(--accent-crimson)] text-white flex flex-wrap items-center justify-between text-[12px] font-sans gap-2">
          <div className="flex items-center gap-2 min-w-0">
            <AlertTriangle size={15} className="shrink-0" />
            <span className="font-mono font-semibold tracking-wider shrink-0">
              [NUMERICAL OVERFLOW // DIVERGED]
            </span>
            <span>Gradient explosion or NaN detected. Last valid frame frozen.</span>
          </div>
          <button
            onClick={onClampAndRecover}
            className="transport-btn text-[11px] bg-[var(--bg-canvas)] text-[var(--text-primary)] font-semibold border border-[var(--border-strong)] hover:bg-[var(--bg-active)] shrink-0"
          >
            <RefreshCw size={12} />
            <span>Clamp & Recover (Halve η)</span>
          </button>
        </div>
      )}

      {/* 4. Main Interactive Canvas Stage Viewport */}
      <div className="flex-1 relative min-h-0 min-w-0">
        <Suspense fallback={<div className="w-full h-full flex items-center justify-center font-mono text-[11px] text-[var(--text-muted)]">Loading stage canvas...</div>}>
          {algorithm.stage === 'tree-partition' ? (
            <Stage02Canvas
              theme={theme}
              trainData={trainData}
              testData={testData}
              artifacts={artifacts}
              gridValues={gridValues}
              gridResolution={gridResolution}
              dirtyManager={dirtyManager}
              algorithm={algorithm}
              datasetTaskType={datasetTaskType}
              hyperparams={hyperparams}
              pointerMode={pointerMode}
              activeClass={activeClass}
              showTestData={showTestData}
              onToggleShowTestData={onToggleShowTestData}
              onAddPoint={onAddPoint}
              onUpdatePoint={onUpdatePoint}
              onDeletePoint={onDeletePoint}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
              predictValueAtPoint={predictValueAtPoint}
              onUpdateHyperparam={onUpdateHyperparam}
            />
          ) : algorithm.stage === 'unsupervised-2d' ? (
            <Stage03Canvas
              theme={theme}
              trainData={trainData}
              testData={testData}
              artifacts={artifacts}
              gridValues={gridValues}
              gridResolution={gridResolution}
              dirtyManager={dirtyManager}
              algorithm={algorithm}
              datasetTaskType={datasetTaskType}
              hyperparams={hyperparams}
              pointerMode={pointerMode}
              activeClass={activeClass}
              onAddPoint={onAddPoint}
              onUpdatePoint={onUpdatePoint}
              onDeletePoint={onDeletePoint}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
              predictValueAtPoint={predictValueAtPoint}
              onUpdateHyperparam={onUpdateHyperparam}
              onSyncCentroids={onSyncCentroids}
            />
          ) : algorithm.stage === 'neural-graph' ? (
            <Stage04Canvas
              theme={theme}
              trainData={trainData}
              testData={testData}
              artifacts={artifacts}
              gridValues={gridValues}
              gridResolution={gridResolution}
              dirtyManager={dirtyManager}
              algorithm={algorithm}
              datasetTaskType={datasetTaskType}
              hyperparams={hyperparams}
              pointerMode={pointerMode}
              activeClass={activeClass}
              showTestData={showTestData}
              onToggleShowTestData={onToggleShowTestData}
              onAddPoint={onAddPoint}
              onUpdatePoint={onUpdatePoint}
              onDeletePoint={onDeletePoint}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
              predictValueAtPoint={predictValueAtPoint}
              onUpdateHyperparam={onUpdateHyperparam}
            />
          ) : algorithm.stage === 'vision-grid' ? (
            <Stage05Canvas
              theme={theme}
              artifacts={artifacts}
              dirtyManager={dirtyManager}
              algorithm={algorithm}
              hyperparams={hyperparams}
              inputGrid={inputGrid}
              onUpdateInputGrid={onUpdateInputGrid}
              onUpdateHyperparam={onUpdateHyperparam}
            />
          ) : algorithm.stage === 'sequence-attn' ? (
            <Stage06Canvas
              theme={theme}
              artifacts={artifacts}
              dirtyManager={dirtyManager}
              algorithm={algorithm}
              hyperparams={hyperparams}
              sequenceDataset={sequenceDataset}
              onSelectToken={onSelectToken}
              onPinGate={onPinGate}
              onUpdateHyperparam={onUpdateHyperparam}
            />
          ) : algorithm.stage === 'rl-gridworld' ? (
            <Stage07Canvas
              theme={theme}
              artifacts={artifacts}
              dirtyManager={dirtyManager}
              algorithm={algorithm}
              hyperparams={hyperparams}
              rlGridworld={rlGridworld}
              activeBrush={activeBrush}
              onPaintTile={onPaintTile}
              onUpdateHyperparam={onUpdateHyperparam}
            />
          ) : (
            <Stage01Canvas
              algorithm={algorithm}
              taskType={algorithm.taskType}
              datasetTaskType={datasetTaskType}
              theme={theme}
              trainData={trainData}
              testData={testData}
              artifacts={artifacts}
              gridValues={gridValues}
              gridResolution={gridResolution}
              dirtyManager={dirtyManager}
              highlightTarget={activeHighlight}
              pointerMode={pointerMode}
              activeClass={activeClass}
              showTestData={showTestData}
              onToggleShowTestData={onToggleShowTestData}
              onAddPoint={onAddPoint}
              onUpdatePoint={onUpdatePoint}
              onDeletePoint={onDeletePoint}
              onDragStart={onDragStart}
              onDragEnd={onDragEnd}
              predictValueAtPoint={predictValueAtPoint}
            />
          )}
        </Suspense>
      </div>
    </main>
  );
};
