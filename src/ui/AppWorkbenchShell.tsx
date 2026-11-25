import React, { useState, useEffect, useRef, useCallback, lazy, Suspense } from 'react';
import { TopControlBar } from './TopControlBar';
import { DataSidebar } from './DataSidebar';
import { InteractiveStageContainer } from './InteractiveStageContainer';
import { TelemetrySidebar } from './TelemetrySidebar';
import { CoachmarkStrip } from './CoachmarkStrip';

const CommandPalette = lazy(() => import('./CommandPalette').then((m) => ({ default: m.CommandPalette })));

import { ThemeMode } from '../types/theme';
import {
  DatasetOptions,
  DataPoint2D,
  PointerToolMode,
  generateTabular2dDataset,
  getDatasetTaskType,
} from '../datasets/tabular2d';
import {
  AlgorithmPlugin,
  VisualArtifact,
  TelemetryMetrics,
  ExplanationPayload,
  SpatialVisionDataset,
  SequenceDataset,
  SequencePresetKind,
  RLGridworldDataset,
  GridworldPresetKind,
  GridworldTileKind,
} from '../types/algorithm';
import { getPlugin } from '../algorithms/registry';
import {
  generateSpatialVisionDataset,
  getPresetGrid,
  shiftGridRight,
  applyNoiseAndContrast,
  VisionPresetKind,
} from '../datasets/visionGrid';
import {
  generateSequenceDataset,
  toggleTokenMask,
} from '../datasets/sequenceTokens';
import {
  createRLGridworldDataset,
  paintGridworldTile,
} from '../datasets/rlGridworld';
import { HistoryRingBuffer } from '../utils/ringBuffer';
import { DirtyManager } from '../rendering/dirty-manager';
import { useEngineWorker } from '../worker/useEngineWorker';
import { serializeToUrlHash, parseUrlHash } from '../utils/urlState';
import { clampThreshold } from '../algorithms/stage01/logisticRegression';

export const AppWorkbenchShell: React.FC = () => {
  const initialUrlState = useRef(parseUrlHash()).current;

  // Theme: Paper / Carbon
  const [theme, setTheme] = useState<ThemeMode>(() => {
    if (initialUrlState.theme) return initialUrlState.theme;
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('mlcanvas-theme') as ThemeMode;
      if (saved === 'carbon' || saved === 'paper') return saved;
    }
    return 'paper';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('mlcanvas-theme', theme);
  }, [theme]);

  // Reduced motion
  const [reducedMotion, setReducedMotion] = useState(false);

  // Selected Algorithm
  const [selectedAlgorithmId, setSelectedAlgorithmId] = useState<string>(
    initialUrlState.algo || 'linear-svm'
  );
  const currentAlgorithm: AlgorithmPlugin =
    getPlugin(selectedAlgorithmId) || getPlugin('linear-svm')!;
  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState(false);

  // Hyperparameters (seeded with initial plugin defaults overlaid by URL params)
  const [hyperparams, setHyperparams] = useState<Record<string, number | string>>(() => {
    const initialPlugin = getPlugin(initialUrlState.algo || 'linear-svm');
    const pluginDefaults: Record<string, number | string> = {};
    if (initialPlugin) {
      for (const hp of [...initialPlugin.primaryParams, ...initialPlugin.advancedParams]) {
        pluginDefaults[hp.id] = hp.default;
      }
    }
    return {
      ...pluginDefaults,
      learningRate:
        typeof initialUrlState.lr === 'number' && Number.isFinite(initialUrlState.lr)
          ? initialUrlState.lr
          : (pluginDefaults.learningRate ?? 0.03),
      cPenalty:
        typeof initialUrlState.c === 'number' && Number.isFinite(initialUrlState.c)
          ? initialUrlState.c
          : (pluginDefaults.cPenalty ?? 1.0),
      lambdaPenalty:
        typeof initialUrlState.lambda === 'number' && Number.isFinite(initialUrlState.lambda)
          ? initialUrlState.lambda
          : (pluginDefaults.lambdaPenalty ?? 0.1),
      degree:
        typeof initialUrlState.degree === 'number' && Number.isFinite(initialUrlState.degree)
          ? initialUrlState.degree
          : (pluginDefaults.degree ?? 3),
      regularization:
        typeof initialUrlState.reg === 'number' && Number.isFinite(initialUrlState.reg)
          ? initialUrlState.reg
          : (pluginDefaults.regularization ?? 0.001),
      solver: initialUrlState.solver ?? (pluginDefaults.solver ?? 'sgd'),
      penaltyType: pluginDefaults.penaltyType ?? 'l1',
      batchPasses: pluginDefaults.batchPasses ?? 5,
      cyclesPerStep: pluginDefaults.cyclesPerStep ?? 5,
    };
  });
  const hyperparamsRef = useRef(hyperparams);
  hyperparamsRef.current = hyperparams;

  const isCurrentAlgoRegression = currentAlgorithm.taskType === 'regression';

  // Dataset Options & Points
  const [datasetOptions, setDatasetOptions] = useState<DatasetOptions>({
    kind: (initialUrlState.dataset as any) || (isCurrentAlgoRegression ? 'sine-wave' : 'linear-blobs'),
    sampleCount: 160,
    noise: initialUrlState.noise ?? 0.1,
    testSplit: initialUrlState.split ?? 0.2,
    seed: initialUrlState.seed ?? 42,
    numClasses: 2,
    scaling: initialUrlState.scale ?? 'raw',
    stretchX2: initialUrlState.stretch ?? false,
  });

  const [trainData, setTrainData] = useState<DataPoint2D[]>([]);
  const [testData, setTestData] = useState<DataPoint2D[]>([]);

  // Stage 05 Spatial Vision Dataset
  const [spatialVision, setSpatialVisionState] = useState<SpatialVisionDataset>(() =>
    generateSpatialVisionDataset({ preset: 'digit' })
  );

  // Stage 06 Sequence Dataset
  const [sequenceTokens, setSequenceTokensState] = useState<SequenceDataset>(() =>
    generateSequenceDataset({ preset: 'coreference' })
  );

  // Stage 07 RL Gridworld Dataset & Brush
  const [rlGridworld, setRLGridworldState] = useState<RLGridworldDataset>(() =>
    createRLGridworldDataset({ preset: 'cliff-walking' })
  );
  const [activeBrush, setActiveBrush] = useState<GridworldTileKind>('wall');

  // Pointer Tool Mode & Active Class
  const [pointerMode, setPointerMode] = useState<PointerToolMode>('add-drag');
  const [activeClass, setActiveClass] = useState<number>(1);
  const [showTestData, setShowTestData] = useState<boolean>(true);

  // Decision Threshold tau
  const [decisionThreshold, setDecisionThreshold] = useState<number>(0);

  // Memory & Dirty Flag Management
  const ringBufferRef = useRef<HistoryRingBuffer>(new HistoryRingBuffer(300, 16));
  const dirtyManagerRef = useRef<DirtyManager>(new DirtyManager());

  // Model Artifacts & Explanation State
  const [artifacts, setArtifacts] = useState<VisualArtifact[]>([]);
  const [latestMetrics, setLatestMetrics] = useState<TelemetryMetrics | null>(null);
  const [latestExplanation, setLatestExplanation] = useState<ExplanationPayload | null>(null);
  const [isConverged, setIsConverged] = useState(false);
  const latestModelStateRef = useRef<any>(null);

  // Offscreen Heatmap Buffer
  const [gridValues, setGridValues] = useState<Float32Array | null>(null);
  const [gridResolution, setGridResolution] = useState<number>(96);

  // Timeline Scrubber State (null = Live execution mode)
  const [scrubIndex, setScrubIndex] = useState<number | null>(null);

  // Web Worker Hook
  const {
    isRunning,
    epoch,
    circuitBreakerTripped,
    initAlgorithm,
    setDataset,
    updateHyperparams,
    step,
    start,
    pause,
    reset,
    evalGrid,
    clampAndRecover,
    syncCentroids,
    setSpatialVision,
    setSequenceTokens,
    setRLGridworld,
    pinGateValve,
  } = useEngineWorker({
    dirtyManager: dirtyManagerRef.current,
    autoPauseOnConverged:
      currentAlgorithm.stage === 'tree-partition' ||
      currentAlgorithm.stage === 'neural-graph',
    onInitState: (initExplanation) => {
      setScrubIndex(null);
      setArtifacts([]);
      setGridValues(null);
      setLatestMetrics(null);
      setIsConverged(false);
      if (initExplanation) {
        setLatestExplanation(initExplanation);
      }
      ringBufferRef.current.clear();
      dirtyManagerRef.current.markDirty();
    },
    onStepResult: (result, stepArtifacts, stepExplanation, serialized, gridBuffer) => {
      latestModelStateRef.current = result.state;
      setArtifacts(stepArtifacts);
      setLatestMetrics(result.metrics);
      setIsConverged(Boolean(result.converged));
      if (stepExplanation) {
        setLatestExplanation(stepExplanation);
      }

      const seqArt = stepArtifacts.find((a) => a.type === 'sequence-gate-attn') as any;
      if (seqArt && typeof seqArt.selectedStep === 'number') {
        const nextIdx = seqArt.selectedStep;
        setSequenceTokensState((prev) =>
          prev.selectedTokenIndex === nextIdx ? prev : { ...prev, selectedTokenIndex: nextIdx }
        );
      }

      if (gridBuffer && gridBuffer.length > 0) {
        setGridValues(gridBuffer);
      } else {
        setGridValues(null);
      }

      // Add to ring buffer with immutable artifacts, metrics, explanation, and gridBuffer
      ringBufferRef.current.push({
        epoch: result.epoch,
        trainLoss: result.metrics.trainLoss,
        testLoss: result.metrics.testLoss,
        trainAccuracy: result.metrics.trainAccuracy,
        testAccuracy: result.metrics.testAccuracy,
        weights: serialized ? serialized.subarray(4, 6) : new Float32Array([0, 0]),
        bias: serialized ? serialized[1] : 0,
        margin: result.metrics.marginWidth,
        gradientNorm: result.metrics.gradientNorm,
        metrics: result.metrics,
        artifacts: stepArtifacts,
        explanation: stepExplanation,
        gridValues: gridBuffer && gridBuffer.length > 0 ? gridBuffer : undefined,
        modelState: result.state,
      });

      dirtyManagerRef.current.markDirty();
    },
    onGridResult: (buffer) => {
      setGridValues(buffer);
      dirtyManagerRef.current.markDirty();
    },
    onInferenceResult: (result, infArtifacts, infExplanation, _serialized, gridBuffer) => {
      latestModelStateRef.current = result.state;
      setArtifacts(infArtifacts);
      setLatestMetrics(result.metrics);
      if (infExplanation) {
        setLatestExplanation(infExplanation);
      }
      const seqArt = infArtifacts.find((a) => a.type === 'sequence-gate-attn') as any;
      if (seqArt && typeof seqArt.selectedStep === 'number') {
        const nextIdx = seqArt.selectedStep;
        setSequenceTokensState((prev) =>
          prev.selectedTokenIndex === nextIdx ? prev : { ...prev, selectedTokenIndex: nextIdx }
        );
      }
      if (gridBuffer && gridBuffer.length > 0) {
        setGridValues(gridBuffer);
      } else {
        setGridValues(null);
      }
      dirtyManagerRef.current.markDirty();
    },
  });

  // Re-generate dataset
  const regenerateDataset = useCallback(() => {
    if (isRunning) pause();
    setScrubIndex(null);
    const { train, test } = generateTabular2dDataset(datasetOptions);
    setTrainData(train);
    setTestData(test);
    setArtifacts([]);
    setGridValues(null);
    setLatestMetrics(null);
    setIsConverged(false);
    ringBufferRef.current.clear();
    setDataset(train, test, datasetOptions.seed, false);
  }, [isRunning, pause, datasetOptions, setDataset]);

  // Initial mount
  useEffect(() => {
    const plugin = getPlugin(selectedAlgorithmId);
    let options = datasetOptions;
    const isAlgoRegression = plugin?.taskType === 'regression';
    const isDatasetRegression = getDatasetTaskType(options.kind) === 'regression';
    if (!initialUrlState.dataset) {
      if (isAlgoRegression && !isDatasetRegression) {
        options = { ...options, kind: 'sine-wave' };
        setDatasetOptions(options);
      } else if (!isAlgoRegression && isDatasetRegression) {
        options = { ...options, kind: 'linear-blobs' };
        setDatasetOptions(options);
      }
    }

    if (selectedAlgorithmId === 'logistic-regression' || selectedAlgorithmId === 'decision-tree') {
      setDecisionThreshold(0.50);
    }

    const { train, test } = generateTabular2dDataset(options);
    setTrainData(train);
    setTestData(test);
    setArtifacts([]);
    setGridValues(null);
    setLatestMetrics(null);

    initAlgorithm(
      selectedAlgorithmId,
      hyperparams,
      train,
      test,
      options.seed,
      spatialVision,
      sequenceTokens,
      rlGridworld
    );
  }, []);

  const lastAlgoRef = useRef(selectedAlgorithmId);
  const handleSelectAlgorithmRef = useRef<((id: string) => void) | null>(null);
  const handleUpdateDatasetOptionsRef = useRef<((partial: Partial<DatasetOptions>) => void) | null>(null);

  // Update URL hash on notable state change
  useEffect(() => {
    const hash = serializeToUrlHash({
      algo: selectedAlgorithmId,
      dataset: datasetOptions.kind,
      seed: datasetOptions.seed,
      noise: datasetOptions.noise,
      split: datasetOptions.testSplit,
      scale: datasetOptions.scaling,
      stretch: datasetOptions.stretchX2,
      theme,
      lr: typeof hyperparams.learningRate === 'number' && Number.isFinite(hyperparams.learningRate)
        ? Number(hyperparams.learningRate)
        : undefined,
      c: typeof hyperparams.cPenalty === 'number' && Number.isFinite(hyperparams.cPenalty)
        ? Number(hyperparams.cPenalty)
        : undefined,
      lambda: typeof hyperparams.lambdaPenalty === 'number' && Number.isFinite(hyperparams.lambdaPenalty)
        ? Number(hyperparams.lambdaPenalty)
        : undefined,
      degree: typeof hyperparams.degree === 'number' && Number.isFinite(hyperparams.degree)
        ? Number(hyperparams.degree)
        : undefined,
      reg: typeof hyperparams.regularization === 'number' && Number.isFinite(hyperparams.regularization)
        ? Number(hyperparams.regularization)
        : undefined,
      solver: typeof hyperparams.solver === 'string'
        ? String(hyperparams.solver)
        : undefined,
    });
    if (window.location.hash !== hash) {
      if (lastAlgoRef.current !== selectedAlgorithmId) {
        window.history.pushState(null, '', hash);
        lastAlgoRef.current = selectedAlgorithmId;
      } else {
        window.history.replaceState(null, '', hash);
      }
    }
  }, [selectedAlgorithmId, datasetOptions, theme, hyperparams]);

  // Handle browser back/forward navigation (popstate / hashchange)
  useEffect(() => {
    const handleLocationChange = () => {
      const parsed = parseUrlHash();
      if (parsed.algo && parsed.algo !== selectedAlgorithmId) {
        lastAlgoRef.current = parsed.algo;
        handleSelectAlgorithmRef.current?.(parsed.algo);
      }
      if (parsed.dataset && parsed.dataset !== datasetOptions.kind) {
        handleUpdateDatasetOptionsRef.current?.({ kind: parsed.dataset as any });
      }
      if (parsed.theme && parsed.theme !== theme) {
        setTheme(parsed.theme);
      }
      const hyperUpdates: Record<string, number | string> = {};
      if (typeof parsed.lr === 'number' && Number.isFinite(parsed.lr)) hyperUpdates.learningRate = parsed.lr;
      if (typeof parsed.c === 'number' && Number.isFinite(parsed.c)) hyperUpdates.cPenalty = parsed.c;
      if (typeof parsed.lambda === 'number' && Number.isFinite(parsed.lambda)) hyperUpdates.lambdaPenalty = parsed.lambda;
      if (typeof parsed.degree === 'number' && Number.isFinite(parsed.degree)) hyperUpdates.degree = parsed.degree;
      if (typeof parsed.reg === 'number' && Number.isFinite(parsed.reg)) hyperUpdates.regularization = parsed.reg;
      if (parsed.solver) hyperUpdates.solver = parsed.solver;
      if (Object.keys(hyperUpdates).length > 0) {
        setHyperparams((prev) => ({ ...prev, ...hyperUpdates }));
      }
    };

    window.addEventListener('hashchange', handleLocationChange);
    window.addEventListener('popstate', handleLocationChange);
    return () => {
      window.removeEventListener('hashchange', handleLocationChange);
      window.removeEventListener('popstate', handleLocationChange);
    };
  }, [selectedAlgorithmId, datasetOptions.kind, theme, setTheme]);

  // When dataset configuration changes
  const handleUpdateDatasetOptions = (partial: Partial<DatasetOptions>) => {
    if (isRunning) pause();
    setScrubIndex(null);
    const updated = { ...datasetOptions, ...partial };
    setDatasetOptions(updated);

    if (currentAlgorithm.stage === 'tree-partition') {
      const nextTask = getDatasetTaskType(updated.kind);
      const nextHyperparams: Record<string, number | string> = {
        ...hyperparamsRef.current,
        task: nextTask,
      };
      if (currentAlgorithm.id === 'decision-tree') {
        nextHyperparams.criterion =
          nextTask === 'regression'
            ? 'mse'
            : hyperparamsRef.current.criterion === 'mse' || !hyperparamsRef.current.criterion
            ? 'gini'
            : hyperparamsRef.current.criterion;
      }
      if (currentAlgorithm.id === 'stacking') {
        nextHyperparams.metaLearner = nextTask === 'regression' ? 'ridge' : 'logistic';
      }
      hyperparamsRef.current = nextHyperparams;
      setHyperparams(nextHyperparams);
      updateHyperparams(nextHyperparams);
    }

    const { train, test } = generateTabular2dDataset(updated);
    setTrainData(train);
    setTestData(test);
    setArtifacts([]);
    setGridValues(null);
    setLatestMetrics(null);
    setIsConverged(false);
    ringBufferRef.current.clear();
    setDataset(train, test, updated.seed, false);
  };
  handleUpdateDatasetOptionsRef.current = handleUpdateDatasetOptions;

  // Algorithm Switching
  const handleSelectAlgorithm = async (algoId: string) => {
    if (isRunning) pause();
    setScrubIndex(null);
    setSelectedAlgorithmId(algoId);
    setArtifacts([]);
    setGridValues(null);
    setLatestMetrics(null);
    setIsConverged(false);
    ringBufferRef.current.clear();

    const plugin = getPlugin(algoId);
    if (!plugin) return;

    // Check dataset compatibility with taskType
    let currentTrain = trainData;
    let currentTest = testData;
    let currentOptions = datasetOptions;
    const currentTask = getDatasetTaskType(datasetOptions.kind);
    const isTabular2dStage =
      plugin.stage === 'continuous-2d' ||
      plugin.stage === 'tree-partition' ||
      plugin.stage === 'unsupervised-2d' ||
      plugin.stage === 'neural-graph';
    if (!isTabular2dStage) {
      if (plugin.taskType === 'regression' && currentTask !== 'regression') {
        currentOptions = { ...datasetOptions, kind: 'sine-wave' };
        setDatasetOptions(currentOptions);
        const generated = generateTabular2dDataset(currentOptions);
        currentTrain = generated.train;
        currentTest = generated.test;
        setTrainData(currentTrain);
        setTestData(currentTest);
      } else if (plugin.taskType === 'classification' && currentTask !== 'classification') {
        currentOptions = { ...datasetOptions, kind: 'linear-blobs' };
        setDatasetOptions(currentOptions);
        const generated = generateTabular2dDataset(currentOptions);
        currentTrain = generated.train;
        currentTest = generated.test;
        setTrainData(currentTrain);
        setTestData(currentTest);
      }
    }

    if (algoId === 'logistic-regression' || algoId === 'decision-tree') {
      setDecisionThreshold(0.50);
    } else {
      setDecisionThreshold(0.0);
    }

    // Set parameter defaults for new algorithm
    const defaults: Record<string, number | string> = {};
    for (const hp of [...plugin.primaryParams, ...plugin.advancedParams]) {
      defaults[hp.id] = hp.default;
    }
    if (plugin.stage === 'tree-partition') {
      const activeTask = getDatasetTaskType(currentOptions.kind);
      defaults.task = activeTask;
      if (algoId === 'decision-tree' && activeTask === 'regression') {
        defaults.criterion = 'mse';
      }
      if (algoId === 'stacking' && activeTask === 'regression') {
        defaults.metaLearner = 'ridge';
      }
    }
    hyperparamsRef.current = defaults;
    setHyperparams(defaults);

    initAlgorithm(
      algoId,
      defaults,
      currentTrain,
      currentTest,
      currentOptions.seed,
      spatialVision,
      sequenceTokens,
      rlGridworld
    );
  };
  handleSelectAlgorithmRef.current = handleSelectAlgorithm;

  const handleSelectVisionPreset = useCallback(
    (preset: VisionPresetKind, variant: 'default' | 'digit0' | 'digit7' = 'default') => {
      const base = getPresetGrid(preset, variant);
      const processed = applyNoiseAndContrast(base, spatialVision.noise, spatialVision.contrast, spatialVision.inverted);
      const updated: SpatialVisionDataset = {
        ...spatialVision,
        preset,
        inputGrid: processed,
      };
      setSpatialVisionState(updated);
      setSpatialVision(updated, false);
      dirtyManagerRef.current.markDirty();
    },
    [spatialVision, setSpatialVision]
  );

  const handleUpdateSpatialVision = useCallback(
    (updater: (prev: SpatialVisionDataset) => SpatialVisionDataset) => {
      setSpatialVisionState((prev) => {
        const next = updater(prev);
        const base = getPresetGrid(next.preset);
        const processed = applyNoiseAndContrast(base, next.noise, next.contrast, next.inverted);
        const full = { ...next, inputGrid: processed };
        setSpatialVision(full, false);
        return full;
      });
      dirtyManagerRef.current.markDirty();
    },
    [setSpatialVision]
  );

  const handleShiftInputRight = useCallback(() => {
    setSpatialVisionState((prev) => {
      const shifted = shiftGridRight(prev.inputGrid, 1);
      const next: SpatialVisionDataset = { ...prev, preset: 'custom', inputGrid: shifted };
      setSpatialVision(next, false);
      return next;
    });
    dirtyManagerRef.current.markDirty();
  }, [setSpatialVision]);

  const handleUpdateInputGrid = useCallback(
    (newGrid: Float32Array) => {
      setSpatialVisionState((prev) => {
        const next: SpatialVisionDataset = { ...prev, preset: 'custom', inputGrid: newGrid };
        setSpatialVision(next, false);
        return next;
      });
      dirtyManagerRef.current.markDirty();
    },
    [setSpatialVision]
  );

  const handleSelectSequencePreset = useCallback(
    (preset: SequencePresetKind) => {
      const updated = generateSequenceDataset({ preset, noise: sequenceTokens.noise });
      setSequenceTokensState(updated);
      setSequenceTokens(updated, true);
      dirtyManagerRef.current.markDirty();
    },
    [sequenceTokens.noise, setSequenceTokens]
  );

  const handleUpdateSequenceDataset = useCallback(
    (updater: (prev: SequenceDataset) => SequenceDataset) => {
      setSequenceTokensState((prev) => {
        const next = updater(prev);
        const generated = generateSequenceDataset({
          preset: next.preset,
          length: next.length,
          noise: next.noise,
          selectedTokenIndex: next.selectedTokenIndex,
        });
        setSequenceTokens(generated, true);
        return generated;
      });
      dirtyManagerRef.current.markDirty();
    },
    [setSequenceTokens]
  );

  const handleToggleTokenMask = useCallback(
    (index: number) => {
      setSequenceTokensState((prev) => {
        const next = toggleTokenMask(prev, index);
        setSequenceTokens(next, true);
        return next;
      });
      dirtyManagerRef.current.markDirty();
    },
    [setSequenceTokens]
  );

  const handleSelectFocusToken = useCallback(
    (index: number) => {
      setSequenceTokensState((prev) => {
        const next = { ...prev, selectedTokenIndex: index };
        setSequenceTokens(next, true);
        return next;
      });
      dirtyManagerRef.current.markDirty();
    },
    [setSequenceTokens]
  );

  const handleSelectGridworldPreset = useCallback(
    (preset: GridworldPresetKind) => {
      setScrubIndex(null);
      setIsConverged(false);
      ringBufferRef.current.clear();
      const next = createRLGridworldDataset({
        preset,
        seed: datasetOptions.seed,
      });
      setRLGridworldState(next);
      setRLGridworld(next, true, false);
      dirtyManagerRef.current.markDirty();
    },
    [datasetOptions.seed, setRLGridworld]
  );

  const handleUpdateRLGridworld = useCallback(
    (updater: (prev: RLGridworldDataset) => RLGridworldDataset) => {
      setScrubIndex(null);
      setIsConverged(false);
      ringBufferRef.current.clear();
      setRLGridworldState((prev) => {
        const next = updater(prev);
        setRLGridworld(next, true, false);
        return next;
      });
      dirtyManagerRef.current.markDirty();
    },
    [setRLGridworld]
  );

  const handlePaintGridworldTile = useCallback(
    (row: number, col: number, tile: GridworldTileKind) => {
      setRLGridworldState((prev) => {
        const next = paintGridworldTile(prev, row, col, tile);
        // In-situ adaptation: preserves learned Q-values/weights while resetting agent to start!
        setRLGridworld(next, true, true);
        return next;
      });
      dirtyManagerRef.current.markDirty();
    },
    [setRLGridworld]
  );

  const handleReset = useCallback(() => {
    if (isRunning) pause();
    setScrubIndex(null);
    setArtifacts([]);
    setGridValues(null);
    setLatestMetrics(null);
    setIsConverged(false);
    ringBufferRef.current.clear();
    reset();
  }, [isRunning, pause, reset]);

  const handleScrubIndexChange = useCallback(
    (newIndex: number | null) => {
      if (newIndex !== null && isRunning) {
        pause();
      }
      setScrubIndex(newIndex);
      dirtyManagerRef.current.markDirty();
    },
    [isRunning, pause]
  );

  const handleReturnToLive = useCallback(() => {
    setScrubIndex(null);
    dirtyManagerRef.current.markDirty();
  }, []);

  const handleToggleShowTestData = useCallback(() => {
    setShowTestData((prev) => !prev);
    dirtyManagerRef.current.markDirty();
  }, []);

  const handleTogglePlay = useCallback(() => {
    if (scrubIndex !== null) {
      setScrubIndex(null);
      dirtyManagerRef.current.markDirty();
    }
    if (isRunning) {
      pause();
    } else {
      start();
    }
  }, [scrubIndex, isRunning, pause, start]);

  const handleStep = useCallback(() => {
    if (scrubIndex !== null) {
      setScrubIndex(null);
      dirtyManagerRef.current.markDirty();
    }
    step();
  }, [scrubIndex, step]);

  const handleUpdateThreshold = useCallback(
    (nextThreshold: number) => {
      if (currentAlgorithm.id === 'logistic-regression' || currentAlgorithm.id === 'decision-tree') {
        const clamped =
          currentAlgorithm.id === 'decision-tree'
            ? Math.max(0.05, Math.min(0.95, Number(nextThreshold) || 0.5))
            : clampThreshold(nextThreshold);
        const updated = { ...hyperparamsRef.current, threshold: clamped };
        hyperparamsRef.current = updated;
        setHyperparams(updated);
        updateHyperparams(updated, true);
        dirtyManagerRef.current.markDirty();
        return;
      }
      setDecisionThreshold(nextThreshold);
    },
    [currentAlgorithm.id, updateHyperparams]
  );

  const handleUpdateHyperparam = useCallback(
    (keyOrBatch: string | Record<string, number | string>, value?: number | string) => {
      const updates: Record<string, number | string> =
        typeof keyOrBatch === 'string' ? { [keyOrBatch]: value! } : keyOrBatch;
      const keys = Object.keys(updates);

      // Decision threshold tau is an inference-time cutoff on Logistic Regression / Decision Tree:
      // never reset weights/epoch or trigger a training step() when updating it.
      if (
        (currentAlgorithm.id === 'logistic-regression' || currentAlgorithm.id === 'decision-tree') &&
        keys.length === 1 &&
        keys[0] === 'threshold'
      ) {
        handleUpdateThreshold(Number(updates.threshold));
        return;
      }

      const normalizedUpdates = { ...updates };
      if (
        currentAlgorithm.id === 'logistic-regression' &&
        'threshold' in normalizedUpdates
      ) {
        normalizedUpdates.threshold = clampThreshold(normalizedUpdates.threshold);
      }

      const updated = { ...hyperparamsRef.current, ...normalizedUpdates };
      hyperparamsRef.current = updated;
      setHyperparams(updated);
      if (!isRunning && currentAlgorithm.stage !== 'rl-gridworld') {
        setScrubIndex(null);
        setIsConverged(false);
        ringBufferRef.current.clear();
      }
      updateHyperparams(updated);
      if (epoch > 0 && !isRunning && currentAlgorithm.executionMode !== 'phase-walkthrough') {
        step();
      }
    },
    [currentAlgorithm.executionMode, currentAlgorithm.id, currentAlgorithm.stage, epoch, handleUpdateThreshold, isRunning, step, updateHyperparams]
  );

  // Point manipulation handlers
  const handleAddPoint = (pt: { x1: number; x2: number; y: number }) => {
    const newPt: DataPoint2D = {
      id: Date.now(),
      x1: pt.x1,
      x2: pt.x2,
      y: pt.y,
      isTest: false,
    };
    const updated = [...trainData, newPt];
    setTrainData(updated);
    const isLive = epoch > 0;
    setDataset(updated, testData, datasetOptions.seed, isLive);
    if (isLive) {
      evalGrid(gridResolution, [-6, 6, -6, 6]);
    }
    dirtyManagerRef.current.markDirty();
  };

  const handleUpdatePoint = (pt: DataPoint2D) => {
    const updateInList = (list: DataPoint2D[]) =>
      list.map((p) => (p.id === pt.id ? pt : p));
    const nextTrain = updateInList(trainData);
    const nextTest = updateInList(testData);
    setTrainData(nextTrain);
    setTestData(nextTest);
    const isLive = epoch > 0;
    setDataset(nextTrain, nextTest, datasetOptions.seed, isLive);
    if (isLive) {
      evalGrid(gridResolution, [-6, 6, -6, 6]);
    }
    dirtyManagerRef.current.markDirty();
  };

  const handleDeletePoint = (id: number) => {
    const nextTrain = trainData.filter((p) => p.id !== id);
    const nextTest = testData.filter((p) => p.id !== id);
    setTrainData(nextTrain);
    setTestData(nextTest);
    const isLive = epoch > 0;
    setDataset(nextTrain, nextTest, datasetOptions.seed, isLive);
    if (isLive) {
      evalGrid(gridResolution, [-6, 6, -6, 6]);
    }
    dirtyManagerRef.current.markDirty();
  };

  const handleDragStart = () => {
    if (epoch > 0) {
      setGridResolution(64);
      evalGrid(64, [-6, 6, -6, 6]);
    }
  };

  const handleDragEnd = () => {
    if (epoch > 0) {
      setGridResolution(96);
      evalGrid(96, [-6, 6, -6, 6]);
    }
  };


  const handleCopyShareLink = () => {
    navigator.clipboard.writeText(window.location.href);
  };

  const handleExportPng = () => {
    let canvas = document.querySelector('.canvas-main') as HTMLCanvasElement | null;
    if (!canvas) {
      canvas = document.querySelector('canvas') as HTMLCanvasElement | null;
    }
    if (canvas) {
      try {
        const link = document.createElement('a');
        link.download = `mlcanvas-${currentAlgorithm.id}-${Date.now()}.png`;
        link.href = canvas.toDataURL('image/png');
        link.click();
      } catch (err) {
        console.error('Failed to export canvas image:', err);
      }
      return;
    }
    alert('Export is currently available for 2D Canvas visualizations (Stages 01–04). Stage 05 uses interactive SVG/DOM layers.');
  };

  // Derived state: Scrubbed vs Live
  const scrubbedSnapshot = scrubIndex !== null ? ringBufferRef.current.get(scrubIndex) : null;
  const displayedEpoch = scrubbedSnapshot ? scrubbedSnapshot.epoch : epoch;
  const displayedArtifacts = scrubbedSnapshot?.artifacts ?? artifacts;
  const displayedMetrics = scrubbedSnapshot?.metrics ?? latestMetrics;
  const displayedExplanation = scrubbedSnapshot?.explanation ?? latestExplanation;
  const displayedGridValues = scrubbedSnapshot ? (scrubbedSnapshot.gridValues ?? null) : gridValues;

  const effectiveDecisionThreshold =
    currentAlgorithm.id === 'logistic-regression'
      ? clampThreshold(hyperparams.threshold ?? 0.5)
      : currentAlgorithm.id === 'decision-tree'
      ? Math.max(0.05, Math.min(0.95, Number(hyperparams.threshold ?? 0.5)))
      : decisionThreshold;

  const predictValueAtPoint = useCallback(
    (x1: number, x2: number): number => {
      try {
        const state = scrubbedSnapshot?.modelState ?? latestModelStateRef.current;
        if (!state) return 0;
        const input = new Float32Array([x1, x2]);
        const pred = (currentAlgorithm as any).predictWithDataset
          ? (currentAlgorithm as any).predictWithDataset(input, state, { train: trainData, test: testData })
          : currentAlgorithm.predict(input, state, hyperparams);
        const val = typeof pred === 'number' ? pred : pred[0];
        if (currentAlgorithm.id === 'logistic-regression' || currentAlgorithm.id === 'decision-tree') {
          return val;
        }
        return val - effectiveDecisionThreshold;
      } catch {
        return 0;
      }
    },
    [scrubbedSnapshot, currentAlgorithm, hyperparams, effectiveDecisionThreshold, trainData, testData]
  );

  // Global Keyboard Shortcuts
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
      } else if (e.code === 'Space') {
        e.preventDefault();
        if (currentAlgorithm.executionMode === 'closed-form') return;
        handleTogglePlay();
      } else if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        handleReset();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        if (currentAlgorithm.executionMode === 'closed-form') return;
        handleStep();
      }
    };
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [handleTogglePlay, handleReset, handleStep, currentAlgorithm.executionMode]);

  const isCrossTaskMismatch =
    currentAlgorithm.stage !== 'vision-grid' &&
    currentAlgorithm.stage !== 'sequence-attn' &&
    currentAlgorithm.stage !== 'unsupervised-2d' &&
    currentAlgorithm.stage !== 'tree-partition' &&
    currentAlgorithm.stage !== 'neural-graph' &&
    currentAlgorithm.id !== 'knn' &&
    ((currentAlgorithm.taskType === 'regression' && getDatasetTaskType(datasetOptions.kind) === 'classification') ||
      (currentAlgorithm.taskType === 'classification' && getDatasetTaskType(datasetOptions.kind) === 'regression'));

  const crossTaskMessage =
    !isCrossTaskMismatch
      ? undefined
      : currentAlgorithm.taskType === 'regression' && getDatasetTaskType(datasetOptions.kind) === 'classification'
      ? 'Cross-Task Mode: Regression model is fitting binary class labels y ∈ {-1, +1} as continuous numerical targets.'
      : currentAlgorithm.taskType === 'classification' && getDatasetTaskType(datasetOptions.kind) === 'regression'
      ? 'Cross-Task Mode: Classifier is binarizing continuous targets around threshold zero.'
      : undefined;

  return (
    <div className={`workbench-shell ${reducedMotion ? 'reduced-motion' : ''}`} data-node-id="3:13">
      {/* 3-Step Calibration Coachmark Strip */}
      <CoachmarkStrip />

      {/* Unified Top Control Bar (Figma 3:14) */}
      <TopControlBar
        isRunning={isRunning}
        epoch={displayedEpoch}
        rlEpisodeCount={
          currentAlgorithm.stage === 'rl-gridworld'
            ? Number(displayedMetrics?.custom?.totalEpisodes ?? 0)
            : undefined
        }
        isScrubbing={scrubIndex !== null}
        isConverged={isConverged}
        currentAlgorithm={currentAlgorithm}
        datasetTaskType={getDatasetTaskType(datasetOptions.kind)}
        hyperparams={hyperparams}
        theme={theme}
        reducedMotion={reducedMotion}
        onTogglePlay={handleTogglePlay}
        onReset={handleReset}
        onStep={handleStep}
        onOpenAlgorithmModal={() => setIsCommandPaletteOpen(true)}
        onUpdateHyperparam={handleUpdateHyperparam}
        onToggleTheme={() => setTheme(theme === 'paper' ? 'carbon' : 'paper')}
        onToggleReducedMotion={() => setReducedMotion(!reducedMotion)}
        onCopyShareLink={handleCopyShareLink}
        onExportPng={handleExportPng}
      />

      {/* 3-Column Workspace Grid (Figma 3:17) */}
      <div className="workspace-grid" data-node-id="3:17">
        {/* Left Column: DataManifoldSidebar (Figma 3:18) */}
        <DataSidebar
          options={datasetOptions}
          theme={theme}
          pointerMode={pointerMode}
          activeClass={activeClass}
          hyperparams={hyperparams}
          advancedParams={currentAlgorithm.advancedParams}
          stage={currentAlgorithm.stage}
          algorithmId={currentAlgorithm.id}
          spatialVision={spatialVision}
          sequenceDataset={sequenceTokens}
          rlGridworld={rlGridworld}
          activeBrush={activeBrush}
          onUpdateOptions={handleUpdateDatasetOptions}
          onSetPointerMode={setPointerMode}
          onSetActiveClass={setActiveClass}
          onRegenerate={regenerateDataset}
          onUpdateHyperparam={handleUpdateHyperparam}
          onSelectVisionPreset={handleSelectVisionPreset}
          onUpdateSpatialVision={handleUpdateSpatialVision}
          onShiftInputRight={handleShiftInputRight}
          onSelectSequencePreset={handleSelectSequencePreset}
          onUpdateSequenceDataset={handleUpdateSequenceDataset}
          onToggleTokenMask={handleToggleTokenMask}
          onSelectFocusToken={handleSelectFocusToken}
          onSelectGridworldPreset={handleSelectGridworldPreset}
          onUpdateRLGridworld={handleUpdateRLGridworld}
          onSelectBrush={setActiveBrush}
        />

        {/* Center Column: InteractiveStageContainer (Figma 3:20) */}
        <InteractiveStageContainer
          algorithm={currentAlgorithm}
          theme={theme}
          datasetTaskType={getDatasetTaskType(datasetOptions.kind)}
          trainData={trainData}
          testData={testData}
          artifacts={displayedArtifacts}
          gridValues={displayedGridValues}
          gridResolution={gridResolution}
          dirtyManager={dirtyManagerRef.current}
          circuitBreakerTripped={circuitBreakerTripped}
          isCrossTaskMismatch={isCrossTaskMismatch}
          crossTaskMessage={crossTaskMessage}
          pointerMode={pointerMode}
          activeClass={activeClass}
          showTestData={showTestData}
          onToggleShowTestData={handleToggleShowTestData}
          hyperparams={hyperparams}
          onUpdateHyperparam={handleUpdateHyperparam}
          onAddPoint={handleAddPoint}
          onUpdatePoint={handleUpdatePoint}
          onDeletePoint={handleDeletePoint}
          onClampAndRecover={clampAndRecover}
          onDragStart={handleDragStart}
          onDragEnd={handleDragEnd}
          predictValueAtPoint={predictValueAtPoint}
          onSyncCentroids={syncCentroids}
          inputGrid={spatialVision.inputGrid}
          sequenceDataset={sequenceTokens}
          rlGridworld={rlGridworld}
          activeBrush={activeBrush}
          onUpdateInputGrid={handleUpdateInputGrid}
          onSelectToken={handleSelectFocusToken}
          onPinGate={pinGateValve}
          onPaintTile={handlePaintGridworldTile}
          scrubbedEpoch={scrubIndex !== null ? displayedEpoch : null}
          onReturnToLive={handleReturnToLive}
        />

        {/* Right Column: TelemetrySidebar (Figma 3:22) */}
        <TelemetrySidebar
          algorithm={currentAlgorithm}
          datasetTaskType={getDatasetTaskType(datasetOptions.kind)}
          metrics={displayedMetrics}
          historyBuffer={ringBufferRef.current}
          theme={theme}
          decisionThreshold={effectiveDecisionThreshold}
          onUpdateThreshold={handleUpdateThreshold}
          isConverged={isConverged}
          explanation={displayedExplanation}
          currentEpoch={epoch}
          rlEpisodeCount={
            currentAlgorithm.stage === 'rl-gridworld'
              ? Number(displayedMetrics?.custom?.totalEpisodes ?? 0)
              : undefined
          }
          scrubIndex={scrubIndex}
          onScrubIndexChange={handleScrubIndexChange}
          onReturnToLive={handleReturnToLive}
          onPause={pause}
          isRunning={isRunning}
        />
      </div>

      {/* Cmd+K Spotlight Command Palette */}
      {isCommandPaletteOpen && (
        <Suspense fallback={null}>
          <CommandPalette
            isOpen={isCommandPaletteOpen}
            onClose={() => setIsCommandPaletteOpen(false)}
            selectedAlgorithmId={selectedAlgorithmId}
            onSelectAlgorithm={handleSelectAlgorithm}
          />
        </Suspense>
      )}
    </div>
  );
};
