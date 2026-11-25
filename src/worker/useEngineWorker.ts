import { useRef, useEffect, useState, useCallback } from 'react';
import { DataPoint2D } from '../datasets/tabular2d';
import { StepResult, VisualArtifact, ExplanationPayload, SpatialVisionDataset, SequenceDataset, RLGridworldDataset } from '../types/algorithm';
import { DirtyManager } from '../rendering/dirty-manager';

export interface UseEngineWorkerOptions {
  dirtyManager: DirtyManager;
  onStepResult?: (
    result: StepResult,
    artifacts: VisualArtifact[],
    explanation: ExplanationPayload,
    serializedState?: Float32Array,
    gridBuffer?: Float32Array,
    gridResolution?: number,
    gridBounds?: [number, number, number, number]
  ) => void;
  onInferenceResult?: (
    result: StepResult,
    artifacts: VisualArtifact[],
    explanation: ExplanationPayload,
    serializedState?: Float32Array,
    gridBuffer?: Float32Array,
    gridResolution?: number,
    gridBounds?: [number, number, number, number]
  ) => void;
  onGridResult?: (buffer: Float32Array, resolution: number, bounds: [number, number, number, number]) => void;
  onInitState?: (explanation: ExplanationPayload) => void;
  autoPauseOnConverged?: boolean;
}

export function useEngineWorker({
  dirtyManager,
  onStepResult,
  onInferenceResult,
  onGridResult,
  onInitState,
  autoPauseOnConverged = false,
}: UseEngineWorkerOptions) {
  const workerRef = useRef<Worker | null>(null);
  const isRunningRef = useRef(false);
  const rafIdRef = useRef<number | null>(null);
  const stepTimerRef = useRef<number | null>(null);
  const stepStartTimeRef = useRef<number>(0);

  const [isRunning, setIsRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [circuitBreakerTripped, setCircuitBreakerTripped] = useState(false);
  const autoPausedByVisibilityRef = useRef(false);

  // Keep callback refs up to date without re-instantiating the worker
  const onStepResultRef = useRef(onStepResult);
  onStepResultRef.current = onStepResult;

  const onInferenceResultRef = useRef(onInferenceResult);
  onInferenceResultRef.current = onInferenceResult;

  const onGridResultRef = useRef(onGridResult);
  onGridResultRef.current = onGridResult;

  const onInitStateRef = useRef(onInitState);
  onInitStateRef.current = onInitState;

  const dirtyManagerRef = useRef(dirtyManager);
  dirtyManagerRef.current = dirtyManager;

  const autoPauseOnConvergedRef = useRef(autoPauseOnConverged);
  autoPauseOnConvergedRef.current = autoPauseOnConverged;

  const lastInitMsgRef = useRef<any>(null);

  const attachWorkerListeners = useCallback((worker: Worker) => {
    worker.onmessage = (event: MessageEvent) => {
      const msg = event.data;
      switch (msg.type) {
        case 'INIT_STATE': {
          setEpoch(0);
          isRunningRef.current = false;
          setIsRunning(false);
          if (rafIdRef.current !== null) {
            cancelAnimationFrame(rafIdRef.current);
            rafIdRef.current = null;
          }
          if (onInitStateRef.current) {
            onInitStateRef.current(msg.explanation);
          }
          dirtyManagerRef.current.markDirty();
          break;
        }

        case 'STEP_RESULT':
        case 'STEP_DONE': {
          const currentEpoch = msg.result.epoch;
          setEpoch(currentEpoch);

          const gridBuf = msg.gridBuffer ? new Float32Array(msg.gridBuffer) : undefined;

          if (onStepResultRef.current) {
            onStepResultRef.current(
              msg.result,
              msg.artifacts || [],
              msg.explanation,
              msg.serializedState ? new Float32Array(msg.serializedState) : undefined,
              gridBuf,
              msg.gridResolution,
              msg.gridBounds
            );
          }
          dirtyManagerRef.current.markDirty();

          // TensorFlow Playground continuous loop: paced at steady 100ms interval (10 ticks/sec)
          // For finite-round algorithms (e.g. Stage 02 boosting), stop cleanly on convergence.
          if (isRunningRef.current && autoPauseOnConvergedRef.current && msg.result?.converged) {
            isRunningRef.current = false;
            setIsRunning(false);
            workerRef.current?.postMessage({ type: 'PAUSE' });
          } else if (isRunningRef.current) {
            const elapsed = performance.now() - stepStartTimeRef.current;
            const targetInterval = 100; // 100ms cadence
            const delay = Math.max(0, targetInterval - elapsed);

            stepTimerRef.current = window.setTimeout(() => {
              if (isRunningRef.current && workerRef.current) {
                stepStartTimeRef.current = performance.now();
                workerRef.current.postMessage({ type: 'STEP' });
              }
            }, delay);
          }
          break;
        }

        case 'INFERENCE_RESULT': {
          const currentEpoch = msg.result?.epoch;
          if (typeof currentEpoch === 'number') {
            setEpoch(currentEpoch);
          }

          const gridBuf = msg.gridBuffer ? new Float32Array(msg.gridBuffer) : undefined;

          if (onInferenceResultRef.current) {
            onInferenceResultRef.current(
              msg.result,
              msg.artifacts || [],
              msg.explanation,
              msg.serializedState ? new Float32Array(msg.serializedState) : undefined,
              gridBuf,
              msg.gridResolution,
              msg.gridBounds
            );
          }
          dirtyManagerRef.current.markDirty();
          break;
        }

        case 'GRID_EVAL_RESULT': {
          if (onGridResultRef.current) {
            const arr =
              msg.buffer instanceof Float32Array
                ? new Float32Array(msg.buffer)
                : new Float32Array(msg.buffer);
            onGridResultRef.current(arr, msg.resolution, msg.bounds);
          }
          dirtyManagerRef.current.markDirty();
          break;
        }

        case 'CIRCUIT_BREAKER_TRIPPED': {
          setCircuitBreakerTripped(true);
          // Pure infinite loop: continue execution without auto-pausing on safety trips with 100ms pacing
          if (isRunningRef.current) {
            stepTimerRef.current = window.setTimeout(() => {
              if (isRunningRef.current && workerRef.current) {
                stepStartTimeRef.current = performance.now();
                workerRef.current.postMessage({ type: 'STEP' });
              }
            }, 100);
          }
          dirtyManagerRef.current.markDirty();
          break;
        }

        case 'STATE_CHANGED': {
          dirtyManagerRef.current.markDirty();
          break;
        }
      }
    };

    worker.onerror = (err) => {
      console.error('[EngineWorker Error]', err);
    };
  }, []);

  const getWorker = useCallback((): Worker | null => {
    if (typeof window === 'undefined') return null;
    if (!workerRef.current) {
      const worker = new Worker(
        new URL('./engine.worker.ts', import.meta.url),
        { type: 'module' }
      );
      attachWorkerListeners(worker);
      workerRef.current = worker;
      if (lastInitMsgRef.current) {
        worker.postMessage(lastInitMsgRef.current);
      }
    }
    return workerRef.current;
  }, [attachWorkerListeners]);

  const postToWorker = useCallback(
    (msg: any) => {
      const worker = getWorker();
      worker?.postMessage(msg);
    },
    [getWorker]
  );

  useEffect(() => {
    if (typeof window === 'undefined') return;

    const handleVisibilityChange = () => {
      if (document.hidden) {
        if (isRunningRef.current) {
          autoPausedByVisibilityRef.current = true;
          isRunningRef.current = false;
          setIsRunning(false);
          if (rafIdRef.current !== null) {
            cancelAnimationFrame(rafIdRef.current);
            rafIdRef.current = null;
          }
        }
      } else {
        if (autoPausedByVisibilityRef.current) {
          autoPausedByVisibilityRef.current = false;
          isRunningRef.current = true;
          setIsRunning(true);
          postToWorker({ type: 'STEP' });
        }
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      if (stepTimerRef.current !== null) {
        clearTimeout(stepTimerRef.current);
        stepTimerRef.current = null;
      }
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      workerRef.current?.terminate();
      workerRef.current = null;
    };
  }, [getWorker, postToWorker]);

  const initAlgorithm = useCallback(
    (
      algorithmId: string,
      hyperparams: Record<string, number | string>,
      trainData: DataPoint2D[],
      testData: DataPoint2D[],
      seed: number = 42,
      spatialVision?: SpatialVisionDataset,
      sequenceTokens?: SequenceDataset,
      rlGridworld?: RLGridworldDataset
    ) => {
      isRunningRef.current = false;
      autoPausedByVisibilityRef.current = false;
      setIsRunning(false);
      if (stepTimerRef.current !== null) {
        clearTimeout(stepTimerRef.current);
        stepTimerRef.current = null;
      }
      if (rafIdRef.current !== null) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
      setCircuitBreakerTripped(false);
      setEpoch(0);
      const initMsg = {
        type: 'INIT_ALGORITHM',
        algorithmId,
        hyperparams,
        trainData,
        testData,
        seed,
        spatialVision: spatialVision
          ? {
              ...spatialVision,
              inputGrid: new Float32Array(spatialVision.inputGrid),
            }
          : undefined,
        sequenceTokens: sequenceTokens
          ? {
              ...sequenceTokens,
              sequence: sequenceTokens.sequence.map((s) => ({
                ...s,
                embedding: new Float32Array(s.embedding),
              })),
              targetSequence: sequenceTokens.targetSequence
                ? sequenceTokens.targetSequence.map((s) => ({
                    ...s,
                    embedding: new Float32Array(s.embedding),
                  }))
                : undefined,
            }
          : undefined,
        rlGridworld: rlGridworld
          ? {
              ...rlGridworld,
              tiles: new Uint8Array(rlGridworld.tiles),
              startPos: [...rlGridworld.startPos],
              goalPos: [...rlGridworld.goalPos],
            }
          : undefined,
      };
      lastInitMsgRef.current = initMsg;
      postToWorker(initMsg);
    },
    [postToWorker]
  );

  const setSequenceTokens = useCallback(
    (sequenceTokens: SequenceDataset, liveUpdate: boolean = true) => {
      postToWorker({
        type: 'SET_SEQUENCE_TOKENS',
        sequenceTokens: {
          ...sequenceTokens,
          sequence: sequenceTokens.sequence.map((s) => ({
            ...s,
            embedding: new Float32Array(s.embedding),
          })),
          targetSequence: sequenceTokens.targetSequence
            ? sequenceTokens.targetSequence.map((s) => ({
                ...s,
                embedding: new Float32Array(s.embedding),
              }))
            : undefined,
        },
        liveUpdate,
      });
    },
    [postToWorker]
  );

  const pinGateValve = useCallback(
    (stepIndex: number, gateId: string, value?: number) => {
      postToWorker({
        type: 'PIN_GATE_VALVE',
        stepIndex,
        gateId,
        value,
      });
    },
    [postToWorker]
  );

  const setSpatialVision = useCallback(
    (spatialVision: SpatialVisionDataset, liveUpdate: boolean = true) => {
      postToWorker({
        type: 'SET_SPATIAL_VISION',
        spatialVision: {
          ...spatialVision,
          inputGrid: new Float32Array(spatialVision.inputGrid),
        },
        liveUpdate,
      });
    },
    [postToWorker]
  );

  const setRLGridworld = useCallback(
    (rlGridworld: RLGridworldDataset, liveUpdate: boolean = true, preserveState: boolean = false) => {
      postToWorker({
        type: 'SET_RL_GRIDWORLD',
        rlGridworld: {
          ...rlGridworld,
          tiles: new Uint8Array(rlGridworld.tiles),
          startPos: [...rlGridworld.startPos],
          goalPos: [...rlGridworld.goalPos],
        },
        liveUpdate,
        preserveState,
      });
    },
    [postToWorker]
  );

  const setDataset = useCallback(
    (
      trainData: DataPoint2D[],
      testData: DataPoint2D[],
      seed: number = 42,
      liveUpdate: boolean = false
    ) => {
      setCircuitBreakerTripped(false);
      postToWorker({
        type: 'SET_DATASET',
        trainData,
        testData,
        seed,
        liveUpdate,
      });
    },
    [postToWorker]
  );

  const updateHyperparams = useCallback(
    (hyperparams: Record<string, number | string>, inferenceOnly: boolean = false) => {
      postToWorker({
        type: 'UPDATE_HYPERPARAMS',
        hyperparams,
        inferenceOnly,
      });
    },
    [postToWorker]
  );

  const step = useCallback(() => {
    if (isRunningRef.current) return;
    postToWorker({ type: 'STEP' });
  }, [postToWorker]);

  const start = useCallback(() => {
    if (isRunningRef.current) return;
    setCircuitBreakerTripped(false);
    isRunningRef.current = true;
    setIsRunning(true);
    stepStartTimeRef.current = performance.now();
    postToWorker({ type: 'STEP' });
  }, [postToWorker]);

  const pause = useCallback(() => {
    isRunningRef.current = false;
    setIsRunning(false);
    if (stepTimerRef.current !== null) {
      clearTimeout(stepTimerRef.current);
      stepTimerRef.current = null;
    }
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    postToWorker({ type: 'PAUSE' });
  }, [postToWorker]);

  const reset = useCallback(() => {
    isRunningRef.current = false;
    setIsRunning(false);
    if (stepTimerRef.current !== null) {
      clearTimeout(stepTimerRef.current);
      stepTimerRef.current = null;
    }
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    setCircuitBreakerTripped(false);
    setEpoch(0);
    postToWorker({ type: 'RESET' });
  }, [postToWorker]);

  const evalGrid = useCallback(
    (resolution: number = 96, bounds: [number, number, number, number] = [-6, 6, -6, 6]) => {
      postToWorker({
        type: 'EVAL_GRID',
        resolution,
        bounds,
      });
    },
    [postToWorker]
  );

  const clampAndRecover = useCallback(() => {
    setCircuitBreakerTripped(false);
    postToWorker({
      type: 'CLAMP_RECOVER',
    });
  }, [postToWorker]);

  const restoreSnapshot = useCallback(
    (snapshotBuffer: Float32Array) => {
      postToWorker({
        type: 'RESTORE_SNAPSHOT',
        snapshotBuffer: new Float32Array(snapshotBuffer),
      });
    },
    [postToWorker]
  );

  const syncCentroids = useCallback(
    (centroids: [number, number][]) => {
      postToWorker({
        type: 'SYNC_CENTROIDS',
        centroids: centroids.map((c) => [...c]),
      });
    },
    [postToWorker]
  );

  return {
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
    restoreSnapshot,
    syncCentroids,
    setSpatialVision,
    setSequenceTokens,
    setRLGridworld,
    pinGateValve,
  };
}
