import React, { useState, useMemo } from 'react';
import {
  AlgorithmPlugin,
  VisualArtifact,
  RLGridworldArtifact,
  RLGridworldDataset,
  GridworldTileKind,
} from '../types/algorithm';
import { ThemeMode } from '../types/theme';
import { DirtyManager } from './dirty-manager';
import {
  GRID_SIZE,
  NUM_STATES,
  NUM_ACTIONS,
  ACTION_NAMES,
  ACTION_SYMBOLS,
  TILE_VALUES,
} from '../datasets/rlGridworld';
import { renderLatex } from '../utils/katex-render';
import {
  Activity,
  Compass,
  Database,
  ArrowRight,
  TrendingUp,
} from 'lucide-react';

export interface Stage07CanvasProps {
  theme: ThemeMode;
  artifacts: VisualArtifact[];
  dirtyManager: DirtyManager;
  algorithm: AlgorithmPlugin;
  hyperparams?: Record<string, number | string>;
  rlGridworld?: RLGridworldDataset;
  activeBrush?: GridworldTileKind;
  onPaintTile?: (row: number, col: number, tile: GridworldTileKind) => void;
  onUpdateHyperparam?: (keyOrBatch: string | Record<string, number | string>, value?: number | string) => void;
}

export const Stage07Canvas: React.FC<Stage07CanvasProps> = ({
  theme,
  artifacts,
  dirtyManager,
  algorithm,
  hyperparams = {},
  rlGridworld,
  activeBrush = 'wall',
  onPaintTile,
  onUpdateHyperparam,
}) => {
  const isCarbon = theme === 'carbon';

  // Find active Stage 07 artifact
  const rlArtifact = useMemo(() => {
    return artifacts.find((a): a is RLGridworldArtifact => a.type === 'rl-gridworld-overlay');
  }, [artifacts]);

  // Selected or hovered cell for inspection
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);

  // Interaction Mode: Inspect Cell vs Paint Tiles
  const [interactionMode, setInteractionMode] = useState<'inspect' | 'paint'>('inspect');

  // Rollout speed toggle
  const currentSpeed = (hyperparams.rolloutSpeed ?? rlArtifact?.rolloutSpeed ?? 'single-step') as 'single-step' | 'full-episode';

  const tiles = rlArtifact?.tiles ?? rlGridworld?.tiles ?? new Uint8Array(NUM_STATES);
  const qValues = rlArtifact?.qValues ?? new Float32Array(NUM_STATES * NUM_ACTIONS);
  const policyProbs = rlArtifact?.policyProbabilities;
  const policyLogits = rlArtifact?.policyLogits;
  const greedyActions = rlArtifact?.greedyActions ?? new Int32Array(NUM_STATES);
  const agentState = rlArtifact?.agentState ?? {
    row: rlGridworld?.startPos[0] ?? 0,
    col: rlGridworld?.startPos[1] ?? 0,
    lastAction: -1,
    lastReward: 0,
    cumulativeReturn: 0,
    steps: 0,
    isTerminal: false,
  };
  const lastTransition = rlArtifact?.lastTransition;
  const trajectory = rlArtifact?.trajectory ?? [];
  const replaySamples = rlArtifact?.replayBufferSample ?? [];
  const targetSync = rlArtifact?.targetNetworkSync;

  const isPolicyGradient = algorithm.id === 'policy-gradient';
  const isDQN = algorithm.id === 'dqn';
  const isQLearning = algorithm.id === 'q-learning';

  // Active inspected cell: hovered > selected > agent cell
  const activeInspectedCell = hoveredCell ?? selectedCell ?? { row: agentState.row, col: agentState.col };
  const inspectedStateIdx = activeInspectedCell.row * GRID_SIZE + activeInspectedCell.col;

  // Compute Q-value min and max for dynamic normalization
  const { minQ, maxQ } = useMemo(() => {
    let min = -10.0;
    let max = 15.0;
    if (qValues) {
      for (let i = 0; i < qValues.length; i++) {
        const v = qValues[i];
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }
    return { minQ: min, maxQ: max };
  }, [qValues]);

  // Handle cell click (selection or tile painting)
  const handleCellClick = (r: number, c: number) => {
    setSelectedCell({ row: r, col: c });
    if (interactionMode === 'paint' && onPaintTile && activeBrush) {
      onPaintTile(r, c, activeBrush);
    }
    dirtyManager.markDirty();
  };

  // Triangle color helper
  const getTriangleFill = (q: number, prob?: number) => {
    if (prob !== undefined) {
      // Policy gradient probability: intensity scales with prob in [0, 1]
      const alpha = Math.max(0.08, prob * 0.85);
      return isCarbon ? `rgba(82, 118, 184, ${alpha})` : `rgba(30, 64, 175, ${alpha})`;
    }

    if (Math.abs(q) < 0.001) {
      return isCarbon ? 'rgba(46, 46, 43, 0.4)' : 'rgba(220, 218, 211, 0.4)';
    }

    if (q > 0) {
      const norm = Math.min(1.0, q / Math.max(1.0, maxQ));
      const alpha = 0.12 + norm * 0.65;
      return isCarbon ? `rgba(77, 143, 104, ${alpha})` : `rgba(21, 128, 61, ${alpha})`;
    } else {
      const norm = Math.min(1.0, Math.abs(q) / Math.max(1.0, Math.abs(minQ)));
      const alpha = 0.15 + norm * 0.7;
      return isCarbon ? `rgba(194, 103, 82, ${alpha})` : `rgba(217, 78, 52, ${alpha})`;
    }
  };

  return (
    <div className="w-full h-full flex flex-col overflow-hidden bg-[var(--bg-primary)] select-none font-sans">
      {/* 1. Canvas Sub-Header: Mode Banner + Mode Toggle + Rollout Cadence Toggle */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2 border-b border-[var(--border-hairline)] bg-[var(--bg-canvas)] flex-shrink-0 min-w-0">
        <div className="flex flex-wrap items-center gap-2.5 min-w-0">
          <div className="flex items-center gap-1.5 text-[11px] font-mono font-semibold uppercase tracking-wider text-[var(--text-primary)] shrink-0">
            <Compass size={13} className="text-[var(--pigment-class-b)]" />
            <span>Stage 07: Split-Quadrant MDP Arena (6×6 Grid)</span>
          </div>
          <span className="font-mono text-[9px] px-1.5 py-0.5 rounded border border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[var(--text-secondary)] shrink-0">
            {rlGridworld?.preset ? rlGridworld.preset.toUpperCase() : 'CLIFF-WALKING'}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Interaction Mode Toggle: Inspect Cell vs Paint Tiles */}
          <div className="flex items-center gap-1.5 font-mono text-[11px] shrink-0">
            <span className="text-[10px] text-[var(--text-muted)] uppercase whitespace-nowrap">Mode:</span>
            <div className="flex items-center gap-1 rounded border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5">
              <button
                className={`px-2 py-0.5 text-[10px] font-mono whitespace-nowrap transition-colors flex items-center gap-1 ${
                  interactionMode === 'inspect'
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] font-medium shadow-xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
                onClick={() => setInteractionMode('inspect')}
                title="Inspect mode: click any cell to pin its Q-values or action probabilities without altering tiles"
              >
                Inspect 🔍
              </button>
              <button
                className={`px-2 py-0.5 text-[10px] font-mono whitespace-nowrap transition-colors flex items-center gap-1 ${
                  interactionMode === 'paint'
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] font-medium shadow-xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
                onClick={() => setInteractionMode('paint')}
                title={`Paint mode: click cells to paint ${activeBrush} tiles`}
              >
                Paint 🖌 ({activeBrush})
              </button>
            </div>
          </div>

          {/* Rollout Speed Selector */}
          <div className="flex items-center gap-1.5 font-mono text-[11px] shrink-0">
            <span className="text-[10px] text-[var(--text-muted)] uppercase whitespace-nowrap">Cadence:</span>
            <div className="flex items-center gap-1 rounded border border-[var(--border-hairline)] bg-[var(--bg-subtle)] p-0.5">
              <button
                className={`px-2 py-0.5 text-[10px] font-mono whitespace-nowrap transition-colors ${
                  currentSpeed === 'single-step'
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] font-medium shadow-xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
                onClick={() => onUpdateHyperparam?.('rolloutSpeed', 'single-step')}
                title="1 Environment Transition per tick: visualizes exact Bellman backup arc"
              >
                1 Step / Frame
              </button>
              <button
                className={`px-2 py-0.5 text-[10px] font-mono whitespace-nowrap transition-colors ${
                  currentSpeed === 'full-episode'
                    ? 'bg-[var(--text-primary)] text-[var(--text-inverse)] font-medium shadow-xs'
                    : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
                }`}
                onClick={() => onUpdateHyperparam?.('rolloutSpeed', 'full-episode')}
                title="1 Full Episode Rollout per tick: fast training at 60 FPS"
              >
                1 Episode / Frame
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* 2. Main Synchronized 2-Zone Body */}
      <div className="flex-1 flex min-h-0 min-w-0 overflow-hidden">
        {/* ============================================================ */}
        {/* LEFT / CENTER VIEWPORT (65% width): Main Interactive MDP Grid */}
        {/* ============================================================ */}
        <div style={{ flex: '65 1 0%' }} className="h-full min-w-0 flex flex-col border-r border-[var(--border-hairline)] p-4 overflow-y-auto">
          <div className="flex-1 flex items-center justify-center min-h-[380px]">
            <div className="relative w-full max-w-[560px] aspect-square border border-[var(--border-hairline)] bg-[var(--bg-canvas)] shadow-xs">
              {/* SVG 6x6 Grid with 4-Triangle Split Cells */}
              <svg
                viewBox="0 0 600 600"
                className="w-full h-full block"
                style={{ shapeRendering: 'geometricPrecision' }}
              >
                <defs>
                  {/* Arrowhead marker for Bellman backup arc */}
                  <marker
                    id="td-arc-arrow"
                    viewBox="0 0 10 10"
                    refX="6"
                    refY="5"
                    markerWidth="5"
                    markerHeight="5"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill={isCarbon ? '#C26752' : '#D94E34'} />
                  </marker>
                  {/* Glowing agent filter */}
                  <filter id="agent-glow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur in="SourceGraphic" stdDeviation="3" />
                  </filter>
                </defs>

                {/* Draw 36 Grid Cells */}
                {Array.from({ length: GRID_SIZE }).map((_, r) =>
                  Array.from({ length: GRID_SIZE }).map((_, c) => {
                    const idx = r * GRID_SIZE + c;
                    const tileType = tiles[idx];
                    const isWall = tileType === TILE_VALUES.wall;
                    const isLava = tileType === TILE_VALUES.lava;
                    const isGoal = tileType === TILE_VALUES.goal;
                    const isBait = tileType === TILE_VALUES.bait;
                    const isStart = tileType === TILE_VALUES.start;
                    const isAgent = agentState.row === r && agentState.col === c;
                    const isInspected = activeInspectedCell.row === r && activeInspectedCell.col === c;

                    const cellX = c * 100;
                    const cellY = r * 100;
                    const centerX = cellX + 50;
                    const centerY = cellY + 50;

                    // Q-values for 4 actions: [Up, Right, Down, Left]
                    const qUp = qValues[idx * NUM_ACTIONS + 0];
                    const qRight = qValues[idx * NUM_ACTIONS + 1];
                    const qDown = qValues[idx * NUM_ACTIONS + 2];
                    const qLeft = qValues[idx * NUM_ACTIONS + 3];

                    const pUp = policyProbs ? policyProbs[idx * NUM_ACTIONS + 0] : undefined;
                    const pRight = policyProbs ? policyProbs[idx * NUM_ACTIONS + 1] : undefined;
                    const pDown = policyProbs ? policyProbs[idx * NUM_ACTIONS + 2] : undefined;
                    const pLeft = policyProbs ? policyProbs[idx * NUM_ACTIONS + 3] : undefined;

                    const greedyA = greedyActions[idx];

                    return (
                      <g
                        key={`cell-${r}-${c}`}
                        className="cursor-pointer transition-opacity"
                        onClick={() => handleCellClick(r, c)}
                        onMouseEnter={() => setHoveredCell({ row: r, col: c })}
                        onMouseLeave={() => setHoveredCell(null)}
                      >
                        {/* Cell Background */}
                        <rect
                          x={cellX}
                          y={cellY}
                          width={100}
                          height={100}
                          fill={
                            isWall
                              ? isCarbon
                                ? '#20201E'
                                : '#E5E3DC'
                              : isCarbon
                              ? '#181817'
                              : '#FAF9F5'
                          }
                          stroke={isCarbon ? '#2E2E2B' : '#DCDAD3'}
                          strokeWidth={isInspected ? 2 : 1}
                        />

                        {/* Special Tile Overlays */}
                        {isWall ? (
                          <g>
                            {/* Cross-hatch diagonal masonry lines */}
                            <line
                              x1={cellX}
                              y1={cellY}
                              x2={cellX + 100}
                              y2={cellY + 100}
                              stroke={isCarbon ? '#333330' : '#CEC9BE'}
                              strokeWidth={1.5}
                            />
                            <line
                              x1={cellX + 100}
                              y1={cellY}
                              x2={cellX}
                              y2={cellY + 100}
                              stroke={isCarbon ? '#333330' : '#CEC9BE'}
                              strokeWidth={1.5}
                            />
                            <text
                              x={centerX}
                              y={centerY + 4}
                              textAnchor="middle"
                              fill={isCarbon ? '#686763' : '#8A8884'}
                              fontSize="12"
                              fontFamily="JetBrains Mono"
                              fontWeight="bold"
                            >
                              WALL
                            </text>
                          </g>
                        ) : isLava ? (
                          <g>
                            <rect
                              x={cellX + 2}
                              y={cellY + 2}
                              width={96}
                              height={96}
                              fill={isCarbon ? 'rgba(239, 68, 68, 0.22)' : 'rgba(217, 78, 52, 0.18)'}
                            />
                            <text
                              x={centerX}
                              y={centerY - 8}
                              textAnchor="middle"
                              fill={isCarbon ? '#EF4444' : '#D94E34'}
                              fontSize="12"
                              fontWeight="bold"
                              fontFamily="JetBrains Mono"
                            >
                              LAVA
                            </text>
                            <text
                              x={centerX}
                              y={centerY + 12}
                              textAnchor="middle"
                              fill={isCarbon ? '#EF4444' : '#D94E34'}
                              fontSize="10"
                              fontFamily="JetBrains Mono"
                              fontWeight="600"
                            >
                              -10
                            </text>
                          </g>
                        ) : isGoal ? (
                          <g>
                            <rect
                              x={cellX + 2}
                              y={cellY + 2}
                              width={96}
                              height={96}
                              fill={isCarbon ? 'rgba(77, 143, 104, 0.28)' : 'rgba(21, 128, 61, 0.22)'}
                            />
                            <circle
                              cx={centerX}
                              cy={centerY}
                              r={28}
                              fill="none"
                              stroke={isCarbon ? '#4D8F68' : '#15803D'}
                              strokeWidth={1.5}
                              strokeDasharray="3 3"
                            />
                            <text
                              x={centerX}
                              y={centerY - 6}
                              textAnchor="middle"
                              fill={isCarbon ? '#4D8F68' : '#15803D'}
                              fontSize="12"
                              fontWeight="bold"
                              fontFamily="JetBrains Mono"
                            >
                              GOAL
                            </text>
                            <text
                              x={centerX}
                              y={centerY + 12}
                              textAnchor="middle"
                              fill={isCarbon ? '#4D8F68' : '#15803D'}
                              fontSize="11"
                              fontWeight="bold"
                              fontFamily="JetBrains Mono"
                            >
                              {rlGridworld?.preset === 'twin-treasure' ? '+20' : '+10'}
                            </text>
                          </g>
                        ) : isBait ? (
                          <g>
                            <rect
                              x={cellX + 2}
                              y={cellY + 2}
                              width={96}
                              height={96}
                              fill={isCarbon ? 'rgba(217, 119, 6, 0.25)' : 'rgba(180, 83, 9, 0.2)'}
                            />
                            <text
                              x={centerX}
                              y={centerY - 6}
                              textAnchor="middle"
                              fill={isCarbon ? '#D97706' : '#B45309'}
                              fontSize="11"
                              fontWeight="bold"
                              fontFamily="JetBrains Mono"
                            >
                              BAIT
                            </text>
                            <text
                              x={centerX}
                              y={centerY + 12}
                              textAnchor="middle"
                              fill={isCarbon ? '#D97706' : '#B45309'}
                              fontSize="11"
                              fontWeight="bold"
                              fontFamily="JetBrains Mono"
                            >
                              +2
                            </text>
                          </g>
                        ) : (
                          // 4-Triangle Split Cell Rendering for Standard Navigation Tiles
                          <g>
                            {/* North Triangle (Action 0) */}
                            <polygon
                              points={`${cellX},${cellY} ${cellX + 100},${cellY} ${centerX},${centerY}`}
                              fill={getTriangleFill(qUp, pUp)}
                              stroke={isCarbon ? '#2E2E2B' : '#DCDAD3'}
                              strokeWidth={0.5}
                            />
                            {/* East Triangle (Action 1) */}
                            <polygon
                              points={`${cellX + 100},${cellY} ${cellX + 100},${cellY + 100} ${centerX},${centerY}`}
                              fill={getTriangleFill(qRight, pRight)}
                              stroke={isCarbon ? '#2E2E2B' : '#DCDAD3'}
                              strokeWidth={0.5}
                            />
                            {/* South Triangle (Action 2) */}
                            <polygon
                              points={`${cellX + 100},${cellY + 100} ${cellX},${cellY + 100} ${centerX},${centerY}`}
                              fill={getTriangleFill(qDown, pDown)}
                              stroke={isCarbon ? '#2E2E2B' : '#DCDAD3'}
                              strokeWidth={0.5}
                            />
                            {/* West Triangle (Action 3) */}
                            <polygon
                              points={`${cellX},${cellY + 100} ${cellX},${cellY} ${centerX},${centerY}`}
                              fill={getTriangleFill(qLeft, pLeft)}
                              stroke={isCarbon ? '#2E2E2B' : '#DCDAD3'}
                              strokeWidth={0.5}
                            />

                            {/* Q-Value / Policy Probability Text Readouts inside triangles */}
                            {isPolicyGradient ? (
                              // Percentage readout for REINFORCE
                              <g fontSize="7.5" fontFamily="JetBrains Mono" textAnchor="middle" fill={isCarbon ? '#D4D3CD' : '#141413'}>
                                <text x={centerX} y={cellY + 15}>{((pUp ?? 0.25) * 100).toFixed(0)}%</text>
                                <text x={cellX + 84} y={centerY + 3}>{((pRight ?? 0.25) * 100).toFixed(0)}%</text>
                                <text x={centerX} y={cellY + 92}>{((pDown ?? 0.25) * 100).toFixed(0)}%</text>
                                <text x={cellX + 16} y={centerY + 3}>{((pLeft ?? 0.25) * 100).toFixed(0)}%</text>
                              </g>
                            ) : (
                              // Numerical Q-value readout for Q-Learning & DQN
                              <g fontSize="7.5" fontFamily="JetBrains Mono" textAnchor="middle" fill={isCarbon ? '#D4D3CD' : '#141413'}>
                                <text x={centerX} y={cellY + 15}>{qUp.toFixed(1)}</text>
                                <text x={cellX + 83} y={centerY + 3}>{qRight.toFixed(1)}</text>
                                <text x={centerX} y={cellY + 92}>{qDown.toFixed(1)}</text>
                                <text x={cellX + 17} y={centerY + 3}>{qLeft.toFixed(1)}</text>
                              </g>
                            )}

                            {/* Center Greedy Policy Direction Arrow */}
                            <circle cx={centerX} cy={centerY} r={8} fill={isCarbon ? '#181817' : '#FAF9F5'} stroke={isCarbon ? '#3E3E3B' : '#C5C3BB'} strokeWidth={0.75} />
                            <text
                              x={centerX}
                              y={centerY + 3.5}
                              textAnchor="middle"
                              fontSize="10"
                              fontFamily="JetBrains Mono"
                              fontWeight="bold"
                              fill={isCarbon ? '#D4D3CD' : '#141413'}
                            >
                              {ACTION_SYMBOLS[greedyA]}
                            </text>
                          </g>
                        )}

                        {/* Start Position Marker */}
                        {isStart && !isAgent && (
                          <g>
                            <circle
                              cx={centerX}
                              cy={centerY}
                              r={14}
                              fill={isCarbon ? 'rgba(82, 118, 184, 0.2)' : 'rgba(30, 64, 175, 0.15)'}
                              stroke={isCarbon ? '#5276B8' : '#1E40AF'}
                              strokeWidth={1.5}
                            />
                            <text
                              x={centerX}
                              y={centerY + 3.5}
                              textAnchor="middle"
                              fontSize="9"
                              fontFamily="JetBrains Mono"
                              fontWeight="bold"
                              fill={isCarbon ? '#5276B8' : '#1E40AF'}
                            >
                              S
                            </text>
                          </g>
                        )}

                        {/* Coordinate Badge in Corner */}
                        <text
                          x={cellX + 4}
                          y={cellY + 10}
                          fontSize="7"
                          fontFamily="JetBrains Mono"
                          fill={isCarbon ? '#686763' : '#8A8884'}
                          opacity={0.8}
                        >
                          {r},{c}
                        </text>

                        {/* Inspection Box Highlight */}
                        {isInspected && (
                          <rect
                            x={cellX + 1}
                            y={cellY + 1}
                            width={98}
                            height={98}
                            fill="none"
                            stroke={isCarbon ? '#D4D3CD' : '#141413'}
                            strokeWidth={2}
                          />
                        )}
                      </g>
                    );
                  })
                )}

                {/* Trajectory Trail (Dashed Line connecting past steps) */}
                {trajectory.length > 1 && (
                  <path
                    d={trajectory.reduce((acc, pt, i) => {
                      const px = pt.col * 100 + 50;
                      const py = pt.row * 100 + 50;
                      return i === 0 ? `M ${px} ${py}` : `${acc} L ${px} ${py}`;
                    }, '')}
                    fill="none"
                    stroke={isCarbon ? 'rgba(82, 118, 184, 0.6)' : 'rgba(30, 64, 175, 0.5)'}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                  />
                )}

                {/* Bellman Backup Arc (TD Flash connecting s to s' with arrowhead) */}
                {lastTransition && currentSpeed === 'single-step' && (
                  <g className="transition-all animate-pulse">
                    {(() => {
                      const sR = Math.floor(lastTransition.state / GRID_SIZE);
                      const sC = lastTransition.state % GRID_SIZE;
                      const nR = Math.floor(lastTransition.nextState / GRID_SIZE);
                      const nC = lastTransition.nextState % GRID_SIZE;
                      const fromX = sC * 100 + 50;
                      const fromY = sR * 100 + 50;
                      const toX = nC * 100 + 50;
                      const toY = nR * 100 + 50;
                      const midX = (fromX + toX) / 2;
                      const midY = (fromY + toY) / 2 - 25;

                      return (
                        <>
                          <path
                            d={`M ${fromX} ${fromY} Q ${midX} ${midY} ${toX} ${toY}`}
                            fill="none"
                            stroke={isCarbon ? '#C26752' : '#D94E34'}
                            strokeWidth={2.5}
                            markerEnd="url(#td-arc-arrow)"
                          />
                          <circle cx={fromX} cy={fromY} r={3.5} fill={isCarbon ? '#C26752' : '#D94E34'} />
                        </>
                      );
                    })()}
                  </g>
                )}

                {/* Agent Token Marker ● */}
                <g className="transition-all duration-150">
                  <circle
                    cx={agentState.col * 100 + 50}
                    cy={agentState.row * 100 + 50}
                    r={12}
                    fill={isCarbon ? '#C26752' : '#D94E34'}
                    stroke="#FFFFFF"
                    strokeWidth={2}
                    className="drop-shadow-sm"
                  />
                  <circle
                    cx={agentState.col * 100 + 50}
                    cy={agentState.row * 100 + 50}
                    r={5}
                    fill="#FFFFFF"
                  />
                </g>
              </svg>
            </div>
          </div>

          {/* Bottom Live Formula Pill / Status Bar */}
          <div className="mt-3 p-2.5 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] flex flex-wrap items-center justify-between gap-2 text-[11px] font-mono">
            <div className="flex items-center gap-3">
              <span className="text-[var(--text-muted)] uppercase">Agent Position:</span>
              <span className="font-semibold text-[var(--text-primary)]">
                ({agentState.row}, {agentState.col})
              </span>
              <span className="text-[var(--border-hairline)]">|</span>
              <span className="text-[var(--text-muted)] uppercase">Last Reward:</span>
              <span
                className={`font-semibold ${
                  agentState.lastReward > 0
                    ? 'text-[var(--pigment-class-b)]'
                    : agentState.lastReward < 0
                    ? 'text-[var(--pigment-class-a)]'
                    : 'text-[var(--text-primary)]'
                }`}
              >
                {agentState.lastReward >= 0 ? `+${agentState.lastReward.toFixed(2)}` : agentState.lastReward.toFixed(2)}
              </span>
            </div>

            {lastTransition && (
              <div className="flex items-center gap-2">
                <span className="text-[var(--text-muted)]">TD Error δ:</span>
                <span className="font-semibold text-[var(--pigment-class-a)]">
                  {lastTransition.tdError >= 0
                    ? `+${lastTransition.tdError.toFixed(3)}`
                    : lastTransition.tdError.toFixed(3)}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* ============================================================ */}
        {/* RIGHT INSET VIEWPORT (35% width): Algorithm Inspector        */}
        {/* ============================================================ */}
        <div style={{ flex: '35 1 0%' }} className="h-full min-w-0 flex flex-col p-4 overflow-y-auto bg-[var(--bg-primary)]">
          {isQLearning ? (
            /* --- Q-LEARNING INSPECTOR --- */
            <div className="flex flex-col gap-4">
              <div className="panel-header gap-2">
                <span className="panel-title font-mono tracking-wider flex items-center gap-1.5 min-w-0">
                  <Activity size={12} className="text-[var(--pigment-class-a)] shrink-0" />
                  <span className="truncate">BELLMAN BACKUP INSPECTOR</span>
                </span>
                <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] whitespace-nowrap shrink-0">
                  Cell ({activeInspectedCell.row}, {activeInspectedCell.col})
                </span>
              </div>

              {/* 4 Directional Q-Value Bars for Selected Cell */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-2.5">
                <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                  Action Values Q(s, a) for State {inspectedStateIdx}
                </span>

                {[
                  { name: 'North ▲', a: 0, val: qValues[inspectedStateIdx * NUM_ACTIONS + 0] },
                  { name: 'East ▶', a: 1, val: qValues[inspectedStateIdx * NUM_ACTIONS + 1] },
                  { name: 'South ▼', a: 2, val: qValues[inspectedStateIdx * NUM_ACTIONS + 2] },
                  { name: 'West ◀', a: 3, val: qValues[inspectedStateIdx * NUM_ACTIONS + 3] },
                ].map((item) => {
                  const isGreedy = greedyActions[inspectedStateIdx] === item.a;
                  const isPositive = item.val >= 0;
                  const absVal = Math.min(100, (Math.abs(item.val) / Math.max(1, maxQ)) * 100);

                  return (
                    <div key={item.name} className="flex flex-col gap-1 text-[11px] font-mono">
                      <div className="flex justify-between items-center">
                        <span className={`flex items-center gap-1.5 ${isGreedy ? 'font-bold text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
                          {item.name}
                          {isGreedy && <span className="text-[9px] px-1 bg-[var(--bg-active)] rounded">argMax</span>}
                        </span>
                        <span className={`font-semibold ${isPositive ? 'text-[var(--pigment-class-b)]' : 'text-[var(--pigment-class-a)]'}`}>
                          {item.val >= 0 ? `+${item.val.toFixed(3)}` : item.val.toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full h-1.5 rounded-xs bg-[var(--bg-subtle)] overflow-hidden">
                        <div
                          className={`h-full transition-all duration-100 ${isPositive ? 'bg-[var(--pigment-class-b)]' : 'bg-[var(--pigment-class-a)]'}`}
                          style={{ width: `${Math.max(4, absVal)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Exact Formula Plug-in Card */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-2">
                <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                  Bellman Optimality Update Plug-In
                </span>
                <div
                  className="p-2 rounded bg-[var(--bg-subtle)] overflow-x-auto text-[11px]"
                  dangerouslySetInnerHTML={{
                    __html: renderLatex(
                      'Q(s, a) \\leftarrow Q(s, a) + \\alpha \\Big[r + \\gamma \\max_{a\'} Q(s\', a\') - Q(s, a)\\Big]'
                    ),
                  }}
                />
                {lastTransition && (
                  <div className="font-mono text-[10px] space-y-1 text-[var(--text-secondary)] pt-1 border-t border-[var(--border-hairline)]">
                    <div>s: {lastTransition.state} → a: {ACTION_NAMES[lastTransition.action]}</div>
                    <div>r: {lastTransition.reward.toFixed(2)} | target y: {lastTransition.targetQ.toFixed(3)}</div>
                    <div className="text-[var(--pigment-class-a)] font-semibold">
                      TD Error δ = {lastTransition.tdError.toFixed(3)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : isDQN ? (
            /* --- DQN INSPECTOR --- */
            <div className="flex flex-col gap-4">
              <div className="panel-header gap-2">
                <span className="panel-title font-mono tracking-wider flex items-center gap-1.5 min-w-0">
                  <Database size={12} className="text-[var(--pigment-class-b)] shrink-0" />
                  <span className="truncate">DQN NEURAL ACTION-VALUES & REPLAY</span>
                </span>
                <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] whitespace-nowrap shrink-0">
                  Cell ({activeInspectedCell.row}, {activeInspectedCell.col})
                </span>
              </div>

              {/* 4 Directional Neural Q-Value Bars for Selected Cell */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-2.5">
                <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                  Neural Function Q_θ(s, a) for State {inspectedStateIdx}
                </span>

                {[
                  { name: 'North ▲', a: 0, val: qValues[inspectedStateIdx * NUM_ACTIONS + 0] },
                  { name: 'East ▶', a: 1, val: qValues[inspectedStateIdx * NUM_ACTIONS + 1] },
                  { name: 'South ▼', a: 2, val: qValues[inspectedStateIdx * NUM_ACTIONS + 2] },
                  { name: 'West ◀', a: 3, val: qValues[inspectedStateIdx * NUM_ACTIONS + 3] },
                ].map((item) => {
                  const isGreedy = greedyActions[inspectedStateIdx] === item.a;
                  const isPositive = item.val >= 0;
                  const absVal = Math.min(100, (Math.abs(item.val) / Math.max(1, maxQ)) * 100);

                  return (
                    <div key={item.name} className="flex flex-col gap-1 text-[11px] font-mono">
                      <div className="flex justify-between items-center">
                        <span className={`flex items-center gap-1.5 ${isGreedy ? 'font-bold text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
                          {item.name}
                          {isGreedy && <span className="text-[9px] px-1 bg-[var(--bg-active)] rounded">argMax</span>}
                        </span>
                        <span className={`font-semibold ${isPositive ? 'text-[var(--pigment-class-b)]' : 'text-[var(--pigment-class-a)]'}`}>
                          {item.val >= 0 ? `+${item.val.toFixed(3)}` : item.val.toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full h-1.5 rounded-xs bg-[var(--bg-subtle)] overflow-hidden">
                        <div
                          className={`h-full transition-all duration-100 ${isPositive ? 'bg-[var(--pigment-class-b)]' : 'bg-[var(--pigment-class-a)]'}`}
                          style={{ width: `${Math.max(4, absVal)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Target Network Sync Gauge */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-2">
                <div className="flex justify-between items-center text-[10px] font-mono">
                  <span className="text-[var(--text-muted)] uppercase">
                    Target Sync <span className="normal-case">θ⁻ ← θ</span>:
                  </span>
                  <span className="font-semibold text-[var(--text-primary)]">
                    {targetSync ? `${targetSync.stepCounter} / ${targetSync.syncInterval} Steps` : '—'}
                  </span>
                </div>
                <div className="w-full h-2 rounded-xs bg-[var(--bg-subtle)] overflow-hidden">
                  <div
                    className="h-full bg-[var(--pigment-class-b)] transition-all duration-75"
                    style={{ width: `${(targetSync?.syncProgress ?? 0) * 100}%` }}
                  />
                </div>
                <span className="text-[9px] font-mono text-[var(--text-secondary)] block">
                  Holds Bellman targets fixed for {targetSync?.syncInterval ?? 20} steps to prevent deadly moving target divergence.
                </span>
              </div>

              {/* Experience Replay Buffer Mini Strip */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-2">
                <div className="flex justify-between items-center text-[10px] font-mono">
                  <span className="text-[var(--text-muted)] uppercase">Replay Buffer Sample (Mini-Batch)</span>
                  <span className="text-[9px] text-[var(--pigment-class-a)] font-semibold">16 Random Draws</span>
                </div>

                <div className="max-h-[220px] overflow-y-auto space-y-1 font-mono text-[10px]">
                  {replaySamples.length > 0 ? (
                    replaySamples.slice(-12).reverse().map((sample, idx) => (
                      <div
                        key={idx}
                        className={`p-1.5 rounded flex items-center justify-between border ${
                          sample.isSampled
                            ? 'border-[var(--pigment-class-a)] bg-[var(--fill-pos)]'
                            : 'border-[var(--border-hairline)] bg-[var(--bg-subtle)] opacity-75'
                        }`}
                      >
                        <div className="flex items-center gap-1.5 min-w-0">
                          <span className="font-semibold text-[var(--text-primary)]">s:{sample.s}</span>
                          <ArrowRight size={10} className="text-[var(--text-muted)] shrink-0" />
                          <span className="text-[var(--text-secondary)]">a:{ACTION_SYMBOLS[sample.a]}</span>
                        </div>
                        <div className="flex items-center gap-2 shrink-0">
                          <span className={sample.r > 0 ? 'text-[var(--pigment-class-b)]' : sample.r < 0 ? 'text-[var(--pigment-class-a)]' : 'text-[var(--text-muted)]'}>
                            r:{sample.r.toFixed(1)}
                          </span>
                          <span className="text-[var(--text-muted)]">s':{sample.nextS}</span>
                          {sample.done && <span className="text-[8px] px-1 bg-[var(--bg-active)] rounded">done</span>}
                        </div>
                      </div>
                    ))
                  ) : (
                    <div className="text-[10px] text-[var(--text-muted)] py-4 text-center">
                      Buffer filling... Run transitions to populate replay memory.
                    </div>
                  )}
                </div>
              </div>

              {/* DQN Bellman MSE Loss Card */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-2">
                <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                  DQN Bellman MSE Loss
                </span>
                <div
                  className="p-2 rounded bg-[var(--bg-subtle)] overflow-x-auto text-[10px] [&_.katex]:text-[0.88em]"
                  dangerouslySetInnerHTML={{
                    __html: renderLatex(
                      'L(\\theta) = \\mathbb{E} \\Big[\\Big(r + \\gamma \\max_{a\'} Q_{\\theta^-}(s\', a\') - Q_\\theta(s, a)\\Big)^2\\Big]'
                    ),
                  }}
                />
                {lastTransition && (
                  <div className="font-mono text-[10px] space-y-1 text-[var(--text-secondary)] pt-1 border-t border-[var(--border-hairline)]">
                    <div>s: {lastTransition.state} → a: {ACTION_NAMES[lastTransition.action]}</div>
                    <div>r: {lastTransition.reward.toFixed(2)} | target y: {lastTransition.targetQ.toFixed(3)}</div>
                    <div className="text-[var(--pigment-class-a)] font-semibold">
                      TD Error δ = {lastTransition.tdError.toFixed(3)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : (
            /* --- REINFORCE POLICY GRADIENT INSPECTOR --- */
            <div className="flex flex-col gap-4">
              <div className="panel-header gap-2">
                <span className="panel-title font-mono tracking-wider flex items-center gap-1.5 min-w-0">
                  <TrendingUp size={12} className="text-[var(--pigment-class-b)] shrink-0" />
                  <span className="truncate">REINFORCE POLICY & TRAJECTORY</span>
                </span>
                <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg-subtle)] border border-[var(--border-hairline)] whitespace-nowrap shrink-0">
                  Cell ({activeInspectedCell.row}, {activeInspectedCell.col})
                </span>
              </div>

              {/* 4 Directional Policy Probability Bars for Selected Cell */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-2.5">
                <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                  Policy Probabilities π_θ(a|s) for State {inspectedStateIdx}
                </span>

                {[
                  {
                    name: 'North ▲',
                    a: 0,
                    prob: policyProbs ? policyProbs[inspectedStateIdx * NUM_ACTIONS + 0] : 0.25,
                    logit: policyLogits ? policyLogits[inspectedStateIdx * NUM_ACTIONS + 0] : 0.0,
                  },
                  {
                    name: 'East ▶',
                    a: 1,
                    prob: policyProbs ? policyProbs[inspectedStateIdx * NUM_ACTIONS + 1] : 0.25,
                    logit: policyLogits ? policyLogits[inspectedStateIdx * NUM_ACTIONS + 1] : 0.0,
                  },
                  {
                    name: 'South ▼',
                    a: 2,
                    prob: policyProbs ? policyProbs[inspectedStateIdx * NUM_ACTIONS + 2] : 0.25,
                    logit: policyLogits ? policyLogits[inspectedStateIdx * NUM_ACTIONS + 2] : 0.0,
                  },
                  {
                    name: 'West ◀',
                    a: 3,
                    prob: policyProbs ? policyProbs[inspectedStateIdx * NUM_ACTIONS + 3] : 0.25,
                    logit: policyLogits ? policyLogits[inspectedStateIdx * NUM_ACTIONS + 3] : 0.0,
                  },
                ].map((item) => {
                  const isGreedy = greedyActions[inspectedStateIdx] === item.a;
                  const pct = Math.max(0, Math.min(100, item.prob * 100));

                  return (
                    <div key={item.name} className="flex flex-col gap-1 text-[11px] font-mono">
                      <div className="flex justify-between items-center">
                        <span className={`flex items-center gap-1.5 ${isGreedy ? 'font-bold text-[var(--text-primary)]' : 'text-[var(--text-secondary)]'}`}>
                          {item.name}
                          {isGreedy && <span className="text-[9px] px-1 bg-[var(--bg-active)] rounded">argMax</span>}
                        </span>
                        <div className="flex items-center gap-2">
                          <span className="text-[9px] text-[var(--text-muted)]">
                            θ:{item.logit >= 0 ? `+${item.logit.toFixed(2)}` : item.logit.toFixed(2)}
                          </span>
                          <span className="font-semibold text-[var(--pigment-class-b)]">
                            {pct.toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      <div className="w-full h-1.5 rounded-xs bg-[var(--bg-subtle)] overflow-hidden">
                        <div
                          className="h-full bg-[var(--pigment-class-b)] transition-all duration-100"
                          style={{ width: `${Math.max(4, pct)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Trajectory Step Ribbon */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-2">
                <div className="flex justify-between items-center text-[10px] font-mono">
                  <span className="text-[var(--text-muted)] uppercase">Trajectory Ribbon ({trajectory.length} Steps)</span>
                  <span className="text-[9px] text-[var(--text-secondary)] font-semibold">G_t & Advantage A_t</span>
                </div>

                <div className="max-h-[200px] overflow-y-auto space-y-1.5 font-mono text-[10px]">
                  {trajectory.length > 0 ? (
                    trajectory.map((step, idx) => {
                      const isPositive = step.advantage >= 0;
                      return (
                        <div
                          key={idx}
                          className={`p-1.5 rounded flex items-center justify-between border ${
                            isPositive
                              ? 'border-[var(--pigment-class-b)] bg-[var(--fill-neg)]'
                              : 'border-[var(--pigment-class-a)] bg-[var(--fill-pos)]'
                          }`}
                        >
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className="text-[9px] text-[var(--text-muted)] shrink-0">t={idx}</span>
                            <span className="font-semibold text-[var(--text-primary)] truncate">
                              ({step.row},{step.col})
                            </span>
                            <span className="text-[var(--text-secondary)] shrink-0">{ACTION_SYMBOLS[step.action]}</span>
                          </div>
                          <div className="flex items-center gap-2 shrink-0">
                            <span className="text-[var(--text-secondary)]">G:{step.gValue.toFixed(2)}</span>
                            <span
                              className={`font-semibold ${
                                isPositive ? 'text-[var(--pigment-class-b)]' : 'text-[var(--pigment-class-a)]'
                              }`}
                            >
                              A:{isPositive ? `+${step.advantage.toFixed(2)}` : step.advantage.toFixed(2)}
                            </span>
                          </div>
                        </div>
                      );
                    })
                  ) : (
                    <div className="text-[10px] text-[var(--text-muted)] py-6 text-center">
                      Click Step +1 or Play to sample a Monte-Carlo episode rollout.
                    </div>
                  )}
                </div>
              </div>

              {/* Policy Gradient Math Card */}
              <div className="p-3 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] space-y-1.5">
                <span className="text-[10px] font-mono text-[var(--text-muted)] uppercase">
                  Policy Gradient Likelihood Ratio
                </span>
                <div
                  className="p-2 rounded bg-[var(--bg-subtle)] overflow-x-auto text-[10px] [&_.katex]:text-[0.86em]"
                  dangerouslySetInnerHTML={{
                    __html: renderLatex(
                      '\\theta \\leftarrow \\theta + \\alpha \\sum_t \\nabla_\\theta \\ln \\pi(a_t|s_t) \\Big(G_t - b\\Big) + \\beta_{\\text{ent}} \\nabla_\\theta H(\\pi)'
                    ),
                  }}
                />
                <span className="text-[9px] font-mono text-[var(--text-secondary)] block">
                  Trajectories with G_t &gt; b get reinforced; below-average trajectories get suppressed.
                </span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 3. Standardized Swiss Bottom Status & Legend Bar (Zone D) */}
      <footer className="w-full flex flex-wrap items-center justify-between px-3 py-1.5 border-t border-[var(--border-hairline)] bg-[var(--bg-subtle)] text-[10px] gap-2 select-none shrink-0 font-mono">
        {/* Left: Quick Instructions */}
        <div className="flex flex-wrap items-center gap-3 text-[var(--text-secondary)] text-[9px]">
          <span className="flex items-center gap-1">
            <span className="text-[var(--pigment-class-b)]">✦</span>
            <span>Click cell to inspect action values</span>
          </span>
          <span className="text-[var(--border-hairline)]">|</span>
          <span className="flex items-center gap-1">
            <span>🖌 Paint tiles via sidebar brush</span>
          </span>
          <span className="text-[var(--border-hairline)]">|</span>
          <span className="flex items-center gap-1">
            <span className="text-[var(--pigment-class-a)]">⚡</span>
            <span>Single-step visualizes Bellman backup arc & TD error</span>
          </span>
        </div>

        {/* Center: Swatches & Legend */}
        <div className="flex flex-wrap items-center gap-3 text-[var(--text-secondary)] text-[9px]">
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2.5 h-2.5 rounded-xs bg-[var(--pigment-class-b)]/25 border border-[var(--pigment-class-b)]" />
            <span>Goal (+10/+20)</span>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2.5 h-2.5 rounded-xs bg-[var(--pigment-class-a)]/25 border border-[var(--pigment-class-a)]" />
            <span>Lava / Cliff (-10)</span>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2.5 h-2.5 rounded-xs bg-amber-500/25 border border-amber-500" />
            <span>Bait (+2)</span>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2.5 h-2.5 rounded-full bg-[var(--pigment-class-a)] border border-white" />
            <span>Agent ●</span>
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <span className="w-2.5 h-2.5 rounded-xs bg-[#333330] border border-[var(--border-hairline)]" />
            <span>Wall #</span>
          </div>
        </div>

        {/* Right: Architecture & Cadence Metadata */}
        <div className="flex items-center gap-2 text-[var(--text-muted)] text-[9px]">
          <span className="px-1.5 py-0.5 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] text-[var(--text-primary)] font-semibold">
            {algorithm.name}
          </span>
          <span className="px-1.5 py-0.5 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] text-[var(--text-secondary)]">
            6×6 Grid (36 States)
          </span>
          <span className="px-1.5 py-0.5 rounded border border-[var(--border-hairline)] bg-[var(--bg-canvas)] text-[var(--text-secondary)]">
            {currentSpeed === 'single-step' ? '1 Step / Frame' : '1 Episode / Frame'}
          </span>
        </div>
      </footer>
    </div>
  );
};
