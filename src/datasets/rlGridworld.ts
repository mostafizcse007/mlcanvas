import {
  GridworldPresetKind,
  GridworldTileKind,
  RLGridworldDataset,
} from '../types/algorithm';

export const GRID_SIZE = 6;
export const NUM_STATES = GRID_SIZE * GRID_SIZE; // 36
export const NUM_ACTIONS = 4; // 0: Up (North), 1: Right (East), 2: Down (South), 3: Left (West)

export const ACTION_DELTAS: [number, number][] = [
  [-1, 0], // 0: Up
  [0, 1],  // 1: Right
  [1, 0],  // 2: Down
  [0, -1], // 3: Left
];

export const ACTION_NAMES = ['North ▲', 'East ▶', 'South ▼', 'West ◀'];
export const ACTION_SYMBOLS = ['▲', '▶', '▼', '◀'];

export const TILE_VALUES: Record<GridworldTileKind, number> = {
  empty: 0,
  wall: 1,
  lava: 2,
  goal: 3,
  bait: 4,
  start: 5,
};

export const TILE_NAMES: Record<number, GridworldTileKind> = {
  0: 'empty',
  1: 'wall',
  2: 'lava',
  3: 'goal',
  4: 'bait',
  5: 'start',
};

/** Canonical descriptions for the 4 presets */
export const GRIDWORLD_PRESET_INFO: Record<
  GridworldPresetKind,
  { name: string; shortDesc: string; defaultPenalty: number; defaultSlip: number }
> = {
  'cliff-walking': {
    name: 'Classic Cliff Walking',
    shortDesc: 'Safe top route vs. optimal risky edge along the lethal Lava Cliff (-10).',
    defaultPenalty: -0.04,
    defaultSlip: 0.0,
  },
  'frozen-lake': {
    name: 'Frozen Lake (Stochastic)',
    shortDesc: 'Slippery ice tiles with perpendicular slide probability P(slip) = 20%.',
    defaultPenalty: -0.04,
    defaultSlip: 0.2,
  },
  'twin-treasure': {
    name: 'Twin Treasure Trap',
    shortDesc: 'Myopic bait (+2, 2 steps) vs. distant optimal goal (+20, 8 steps) to test γ.',
    defaultPenalty: -0.04,
    defaultSlip: 0.0,
  },
  'four-rooms': {
    name: 'Four-Rooms Bottleneck',
    shortDesc: 'Sub-goal discovery through 4 partitioned quadrants with narrow doorway bottlenecks.',
    defaultPenalty: -0.04,
    defaultSlip: 0.0,
  },
};

/** Generate the tile matrix for each preset */
export function generatePresetTiles(
  preset: GridworldPresetKind
): { tiles: Uint8Array; startPos: [number, number]; goalPos: [number, number] } {
  const tiles = new Uint8Array(NUM_STATES); // default all 0 (empty)
  let startPos: [number, number] = [0, 0];
  let goalPos: [number, number] = [GRID_SIZE - 1, GRID_SIZE - 1];

  switch (preset) {
    case 'cliff-walking': {
      // Start at bottom-left (5, 0), Goal at bottom-right (5, 5)
      // Bottom edge row 5, cols 1..4 are Lava (-10)
      startPos = [5, 0];
      goalPos = [5, 5];
      tiles[5 * GRID_SIZE + 0] = TILE_VALUES.start;
      tiles[5 * GRID_SIZE + 5] = TILE_VALUES.goal;
      for (let c = 1; c <= 4; c++) {
        tiles[5 * GRID_SIZE + c] = TILE_VALUES.lava;
      }
      break;
    }

    case 'frozen-lake': {
      // Start top-left (0, 0), Goal bottom-right (5, 5)
      startPos = [0, 0];
      goalPos = [5, 5];
      tiles[0 * GRID_SIZE + 0] = TILE_VALUES.start;
      tiles[5 * GRID_SIZE + 5] = TILE_VALUES.goal;

      // Hole positions
      const holes: [number, number][] = [
        [1, 1],
        [1, 3],
        [2, 4],
        [3, 1],
        [4, 3],
        [4, 5],
      ];
      for (const [r, c] of holes) {
        tiles[r * GRID_SIZE + c] = TILE_VALUES.lava;
      }
      break;
    }

    case 'twin-treasure': {
      // Start at center-bottom (3, 2)
      // Small Bait (+2) at (3, 0) - distance 2
      // Large Goal (+20) at (0, 5) - distance 8
      startPos = [3, 2];
      goalPos = [0, 5];
      tiles[startPos[0] * GRID_SIZE + startPos[1]] = TILE_VALUES.start;
      tiles[goalPos[0] * GRID_SIZE + goalPos[1]] = TILE_VALUES.goal;
      tiles[3 * GRID_SIZE + 0] = TILE_VALUES.bait;

      // Small divider wall so paths separate cleanly
      tiles[2 * GRID_SIZE + 1] = TILE_VALUES.wall;
      tiles[2 * GRID_SIZE + 2] = TILE_VALUES.wall;
      tiles[2 * GRID_SIZE + 3] = TILE_VALUES.wall;
      break;
    }

    case 'four-rooms': {
      // 4 rooms partitioned by row 2 and col 2
      // Doorways at (2, 1), (2, 4), (1, 2), (4, 2)
      startPos = [0, 0];
      goalPos = [5, 5];

      // Walls along row 2 and col 2
      for (let c = 0; c < GRID_SIZE; c++) {
        tiles[2 * GRID_SIZE + c] = TILE_VALUES.wall;
      }
      for (let r = 0; r < GRID_SIZE; r++) {
        tiles[r * GRID_SIZE + 2] = TILE_VALUES.wall;
      }

      // Cut 4 doorways
      tiles[2 * GRID_SIZE + 1] = TILE_VALUES.empty; // doorway horizontal room 1-3
      tiles[2 * GRID_SIZE + 4] = TILE_VALUES.empty; // doorway horizontal room 2-4
      tiles[1 * GRID_SIZE + 2] = TILE_VALUES.empty; // doorway vertical room 1-2
      tiles[4 * GRID_SIZE + 2] = TILE_VALUES.empty; // doorway vertical room 3-4

      tiles[startPos[0] * GRID_SIZE + startPos[1]] = TILE_VALUES.start;
      tiles[goalPos[0] * GRID_SIZE + goalPos[1]] = TILE_VALUES.goal;
      break;
    }
  }

  return { tiles, startPos, goalPos };
}

/** Factory for creating initial RL Gridworld Dataset */
export function createRLGridworldDataset(
  options: Partial<RLGridworldDataset> = {}
): RLGridworldDataset {
  const preset: GridworldPresetKind = options.preset || 'cliff-walking';
  const info = GRIDWORLD_PRESET_INFO[preset];
  const { tiles, startPos, goalPos } = generatePresetTiles(preset);

  return {
    preset,
    gridSize: GRID_SIZE,
    tiles: options.tiles ? new Uint8Array(options.tiles) : tiles,
    startPos: options.startPos ? [...options.startPos] : startPos,
    goalPos: options.goalPos ? [...options.goalPos] : goalPos,
    livingPenalty: options.livingPenalty ?? info.defaultPenalty,
    slipProbability: options.slipProbability ?? info.defaultSlip,
    seed: options.seed ?? 42,
  };
}

/** Step environment transition dynamics */
export function stepEnvironment(
  row: number,
  col: number,
  action: number,
  dataset: RLGridworldDataset,
  rng?: () => number
): {
  nextRow: number;
  nextCol: number;
  nextState: number;
  reward: number;
  done: boolean;
  isCliff: boolean;
  isGoal: boolean;
  isBait: boolean;
  slipped: boolean;
} {
  let effectiveAction = action;
  let slipped = false;

  // Stochastic slip check
  if (dataset.slipProbability > 0 && rng) {
    const p = rng();
    if (p < dataset.slipProbability) {
      slipped = true;
      // Perpendicular slip (+1 or -1 modulo 4)
      const slipDir = rng() < 0.5 ? 1 : 3;
      effectiveAction = (action + slipDir) % 4;
    }
  }

  const [dr, dc] = ACTION_DELTAS[effectiveAction];
  let targetRow = row + dr;
  let targetCol = col + dc;

  // Boundary check (bumping into border keeps agent in place)
  if (targetRow < 0 || targetRow >= GRID_SIZE || targetCol < 0 || targetCol >= GRID_SIZE) {
    targetRow = row;
    targetCol = col;
  }

  const targetIdx = targetRow * GRID_SIZE + targetCol;
  const tileType = dataset.tiles[targetIdx];

  // Wall check (bumping into wall keeps agent in place)
  if (tileType === TILE_VALUES.wall) {
    targetRow = row;
    targetCol = col;
  }

  const finalIdx = targetRow * GRID_SIZE + targetCol;
  const finalTile = dataset.tiles[finalIdx];

  let reward = dataset.livingPenalty;
  let done = false;
  let isCliff = false;
  let isGoal = false;
  let isBait = false;

  if (finalTile === TILE_VALUES.goal) {
    // Goal Reward: +10 on standard, or +20 on twin-treasure
    reward += dataset.preset === 'twin-treasure' ? 20.0 : 10.0;
    done = true;
    isGoal = true;
  } else if (finalTile === TILE_VALUES.bait) {
    // Bait Reward: +2 (immediate trap bait)
    reward += 2.0;
    done = true;
    isBait = true;
  } else if (finalTile === TILE_VALUES.lava) {
    // Lava/Cliff Penalty: -10.0 and terminal episode end
    reward += -10.0;
    done = true;
    isCliff = true;
  }

  return {
    nextRow: targetRow,
    nextCol: targetCol,
    nextState: finalIdx,
    reward,
    done,
    isCliff,
    isGoal,
    isBait,
    slipped,
  };
}

/** Paint tile on Gridworld */
export function paintGridworldTile(
  dataset: RLGridworldDataset,
  row: number,
  col: number,
  tileKind: GridworldTileKind
): RLGridworldDataset {
  const nextTiles = new Uint8Array(dataset.tiles);
  const idx = row * GRID_SIZE + col;
  let nextStart = [...dataset.startPos] as [number, number];
  let nextGoal = [...dataset.goalPos] as [number, number];

  if (tileKind === 'start') {
    // Clear old start
    for (let i = 0; i < NUM_STATES; i++) {
      if (nextTiles[i] === TILE_VALUES.start) nextTiles[i] = TILE_VALUES.empty;
    }
    nextTiles[idx] = TILE_VALUES.start;
    nextStart = [row, col];
  } else if (tileKind === 'goal') {
    // Clear old goal
    for (let i = 0; i < NUM_STATES; i++) {
      if (nextTiles[i] === TILE_VALUES.goal) nextTiles[i] = TILE_VALUES.empty;
    }
    nextTiles[idx] = TILE_VALUES.goal;
    nextGoal = [row, col];
  } else {
    // Don't overwrite if it was start or goal unless explicitly replacing
    nextTiles[idx] = TILE_VALUES[tileKind];
  }

  return {
    ...dataset,
    tiles: nextTiles,
    startPos: nextStart,
    goalPos: nextGoal,
  };
}

