import { SequenceDataset, SequencePresetKind, SequenceToken } from '../types/algorithm';

// Curated 8D semantic feature coordinates:
// [0: Content/Noun, 1: Entity-A, 2: Entity-B, 3: Open/Close-Punct, 4: Bracket-Style, 5: Valence, 6: Inversion, 7: Verb/Action]
const SEMANTIC_DICTIONARY: Record<string, number[]> = {
  // 1. Long-Range Coreference
  'The':      [0.10,  0.05,  0.05,  0.00,  0.00,  0.00,  0.00,  0.00],
  'trophy':   [0.90,  0.88,  0.10,  0.00,  0.00,  0.20,  0.00,  0.05],
  'did':      [0.20,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.50],
  'not':      [0.30,  0.00,  0.00,  0.00,  0.00, -0.40,  0.95,  0.10],
  'fit':      [0.60,  0.30,  0.30,  0.00,  0.00, -0.20,  0.10,  0.85],
  'the':      [0.10,  0.05,  0.05,  0.00,  0.00,  0.00,  0.00,  0.00],
  'suitcase': [0.85,  0.15,  0.85,  0.00,  0.00,  0.05,  0.00,  0.05],
  'because':  [0.40,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.20],
  'it':       [0.75,  0.70,  0.60,  0.00,  0.00,  0.00,  0.00,  0.00], // Anaphoric pronoun resolving to trophy or suitcase
  'was':      [0.20,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.40],
  'big':      [0.70,  0.80,  0.10,  0.00,  0.00,  0.10,  0.00,  0.10],

  // 2. Bracket / Parity Memory
  '[':        [0.05,  0.00,  0.00,  0.95,  0.90,  0.00,  0.00,  0.00], // Open Square
  'A':        [0.80,  0.75,  0.10,  0.00,  0.00,  0.20,  0.00,  0.00],
  'B':        [0.80,  0.60,  0.25,  0.00,  0.00,  0.10,  0.00,  0.00],
  '(':        [0.05,  0.00,  0.00,  0.90, -0.90,  0.00,  0.00,  0.00], // Open Round
  'C':        [0.80,  0.25,  0.60,  0.00,  0.00,  0.10,  0.00,  0.00],
  ')':        [0.05,  0.00,  0.00, -0.90, -0.90,  0.00,  0.00,  0.00], // Close Round
  'D':        [0.80,  0.10,  0.75,  0.00,  0.00,  0.20,  0.00,  0.00],
  ']':        [0.05,  0.00,  0.00, -0.95,  0.90,  0.00,  0.00,  0.00], // Close Square

  // 3. Sentiment Negation Flip
  'Not':       [0.30,  0.00,  0.00,  0.00,  0.00, -0.30,  0.95,  0.10],
  'only':      [0.25,  0.00,  0.00,  0.00,  0.00,  0.00,  0.40,  0.00],
  'plot':      [0.85,  0.20,  0.20,  0.00,  0.00,  0.00,  0.00,  0.00],
  'slow':      [0.75,  0.10,  0.10,  0.00,  0.00, -0.80,  0.10,  0.10],
  'but':       [0.35,  0.00,  0.00,  0.00,  0.00,  0.10,  0.50,  0.00],
  'actually':  [0.40,  0.00,  0.00,  0.00,  0.00,  0.30,  0.00,  0.00],
  'brilliant': [0.95,  0.30,  0.10,  0.00,  0.00,  0.98,  0.00,  0.00],

  // 4. Seq2Seq Translation Pair
  // Source Bengali
  'Ami':      [0.20,  1.60, -0.80,  0.00,  0.00,  0.00,  0.00, -0.80], // "I"
  'bhat':     [0.20, -0.80,  1.60,  0.00,  0.00,  0.00,  0.00, -0.80], // "rice"
  'khai':     [0.20, -0.80, -0.80,  0.00,  0.00,  0.00,  0.00,  1.80], // "eat"
  // Target English
  '<bos>':    [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
  'I':        [0.20,  1.60, -0.80,  0.00,  0.00,  0.00,  0.00, -0.80], // Align with Ami
  'eat':      [0.20, -0.80, -0.80,  0.00,  0.00,  0.00,  0.00,  1.80], // Align with khai
  'rice':     [0.20, -0.80,  1.60,  0.00,  0.00,  0.00,  0.00, -0.80], // Align with bhat
  '<eos>':    [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
  '[MASK]':   [0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00],
};

export const PRESET_SEQUENCES: Record<SequencePresetKind, { tokens: string[]; targetTokens?: string[]; description: string }> = {
  coreference: {
    tokens: ['The', 'trophy', 'did', 'not', 'fit', 'the', 'suitcase', 'because', 'it', 'was', 'big'],
    targetTokens: ['<bos>', 'The', 'trophy', 'did', 'not', 'fit'],
    description: 'Long-Range Coreference: tests if "it" resolves to "trophy" or "suitcase" over 11 steps.',
  },
  bracket: {
    tokens: ['[', 'A', 'B', '(', 'C', ')', 'D', ']'],
    targetTokens: ['<bos>', '[', '(', ')', ']'],
    description: 'Bracket / Parity Memory: tests if opening bracket [ at t=0 survives to t=7.',
  },
  sentiment: {
    tokens: ['Not', 'only', 'was', 'the', 'plot', 'slow', 'but', 'actually', 'brilliant'],
    targetTokens: ['<bos>', 'plot', 'slow', 'but', 'brilliant'],
    description: 'Sentiment Negation Flip: tests recurrent gate dynamics as valence inverts.',
  },
  seq2seq: {
    tokens: ['Ami', 'bhat', 'khai'],
    targetTokens: ['<bos>', 'I', 'eat', 'rice'],
    description: 'Seq2Seq Translation Pair: Bengali "Ami bhat khai" -> English "<bos> I eat rice".',
  },
};

/** Deterministic pseudo-random number generator */
function seededRandom(seed: number): () => number {
  let s = seed % 2147483647;
  if (s <= 0) s += 2147483646;
  return () => {
    s = (s * 16807) % 2147483647;
    return (s - 1) / 2147483646;
  };
}

/** Standard Normal Box-Muller transform */
function normalRandom(rng: () => number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

/** Deterministic hash for unseen custom token strings */
function hashTokenToVector(text: string): number[] {
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  const vec = new Array(8);
  for (let i = 0; i < 8; i++) {
    h = Math.imul(h ^ (i * 0x5bd1e995), 0x01000193);
    vec[i] = ((h & 0xffff) / 0xffff) * 2.0 - 1.0;
  }
  return vec;
}

/** Retrieve base 8D embedding for any token */
export function getTokenBaseEmbedding(text: string): Float32Array {
  const base = SEMANTIC_DICTIONARY[text] || hashTokenToVector(text);
  const out = new Float32Array(8);
  for (let i = 0; i < 8; i++) {
    out[i] = base[i];
  }
  return out;
}

/** Build sequence token with embedding noise and mask support */
export function buildSequenceToken(id: number, text: string, noise: number = 0, isMasked: boolean = false, seed: number = 42): SequenceToken {
  const base = getTokenBaseEmbedding(text);
  const out = new Float32Array(8);
  const rng = seededRandom(seed + id * 101);

  if (isMasked) {
    // Zero out / neutral mask embedding
    return { id, text: `[MASK]`, embedding: out, isMasked: true };
  }

  for (let i = 0; i < 8; i++) {
    const n = noise > 0 ? normalRandom(rng) * noise : 0;
    out[i] = Math.max(-2.0, Math.min(2.0, base[i] + n));
  }

  return {
    id,
    text,
    embedding: out,
    isMasked: false,
  };
}

export interface GenerateSequenceOptions {
  preset?: SequencePresetKind;
  length?: number;
  noise?: number;
  selectedTokenIndex?: number;
  seed?: number;
}

/** Generate a complete SequenceDataset */
export function generateSequenceDataset(options: GenerateSequenceOptions = {}): SequenceDataset {
  const preset = options.preset ?? 'coreference';
  const noise = Math.max(0, Math.min(0.5, options.noise ?? 0));
  const seed = options.seed ?? 42;

  const presetDef = PRESET_SEQUENCES[preset];
  const maxAvailable = presetDef.tokens.length;
  // Length clamped between 4 and 10 (or up to maxAvailable for 3-token seq2seq source)
  const defaultLen = preset === 'seq2seq' ? presetDef.tokens.length : Math.min(maxAvailable, options.length ?? Math.min(10, maxAvailable));
  const length = preset === 'seq2seq' ? presetDef.tokens.length : Math.max(4, Math.min(10, defaultLen));

  const sequence: SequenceToken[] = [];
  for (let i = 0; i < length; i++) {
    const tokenText = presetDef.tokens[i] ?? `tok_${i}`;
    sequence.push(buildSequenceToken(i, tokenText, noise, false, seed));
  }

  let targetSequence: SequenceToken[] | undefined = undefined;
  if (presetDef.targetTokens) {
    targetSequence = presetDef.targetTokens.map((t, idx) =>
      buildSequenceToken(idx, t, noise, false, seed + 500)
    );
  }

  const selectedTokenIndex = Math.max(
    0,
    Math.min(sequence.length - 1, options.selectedTokenIndex ?? (preset === 'coreference' ? 8 : 0))
  );

  return {
    preset,
    sequence,
    targetSequence,
    length,
    noise,
    selectedTokenIndex,
  };
}

/** Perturb or shift a token's embedding values */
export function perturbTokenEmbedding(token: SequenceToken, dimIndex: number, delta: number): SequenceToken {
  const nextEmb = new Float32Array(token.embedding);
  if (dimIndex >= 0 && dimIndex < nextEmb.length) {
    nextEmb[dimIndex] = Math.max(-2.5, Math.min(2.5, nextEmb[dimIndex] + delta));
  }
  return {
    ...token,
    embedding: nextEmb,
  };
}

/** Toggle mask on a token in the dataset */
export function toggleTokenMask(dataset: SequenceDataset, tokenIndex: number): SequenceDataset {
  if (tokenIndex < 0 || tokenIndex >= dataset.sequence.length) return dataset;
  const nextSeq = dataset.sequence.map((tok, idx) => {
    if (idx !== tokenIndex) return tok;
    const isNowMasked = !tok.isMasked;
    const origText = PRESET_SEQUENCES[dataset.preset].tokens[idx] ?? tok.text;
    return buildSequenceToken(tok.id, origText, dataset.noise, isNowMasked, 42);
  });

  return {
    ...dataset,
    sequence: nextSeq,
  };
}
