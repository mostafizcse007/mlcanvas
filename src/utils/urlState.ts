import { ThemeMode } from '../types/theme';

export interface SerializedAppState {
  algo?: string;
  dataset?: string;
  seed?: number;
  noise?: number;
  split?: number;
  scale?: 'raw' | 'zscore' | 'minmax';
  stretch?: boolean;
  theme?: ThemeMode;
  lr?: number;
  c?: number;
  lambda?: number;
  degree?: number;
  reg?: number;
  solver?: string;
}

export function serializeToUrlHash(state: SerializedAppState): string {
  const params = new URLSearchParams();
  if (state.algo) params.set('algo', state.algo);
  if (state.dataset) params.set('data', state.dataset);
  if (typeof state.seed === 'number' && Number.isFinite(state.seed)) params.set('seed', state.seed.toString());
  if (typeof state.noise === 'number' && Number.isFinite(state.noise)) params.set('noise', Math.round(state.noise * 100).toString());
  if (typeof state.split === 'number' && Number.isFinite(state.split)) params.set('split', Math.round(state.split * 100).toString());
  if (state.scale) params.set('scale', state.scale);
  if (state.stretch) params.set('stretch', '1');
  if (state.theme) params.set('theme', state.theme);
  if (typeof state.lr === 'number' && Number.isFinite(state.lr)) params.set('lr', state.lr.toString());
  if (typeof state.c === 'number' && Number.isFinite(state.c)) params.set('c', state.c.toString());
  if (typeof state.lambda === 'number' && Number.isFinite(state.lambda)) params.set('lam', state.lambda.toString());
  if (typeof state.degree === 'number' && Number.isFinite(state.degree)) params.set('deg', state.degree.toString());
  if (typeof state.reg === 'number' && Number.isFinite(state.reg)) params.set('reg', state.reg.toString());
  if (state.solver) params.set('sol', state.solver);

  return '#' + params.toString();
}

export function parseUrlHash(): SerializedAppState {
  if (typeof window === 'undefined') return {};
  const hash = window.location.hash.replace(/^#/, '');
  if (!hash) return {};

  const params = new URLSearchParams(hash);
  const result: SerializedAppState = {};

  if (params.has('algo')) result.algo = params.get('algo')!;
  if (params.has('data')) result.dataset = params.get('data')!;
  if (params.has('seed')) {
    const v = parseInt(params.get('seed')!, 10);
    if (Number.isFinite(v)) result.seed = v;
  }
  if (params.has('noise')) {
    const v = parseInt(params.get('noise')!, 10) / 100;
    if (Number.isFinite(v)) result.noise = v;
  }
  if (params.has('split')) {
    const v = parseInt(params.get('split')!, 10) / 100;
    if (Number.isFinite(v)) result.split = v;
  }
  if (params.has('scale')) result.scale = params.get('scale') as 'raw' | 'zscore' | 'minmax';
  if (params.has('stretch')) result.stretch = params.get('stretch') === '1';
  if (params.has('theme')) result.theme = params.get('theme') as ThemeMode;
  if (params.has('lr')) {
    const v = parseFloat(params.get('lr')!);
    if (Number.isFinite(v)) result.lr = v;
  }
  if (params.has('c')) {
    const v = parseFloat(params.get('c')!);
    if (Number.isFinite(v)) result.c = v;
  }
  if (params.has('lam')) {
    const v = parseFloat(params.get('lam')!);
    if (Number.isFinite(v)) result.lambda = v;
  }
  if (params.has('deg')) {
    const v = parseInt(params.get('deg')!, 10);
    if (Number.isFinite(v)) result.degree = v;
  }
  if (params.has('reg')) {
    const v = parseFloat(params.get('reg')!);
    if (Number.isFinite(v)) result.reg = v;
  }
  if (params.has('sol')) {
    result.solver = params.get('sol')!;
  }

  return result;
}

