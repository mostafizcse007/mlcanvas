import { ThemeMode } from '../types/theme';

export interface RGBA {
  r: number;
  g: number;
  b: number;
  a: number;
}

export function getHeatmapPixelColor(
  value: number, // e.g. decision boundary score from -3 to +3
  theme: ThemeMode,
  discretize: boolean = false
): [number, number, number, number] {
  // Sigmoid / tanh soft squashing or sharp binary discretization
  const t = discretize ? (value >= 0 ? 1.0 : -1.0) : Math.tanh(value); // -1.0 to +1.0

  if (theme === 'carbon') {
    // Carbon palette: Class A [229, 131, 68], Class B [78, 137, 184], Neutral [24, 24, 23]
    if (t > 0) {
      // Interpolate from neutral carbon to Class A
      const alpha = discretize ? 0.75 : Math.min(0.65, Math.abs(t) * 0.65);
      return [229, 131, 68, Math.floor(alpha * 255)];
    } else {
      // Interpolate from neutral carbon to Class B
      const alpha = discretize ? 0.75 : Math.min(0.65, Math.abs(t) * 0.65);
      return [78, 137, 184, Math.floor(alpha * 255)];
    }
  } else {
    // Paper palette: Class A [200, 90, 23], Class B [43, 91, 132]
    if (t > 0) {
      const alpha = discretize ? 0.65 : Math.min(0.55, Math.abs(t) * 0.55);
      return [200, 90, 23, Math.floor(alpha * 255)];
    } else {
      const alpha = discretize ? 0.65 : Math.min(0.55, Math.abs(t) * 0.55);
      return [43, 91, 132, Math.floor(alpha * 255)];
    }
  }
}

export function populateHeatmapImageData(
  imageData: ImageData,
  gridValues: Float32Array,
  theme: ThemeMode,
  discretize: boolean = false
): void {
  const data = imageData.data;
  const count = Math.min(gridValues.length, imageData.width * imageData.height);

  for (let i = 0; i < count; i++) {
    const val = gridValues[i];
    const [r, g, b, a] = getHeatmapPixelColor(val, theme, discretize);
    const offset = i * 4;
    data[offset] = r;
    data[offset + 1] = g;
    data[offset + 2] = b;
    data[offset + 3] = a;
  }
}

export function getDensityPixelColor(
  normalized: number, // 0.0 to 1.0
  theme: ThemeMode
): [number, number, number, number] {
  const t = Math.max(0, Math.min(1, normalized));
  if (theme === 'carbon') {
    // Carbon: Dark neutral [24, 24, 23] -> Vibrant Amber [245, 158, 11]
    const r = Math.floor(24 + (245 - 24) * t);
    const g = Math.floor(24 + (158 - 24) * t);
    const b = Math.floor(23 + (11 - 23) * t);
    const alpha = Math.min(0.85, 0.15 + t * 0.7);
    return [r, g, b, Math.floor(alpha * 255)];
  } else {
    // Paper: Light neutral [250, 249, 245] -> Warm Amber [217, 119, 6]
    const r = Math.floor(250 + (217 - 250) * t);
    const g = Math.floor(249 + (119 - 249) * t);
    const b = Math.floor(245 + (6 - 245) * t);
    const alpha = Math.min(0.85, 0.15 + t * 0.7);
    return [r, g, b, Math.floor(alpha * 255)];
  }
}


