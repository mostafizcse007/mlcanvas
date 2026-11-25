import { DataPoint2D } from '../datasets/tabular2d';

/**
 * Returns only the data points that should be rendered or hit-tested on a 2D/1D canvas.
 * When showTestData is false, testData entries and any trainData entries flagged with
 * isTest === true are strictly omitted.
 */
export function getVisibleDataPoints(
  trainData: DataPoint2D[],
  testData: DataPoint2D[],
  showTestData: boolean
): DataPoint2D[] {
  const visibleTrain = showTestData
    ? trainData
    : trainData.filter((pt) => !pt.isTest);

  if (!showTestData || testData.length === 0) {
    return visibleTrain;
  }

  const normalizedTest = testData.map((pt) =>
    pt.isTest ? pt : { ...pt, isTest: true }
  );
  return [...visibleTrain, ...normalizedTest];
}

export interface HitTestOptions {
  px: number;
  py: number;
  width: number;
  height: number;
  hitRadiusPx: number;
  toPxX: (x1: number, width: number) => number;
  toPxY: (x2: number, height: number) => number;
  searchFromEnd?: boolean;
  strictLessThan?: boolean;
}

/**
 * Performs pointer hit-testing strictly against visible points.
 */
export function hitTestDataPoints(
  points: DataPoint2D[],
  options: HitTestOptions
): DataPoint2D | null {
  const {
    px,
    py,
    width,
    height,
    hitRadiusPx,
    toPxX,
    toPxY,
    searchFromEnd = false,
    strictLessThan = false,
  } = options;
  const r2 = hitRadiusPx * hitRadiusPx;
  const start = searchFromEnd ? points.length - 1 : 0;
  const step = searchFromEnd ? -1 : 1;

  for (let i = start; searchFromEnd ? i >= 0 : i < points.length; i += step) {
    const pt = points[i];
    const ptPx = toPxX(pt.x1, width);
    const ptPy = toPxY(pt.x2, height);
    const dx = ptPx - px;
    const dy = ptPy - py;
    const dist2 = dx * dx + dy * dy;
    if (strictLessThan ? dist2 < r2 : dist2 <= r2) {
      return pt;
    }
  }
  return null;
}

export interface DrawDataPointOptions {
  px: number;
  py: number;
  isTest: boolean;
  color: string;
  isCarbon: boolean;
  trainRadius?: number;
  testRadius?: number;
  trainStrokeColor?: string;
  trainStrokeWidth?: number;
  testStrokeWidth?: number;
  innerDotRadius?: number;
}

/**
 * Renders a single 2D/1D dataset observation: solid matte circle for training points
 * and a hollow ringed circle (with optional center dot) for test-split points.
 */
export function drawStandardDataPoint(
  ctx: CanvasRenderingContext2D,
  options: DrawDataPointOptions
): void {
  const {
    px,
    py,
    isTest,
    color,
    isCarbon,
    trainRadius = 4.0,
    testRadius = 4.5,
    trainStrokeColor = isCarbon ? '#181817' : '#FAF9F5',
    trainStrokeWidth = 1.0,
    testStrokeWidth = 2.0,
    innerDotRadius = 1.2,
  } = options;

  ctx.beginPath();
  if (isTest) {
    ctx.arc(px, py, testRadius, 0, Math.PI * 2);
    ctx.fillStyle = isCarbon ? '#131312' : '#FAF9F5';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = testStrokeWidth;
    ctx.stroke();

    if (innerDotRadius > 0) {
      ctx.beginPath();
      ctx.arc(px, py, innerDotRadius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
  } else {
    ctx.arc(px, py, trainRadius, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = trainStrokeColor;
    ctx.lineWidth = trainStrokeWidth;
    ctx.stroke();
  }
}

export interface RenderDataPoints2DParams {
  trainData: DataPoint2D[];
  testData: DataPoint2D[];
  showTestData: boolean;
  width: number;
  height: number;
  toPxX: (x1: number, width: number) => number;
  toPxY: (x2: number, height: number) => number;
  getColor: (pt: DataPoint2D) => string;
  isCarbon: boolean;
  trainRadius?: number;
  testRadius?: number;
  trainStrokeColor?: string;
  trainStrokeWidth?: number;
  testStrokeWidth?: number;
  innerDotRadius?: number;
  beforeDrawPoint?: (
    pt: DataPoint2D,
    px: number,
    py: number,
    color: string
  ) => { trainRadius?: number; testRadius?: number } | void;
}

/**
 * Filters and renders visible train and test points onto a 2D canvas context.
 */
export function renderDataPoints2D(
  ctx: CanvasRenderingContext2D,
  params: RenderDataPoints2DParams
): DataPoint2D[] {
  const {
    trainData,
    testData,
    showTestData,
    width,
    height,
    toPxX,
    toPxY,
    getColor,
    isCarbon,
    trainRadius = 4.0,
    testRadius = 4.5,
    trainStrokeColor,
    trainStrokeWidth,
    testStrokeWidth,
    innerDotRadius,
    beforeDrawPoint,
  } = params;

  const visiblePoints = getVisibleDataPoints(trainData, testData, showTestData);
  for (let i = 0; i < visiblePoints.length; i++) {
    const pt = visiblePoints[i];
    const px = toPxX(pt.x1, width);
    const py = toPxY(pt.x2, height);
    const color = getColor(pt);
    const overrides = beforeDrawPoint ? beforeDrawPoint(pt, px, py, color) : undefined;

    drawStandardDataPoint(ctx, {
      px,
      py,
      isTest: Boolean(pt.isTest),
      color,
      isCarbon,
      trainRadius: overrides?.trainRadius ?? trainRadius,
      testRadius: overrides?.testRadius ?? testRadius,
      trainStrokeColor,
      trainStrokeWidth,
      testStrokeWidth,
      innerDotRadius,
    });
  }

  return visiblePoints;
}
