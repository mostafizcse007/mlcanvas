import { DatasetType } from '../types/dataset';
import { generateDataset } from './generators';

export function renderDatasetThumbnail(
  canvas: HTMLCanvasElement,
  type: DatasetType,
  isCarbon: boolean
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;

  // Background
  ctx.fillStyle = isCarbon ? '#141413' : '#faf9f6';
  ctx.fillRect(0, 0, w, h);

  // Border hairline
  ctx.strokeStyle = isCarbon ? '#323230' : '#dcdad3';
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

  // Generate lightweight sample
  const { train } = generateDataset({
    type,
    sampleCount: 60,
    noise: 0.05,
    testRatio: 0,
  });

  const pigmentA = isCarbon ? '#e58344' : '#c85a17';
  const pigmentB = isCarbon ? '#4e89b8' : '#2b5b84';

  const scale = (w * 0.42) / 6.0;
  const cx = w / 2;
  const cy = h / 2;

  for (let i = 0; i < train.length; i++) {
    const p = train[i];
    const px = cx + p.x1 * scale;
    const py = cy - p.x2 * scale;

    ctx.fillStyle = (type === 'linear-regression' ? p.y >= 0 : p.y === 1) ? pigmentA : pigmentB;
    ctx.beginPath();
    ctx.arc(px, py, 1.8, 0, Math.PI * 2);
    ctx.fill();
  }
}

