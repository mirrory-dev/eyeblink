export function getImageData(videoEl: CanvasImageSource): ImageData {
  const width = videoEl.width as number;
  const height = videoEl.height as number;

  const procCanvas = new OffscreenCanvas(width, height);
  const ctx = procCanvas.getContext('2d')!;
  ctx.drawImage(videoEl, 0, 0, procCanvas.width, procCanvas.height);

  const image = ctx.getImageData(0, 0, procCanvas.width, procCanvas.height);
  return image;
}
