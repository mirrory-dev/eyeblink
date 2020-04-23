import * as tf from '@tensorflow/tfjs';
import * as blinkModel from '@prism-3d/eyeblink/dist/umd/eyeblink';
import {Eyeblink} from '@prism-3d/eyeblink/dist/umd/eyeblink';

let predictor: Eyeblink;
let webcam: any;

const rightEyeEl = document.querySelector('#rightEye')! as HTMLDivElement;
const leftEyeEl = document.querySelector('#leftEye')! as HTMLDivElement;
const webcamEl = document.querySelector('#webcam')! as HTMLVideoElement;
const textEl = document.querySelector('#text')! as HTMLDivElement;

const fps = 10;
const fpsInterval = 1000 / fps;
let then = Date.now();
let elapsed = 0;

interface GetImageDataOption {
  width: number;
  height: number;
}

export function getImageData(
  videoEl: CanvasImageSource,
  {width, height}: GetImageDataOption = {
    width: 200,
    height: 200,
  },
): ImageData {
  const procCanvas = new OffscreenCanvas(width, height);
  const ctx = procCanvas.getContext('2d')!;
  ctx.drawImage(videoEl, 0, 0, procCanvas.width, procCanvas.height);

  const image = ctx.getImageData(0, 0, procCanvas.width, procCanvas.height);

  return image;
}

async function animate() {
  requestAnimationFrame(animate);
  const now = Date.now();
  elapsed = now - then;
  if (elapsed > fpsInterval) {
    then = now - (elapsed % fpsInterval);
    const image = getImageData(webcamEl);
    const openness = await predictor.predictEyeOpenness(image);
    if (!openness) return;
    console.log(openness.right, openness.left);
    rightEyeEl.style.height = `${openness.right.openness * 200}px`;
    leftEyeEl.style.height = `${openness.left.openness * 200}px`;
    textEl.innerHTML = `left: ${openness.left.openness} (${openness.left.likelihood}), right: ${openness.right.openness} (${openness.right.likelihood})`;
  }
}

async function init() {
  predictor = await blinkModel.load('models/model.json');
  webcam = await tf.data.webcam(webcamEl);
  animate();
}

init();
