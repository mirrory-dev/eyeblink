import {loadGraphModel} from '@tensorflow/tfjs-converter';
import {load as loadFaceMeshModel} from '@tensorflow-models/facemesh';
import {io} from '@tensorflow/tfjs';

import {Eyeblink} from './eyeblink';

export {Eyeblink, EyeblinkPrediction, BoundingBox} from './eyeblink';
export {getImageData} from './utils/image';

const defaultGraphModelPath =
  'https://VanityXR.github.io/eyeblink/models/model.json';

export async function loadModel(
  graphModelPath: string | io.IOHandler = defaultGraphModelPath,
) {
  return loadGraphModel(graphModelPath);
}

export async function load(
  graphModelPath: string | io.IOHandler = defaultGraphModelPath,
) {
  const blinkModel = await loadModel(graphModelPath);
  const facemeshModel = await loadFaceMeshModel({maxFaces: 1});
  return new Eyeblink(blinkModel, facemeshModel);
}
