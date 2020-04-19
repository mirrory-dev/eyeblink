import * as tf from '@tensorflow/tfjs-converter';
import * as facemesh from '@tensorflow-models/facemesh';

import {Eyeblink} from './eyeblink';

const defaultGraphModelPath =
  'https://unpkg.com/@prism-3d/eyeblink/models/model.json';

export async function load(graphModelPath: string = defaultGraphModelPath) {
  const blinkModel = await tf.loadGraphModel(graphModelPath);
  const facemeshModel = await facemesh.load({maxFaces: 1});

  return new Eyeblink(blinkModel, facemeshModel);
}
