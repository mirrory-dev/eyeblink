import {loadGraphModel} from '@tensorflow/tfjs-converter';
import {load as loadFaceMeshModel} from '@tensorflow-models/facemesh';
import {io} from '@tensorflow/tfjs';
import {resolve} from 'path';

import {Eyeblink} from './eyeblink';

const defaultGraphModelPath = resolve(__dirname, '../models/model.json');

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
