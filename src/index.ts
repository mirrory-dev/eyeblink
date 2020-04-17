import * as tf from '@tensorflow/tfjs-node';
import * as facemesh from '@tensorflow-models/facemesh';
import path from 'path';

import {Eyeblink} from './eyeblink';

const defaultGraphModelPath = path.resolve(__dirname, '../models/model.json');

export async function load(graphModelPath: string = defaultGraphModelPath) {
  const blinkModel = await tf.loadGraphModel(
    'file://' + path.resolve(graphModelPath),
  );
  const facemeshModel = await facemesh.load({maxFaces: 1});
  return new Eyeblink(blinkModel, facemeshModel);
}
