import { io } from '@tensorflow/tfjs';
import { Eyeblink } from './eyeblink';
export { Eyeblink, EyeblinkPrediction, BoundingBox } from './eyeblink';
export { getImageData } from './utils/image';
export declare function loadModel(graphModelPath?: string | io.IOHandler): Promise<import("@tensorflow/tfjs-converter").GraphModel>;
export declare function load(graphModelPath?: string | io.IOHandler): Promise<Eyeblink>;
