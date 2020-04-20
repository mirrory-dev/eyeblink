import { io } from '@tensorflow/tfjs';
import { Eyeblink } from './eyeblink';
export declare function loadModel(graphModelPath?: string | io.IOHandler): Promise<import("@tensorflow/tfjs-converter").GraphModel>;
export declare function load(graphModelPath?: string | io.IOHandler): Promise<Eyeblink>;
