import * as tf from '@tensorflow/tfjs';
import { FaceMesh } from '@tensorflow-models/facemesh';
export interface BoundingBox {
    topLeft: readonly [number, number];
    bottomRight: readonly [number, number];
}
export interface EyeblinkPrediction {
    right: number;
    left: number;
}
export declare class Eyeblink {
    private eyeblinkModel;
    private facemeshModel;
    constructor(eyeblinkModel: tf.GraphModel, facemeshModel: FaceMesh);
    private extractEyeBoundingBox;
    private getPredictionWithinBoundingBox;
    predictEyeOpenness(image: ImageData): Promise<EyeblinkPrediction | null>;
}
