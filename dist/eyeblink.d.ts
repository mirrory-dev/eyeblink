import * as tf from '@tensorflow/tfjs-core';
import { GraphModel } from '@tensorflow/tfjs-converter';
import { FaceMesh, AnnotatedPrediction } from '@tensorflow-models/facemesh';
export interface BoundingBox {
    topLeft: readonly [number, number];
    bottomRight: readonly [number, number];
}
export interface EyeState {
    openness: number;
    likelihood: number;
}
export interface EyeblinkPrediction {
    right: EyeState;
    left: EyeState;
}
export declare class Eyeblink {
    private eyeblinkModel;
    private facemeshModel;
    constructor(eyeblinkModel: GraphModel, facemeshModel: FaceMesh);
    private extractEyeBoundingBox;
    private getPredictionWithinBoundingBox;
    predictEyeOpenness(image: tf.Tensor3D | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, face?: AnnotatedPrediction): Promise<EyeblinkPrediction | null>;
}
