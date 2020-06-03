import * as tf from '@tensorflow/tfjs-core';
import { GraphModel } from '@tensorflow/tfjs-converter';
import { FaceMesh, AnnotatedPrediction } from '@tensorflow-models/facemesh';
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
    constructor(eyeblinkModel: GraphModel, facemeshModel: FaceMesh);
    private extractEyeBoundingBox;
    getPredictionWithinBoundingBox(input: tf.Tensor3D, boundingBoxes: BoundingBox[]): Promise<tf.TypedArray>;
    predictEyeOpenness(image: tf.Tensor3D | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, face?: AnnotatedPrediction): Promise<EyeblinkPrediction | null>;
}
