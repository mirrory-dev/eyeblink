import { io } from '@tensorflow/tfjs';
import * as tf from "@tensorflow/tfjs-core";
import { GraphModel } from "@tensorflow/tfjs-converter";
import { FaceMesh, AnnotatedPrediction } from "@tensorflow-models/facemesh";
interface BoundingBox {
    topLeft: readonly [number, number];
    bottomRight: readonly [number, number];
}
interface EyeblinkPrediction {
    right: number;
    left: number;
}
declare class Eyeblink {
    private eyeblinkModel;
    private facemeshModel;
    constructor(eyeblinkModel: GraphModel, facemeshModel: FaceMesh);
    private extractEyeBoundingBox;
    private getPredictionWithinBoundingBox;
    predictEyeOpenness(image: tf.Tensor3D | ImageData | HTMLImageElement | HTMLCanvasElement | HTMLVideoElement, face?: AnnotatedPrediction): Promise<EyeblinkPrediction | null>;
}
declare function getImageData(videoEl: CanvasImageSource): ImageData;
declare function loadModel(graphModelPath?: string | io.IOHandler): Promise<import("@tensorflow/tfjs-converter").GraphModel>;
declare function load(graphModelPath?: string | io.IOHandler): Promise<Eyeblink>;
export { Eyeblink, EyeblinkPrediction, BoundingBox, getImageData, loadModel, load };
