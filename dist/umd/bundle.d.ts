import * as tf from "@tensorflow/tfjs";
import { FaceMesh } from "@tensorflow-models/facemesh";
interface EyeblinkPrediction {
    right: number;
    left: number;
}
declare class Eyeblink {
    private eyeblinkModel;
    private facemeshModel;
    constructor(eyeblinkModel: tf.GraphModel, facemeshModel: FaceMesh);
    private extractEyeBoundingBox;
    private getPredictionWithinBoundingBox;
    predictEyeOpenness(image: ImageData): Promise<EyeblinkPrediction | null>;
}
declare function load(graphModelPath?: string): Promise<Eyeblink>;
export { load };
