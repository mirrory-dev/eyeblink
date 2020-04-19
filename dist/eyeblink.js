"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs");
class Eyeblink {
    constructor(eyeblinkModel, facemeshModel) {
        this.eyeblinkModel = eyeblinkModel;
        this.facemeshModel = facemeshModel;
    }
    extractEyeBoundingBox(face, [top, right, bottom, left]) {
        const eyeTopY = face.scaledMesh[top][1];
        const eyeRightX = face.scaledMesh[right][0];
        const eyeBottomY = face.scaledMesh[bottom][1];
        const eyeLeftX = face.scaledMesh[left][0];
        const topLeft = [eyeLeftX, eyeTopY];
        const bottomRight = [eyeRightX, eyeBottomY];
        return { topLeft, bottomRight };
    }
    async getPredictionWithinBoundingBox(input, boundingBox) {
        const image = tf.tidy(() => {
            if (!(input instanceof tf.Tensor)) {
                input = tf.browser.fromPixels(input);
            }
            return input.toFloat().expandDims(0);
        });
        const cropped = tf.image
            .cropAndResize(image, [
            [
                boundingBox.topLeft[1] / 200,
                boundingBox.topLeft[0] / 200,
                boundingBox.bottomRight[1] / 200,
                boundingBox.bottomRight[0] / 200,
            ],
        ], [0], [26, 34])
            .squeeze([0]);
        const grayscale = cropped.mean(2).expandDims(2);
        const inputImage = grayscale.expandDims(0).toFloat().div(255);
        const prediction = await this.eyeblinkModel.predict(inputImage).data();
        return prediction[0];
    }
    async predictEyeOpenness(image) {
        const facePredictions = await this.facemeshModel.estimateFaces(image);
        if (facePredictions.length === 0)
            return null;
        const face = facePredictions[0];
        const imageTensor = tf.browser.fromPixels(image);
        const rightEyeMeshIdx = [27, 243, 23, 130];
        const leftEyeMeshIdx = [257, 359, 253, 362];
        const rightEyeBB = this.extractEyeBoundingBox(face, rightEyeMeshIdx);
        const leftEyeBB = this.extractEyeBoundingBox(face, leftEyeMeshIdx);
        const rightEyePred = await this.getPredictionWithinBoundingBox(imageTensor, rightEyeBB);
        const leftEyePred = await this.getPredictionWithinBoundingBox(imageTensor, leftEyeBB);
        imageTensor.dispose();
        return { right: rightEyePred, left: leftEyePred };
    }
}
exports.Eyeblink = Eyeblink;
//# sourceMappingURL=eyeblink.js.map