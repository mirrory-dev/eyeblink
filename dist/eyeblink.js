"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs-core");
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
    async getPredictionWithinBoundingBox(input, boundingBoxes) {
        const boundingBoxesNormalized = boundingBoxes.map((box) => {
            return [
                box.topLeft[1] / input.shape[0],
                box.topLeft[0] / input.shape[1],
                box.bottomRight[1] / input.shape[0],
                box.bottomRight[0] / input.shape[1],
            ];
        });
        const cropped = tf.image
            .cropAndResize(input.expandDims(0), boundingBoxesNormalized, boundingBoxesNormalized.map(() => 0), [26, 34])
            .toFloat();
        const grayscale = cropped.mean(3).expandDims(3);
        const inputImage = grayscale.toFloat().div(255);
        const prediction = await this.eyeblinkModel.predict(inputImage).data();
        return prediction;
    }
    async predictEyeOpenness(image, face) {
        if (!(image instanceof tf.Tensor)) {
            const tensor = tf.browser.fromPixels(image);
            const result = await this.predictEyeOpenness(tensor, face);
            tensor.dispose();
            return result;
        }
        if (!face) {
            const facePredictions = await this.facemeshModel.estimateFaces(image);
            if (facePredictions.length === 0)
                return null;
            face = facePredictions[0];
        }
        const rightEyeMeshIdx = [27, 243, 23, 130];
        const leftEyeMeshIdx = [257, 359, 253, 362];
        const rightEyeBB = this.extractEyeBoundingBox(face, rightEyeMeshIdx);
        const leftEyeBB = this.extractEyeBoundingBox(face, leftEyeMeshIdx);
        const [rightEyePred, leftEyePred,] = await this.getPredictionWithinBoundingBox(image, [
            rightEyeBB,
            leftEyeBB,
        ]);
        return { right: rightEyePred, left: leftEyePred };
    }
}
exports.Eyeblink = Eyeblink;
//# sourceMappingURL=eyeblink.js.map