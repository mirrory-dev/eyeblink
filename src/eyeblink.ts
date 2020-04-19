import * as tf from '@tensorflow/tfjs';
import * as facemesh from '@tensorflow-models/facemesh';
import {FaceMesh} from '@tensorflow-models/facemesh';

export interface BoundingBox {
  topLeft: readonly [number, number];
  bottomRight: readonly [number, number];
}

export interface EyeblinkPrediction {
  right: number;
  left: number;
}

export class Eyeblink {
  private eyeblinkModel: tf.GraphModel;
  private facemeshModel: FaceMesh;

  constructor(eyeblinkModel: tf.GraphModel, facemeshModel: FaceMesh) {
    this.eyeblinkModel = eyeblinkModel;
    this.facemeshModel = facemeshModel;
  }

  private extractEyeBoundingBox(
    face: facemesh.AnnotatedPrediction,
    [top, right, bottom, left]: readonly [number, number, number, number],
  ): BoundingBox {
    const eyeTopY = (face.scaledMesh as any)[top][1] as number;
    const eyeRightX = (face.scaledMesh as any)[right][0] as number;
    const eyeBottomY = (face.scaledMesh as any)[bottom][1] as number;
    const eyeLeftX = (face.scaledMesh as any)[left][0] as number;
    const topLeft = [eyeLeftX, eyeTopY] as const;
    const bottomRight = [eyeRightX, eyeBottomY] as const;
    return {topLeft, bottomRight};
  }

  private async getPredictionWithinBoundingBox(
    input: tf.Tensor3D | ImageData,
    boundingBox: BoundingBox,
  ) {
    const image: tf.Tensor4D = tf.tidy(() => {
      if (!(input instanceof tf.Tensor)) {
        input = tf.browser.fromPixels(input);
      }
      return (input as tf.Tensor).toFloat().expandDims(0);
    });

    const cropped = tf.image
      .cropAndResize(
        image,
        [
          [
            boundingBox.topLeft[1] / 200,
            boundingBox.topLeft[0] / 200,
            boundingBox.bottomRight[1] / 200,
            boundingBox.bottomRight[0] / 200,
          ],
        ],
        [0],
        [26, 34],
      )
      .squeeze([0]);
    const grayscale = cropped.mean(2).expandDims(2);
    const inputImage = grayscale.expandDims(0).toFloat().div(255);
    const prediction = await (this.eyeblinkModel.predict(
      inputImage,
    ) as tf.Tensor).data();
    return prediction[0];
  }

  async predictEyeOpenness(
    image: ImageData,
  ): Promise<EyeblinkPrediction | null> {
    const facePredictions = await this.facemeshModel.estimateFaces(image);
    if (facePredictions.length === 0) return null;
    const face = facePredictions[0];
    const imageTensor = tf.browser.fromPixels(image);

    const rightEyeMeshIdx = [27, 243, 23, 130] as const;
    const leftEyeMeshIdx = [257, 359, 253, 362] as const;
    const rightEyeBB = this.extractEyeBoundingBox(face, rightEyeMeshIdx);
    const leftEyeBB = this.extractEyeBoundingBox(face, leftEyeMeshIdx);
    const rightEyePred = await this.getPredictionWithinBoundingBox(
      imageTensor,
      rightEyeBB,
    );
    const leftEyePred = await this.getPredictionWithinBoundingBox(
      imageTensor,
      leftEyeBB,
    );

    imageTensor.dispose();

    return {right: rightEyePred, left: leftEyePred};
  }
}
