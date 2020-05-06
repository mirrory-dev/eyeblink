import * as tf from "@tensorflow/tfjs-core";
import { GraphModel } from "@tensorflow/tfjs-converter";
import { FaceMesh, AnnotatedPrediction } from "@tensorflow-models/facemesh";

export interface BoundingBox {
  topLeft: readonly [number, number];
  bottomRight: readonly [number, number];
}

export interface EyeblinkPrediction {
  right: number;
  left: number;
}

async function equalizeHist(images: tf.Tensor4D) {
  const data = await images.data();
  for (let i = 0; i < images.shape[0]; ++i) {
    let max = 0;
    let min = 1;
    for (let j = 0; j < images.strides[0]; ++j) {
      let index = i * images.strides[0] + j;
      max = Math.max(max, data[index]);
      min = Math.min(min, data[index]);
    }
    for (let j = 0; j < images.strides[0]; ++j) {
      let index = i * images.strides[0] + j;
      data[index] = (data[index] - min) / (max - min);
    }
  }
  return tf.tensor1d(data).reshape(images.shape);
}

export class Eyeblink {
  private eyeblinkModel: GraphModel;
  private facemeshModel: FaceMesh;

  constructor(eyeblinkModel: GraphModel, facemeshModel: FaceMesh) {
    this.eyeblinkModel = eyeblinkModel;
    this.facemeshModel = facemeshModel;
  }

  private extractEyeBoundingBox(
    face: AnnotatedPrediction,
    [top, right, bottom, left]: readonly [number, number, number, number]
  ): BoundingBox {
    const eyeTopY = (face.scaledMesh as any)[top][1] as number;
    const eyeRightX = (face.scaledMesh as any)[right][0] as number;
    const eyeBottomY = (face.scaledMesh as any)[bottom][1] as number;
    const eyeLeftX = (face.scaledMesh as any)[left][0] as number;
    const topLeft = [eyeLeftX, eyeTopY] as const;
    const bottomRight = [eyeRightX, eyeBottomY] as const;
    return { topLeft, bottomRight };
  }

  async getPredictionWithinBoundingBox(
    input: tf.Tensor3D,
    boundingBoxes: BoundingBox[]
  ) {
    const grayscale: tf.Tensor4D = tf.tidy(() => {
      const boundingBoxesNormalized = boundingBoxes.map((box) => {
        return [
          box.topLeft[1] / input.shape[0],
          box.topLeft[0] / input.shape[1],
          box.bottomRight[1] / input.shape[0],
          box.bottomRight[0] / input.shape[1],
        ];
      });

      const cropped = tf.image
        .cropAndResize(
          input.expandDims(0),
          boundingBoxesNormalized,
          boundingBoxesNormalized.map(() => 0),
          [26, 34]
        )
        .toFloat();
      return cropped.mean(3).expandDims(3).toFloat().div(255);
    });

    const grayscaleEqualized = await equalizeHist(grayscale);
    const prediction = this.eyeblinkModel.predict(
      grayscaleEqualized
    ) as tf.Tensor;
    const result = await prediction.data();

    grayscaleEqualized.dispose();
    prediction.dispose();
    return result;
  }

  async predictEyeOpenness(
    image:
      | tf.Tensor3D
      | ImageData
      | HTMLImageElement
      | HTMLCanvasElement
      | HTMLVideoElement,
    face?: AnnotatedPrediction
  ): Promise<EyeblinkPrediction | null> {
    if (!(image instanceof tf.Tensor)) {
      const tensor = tf.browser.fromPixels(image);
      const result = await this.predictEyeOpenness(tensor, face);
      tensor.dispose();
      return result;
    }

    if (!face) {
      const facePredictions = await this.facemeshModel.estimateFaces(image);
      if (facePredictions.length === 0) return null;
      face = facePredictions[0];
    }

    const rightEyeMeshIdx = [27, 243, 23, 130] as const;
    const leftEyeMeshIdx = [257, 359, 253, 362] as const;
    const rightEyeBB = this.extractEyeBoundingBox(face, rightEyeMeshIdx);
    const leftEyeBB = this.extractEyeBoundingBox(face, leftEyeMeshIdx);
    const [
      rightEyePred,
      leftEyePred,
    ] = await this.getPredictionWithinBoundingBox(image, [
      rightEyeBB,
      leftEyeBB,
    ]);

    return { right: rightEyePred, left: leftEyePred };
  }
}
