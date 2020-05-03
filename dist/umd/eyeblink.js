/**
    * @license
    * Copyright 2020 Prism.
    * Licensed under the Apache License, Version 2.0 (the "License");
    * you may not use this file except in compliance with the License.
    * You may obtain a copy of the License at
    *
    * http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing, software
    * distributed under the License is distributed on an "AS IS" BASIS,
    * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    * See the License for the specific language governing permissions and
    * limitations under the License.
    * =============================================================================
    */
(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports, require('@tensorflow/tfjs-converter'), require('@tensorflow/tfjs-core')) :
    typeof define === 'function' && define.amd ? define(['exports', '@tensorflow/tfjs-converter', '@tensorflow/tfjs-core'], factory) :
    (global = global || self, factory(global.eyeblink = {}, global.tf, global.tfjsCore));
}(this, (function (exports, tfjsConverter, tf) { 'use strict';

    /**
        * @license
        * Copyright 2020 Google LLC. All Rights Reserved.
        * Licensed under the Apache License, Version 2.0 (the "License");
        * you may not use this file except in compliance with the License.
        * You may obtain a copy of the License at
        *
        * http://www.apache.org/licenses/LICENSE-2.0
        *
        * Unless required by applicable law or agreed to in writing, software
        * distributed under the License is distributed on an "AS IS" BASIS,
        * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        * See the License for the specific language governing permissions and
        * limitations under the License.
        * =============================================================================
        */
    const disposeBox=e=>{e.startEndTensor.dispose(),e.startPoint.dispose(),e.endPoint.dispose();},createBox=e=>({startEndTensor:e,startPoint:tf.slice(e,[0,0],[-1,2]),endPoint:tf.slice(e,[0,2],[-1,2])}),scaleBox=(e,t)=>{const s=tf.mul(e.startPoint,t),o=tf.mul(e.endPoint,t),n=tf.concat2d([s,o],1);return createBox(n)},ANCHORS_CONFIG={strides:[8,16],anchors:[2,6]},NUM_LANDMARKS=6;function generateAnchors(e,t,s){const o=[];for(let n=0;n<s.strides.length;n++){const i=s.strides[n],r=Math.floor((t+i-1)/i),a=Math.floor((e+i-1)/i),c=s.anchors[n];for(let e=0;e<r;e++){const t=i*(e+.5);for(let e=0;e<a;e++){const s=i*(e+.5);for(let e=0;e<c;e++)o.push([s,t]);}}}return o}function decodeBounds(e,t,s){const o=tf.slice(e,[0,1],[-1,2]),n=tf.add(o,t),i=tf.slice(e,[0,3],[-1,2]),r=tf.div(i,s),a=tf.div(n,s),c=tf.div(r,2),l=tf.sub(a,c),d=tf.add(a,c),h=tf.mul(l,s),u=tf.mul(d,s);return tf.concat2d([h,u],1)}function getInputTensorDimensions(e){return e instanceof tf.Tensor?[e.shape[0],e.shape[1]]:[e.height,e.width]}function flipFaceHorizontal(e,t){let s,o,n;if(e.topLeft instanceof tf.Tensor&&e.bottomRight instanceof tf.Tensor){const[i,r]=tf.tidy(()=>[tf.concat([tf.sub(t-1,e.topLeft.slice(0,1)),e.topLeft.slice(1,1)]),tf.concat([tf.sub(t-1,e.bottomRight.slice(0,1)),e.bottomRight.slice(1,1)])]);s=i,o=r,null!=e.landmarks&&(n=tf.tidy(()=>{const s=tf.sub(tf.tensor1d([t-1,0]),e.landmarks),o=tf.tensor1d([1,-1]);return tf.mul(s,o)}));}else {const[i,r]=e.topLeft,[a,c]=e.bottomRight;s=[t-1-i,r],o=[t-1-a,c],null!=e.landmarks&&(n=e.landmarks.map(e=>[t-1-e[0],e[1]]));}const i={topLeft:s,bottomRight:o};return null!=n&&(i.landmarks=n),null!=e.probability&&(i.probability=e.probability instanceof tf.Tensor?e.probability.clone():e.probability),i}function scaleBoxFromPrediction(e,t){return tf.tidy(()=>{let s;return s=e.hasOwnProperty("box")?e.box:e,scaleBox(s,t).startEndTensor.squeeze()})}class BlazeFaceModel{constructor(e,t,s,o,n,i){this.blazeFaceModel=e,this.width=t,this.height=s,this.maxFaces=o,this.anchorsData=generateAnchors(t,s,ANCHORS_CONFIG),this.anchors=tf.tensor2d(this.anchorsData),this.inputSizeData=[t,s],this.inputSize=tf.tensor1d([t,s]),this.iouThreshold=n,this.scoreThreshold=i;}async getBoundingBoxes(e,t,s=!0){const[o,n,i]=tf.tidy(()=>{const t=e.resizeBilinear([this.width,this.height]),s=tf.mul(tf.sub(t.div(255),.5),2),o=this.blazeFaceModel.predict(s).squeeze(),n=decodeBounds(o,this.anchors,this.inputSize),i=tf.slice(o,[0,0],[-1,1]);return [o,n,tf.sigmoid(i).squeeze()]}),r=console.warn;console.warn=(()=>{});const a=tf.image.nonMaxSuppression(n,i,this.maxFaces,this.iouThreshold,this.scoreThreshold);console.warn=r;const c=await a.array();a.dispose();let l=c.map(e=>tf.slice(n,[e,0],[1,-1]));t||(l=await Promise.all(l.map(async e=>{const t=await e.array();return e.dispose(),t})));const d=e.shape[1],h=e.shape[2];let u;u=t?tf.div([h,d],this.inputSize):[h/this.inputSizeData[0],d/this.inputSizeData[1]];const p=[];for(let e=0;e<l.length;e++){const n=l[e],r=tf.tidy(()=>{const r=createBox(n instanceof tf.Tensor?n:tf.tensor2d(n));if(!s)return r;const a=c[e];let l;return l=t?this.anchors.slice([a,0],[1,2]):this.anchorsData[a],{box:r,landmarks:tf.slice(o,[a,NUM_LANDMARKS-1],[1,-1]).squeeze().reshape([NUM_LANDMARKS,-1]),probability:tf.slice(i,[a],[1]),anchor:l}});p.push(r);}return n.dispose(),i.dispose(),o.dispose(),{boxes:p,scaleFactor:u}}async estimateFaces(e,t=!1,s=!1,o=!0){const[,n]=getInputTensorDimensions(e),i=tf.tidy(()=>(e instanceof tf.Tensor||(e=tf.browser.fromPixels(e)),e.toFloat().expandDims(0))),{boxes:r,scaleFactor:a}=await this.getBoundingBoxes(i,t,o);return i.dispose(),t?r.map(e=>{const t=scaleBoxFromPrediction(e,a);let i={topLeft:t.slice([0],[2]),bottomRight:t.slice([2],[2])};if(o){const{landmarks:t,probability:s,anchor:o}=e,n=t.add(o).mul(a);i.landmarks=n,i.probability=s;}return s&&(i=flipFaceHorizontal(i,n)),i}):Promise.all(r.map(async e=>{const t=scaleBoxFromPrediction(e,a);let i;if(o){const[s,o,n]=await Promise.all([e.landmarks,t,e.probability].map(async e=>e.array())),r=e.anchor,[c,l]=a,d=s.map(e=>[(e[0]+r[0])*c,(e[1]+r[1])*l]);i={topLeft:o.slice(0,2),bottomRight:o.slice(2),landmarks:d,probability:n},disposeBox(e.box),e.landmarks.dispose(),e.probability.dispose();}else {const e=await t.array();i={topLeft:e.slice(0,2),bottomRight:e.slice(2)};}return t.dispose(),s&&(i=flipFaceHorizontal(i,n)),i}))}}const BLAZEFACE_MODEL_URL="https://tfhub.dev/tensorflow/tfjs-model/blazeface/1/default/1";async function load({maxFaces:e=10,inputWidth:t=128,inputHeight:s=128,iouThreshold:o=.3,scoreThreshold:n=.75}={}){const i=await tfjsConverter.loadGraphModel(BLAZEFACE_MODEL_URL,{fromTFHub:!0});return new BlazeFaceModel(i,t,s,e,o,n)}const MESH_ANNOTATIONS={silhouette:[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109],lipsUpperOuter:[61,185,40,39,37,0,267,269,270,409,291],lipsLowerOuter:[146,91,181,84,17,314,405,321,375,291],lipsUpperInner:[78,191,80,81,82,13,312,311,310,415,308],lipsLowerInner:[78,95,88,178,87,14,317,402,318,324,308],rightEyeUpper0:[246,161,160,159,158,157,173],rightEyeLower0:[33,7,163,144,145,153,154,155,133],rightEyeUpper1:[247,30,29,27,28,56,190],rightEyeLower1:[130,25,110,24,23,22,26,112,243],rightEyeUpper2:[113,225,224,223,222,221,189],rightEyeLower2:[226,31,228,229,230,231,232,233,244],rightEyeLower3:[143,111,117,118,119,120,121,128,245],rightEyebrowUpper:[156,70,63,105,66,107,55,193],rightEyebrowLower:[35,124,46,53,52,65],leftEyeUpper0:[466,388,387,386,385,384,398],leftEyeLower0:[263,249,390,373,374,380,381,382,362],leftEyeUpper1:[467,260,259,257,258,286,414],leftEyeLower1:[359,255,339,254,253,252,256,341,463],leftEyeUpper2:[342,445,444,443,442,441,413],leftEyeLower2:[446,261,448,449,450,451,452,453,464],leftEyeLower3:[372,340,346,347,348,349,350,357,465],leftEyebrowUpper:[383,300,293,334,296,336,285,417],leftEyebrowLower:[265,353,276,283,282,295],midwayBetweenEyes:[168],noseTip:[1],noseBottom:[2],noseRightCorner:[98],noseLeftCorner:[327],rightCheek:[205],leftCheek:[425]};function disposeBox$1(e){null!=e&&null!=e.startPoint&&(e.startEndTensor.dispose(),e.startPoint.dispose(),e.endPoint.dispose());}function createBox$1(e,t,s){return {startEndTensor:e,startPoint:null!=t?t:tf.slice(e,[0,0],[-1,2]),endPoint:null!=s?s:tf.slice(e,[0,2],[-1,2])}}function scaleBoxCoordinates(e,t){const s=tf.mul(e.startPoint,t),o=tf.mul(e.endPoint,t);return createBox$1(tf.concat2d([s,o],1))}function getBoxSize(e){return tf.tidy(()=>{const t=tf.sub(e.endPoint,e.startPoint);return tf.abs(t)})}function getBoxCenter(e){return tf.tidy(()=>{const t=tf.div(tf.sub(e.endPoint,e.startPoint),2);return tf.add(e.startPoint,t)})}function cutBoxFromImageAndResize(e,t,s){const o=t.shape[1],n=t.shape[2],i=e.startEndTensor;return tf.tidy(()=>{const e=tf.concat2d([i.slice([0,1],[-1,1]),i.slice([0,0],[-1,1]),i.slice([0,3],[-1,1]),i.slice([0,2],[-1,1])],0),r=tf.div(e.transpose(),[o,n,o,n]);return tf.image.cropAndResize(t,r,[0],s)})}function enlargeBox(e,t=1.5){return tf.tidy(()=>{const s=getBoxCenter(e),o=getBoxSize(e),n=tf.mul(tf.div(o,2),t),i=tf.sub(s,n),r=tf.add(s,n);return createBox$1(tf.concat2d([i,r],1),i,r)})}const LANDMARKS_COUNT=468,UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD=.25;class Pipeline{constructor(e,t,s,o,n,i){this.regionsOfInterest=[],this.runsWithoutFaceDetector=0,this.boundingBoxDetector=e,this.meshDetector=t,this.meshWidth=s,this.meshHeight=o,this.maxContinuousChecks=n,this.maxFaces=i;}async predict(e){if(this.shouldUpdateRegionsOfInterest()){const t=!0,s=!1,{boxes:o,scaleFactor:n}=await this.boundingBoxDetector.getBoundingBoxes(e,t,s);if(0===o.length)return n.dispose(),this.clearAllRegionsOfInterest(),null;const i=o.map(e=>enlargeBox(scaleBoxCoordinates(e,n)));o.forEach(disposeBox$1),this.updateRegionsOfInterest(i),this.runsWithoutFaceDetector=0;}else this.runsWithoutFaceDetector++;return tf.tidy(()=>this.regionsOfInterest.map((t,s)=>{const o=cutBoxFromImageAndResize(t,e,[this.meshHeight,this.meshWidth]).div(255),[,n,i]=this.meshDetector.predict(o),r=tf.reshape(i,[-1,3]),a=tf.div(getBoxSize(t),[this.meshWidth,this.meshHeight]),c=tf.mul(r,a.concat(tf.tensor2d([1],[1,1]),1)).add(t.startPoint.concat(tf.tensor2d([0],[1,1]),1)),l=this.calculateLandmarksBoundingBox(c);return disposeBox$1(this.regionsOfInterest[s]),this.regionsOfInterest[s]=l,{coords:r,scaledCoords:c,box:l,flag:n.squeeze()}}))}updateRegionsOfInterest(e){for(let t=0;t<e.length;t++){const s=e[t],o=this.regionsOfInterest[t];let n=0;if(o&&o.startPoint){const[e,t,i,r]=s.startEndTensor.arraySync()[0],[a,c,l,d]=o.startEndTensor.arraySync()[0],h=Math.max(e,a),u=Math.max(t,c),p=(Math.min(i,l)-h)*(Math.min(r,d)-u);n=p/((i-e)*(r-t)+(l-a)*(d-t)-p);}n>UPDATE_REGION_OF_INTEREST_IOU_THRESHOLD?disposeBox$1(s):(this.regionsOfInterest[t]=s,disposeBox$1(o));}for(let t=e.length;t<this.regionsOfInterest.length;t++)disposeBox$1(this.regionsOfInterest[t]);this.regionsOfInterest=this.regionsOfInterest.slice(0,e.length);}clearRegionOfInterest(e){null!=this.regionsOfInterest[e]&&(disposeBox$1(this.regionsOfInterest[e]),this.regionsOfInterest=[...this.regionsOfInterest.slice(0,e),...this.regionsOfInterest.slice(e+1)]);}clearAllRegionsOfInterest(){for(let e=0;e<this.regionsOfInterest.length;e++)disposeBox$1(this.regionsOfInterest[e]);this.regionsOfInterest=[];}shouldUpdateRegionsOfInterest(){const e=this.regionsOfInterest.length,t=0===e;return 1===this.maxFaces||t?t:e!==this.maxFaces&&this.runsWithoutFaceDetector>=this.maxContinuousChecks}calculateLandmarksBoundingBox(e){const t=e.slice([0,0],[LANDMARKS_COUNT,1]),s=e.slice([0,1],[LANDMARKS_COUNT,1]);return enlargeBox(createBox$1(tf.stack([t.min(),s.min(),t.max(),s.max()]).expandDims(0)))}}const FACEMESH_GRAPHMODEL_PATH="https://tfhub.dev/mediapipe/tfjs-model/facemesh/1/default/1",MESH_MODEL_INPUT_WIDTH=192,MESH_MODEL_INPUT_HEIGHT=192;async function load$1({maxContinuousChecks:e=5,detectionConfidence:t=.9,maxFaces:s=10,iouThreshold:o=.3,scoreThreshold:n=.75}={}){const[i,r]=await Promise.all([loadDetectorModel(s,o,n),loadMeshModel()]);return new FaceMesh(i,r,e,t,s)}async function loadDetectorModel(e,t,s){return load({maxFaces:e,iouThreshold:t,scoreThreshold:s})}async function loadMeshModel(){return tfjsConverter.loadGraphModel(FACEMESH_GRAPHMODEL_PATH,{fromTFHub:!0})}function getInputTensorDimensions$1(e){return e instanceof tf.Tensor?[e.shape[0],e.shape[1]]:[e.height,e.width]}function flipFaceHorizontal$1(e,t){if(e.mesh instanceof tf.Tensor){const[s,o,n,i]=tf.tidy(()=>{const s=tf.tensor1d([t-1,0,0]),o=tf.tensor1d([1,-1,1]);return tf.tidy(()=>[tf.concat([tf.sub(t-1,e.boundingBox.topLeft.slice(0,1)),e.boundingBox.topLeft.slice(1,1)]),tf.concat([tf.sub(t-1,e.boundingBox.bottomRight.slice(0,1)),e.boundingBox.bottomRight.slice(1,1)]),tf.sub(s,e.mesh).mul(o),tf.sub(s,e.scaledMesh).mul(o)])});return Object.assign({},e,{boundingBox:{topLeft:s,bottomRight:o},mesh:n,scaledMesh:i})}return Object.assign({},e,{boundingBox:{topLeft:[t-1-e.boundingBox.topLeft[0],e.boundingBox.topLeft[1]],bottomRight:[t-1-e.boundingBox.bottomRight[0],e.boundingBox.bottomRight[1]]},mesh:e.mesh.map(e=>{const s=e.slice(0);return s[0]=t-1-e[0],s}),scaledMesh:e.scaledMesh.map(e=>{const s=e.slice(0);return s[0]=t-1-e[0],s})})}class FaceMesh{constructor(e,t,s,o,n){this.pipeline=new Pipeline(e,t,MESH_MODEL_INPUT_WIDTH,MESH_MODEL_INPUT_HEIGHT,s,n),this.detectionConfidence=o;}static getAnnotations(){return MESH_ANNOTATIONS}async estimateFaces(e,t=!1,s=!1){const[,o]=getInputTensorDimensions$1(e),n=tf.tidy(()=>(e instanceof tf.Tensor||(e=tf.browser.fromPixels(e)),e.toFloat().expandDims(0))),i=tf.env().get("WEBGL_PACK_DEPTHWISECONV");tf.env().set("WEBGL_PACK_DEPTHWISECONV",!0);const r=await this.pipeline.predict(n);return tf.env().set("WEBGL_PACK_DEPTHWISECONV",i),n.dispose(),null!=r&&r.length>0?Promise.all(r.map(async(e,n)=>{const{coords:i,scaledCoords:r,box:a,flag:c}=e;let l=[c];t||(l=l.concat([i,r,a.startPoint,a.endPoint]));const d=await Promise.all(l.map(async e=>e.array())),h=d[0];if(c.dispose(),h<this.detectionConfidence&&this.pipeline.clearRegionOfInterest(n),t){const e={faceInViewConfidence:h,mesh:i,scaledMesh:r,boundingBox:{topLeft:a.startPoint.squeeze(),bottomRight:a.endPoint.squeeze()}};return s?flipFaceHorizontal$1(e,o):e}const[u,p,f,m]=d.slice(1);r.dispose(),i.dispose();let g={faceInViewConfidence:h,boundingBox:{topLeft:f,bottomRight:m},mesh:u,scaledMesh:p};s&&(g=flipFaceHorizontal$1(g,o));const b={};for(const e in MESH_ANNOTATIONS)b[e]=MESH_ANNOTATIONS[e].map(e=>g.scaledMesh[e]);return g.annotations=b,g})):[]}}

    function equalizeHist(images) {
        const data = images.dataSync();
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
            const prediction = tf.tidy(() => {
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
                const inputImage = equalizeHist(grayscale.toFloat().div(255));
                return this.eyeblinkModel.predict(inputImage);
            });
            const result = await prediction.data();
            prediction.dispose();
            return result;
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

    function getImageData(videoEl) {
        const width = videoEl.width;
        const height = videoEl.height;
        const procCanvas = new OffscreenCanvas(width, height);
        const ctx = procCanvas.getContext('2d');
        ctx.drawImage(videoEl, 0, 0, procCanvas.width, procCanvas.height);
        const image = ctx.getImageData(0, 0, procCanvas.width, procCanvas.height);
        return image;
    }

    const defaultGraphModelPath = 'https://VanityXR.github.io/eyeblink/models/model.json';
    async function loadModel(graphModelPath = defaultGraphModelPath) {
        return tfjsConverter.loadGraphModel(graphModelPath);
    }
    async function load$2(graphModelPath = defaultGraphModelPath) {
        const blinkModel = await loadModel(graphModelPath);
        const facemeshModel = await load$1({ maxFaces: 1 });
        return new Eyeblink(blinkModel, facemeshModel);
    }

    exports.Eyeblink = Eyeblink;
    exports.getImageData = getImageData;
    exports.load = load$2;
    exports.loadModel = loadModel;

    Object.defineProperty(exports, '__esModule', { value: true });

})));
