"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tfjs_converter_1 = require("@tensorflow/tfjs-converter");
const facemesh_1 = require("@tensorflow-models/facemesh");
const eyeblink_1 = require("./eyeblink");
var eyeblink_2 = require("./eyeblink");
exports.Eyeblink = eyeblink_2.Eyeblink;
var image_1 = require("./utils/image");
exports.getImageData = image_1.getImageData;
const defaultGraphModelPath = 'https://prism-3d.github.io/eyeblink/models/model.json';
async function loadModel(graphModelPath = defaultGraphModelPath) {
    return tfjs_converter_1.loadGraphModel(graphModelPath);
}
exports.loadModel = loadModel;
async function load(graphModelPath = defaultGraphModelPath) {
    const blinkModel = await loadModel(graphModelPath);
    const facemeshModel = await facemesh_1.load({ maxFaces: 1 });
    return new eyeblink_1.Eyeblink(blinkModel, facemeshModel);
}
exports.load = load;
//# sourceMappingURL=browser.js.map