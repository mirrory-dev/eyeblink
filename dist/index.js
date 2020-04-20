"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tfjs_converter_1 = require("@tensorflow/tfjs-converter");
const facemesh_1 = require("@tensorflow-models/facemesh");
const path_1 = require("path");
const eyeblink_1 = require("./eyeblink");
const defaultGraphModelPath = path_1.resolve(__dirname, '../models/model.json');
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
//# sourceMappingURL=index.js.map