"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs-converter");
const facemesh = require("@tensorflow-models/facemesh");
const eyeblink_1 = require("./eyeblink");
const defaultGraphModelPath = 'https://unpkg.com/@prism-3d/eyeblink/models/model.json';
async function load(graphModelPath = defaultGraphModelPath) {
    const blinkModel = await tf.loadGraphModel(graphModelPath);
    const facemeshModel = await facemesh.load({ maxFaces: 1 });
    return new eyeblink_1.Eyeblink(blinkModel, facemeshModel);
}
exports.load = load;
//# sourceMappingURL=browser.js.map