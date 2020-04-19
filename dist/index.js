"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const tf = require("@tensorflow/tfjs-node");
const facemesh = require("@tensorflow-models/facemesh");
const path = require("path");
const eyeblink_1 = require("./eyeblink");
const defaultGraphModelPath = path.resolve(__dirname, '../models/model.json');
async function load(graphModelPath = defaultGraphModelPath) {
    const blinkModel = await tf.loadGraphModel('file://' + path.resolve(graphModelPath));
    const facemeshModel = await facemesh.load({ maxFaces: 1 });
    return new eyeblink_1.Eyeblink(blinkModel, facemeshModel);
}
exports.load = load;
//# sourceMappingURL=index.js.map