"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
function getImageData(videoEl) {
    const width = videoEl.width;
    const height = videoEl.height;
    const procCanvas = new OffscreenCanvas(width, height);
    const ctx = procCanvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, procCanvas.width, procCanvas.height);
    const image = ctx.getImageData(0, 0, procCanvas.width, procCanvas.height);
    return image;
}
exports.getImageData = getImageData;
//# sourceMappingURL=image.js.map