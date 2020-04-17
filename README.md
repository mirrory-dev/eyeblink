# Eye Blink Detector

The eye blink detector that consumes a cropped image of eyes and predicts the eye openness. The original model is derived from [Taehee Lee's Eye Blink Detector](https://github.com/kairess/eye_blink_detector).

## Usage

```
yarn add @prism-3d/eyeblink
```

### Multi-architecture support

#### Node.js

Load tf.js Model from file-system and use Tensorflow C binary to predict.

```js
import * as eyeblinkModel from '@prism-3d/eyeblink';
const model = await eyeblinkModel.load('path/to/tfjs-model');
```

#### Browser

Load tf.js Model from URL and use Tensorflow.js to predict.

```js
import * as eyeblinkModel from '@prism-3d/eyeblink/lib/browser';
const model = await eyeblinkModel.load('https://path/to/tfjs-model');
```

## Demo

```
cd demo
yarn install
yarn start
open https://localhost:5000
```
