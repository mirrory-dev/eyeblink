import node from '@rollup/plugin-node-resolve';
import {terser} from 'rollup-plugin-terser';
import nodeBuiltins from 'rollup-plugin-node-builtins';
import ts from '@wessberg/rollup-plugin-ts';

const PREAMBLE = `/**
    * @license
    * Copyright ${new Date().getFullYear()} Prism.
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
    */`;

function minify() {
  return terser();
}

function config({input, plugins = [], output = {}}) {
  return {
    input,
    plugins: [
      ts({
        tsconfig: (resolvedConfig) => ({...resolvedConfig, module: 'ES2015'}),
      }),
      node(),
      nodeBuiltins(),
      ...plugins,
    ],
    output: {
      banner: PREAMBLE,
      globals: {
        '@tensorflow/tfjs': 'tf',
        '@tensorflow/tfjs-core': 'tfjsCore',
        '@tensorflow/tfjs-converter': 'tf',
      },
      ...output,
    },
    external: [
      '@tensorflow/tfjs',
      '@tensorflow/tfjs-core',
      '@tensorflow/tfjs-converter',
    ],
  };
}

export default [
  config({
    input: 'src/browser.ts',
    output: {format: 'umd', name: 'eyeblink', file: 'dist/umd/eyeblink.js'},
  }),
  config({
    input: 'src/browser.ts',
    plugins: [minify()],
    output: {format: 'umd', name: 'eyeblink', file: 'dist/umd/eyeblink.min.js'},
  }),
];
