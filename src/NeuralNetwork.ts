import { Layer } from './Layer';
import { ActivationFunction } from './ActivationFunction/ActivationFunction';
import { ReLU } from './ActivationFunction/ReLU';
import { Identity } from './ActivationFunction/Identity';
import { softmax, round } from './utils';

// Types
export type ModelState = {
  layers: {
    weights: number[][];
    biases: number[];
  }[];
};

export class NeuralNetwork {
  private layers: Layer[];

  constructor(layerSizes: number[]) {
    this.layers = [];

    // Skip input layer
    for (let i = 1; i < layerSizes.length; i++) {
      let layer: Layer;

      if (i === layerSizes.length - 1) {
        // Use identity function for output layer
        layer = new Layer(layerSizes[i], layerSizes[i - 1], new Identity());
      } else {
        // Use ReLU for hidden layers
        layer = new Layer(layerSizes[i], layerSizes[i - 1], new ReLU());
      }

      this.layers.push(layer);
    }
  }

  train(input: number[], expected: number[], learningRate: number): number {
    const predicted = this.predict(input);
    const targetIndex = expected.findIndex((n) => n === 1);
    const loss = this.loss(predicted, targetIndex);
    const lossGradient = this.getLossGradient(predicted, expected);

    // console.log(` Input: ${input}`);
    // console.log(` Pred.: ${predicted.map((n) => round(n))}`);

    this.backward(lossGradient, learningRate);

    return loss;
  }

  backward(lossGradient: number[], learningRate: number) {
    for (let i = this.layers.length - 1; i >= 0; i--) {
      lossGradient = this.layers[i].backward(lossGradient, learningRate);
    }
  }

  // Cross entropy (softmax prediction, one-hot expected)
  loss(predicted: number[], expectedIndex: number): number {
    const epsilon = 1e-15; // to prevent log(0) error
    const prob = Math.max(predicted[expectedIndex], epsilon);
    return -Math.log(prob);
  }

  predict(input: number[]): number[] {
    let result = input;

    for (const layer of this.layers) {
      result = layer.forward(result);
    }

    return softmax(result);
  }

  getLossGradient(predicted: number[], expected: number[]): number[] {
    let grad = predicted.slice();
    let targetIndex = expected.findIndex((n) => n === 1);
    grad[targetIndex] -= 1;
    return grad;
  }

  // Get output at layer X
  forwardToLayer(input: number[], layer: number): number[] {
    if (layer > this.layers.length || layer < 0) {
      throw new Error(`Layer must be valid layer ID (received ${layer})`);
    }

    let result = input;

    for (let i = 0; i <= layer; i++) {
      result = this.layers[i].forward(result);
    }

    return result;
  }

  getModelState(): ModelState {
    const layers = [];

    for (const layer of this.layers) {
      const result = layers.push({
        weights: layer.getWeights(),
        biases: layer.getBiases(),
      });
    }

    return { layers: layers };
  }

  buildFromSavedModel(savedModel: ModelState) {
    savedModel.layers.forEach((layer, index) => {
      this.layers[index].createFromSavedModel(layer);
      console.log(`Created layer ${index + 1} from file`);
    });
  }

  getLayers(): Layer[] {
    return this.layers;
  }
}
