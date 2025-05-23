import { Layer } from './Layer';
import { ActivationFunction } from './ActivationFunction/ActivationFunction';
import { ReLU } from './ActivationFunction/ReLU';
import { Identity } from './ActivationFunction/Identity';
import { softmax, round, norm } from './utils';

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

  train(inputs: number[][], targets: number[][], learningRate: number): number {
    if (inputs.length !== targets.length) {
      throw new Error(`Input and target of different lengths`);
    }

    const batchLoss = [];
    const accLossGradient: number[] = [];

    // Iterate over each example
    for (let i = 0; i < inputs.length; i++) {
      const input = inputs[i];
      const target = targets[i];

      const predicted = this.predict(input);
      const loss = this.loss(predicted, target);
      batchLoss.push(loss);

      const lossGradient = this.getLossGradient(predicted, target);

      if (accLossGradient.length === 0) {
        for (let j = 0; j < lossGradient.length; j++) {
          accLossGradient.push(0);
        }
      }

      for (let j = 0; j < accLossGradient.length; j++) {
        accLossGradient[j] += lossGradient[j];
      }
    }

    for (let i = 0; i < accLossGradient.length; i++) {
      accLossGradient[i] /= accLossGradient.length;
    }

    this.backward(accLossGradient, learningRate);

    return batchLoss.reduce((sum, val) => (sum += val)) / batchLoss.length;
  }

  backward(lossGradient: number[], learningRate: number) {
    for (let i = this.layers.length - 1; i >= 0; i--) {
      lossGradient = this.layers[i].backward(lossGradient, learningRate);
    }
  }

  // Cross entropy (softmax prediction, one-hot expected)
  loss(predicted: number[], target: number[]): number {
    const targetIndex = target.findIndex((n) => n === 1);
    const epsilon = 1e-15; // to prevent log(0) error
    const prob = Math.max(predicted[targetIndex], epsilon);
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
