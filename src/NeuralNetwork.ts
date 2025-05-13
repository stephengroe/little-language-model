import { Layer } from './Layer';
import { ActivationFunction } from './ActivationFunction/ActivationFunction';
import { ReLU } from './ActivationFunction/ReLU';
import { Identity } from './ActivationFunction/Identity';

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
      if (i === layerSizes.length) {
        // Use identity function for output layer
        this.layers.push(
          new Layer(layerSizes[i], layerSizes[i - 1], new Identity())
        );
      } else {
        // Use ReLU for hidden layers
        this.layers.push(
          new Layer(layerSizes[i], layerSizes[i - 1], new ReLU())
        );
      }
    }
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

  train(input: number[], expected: number[], learningRate: number) {
    const predicted = this.predict(input);
    const loss = this.loss(predicted, expected);
    const lossGradient = this.getLossGradient(predicted, expected);

    this.backward(lossGradient, learningRate);
  }

  trainOnBatch(
    batch: { input: number[][]; target: number[][] },
    learningRate: number
  ) {
    for (let i = 0; i < batch.input.length; i++) {
      this.train(batch.input[i], batch.target[i], learningRate);
    }

    // Apply gradients after processing batch
    for (const layer of this.layers) {
      layer.applyGradients(learningRate, batch.input.length);
    }
  }

  // Back propogation
  backward(lossGradient: number[], learningRate: number) {
    // Move backward across layers
    for (let i = this.layers.length - 1; i >= 0; i--) {
      lossGradient = this.layers[i].calculateGradients(lossGradient);
    }
  }

  // Cross entropy (softmax prediction, one-hot expected)
  loss(predicted: number[], expected: number[]): number {
    // Throw error if predicted and expected arrays don't match up
    if (predicted.length !== expected.length) {
      throw new Error(`Predicted and expected must be of same length`);
    }

    const epsilon = 1e-15; // to prevent log(0) error
    return expected.reduce((sum, actual, i) => {
      return sum - actual * Math.log(Math.max(predicted[i], epsilon));
    });
  }

  predict(input: number[]): number[] {
    let result = input.slice();

    for (const layer of this.layers) {
      result = layer.forward(result);
    }

    return result;
  }

  getLossGradient(predicted: number[], expected: number[]): number[] {
    return predicted.map((prediction, index) => {
      // Calculate derivative
      return prediction - expected[index];
    });
  }

  truncateVector(vector: number[]): number[] {
    return vector.map((num) => Math.round(num * 100) / 100);
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
}
