import { Layer } from './Layer';

export class NeuralNetwork {
  private layers: Layer[];

  constructor(layers: number, depth: number) {
    this.layers = Array.from({ length: layers }, () => {
      return new Layer(depth);
    });
  }

  getLayers(): Layer[] {
    return this.layers;
  }

  train(
    input: number[],
    answerKey: number[],
    epochs: number,
    learningRate: number
  ) {
    for (let i = 0; i < epochs; i++) {
      console.log(`\nEpoch #${i + 1}`);

      const result = this.predict(input);
      const loss = this.loss(result, answerKey);
      console.log(`Loss: ${loss}`);

      this.backward();
    }
  }

  // Back propogation
  backward() {}

  // Mean squared error
  loss(predicted: number[], expected: number[]): number {
    // Throw error if predicted and expected arrays don't match up
    if (predicted.length !== expected.length) {
      throw new Error(`Predicted and expected must be of same length`);
    }

    const squaredErrors = predicted.reduce((acc, value, index) => {
      return acc + Math.pow(value - expected[index], 2);
    }, 0);

    return squaredErrors / predicted.length;
  }

  predict(input: number[]): number[] {
    let result = input.slice();

    console.log(`Neural net input: ${this.truncateVector(input)}`);

    for (const [index, layer] of this.layers.entries()) {
      result = layer.forward(result);
      console.log(
        `Output from layer #${index}: ${this.truncateVector(result)}`
      );
    }

    const adjustedResult = this.applySoftMax(result);

    console.log(`Neural net output: ${this.truncateVector(adjustedResult)}`);

    return adjustedResult;
  }

  getLossGradient(predicted: number[], expected: number[]): number[] {
    return predicted.map((prediction, index) => {
      // Calculate derivative of mean squared error
      return 2 * (prediction - expected[index]);
    });
  }

  applySoftMax(input: number[], temperature: number = 1): number[] {
    // Prevent divide by zero errors
    const safeTemp = Math.max(temperature, 1e-6);
    // Get max to subtract for numerical stability
    const max = Math.max(...input);

    const adjustedInput = input.map((weight) =>
      Math.exp((weight - max) / safeTemp)
    );
    const denominator = adjustedInput.reduce((acc, curr) => (acc += curr), 0);

    return adjustedInput.map((inputNum) => inputNum / denominator);
  }

  truncateVector(vector: number[]): number[] {
    return vector.map((num) => Math.round(num * 100) / 100);
  }
}
