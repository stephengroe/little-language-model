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

  train(input: number[], answerKey: number[]) {
    const result = this.predict(input);
    const loss = this.loss(result, answerKey);

    console.log(`Loss: ${loss}`);
  }

  // Mean squared error
  loss(input: number[], answerKey: number[]): number {
    // Throw error if input and answers don't match up
    if (input.length !== answerKey.length) {
      throw new Error(`Input and answerKey must be of same length`);
    }

    const squaredErrors = input.reduce((acc, value, index) => {
      return acc + Math.pow(value - answerKey[index], 2);
    }, 0);

    return squaredErrors / input.length;
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
