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

  predict(input: number[]): number[] {
    let result = input.slice();

    for (const layer of this.layers) {
      result = layer.forward(result);
    }

    return this.applySoftMax(result);
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

    return adjustedInput.map((expNum) => expNum / denominator);
  }
}
