import { Layer } from './Layer';

export class NeuralNetwork {
  private layers: Layer[];

  constructor(layerSize: number, inputSize: number) {
    this.layers = Array.from({ length: layerSize }, () => new Layer(inputSize));
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

    return input.map((weight) => Math.exp((weight - max) / safeTemp));
  }
}
