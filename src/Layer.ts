import { Node } from './Node';

export class Layer {
  private neurons: Node[] = [];

  constructor(inputSize: number) {
    // Validate input size
    if (inputSize <= 1) {
      throw new Error(`Input size must be greater than one (got ${inputSize})`);
    }

    this.neurons = Array.from({ length: inputSize }, () => new Node(inputSize));
  }

  forward(input: number[]): number[] {
    return this.neurons.map((neuron) => {
      return neuron.getOutput(input);
    });
  }

  getNeurons(): Node[] {
    return this.neurons;
  }
}
