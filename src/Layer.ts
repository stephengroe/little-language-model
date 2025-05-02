import { Neuron } from './Neuron';

export class Layer {
  private neurons: Neuron[] = [];

  constructor(inputSize: number) {
    // Validate input size
    if (inputSize <= 1) {
      throw new Error(`Input size must be greater than one (got ${inputSize})`);
    }

    this.neurons = Array.from(
      { length: inputSize },
      () => new Neuron(inputSize)
    );
  }

  forward(input: number[]): number[] {
    return this.neurons.map((neuron) => {
      return neuron.getOutput(input);
    });
  }

  getNeurons(): Neuron[] {
    return this.neurons;
  }
}
