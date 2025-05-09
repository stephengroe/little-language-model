import { Node } from './Node';

export class Layer {
  private nodes: Node[] = [];

  constructor(layerSize: number, inputSize: number) {
    // Validate input size
    if (inputSize <= 1) {
      throw new Error(`Input size must be greater than one (got ${inputSize})`);
    }

    // Instantiate nodes for this layer with weights based on previous layer
    this.nodes = Array.from({ length: layerSize }, () => new Node(inputSize));
  }

  forward(input: number[]): number[] {
    return this.nodes.map((neuron) => {
      return neuron.getOutput(input);
    });
  }

  gradientDescent(lossGradient: number[], learningRate: number): number[] {
    let nextGradient: number[] = Array.from(
      { length: this.nodes[0].getWeights().length }, // Match length of inputs
      () => 0
    );

    // Update weights and bias at each node and return updated gradient
    for (let i = 0; i < this.nodes.length; i++) {
      nextGradient = this.nodes[i].gradientDescent(
        lossGradient[i],
        learningRate,
        nextGradient
      );
    }

    return nextGradient;
  }

  getNodes(): Node[] {
    return this.nodes;
  }

  getWeights(): number[][] {
    return this.nodes.map((node) => node.getWeights());
  }

  getBiases(): number[] {
    return this.nodes.map((node) => node.getBias());
  }
}
