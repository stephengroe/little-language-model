import { Node } from './Node';

export class Layer {
  private nodes: Node[] = [];

  constructor(layerSize: number, inputSize: number) {
    // Validate input size
    if (inputSize < 1) {
      throw new Error(`Input size must be greater than one (got ${inputSize})`);
    }

    // Instantiate nodes for this layer with weights based on previous layer
    this.nodes = Array.from({ length: layerSize }, () => new Node(inputSize));
  }

  forward(input: number[]): number[] {
    if (input.length !== this.nodes[0].getWeights().length) {
      throw new Error(`Input length does not match weights`);
    }

    return this.nodes.map((node, index) => {
      const output = node.getOutput(input);

      if (Number.isNaN(output)) {
        throw new Error(`Node #${index} output is NaN`);
      }

      return output;
    });
  }

  calculateGradients(lossGradient: number[]): number[] {
    let accumulatedGradient = Array(this.nodes[0].getWeights().length).fill(0);

    // Update weights and bias at each node
    for (let i = 0; i < this.nodes.length; i++) {
      const contrib = this.nodes[i].calculateGradients(lossGradient[i]);

      // Sum all changes
      for (let j = 0; j < accumulatedGradient.length; j++) {
        accumulatedGradient[j] += contrib[j];
      }
    }

    return accumulatedGradient;
  }

  applyGradients(learningRate: number, batchSize: number) {
    for (const node of this.nodes) {
      node.applyGradients(learningRate, batchSize);
    }
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

  createFromSavedModel(savedLayer: { weights: number[][]; biases: number[] }) {
    for (let i = 0; i < savedLayer.weights.length; i++) {
      this.nodes[i].setWeights(savedLayer.weights[i]);
      this.nodes[i].setBias(savedLayer.biases[i]);
    }
  }
}
