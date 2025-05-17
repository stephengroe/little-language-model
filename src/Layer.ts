import { Node } from './Node';
import { ActivationFunction } from './ActivationFunction/ActivationFunction';

export class Layer {
  private nodes: Node[];

  constructor(
    layerDepth: number,
    inputSize: number,
    activationFn: ActivationFunction
  ) {
    if (inputSize < 1) {
      throw new Error(`Input size must be one or more (got ${inputSize})`);
    }

    this.nodes = [];

    for (let i = 0; i < layerDepth; i++) {
      const node = new Node(inputSize, activationFn);
      this.nodes.push(node);
    }
  }

  forward(input: number[]): number[] {
    if (input.length !== this.nodes[0].getWeights().length) {
      throw new Error(`Input length does not match weights`);
    }

    return this.nodes.map((node) => node.forward(input));
  }

  backward(lossGradient: number[], learningRate: number): number[] {
    let accumulatedGradient = Array(this.nodes[0].getWeights().length).fill(0);

    for (let i = 0; i < this.nodes.length; i++) {
      const contrib = this.nodes[i].backward(lossGradient[i], learningRate);

      for (let j = 0; j < accumulatedGradient.length; j++) {
        accumulatedGradient[j] += contrib[j];
      }
    }

    return accumulatedGradient;
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
