import { ActivationFunction } from './ActivationFunction/ActivationFunction';

export class Node {
  private weights: number[];
  private bias: number;
  private inputs: number[];
  private z: number;
  private activation: ActivationFunction;

  constructor(
    inputSize: number,
    activation: ActivationFunction,
    initScale: number = 1
  ) {
    if (inputSize < 1) {
      throw new Error(`Input length must one or more (got ${inputSize})`);
    }

    this.weights = Array.from(
      { length: inputSize },
      () => (Math.random() - 0.5) * initScale
    );
    this.bias = 0;
    this.z = 0;
    this.activation = activation;
    this.inputs = [];
  }

  forward(input: number[]): number {
    this.inputs = input.slice();
    const dot = input.reduce((acc, cur, index) => {
      return acc + cur * this.weights[index];
    }, 0);
    const z = dot + this.bias;
    this.z = z;
    const result = this.activation.apply(z);

    if (Number.isNaN(result)) {
      throw new Error(`Output of node is NaN
        input: ${input}
        weights: ${this.weights}
        dot: ${dot}
        z: ${z}`);
    }

    return result;
  }

  backward(lossGradient: number, learningRate: number): number[] {
    const delta = lossGradient * this.activation.derivative(this.z);

    for (let i = 0; i < this.weights.length; i++) {
      const gradient = delta * this.inputs[i];
      this.weights[i] -= learningRate * gradient;
    }

    this.bias -= learningRate * delta;

    return this.weights.map((weight) => weight * delta);
  }

  getWeights(): number[] {
    return this.weights;
  }

  getBias(): number {
    return this.bias;
  }

  setBias(newBias: number) {
    this.bias = newBias;
  }

  setWeights(newWeights: number[]) {
    this.weights = newWeights;
  }
}
