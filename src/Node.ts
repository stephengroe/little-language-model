export class Node {
  private weights: number[];
  private bias: number;

  constructor(inputSize: number) {
    // Validate input
    if (inputSize <= 1) {
      throw new Error(`Input length must be greater than 1 (got ${inputSize})`);
    }

    this.weights = Array.from({ length: inputSize }, () => Math.random() - 0.5);
    this.bias = 0;
  }

  getDotProduct(inputs: number[]): number {
    // Multiple inputs by weights and sum to a single number
    return inputs.reduce((total, input, index) => {
      return (total += input * this.weights[index]);
    });
  }

  applyActivation(x: number): number {
    // Using ReLU
    return Math.max(0, x);
  }

  getOutput(inputs: number[]): number {
    const dotProduct = this.getDotProduct(inputs);
    return this.applyActivation(dotProduct + this.bias);
  }

  getWeights(): number[] {
    return this.weights;
  }

  getBias(): number {
    return this.bias;
  }
}
