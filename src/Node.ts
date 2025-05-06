export class Node {
  private weights: number[];
  private bias: number;
  private savedInputs: number[];

  constructor(inputSize: number) {
    // Validate input
    if (inputSize <= 1) {
      throw new Error(`Input length must be greater than 1 (got ${inputSize})`);
    }

    this.weights = Array.from({ length: inputSize }, () => Math.random() - 0.5);
    this.bias = 0;
    this.savedInputs = Array.from({ length: inputSize });
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
    // Save inputs
    for (let i = 0; i < inputs.length; i++) {
      this.savedInputs[i] = inputs[i];
    }

    const dotProduct = this.getDotProduct(inputs);
    return this.applyActivation(dotProduct + this.bias);
  }

  gradientDescent(
    lossGradient: number,
    learningRate: number,
    nextGradient: number[]
  ): number[] {
    // Update weights
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] -= learningRate * lossGradient * this.savedInputs[i];
    }

    // Update bias
    this.bias -= learningRate * lossGradient;

    // Update next gradient
    return nextGradient.map((gradient, index) => {
      return gradient + this.weights[index] * lossGradient;
    });
  }

  getWeights(): number[] {
    return this.weights;
  }

  getBias(): number {
    return this.bias;
  }
}
