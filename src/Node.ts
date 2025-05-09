export class Node {
  private weights: number[];
  private bias: number;
  private savedInputs: number[];
  private savedOutput: number;

  constructor(inputSize: number) {
    // Validate input
    if (inputSize <= 1) {
      throw new Error(`Input length must be greater than 1 (got ${inputSize})`);
    }

    this.weights = Array.from({ length: inputSize }, () => Math.random() - 0.5);
    this.savedInputs = Array(inputSize).fill(0);
    this.bias = 0;
    this.savedOutput = 0;
  }

  getDotProduct(inputs: number[]): number {
    // Multiple inputs by weights, add bias, and sum to a single number
    return (
      inputs.reduce((total, input, index) => {
        return (total += input * this.weights[index]);
      }) + this.bias
    );
  }

  applyActivation(x: number): number {
    // Using ReLU
    return Math.max(0, x);
  }

  // ReLU
  activationDerivative(x: number): number {
    return x > 0 ? 1 : 0;
  }

  getOutput(inputs: number[]): number {
    // Save inputs
    for (let i = 0; i < inputs.length; i++) {
      this.savedInputs[i] = inputs[i];
    }

    const dotProduct = this.getDotProduct(inputs);
    const output = this.applyActivation(dotProduct);

    this.savedOutput = output;
    return output;
  }

  gradientDescent(lossGradient: number, learningRate: number): number[] {
    // Get derivative of activation function
    const nodeLossGradient =
      lossGradient * this.activationDerivative(this.savedOutput);

    // Update weights
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] -= learningRate * nodeLossGradient * this.savedInputs[i];
    }

    // Update bias
    this.bias -= learningRate * nodeLossGradient;

    // Update next gradient
    return this.weights.map((weight) => {
      return weight * nodeLossGradient;
    });
  }

  getWeights(): number[] {
    return this.weights;
  }

  getBias(): number {
    return this.bias;
  }
}
