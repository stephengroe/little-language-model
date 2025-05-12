export class Node {
  private weights: number[];
  private weightGradients: number[];
  private bias: number;
  private biasGradient: number;
  private savedInputs: number[];
  private savedOutput: number;

  constructor(inputSize: number) {
    // Validate input
    if (inputSize <= 1) {
      throw new Error(`Input length must be greater than 1 (got ${inputSize})`);
    }

    this.weights = Array.from({ length: inputSize }, () => {
      // He uniform initialization
      return (
        Math.random() * 2 * Math.sqrt(2 / inputSize) - Math.sqrt(2 / inputSize)
      );
    });
    this.weightGradients = Array(inputSize).fill(0);
    this.bias = Math.random() * 0.2 - 0.1; // In range [-0.1, +0.1]
    this.biasGradient = 0;
    this.savedInputs = Array(inputSize).fill(0);
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

  calculateGradients(lossGradient: number): number[] {
    // Get derivative of activation function
    const nodeLossGradient =
      lossGradient * this.activationDerivative(this.savedOutput);

    // Store updated weights
    for (let i = 0; i < this.weightGradients.length; i++) {
      this.weightGradients[i] += nodeLossGradient * this.savedInputs[i];
    }

    // Store updated bias
    this.biasGradient += nodeLossGradient;

    // Return new gradient
    return this.weights.map((weight, index) => weight * lossGradient);
  }

  applyGradients(learningRate: number, batchSize: number) {
    for (let i = 0; i < this.weights.length; i++) {
      const avgGradient = this.weightGradients[i] / batchSize;
      this.weights[i] -= avgGradient * learningRate;
      this.weightGradients[i] = 0; // reset for next batch
    }

    const avgBiasGradient = this.biasGradient / batchSize;
    this.bias -= avgBiasGradient * learningRate;
    this.biasGradient = 0;
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
