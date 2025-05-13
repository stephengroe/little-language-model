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
    // Multiply inputs by weights
    return inputs.reduce((total, input, index) => {
      return (total += input * this.weights[index]);
    });
  }

  getOutput(inputs: number[]): number {
    // Save inputs
    for (let i = 0; i < inputs.length; i++) {
      this.savedInputs[i] = inputs[i];
    }

    return this.getDotProduct(inputs) + this.bias;
  }

  calculateGradients(lossGradient: number): number[] {
    // Get derivative of activation function
    const nodeLossGradient = lossGradient * this.savedOutput;

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
