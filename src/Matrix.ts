import {
  vectorMatrixMultiply,
  softmax,
  toOneHot,
  outerProduct,
  transpose,
} from './utils';

export class Matrix {
  private W1: number[][];
  private W2: number[][];
  private vocabSize: number;
  private input: number[];
  private hidden: number[];
  private output: number[];

  constructor(embeddingSize: number, vocabSize: number) {
    this.vocabSize = vocabSize;
    this.input = [];
    this.hidden = [];
    this.output = [];
    this.W1 = Array.from({ length: vocabSize }, () => {
      return Array.from({ length: embeddingSize }, () => Math.random() - 0.5);
    });
    this.W2 = Array.from({ length: embeddingSize }, () => {
      return Array.from({ length: vocabSize }, () => Math.random() - 0.5);
    });
  }

  forward(x: number[]): number[] {
    this.input = x;
    this.hidden = vectorMatrixMultiply(x, this.W1);
    this.output = vectorMatrixMultiply(this.hidden, this.W2);
    const prediction = softmax(this.output);

    return prediction;
  }

  train(x: number[], targetIndex: number, learningRate: number): number {
    const prediction = this.forward(x);
    const answer = toOneHot(targetIndex, this.vocabSize);
    const loss = this.loss(prediction, answer);

    this.backward(prediction, targetIndex, learningRate);

    return loss;
  }

  loss(predicted: number[], target: number[]): number {
    const targetIndex = target.findIndex((n) => n === 1);
    const epsilon = 1e-15; // to prevent log(0) error
    const prob = Math.max(predicted[targetIndex], epsilon);
    return -Math.log(prob);
  }

  backward(prediction: number[], targetIndex: number, learningRate: number) {
    const target = toOneHot(targetIndex, this.vocabSize);
    const dOutput = prediction.map((p, i) => p - target[i]);

    const W2_grad = outerProduct(this.hidden, dOutput);
    this.W2 = this.applyGradient(this.W2, W2_grad, learningRate);

    const dHidden = vectorMatrixMultiply(dOutput, transpose(this.W2));

    const W1_grad = outerProduct(dHidden, this.input);
    this.W1 = this.applyGradient(this.W1, transpose(W1_grad), learningRate);
  }

  applyGradient(
    W: number[][],
    gradient: number[][],
    learningRate: number
  ): number[][] {
    return W.map((row, i) => {
      return row.map((val, j) => val - learningRate * gradient[i][j]);
    });
  }

  getEmbedding(index: number): number[] {
    return this.W1[index];
  }
}
