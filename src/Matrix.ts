import { softmax, toOneHot, transpose, matrixMultiply } from './utils';

export class Matrix {
  private weights1: number[][];
  private weights2: number[][];
  private vocabSize: number;
  private input: number[][];
  private hidden: number[][];
  private output: number[][];

  constructor(embeddingSize: number, vocabSize: number) {
    this.vocabSize = vocabSize;
    this.input = [];
    this.hidden = [];
    this.output = [];
    this.weights1 = Array.from({ length: vocabSize }, () => {
      return Array.from({ length: embeddingSize }, () => Math.random() - 0.5);
    });
    this.weights2 = Array.from({ length: embeddingSize }, () => {
      return Array.from({ length: vocabSize }, () => Math.random() - 0.5);
    });
  }

  forward(x: number[][]): number[][] {
    this.input = x;
    this.hidden = matrixMultiply(x, this.weights1);
    this.output = matrixMultiply(this.hidden, this.weights2);
    const prediction = this.output.map((row) => softmax(row));

    return prediction;
  }

  loss(predicted: number[][], target: number[][]): number {
    const epsilon = 1e-15; // to prevent log(0) error
    const totalLoss = predicted.reduce((sum, row, i) => {
      const targetIndex = target[i].findIndex((n) => n === 1);
      const prob = Math.max(predicted[i][targetIndex], epsilon);
      return sum + -Math.log(prob);
    }, 0);

    return totalLoss / predicted.length;
  }

  backward(prediction: number[][], target: number[][], learningRate: number) {
    const dOutput = prediction.map((row, i) => {
      return row.map((val, j) => val - target[i][j]);
    });

    const weights2_grad = matrixMultiply(transpose(this.hidden), dOutput);
    this.weights2 = this.applyGradient(
      this.weights2,
      weights2_grad,
      learningRate
    );

    const dHidden = matrixMultiply(dOutput, transpose(this.weights2));

    const weights1_grad = matrixMultiply(transpose(dHidden), this.input);
    this.weights1 = this.applyGradient(
      this.weights1,
      transpose(weights1_grad),
      learningRate
    );
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

  train(
    batchTraining: number[][],
    batchTargets: number[],
    learningRate: number
  ): number {
    const prediction = this.forward(batchTraining);
    const answer = batchTargets.map((i) => toOneHot(i, this.vocabSize));
    const loss = this.loss(prediction, answer);

    this.backward(prediction, answer, learningRate);

    return loss;
  }

  getEmbedding(index: number): number[] {
    return this.weights1[index];
  }
}
