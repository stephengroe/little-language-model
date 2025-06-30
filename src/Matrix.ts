import { matrixVectorMultiply, softmax, toOneHot } from './utils';

export class Matrix {
  private W1: number[][];
  private W2: number[][];
  private vocabSize: number;

  constructor(embeddingSize: number, vocabSize: number) {
    this.vocabSize = vocabSize;
    this.W1 = Array.from({ length: vocabSize }, () => {
      return Array.from({ length: embeddingSize }, () => Math.random() - 0.5);
    });
    this.W2 = Array.from({ length: embeddingSize }, () => {
      return Array.from({ length: vocabSize }, () => Math.random() - 0.5);
    });
  }

  forward(x: number[]): number[] {
    const hidden = matrixVectorMultiply(this.W1, x);
    const output = matrixVectorMultiply(this.W2, hidden);
    const prediction = softmax(output);

    return prediction;
  }

  train(x: number[], targetIndex: number, learningRate: number): number {
    const prediction = this.forward(x);
    const answer = toOneHot(targetIndex, this.vocabSize);
    const loss = this.loss(prediction, answer);

    this.backward(prediction, targetIndex, learningRate);

    return loss;
  }
}
