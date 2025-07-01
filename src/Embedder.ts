import {
  shuffleArray,
  getCosineSimilarity,
  round,
  norm,
  ProgressBar,
  toOneHot,
  saveFile,
} from './utils';
import { TrainingData, TrainingBatch, DataLoader } from './DataLoader';
import { Matrix } from './Matrix';

// For CBOW
export type CBOWPair = {
  input: number[];
  target: number;
};

export class Embedder {
  // Create vocabulary
  private vocabularySize: number;
  private vectorSize: number;
  private trainingData: number[];
  private embeddings: Float32Array;
  private matrix: Matrix;

  // Constructor
  constructor(
    vocabularySize: number,
    vectorSize: number,
    trainingData: number[]
  ) {
    this.vocabularySize = vocabularySize;
    this.vectorSize = vectorSize;
    this.trainingData = trainingData;
    this.embeddings = new Float32Array(vocabularySize * vectorSize);
    this.matrix = new Matrix(vectorSize, vocabularySize);
  }

  // Train
  train(
    batchSize: number,
    contextWindow: number,
    learningRate: number,
    epochs: number,
    logInterval: number = 0.1 // default of 10%
  ) {
    const trainingExamples = this.generateCBOWPairs(contextWindow);
    const dataLoader = new DataLoader(trainingExamples);

    // Epochs
    for (let i = 0; i < epochs; i++) {
      // Linear decay
      const adjustedLearningRate = learningRate * (1 - i / epochs);

      console.log(`\nEpoch #${i + 1}`);
      const batches = dataLoader.batch(batchSize, true);
      const progress = new ProgressBar(batches.length);
      const progressInterval = Math.floor(batches.length * logInterval);

      const epochLoss: number[] = [];

      for (let j = 0; j < batches.length; j++) {
        let { batchInputs, batchTargets } = batches[j];
        const oneHotInput = batchInputs.map((row) =>
          toOneHot(row, this.vocabularySize)
        );

        const loss = this.matrix.train(
          oneHotInput,
          batchTargets,
          adjustedLearningRate
        );
        epochLoss.push(loss);

        if (j % progressInterval === 0 || j === batches.length) {
          progress.update(j, `Loss: ${round(loss, 5)}`);
        }
      }

      const avgLoss =
        epochLoss.reduce((acc, cur) => (acc += cur), 0) / epochLoss.length;
      console.log(`  Epoch av. loss: ${round(avgLoss, 4)}`);
    }
  }

  generateCBOWPairs(contextWindow: number): CBOWPair[] {
    const trainingData: CBOWPair[] = [];

    for (
      let i = contextWindow;
      i < this.trainingData.length - contextWindow;
      i++
    ) {
      const input = [];

      // Iterate over sequences for context window
      for (let j = -contextWindow; j <= contextWindow; j++) {
        if (j === 0) continue; // skip middle token
        input.push(this.trainingData[i + j]);
      }

      trainingData.push({ input, target: this.trainingData[i] });
    }

    return trainingData;
  }

  buildEmbeddings(logInterval: number = 0.01) {
    const progress = new ProgressBar(this.vocabularySize);
    const progressInterval = this.vocabularySize * logInterval;
    // Clear any existing embeddings
    this.embeddings = new Float32Array(this.vocabularySize * this.vectorSize);
    for (let i = 0; i < this.vocabularySize; i++) {
      const vector = this.matrix.getEmbedding(i);

      for (let j = 0; j < this.vectorSize; j++) {
        this.embeddings[i * this.vectorSize + j] = vector[j];
      }

      if (i % progressInterval === 0 || i === this.vocabularySize - 1) {
        progress.update(i);
      }
    }
  }

  getEmbeddings(): Float32Array {
    return this.embeddings;
  }

  getEmbedding(token: number): Float32Array {
    if (!this.embeddings[token * this.vectorSize]) {
      throw new Error(`Token does not exist as embedding (received ${token})`);
    }

    const tokenIndex = token * this.vectorSize;

    return this.embeddings.slice(token, token + this.vectorSize);
  }

  findNearest(
    targetToken: number,
    neighbors: number = 5
  ): { token: number; distance: number }[] {
    if (!this.embeddings[targetToken * this.vectorSize]) {
      throw new Error(`Invalid token (received ${targetToken})`);
    }

    const convertedToken = targetToken * this.vectorSize;
    const targetVector = Array.from(
      this.embeddings.subarray(convertedToken, convertedToken + this.vectorSize)
    );

    // Naive solution, computing all values then sorting
    let similarities: { token: number; distance: number }[] = [];
    for (let i = 0; i < this.vocabularySize; i++) {
      const vector = Array.from(
        this.embeddings.subarray(
          i * this.vectorSize,
          i * this.vectorSize + this.vectorSize
        )
      );

      const similarity: number = getCosineSimilarity(targetVector, vector);
      similarities[i] = { token: i, distance: similarity };
    }

    similarities = similarities.sort((a, b) => b.distance - a.distance);

    return similarities.slice(0, neighbors);
  }
}
