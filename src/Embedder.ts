import { trainingConfig } from './config';
import { ModelState, NeuralNetwork } from './NeuralNetwork';
import { shuffleArray, getCosineSimilarity, round } from './utils';
import { saveFile } from './utils';

// Types
export type TrainingSet = {
  input: number[];
  target: number;
};

export class Embedder {
  // Create vocabulary
  private vocabularySize: number;
  private neuralNet: NeuralNetwork;
  private trainingData: number[];
  private embeddings: Record<number, number[]>;

  // Constructor
  constructor(
    vocabularySize: number,
    vectorSize: number,
    trainingData: number[]
  ) {
    this.vocabularySize = vocabularySize;
    this.trainingData = trainingData;
    this.embeddings = {};

    const neuralNetLayers = [
      this.vocabularySize,
      vectorSize,
      this.vocabularySize,
    ];
    this.neuralNet = new NeuralNetwork(neuralNetLayers);
  }

  async saveNeuralNet() {
    await saveFile(
      './',
      'initial-neural-net.json',
      JSON.stringify(this.neuralNet.getModelState(), null, 2)
    );
  }

  buildFromSavedModel(savedModel: ModelState, vocabularySize: number) {
    const neuralNetLayers = savedModel.layers.map((layer) => {
      return layer.weights.length;
    });
    // Add input layer
    neuralNetLayers.unshift(vocabularySize);
    this.neuralNet = new NeuralNetwork(neuralNetLayers);

    this.neuralNet.buildFromSavedModel(savedModel);
  }

  // Train
  train(
    batchSize: number,
    contextWindow: number,
    learningRate: number,
    epochs: number
  ): ModelState {
    const trainingData = this.generateTrainingData(contextWindow);

    // Epochs
    for (let i = 0; i < epochs; i++) {
      console.log(`\nEpoch #${i + 1}`);
      const epochLoss: number[] = [];

      const shuffledData = shuffleArray(trainingData);
      const tenPercent = Math.ceil(shuffledData.length / 10);

      // Batches
      for (let j = 0; j < shuffledData.length; j += batchSize) {
        const dataSet = shuffledData.slice(j, j + batchSize);
        const vectorizedData = dataSet.map((data) =>
          this.vectorizeTrainingData(data, this.vocabularySize)
        );

        // Datasets
        for (let k = 0; k < vectorizedData.length; k++) {
          const { input, target } = vectorizedData[k];
          const loss = this.neuralNet.train(input, target, learningRate);
          epochLoss.push(loss);
        }

        if (j % tenPercent === 0) {
          const avgLoss =
            epochLoss.reduce((acc, cur) => (acc += cur), 0) / epochLoss.length;
          console.log(
            ` Trained ${round(j / tenPercent)}% | Av. loss: ${round(avgLoss)}`
          );
        }
      }
    }

    return this.neuralNet.getModelState();
  }

  // Build CBOW training groups
  generateTrainingData(contextWindow: number): TrainingSet[] {
    const trainingData: TrainingSet[] = [];

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

  // Transform training data into one-hot
  oneHot(index: number, vocabSize: number): number[] {
    const oneHot = Array.from({ length: vocabSize }, () => 0);
    oneHot[index] = 1;
    return oneHot;
  }

  vectorizeTrainingData(
    trainingData: TrainingSet,
    vocabSize: number
  ): { input: number[]; target: number[] } {
    const { input, target } = trainingData;
    const inputVector = Array.from({ length: vocabSize }, () => 0);
    const targetVector = this.oneHot(target, vocabSize);

    for (let i = 0; i < input.length; i++) {
      inputVector[input[i]] += 1 / input.length;
    }

    return { input: inputVector, target: targetVector };
  }

  vectorizeBatch(
    trainingBatch: TrainingSet[],
    vocabSize: number
  ): { input: number[][]; target: number[][] } {
    let resultInput: number[][] = [];
    let resultTarget: number[][] = [];

    for (let i = 0; i < trainingBatch.length; i++) {
      const item = trainingBatch[i];
      if (
        item.input === undefined ||
        item.target === undefined ||
        item === undefined
      ) {
        console.warn(`Skipping undefined input at index ${i}:`, item);
        continue;
      }

      const { input: vectorizedInput, target: vectorizedTarget } =
        this.vectorizeTrainingData(trainingBatch[i], vocabSize);

      resultInput.push(vectorizedInput);
      resultTarget.push(vectorizedTarget);
    }

    return { input: resultInput, target: resultTarget };
  }

  buildEmbeddings() {
    // Clear any existing embeddings
    this.embeddings = {};
    const embeddings: Record<number, number[]> = {};
    const lastLayer = this.neuralNet.getLayers().length - 1;

    for (let i = 0; i < this.vocabularySize; i++) {
      const vocabOneHot = this.oneHot(i, this.vocabularySize);
      const vector = this.neuralNet.forwardToLayer(vocabOneHot, lastLayer);

      embeddings[i] = vector;
    }

    this.embeddings = embeddings;
  }

  getEmbeddings(): Record<number, number[]> {
    return this.embeddings;
  }

  getEmbedding(token: number): number[] {
    if (!this.embeddings[token]) {
      throw new Error(`Token does not exist as embedding (received ${token})`);
    }

    return this.embeddings[token];
  }

  findNearest(targetToken: number, neighbors: number = 3): number[] {
    if (!this.embeddings[targetToken]) {
      throw new Error(`Invalid token (received ${targetToken})`);
    }

    const targetVector = this.embeddings[targetToken];

    // Naive solution, computing all values then sorting
    const similarities: [string, number][] = Object.entries(
      this.embeddings
    ).map(([token, vector]) => {
      const similarity: number = getCosineSimilarity(targetVector, vector);
      return [token, similarity];
    });

    const nearesetNeighbors: number[] = similarities
      .sort((a, b) => b[1] - a[1])
      .map((val) => Number(val[0]));

    return nearesetNeighbors.slice(0, neighbors);
  }
}
