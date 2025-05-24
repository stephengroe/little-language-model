import { ModelState, NeuralNetwork } from './NeuralNetwork';
import {
  shuffleArray,
  getCosineSimilarity,
  round,
  norm,
  ProgressBar,
  toOneHot,
} from './utils';
import { saveFile } from './utils';
import { TrainingData, TrainingBatch, DataLoader } from './DataLoader';

// For CBOW
export type CBOWPair = {
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

  // Train
  train(
    batchSize: number,
    contextWindow: number,
    learningRate: number,
    epochs: number,
    logInterval: number = 0.1 // default of 10%
  ): ModelState {
    const trainingExamples = this.generateCBOWPairs(contextWindow);

    const dataLoader = new DataLoader(trainingExamples);

    // Epochs
    for (let i = 0; i < epochs; i++) {
      console.log(`\nEpoch #${i + 1}`);
      const batches = dataLoader.batch(batchSize, true);
      const progress = new ProgressBar(batches.length);
      const progressInterval = Math.floor(batches.length * logInterval);

      const epochLoss: number[] = [];

      for (let j = 0; j < batches.length; j++) {
        let { batchInputs, batchTargets } = batches[j];
        const oneHotInputs = batchInputs.map((input) => {
          return toOneHot(input, this.vocabularySize);
        });
        const oneHotTargets = batchTargets.map((target) =>
          toOneHot(target, this.vocabularySize)
        );

        const loss = this.neuralNet.train(
          oneHotInputs,
          oneHotTargets,
          learningRate
        );
        epochLoss.push(loss);

        if (j % progressInterval === 0 || j === batches.length) {
          progress.update(j);
        }
      }

      const avgLoss =
        epochLoss.reduce((acc, cur) => (acc += cur), 0) / epochLoss.length;
      console.log(`  Av. loss: ${round(avgLoss, 4)}`);
    }

    return this.neuralNet.getModelState();
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

  buildEmbeddings() {
    // Clear any existing embeddings
    this.embeddings = {};
    const embeddings: Record<number, number[]> = {};
    const lastLayer = this.neuralNet.getLayers().length - 1;

    for (let i = 0; i < this.vocabularySize; i++) {
      const vocabOneHot = toOneHot(i, this.vocabularySize);
      const vector = this.neuralNet.forwardToLayer(vocabOneHot, lastLayer);

      embeddings[i] = vector;
    }

    this.embeddings = embeddings;
  }

  getEmbeddings(): Record<number, number[]> {
    return this.embeddings;
  }

  getEmbeddingsMatrix(): Float32Array {
    const ids = Object.keys(this.embeddings)
      .map(Number)
      .sort((a, b) => a - b);
    const embeddingSize = this.embeddings[ids[0]].length;
    const flatArray = new Float32Array(ids.length * embeddingSize);

    ids.forEach((id, i) => {
      const vector = this.embeddings[id];
      for (let j = 0; j < embeddingSize; j++) {
        flatArray[i * embeddingSize + j] = vector[j];
      }
    });

    return flatArray;
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
