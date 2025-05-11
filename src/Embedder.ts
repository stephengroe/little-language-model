import { ModelState, NeuralNetwork } from './NeuralNetwork';
import { shuffleArray } from './utils';

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

  // Train
  async train(
    batchSize: number,
    contextWindow: number,
    learningRate: number,
    epochs: number
  ): Promise<ModelState> {
    const trainingData = await this.generateTrainingData(contextWindow);

    for (let i = 0; i < epochs; i++) {
      console.log(`\nEpoch #${i + 1}`);

      // Shuffle data for new epoch
      const shuffledData = trainingData; //shuffleArray(trainingData);

      const totalBatches = Math.round(shuffledData.length / batchSize);
      let batchIndex = 0;
      while (batchIndex < shuffledData.length) {
        console.log(
          `Training batch ${Math.floor(batchIndex / batchSize) + 1}/${totalBatches}`
        );
        const vectorizedTrainingData = await this.vectorizeBatch(
          shuffledData.slice(batchIndex, batchIndex + batchSize),
          this.vocabularySize
        );

        this.neuralNet.trainOnBatch(vectorizedTrainingData, learningRate);

        batchIndex += batchSize;
      }
    }

    return await this.neuralNet.getModelState();
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

  async buildEmbeddings() {
    // Clear any existing embeddings
    this.embeddings = {};
    const embeddings: Record<number, number[]> = {};

    for (let i = 0; i < this.vocabularySize; i++) {
      const vocabOneHot = this.oneHot(i, this.vocabularySize);
      const vector = this.neuralNet.forwardToLayer(vocabOneHot, 0);

      embeddings[i] = vector;
    }

    this.embeddings = embeddings;
  }

  getEmbeddings(): Record<number, number[]> {
    return this.embeddings;
  }
}
