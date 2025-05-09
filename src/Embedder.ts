import { ModelState, NeuralNetwork } from './NeuralNetwork';

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

  // Constructor
  constructor(
    vocabularySize: number,
    vectorSize: number,
    trainingData: number[]
  ) {
    this.vocabularySize = vocabularySize;
    this.trainingData = trainingData;

    const neuralNetLayers = [
      this.vocabularySize,
      vectorSize,
      this.vocabularySize,
    ];
    this.neuralNet = new NeuralNetwork(neuralNetLayers);
  }

  // Train
  async train(
    epochs: number,
    learningRate: number,
    contextWindow: number
  ): Promise<ModelState> {
    const trainingData = await this.generateTrainingData(contextWindow);

    for (let i = 0; i < epochs; i++) {
      console.log(`\nEpoch #${i + 1}`);

      const totalBatches = Math.round(trainingData.length / 100);

      let batchIndex = 0;
      do {
        console.log(`Training batch ${batchIndex / 100 + 1}/${totalBatches}`);
        const vectorizedTrainingData = await this.vectorizeBatch(
          trainingData.slice(batchIndex, batchIndex + 100),
          this.vocabularySize
        );

        this.neuralNet.trainOnBatch(
          vectorizedTrainingData,
          epochs,
          learningRate
        );

        batchIndex += 100;
      } while (batchIndex < trainingData.length);
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
      const { input: vectorizedInput, target: vectorizedTarget } =
        this.vectorizeTrainingData(trainingBatch[i], vocabSize);

      resultInput.push(vectorizedInput);
      resultTarget.push(vectorizedTarget);
    }

    return { input: resultInput, target: resultTarget };
  }
}
