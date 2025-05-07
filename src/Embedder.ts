import { Vocab } from './Vocabulary';

// Types
export type TrainingSet = {
  input: number[];
  target: number;
};

export class Embedder {
  // Create vocabulary
  private embeddings: Map<number, number[]>;

  // Constructor
  constructor(vocabulary: Vocab, dimensions: number) {
    this.embeddings = new Map<number, number[]>();
    this.initializeVectors(vocabulary, dimensions);
  }

  // Initialize embeddings
  initializeVectors(vocabulary: Vocab, dimensions: number) {
    // Iterate over vocabulary
    for (const [token, tokenId] of vocabulary) {
      // Create array of random numbers
      const vectorArray = Array.from({ length: dimensions }, () => {
        return Math.random() - 0.5; // Center around 0
      });
      // Set as new entry in embeddings Map
      this.embeddings.set(tokenId, vectorArray);
    }
  }

  // Build CBOW training groups
  generateTrainingData(corpus: number[], contextWindow: number): TrainingSet[] {
    const trainingData: TrainingSet[] = [];

    for (let i = contextWindow; i < corpus.length - contextWindow; i++) {
      const input = [];

      // Iterate over sequences for context window
      for (let j = -contextWindow; j <= contextWindow; j++) {
        if (j === 0) continue; // skip middle token
        input.push(corpus[i + j]);
      }

      trainingData.push({ input, target: corpus[i] });
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

  // Get embeddings
  getEmbeddings(): Map<number, number[]> {
    return this.embeddings;
  }
}
