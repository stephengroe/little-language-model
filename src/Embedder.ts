import { Vocab } from './Vocabulary';

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

  // Get embeddings
  getEmbeddings(): Map<number, number[]> {
    return this.embeddings;
  }
}
