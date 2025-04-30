import { Vocab } from './Vocabulary';

export class Embedder {
  // Create vocabulary
  private embeddings: Map<string, number[]>;

  // Constructor
  constructor(vocabulary: Vocab, dimensions: number) {
    this.embeddings = new Map<string, number[]>();
    this.initializeEmbeddings(vocabulary, dimensions);
  }

  // Initialize embeddings
  initializeEmbeddings(vocabulary: Vocab, dimensions: number) {
    // Iterate over vocabulary
    for (const [token, tokenId] of vocabulary) {
      // Create array of random numbers
      const embeddingArray = Array.from({ length: dimensions }, () => {
        return Math.random();
      });
      // Set as new entry in embeddings Map
      this.embeddings.set(token, embeddingArray);
    }
  }

  // Get embeddings
  getEmbeddings(): Map<string, number[]> {
    return this.embeddings;
  }
}
