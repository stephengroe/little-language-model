import { CorpusTexts } from './Corpus';

export type Vocab = Map<string, number>;

export class Vocabulary {
  // Create vocabulary
  private vocab: Vocab;

  constructor() {
    this.vocab = new Map();
  }

  // Create list of all unique characters from corpus
  buildFromCorpus(corpus: CorpusTexts): void {
    // Set of unique tokens
    const uniqueTokens = new Set<string>();
    // Iterate over each word
    for (const word of corpus) {
      // Iterate over each token
      for (const token of word) {
        // Add to vocabulary
        // const count = this.vocab.get(token) || 0;
        // this.vocab.set(token, count + 1);
        uniqueTokens.add(token);
      }
    }

    // Convert set to map
    let index = 0;
    for (const token of uniqueTokens) {
      this.vocab.set(token, index);
      index++;
    }
  }

  getVocab() {
    return this.vocab;
  }
}
