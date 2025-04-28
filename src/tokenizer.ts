import { CorpusTexts } from './Corpus';
import { Vocab } from './Vocabulary';

export class Tokenizer {
  // Set up corpus and vocabulary
  private texts: CorpusTexts;
  private vocab: Vocab;

  constructor(texts: CorpusTexts, vocab: Vocab) {
    this.texts = JSON.parse(JSON.stringify(texts));
    this.vocab = new Map<string, number>(vocab);
  }

  // Merge token pairs in corpus
  // NOTE: This mutates in place for performance reasons
  mergeTokenPairInPlace(targetTokenPair: string) {
    // Iterate over all words in corpus
    for (let w = 0; w < this.texts.length; w++) {
      // Reference word directly to allow mutations
      let word = this.texts[w];
      // Sliding window across all letters in each word
      for (let i = 0; i < word.length; i++) {
        // If we're at the last character, move to the next word
        if (i + 1 >= word.length) continue;

        // If token pair matches target token pair
        if (`${word[i]}${word[i + 1]}` === targetTokenPair) {
          // Merge tokens into one
          word.splice(i, 2, targetTokenPair);
          // Skip the merged token
          i += 1;
        }
      }
    }
  }

  // Merge all token pairs
  mergeAllTokenPairs(vocabularySize: number): void {
    // Start count of all tokens
    let totalTokens = 0;

    // While we still have merges left, continue
    while (totalTokens < vocabularySize) {
      // Find most common pair
      const mostCommonPair = this.findMostCommonPair(this.texts);
      // Error handling to prevent infinite loop
      if (!mostCommonPair) break;
      // Merge that pair
      this.mergeTokenPairInPlace(mostCommonPair);
      // Add to vocabulary
      this.vocab.set(mostCommonPair, totalTokens);
      // Increment merged tokens
      totalTokens += 1;
      // Log progress every 100 tokens
      if (totalTokens % 100 === 0) {
        console.log(`Merged token ${totalTokens}/${vocabularySize}`);
      }
    }
  }

  // Create frequency map of token pairs
  buildFrequencyMap(): Map<string, number> {
    // Create new Map to store frequency of token pairs
    const frequencyMap = new Map<string, number>();

    // Iterate over all words in corpus
    for (const word of this.texts) {
      // Sliding window across all letters in each word
      for (let i = 0; i < word.length; i++) {
        // If we're at the last character, move to the next word
        if (i + 1 >= word.length) continue;

        // Otherwise form token pair from adjacent characters
        const tokenPair = `${word[i]}${word[i + 1]}`;

        // Add new entry to frequency map or increment existing
        frequencyMap.set(tokenPair, (frequencyMap.get(tokenPair) ?? 0) + 1);
      }
    }

    return frequencyMap;
  }

  // Find most common adjacent token pair
  findMostCommonPair(corpus: CorpusTexts): string {
    // Build frequency map from the corpus
    const frequencyMap = this.buildFrequencyMap();

    // Set max count and tokens
    let maxCount = -Infinity;
    let maxToken: string = '';

    // Iterate over each token pair in the frequency map
    for (const [token, count] of frequencyMap) {
      // If greater than previous max, set as new max
      if (count > maxCount) {
        maxCount = count;
        maxToken = token;
      }
    }

    return maxToken;
  }

  getVocabulary() {
    return this.vocab;
  }

  getMergedText() {
    return this.texts;
  }
}
