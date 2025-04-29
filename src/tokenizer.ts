import { CorpusTexts } from './Corpus';
import { Vocab } from './Vocabulary';

export class Tokenizer {
  // Set up corpus for merging
  private texts: CorpusTexts;
  // Set up vocabulary
  private vocab: Vocab;
  // Set up index (to store locations of pairs for merging)
  private pairIndex: Map<string, Set<number>>;
  // Set up frequency map
  private frequencyMap: Map<string, number>;

  constructor(texts: CorpusTexts, vocab: Vocab) {
    this.texts = JSON.parse(JSON.stringify(texts));
    this.vocab = new Map<string, number>(vocab);
    this.pairIndex = new Map<string, Set<number>>();
    this.frequencyMap = new Map<string, number>();

    console.log(`Indexing byte pairs...`);
    this.buildPairMapAndIndex();
  }

  // Merge token pairs in corpus
  mergeTokenPair(targetTokenPair: string) {
    // Get positions of all words with target token pair
    const targetWordIndexes = this.pairIndex.get(targetTokenPair);

    // Return if undefined or no indexed words
    if (!targetWordIndexes || targetWordIndexes.size === 0) return;

    // Iterate over target words
    for (const targetWordIndex of targetWordIndexes) {
      // Isolate word in corpus
      let targetWord = this.texts[targetWordIndex];

      // Sliding window across all letters in each word
      for (let i = 0; i < targetWord.length; i++) {
        // If we're at the last character, move to next word
        if (i + 1 >= targetWord.length) continue;

        // If token pair matches target token pair
        if (`${targetWord[i]}${targetWord[i + 1]}` === targetTokenPair) {
          // Merge tokens into one
          targetWord.splice(i, 2, targetTokenPair);
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
      this.mergeTokenPair(mostCommonPair);
      // Add to vocabulary
      this.vocab.set(mostCommonPair, totalTokens);
      // Remove from frequency map
      this.frequencyMap.delete(mostCommonPair);
      // Increment merged tokens
      totalTokens += 1;
      // Log progress every 100 tokens
      if (totalTokens % 100 === 0) {
        console.log(`Merged token ${totalTokens}/${vocabularySize}`);
      }
    }
  }

  // Create frequency map of token pairs
  buildPairMapAndIndex() {
    // Iterate over all words in corpus
    for (const [index, word] of this.texts.entries()) {
      // Sliding window across all letters in each word
      for (let i = 0; i < word.length; i++) {
        // If we're at the last character, move to the next word
        if (i + 1 >= word.length) continue;

        // Otherwise form token pair from adjacent characters
        const tokenPair = `${word[i]}${word[i + 1]}`;

        // Add to frequency map
        this.frequencyMap.set(
          tokenPair,
          (this.frequencyMap.get(tokenPair) ?? 0) + 1
        );

        // Add to index
        if (this.pairIndex.has(tokenPair)) {
          this.pairIndex.get(tokenPair)!.add(index);
        } else {
          this.pairIndex.set(tokenPair, new Set<number>([index]));
        }
      }
    }
  }

  // Find most common adjacent token pair
  findMostCommonPair(corpus: CorpusTexts): string {
    // Set max count and tokens
    let maxCount = -Infinity;
    let maxToken: string = '';

    // Iterate over each token pair in the frequency map
    for (const [token, count] of this.frequencyMap) {
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
