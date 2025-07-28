import { ProgressBar } from './utils';

export class Tokenizer {
  private wordToToken: Map<string, number>;
  private tokenToWord: Map<number, string>;
  private texts: string[][];
  private pairIndex: Map<string, Set<number>>;
  private frequencyMap: Map<string, number>;
  private tokenizedCorpus: number[];
  private tokenCounts: Map<number, number>;

  constructor(texts: string[][]) {
    this.texts = JSON.parse(JSON.stringify(texts));
    this.pairIndex = new Map<string, Set<number>>();
    this.frequencyMap = new Map<string, number>();
    this.tokenizedCorpus = [];
    this.tokenCounts = new Map<number, number>();

    [this.wordToToken, this.tokenToWord] =
      this.buildVocabularyFromCorpus(texts);
    this.indexAllWordTokens();
  }

  buildVocabularyFromCorpus(
    corpus: string[][]
  ): [Map<string, number>, Map<number, string>] {
    const uniqueTokens = new Set<string>();

    for (const word of corpus) {
      for (const token of word) {
        uniqueTokens.add(token);
      }
    }

    const wordToToken = new Map<string, number>();
    const tokenToWord = new Map<number, string>();

    Array.from(uniqueTokens).forEach((word, index) => {
      wordToToken.set(word, index);
      tokenToWord.set(index, word);
    });

    return [wordToToken, tokenToWord];
  }

  // Merge token pairs in a single word
  mergeWordTokenPair(targetTokenPair: string, targetWordIndex: number) {
    let targetWord = this.texts[targetWordIndex];
    let mergeIndexes: number[] = [];

    // Sliding window across all token pairs
    for (let i = 0; i < targetWord.length; i++) {
      // Skip if we're at the last character
      if (i + 1 >= targetWord.length) continue;
      const currentTokenPair = `${targetWord[i]}${targetWord[i + 1]}`;

      // Update frequency map (we'll add later if relevant)
      const frequency = this.frequencyMap.get(currentTokenPair) || 1;
      if (frequency === 1) {
        this.frequencyMap.delete(currentTokenPair);
      } else {
        this.frequencyMap.set(currentTokenPair, frequency - 1);
      }

      this.pairIndex.get(targetTokenPair)?.delete(targetWordIndex);

      // If we've found a pair to merge
      if (currentTokenPair === targetTokenPair) {
        mergeIndexes.push(i);
      }
    }

    let mergeOffset = 0; // Adjust index for words with multiple merges
    for (const mergeIndex of mergeIndexes) {
      targetWord.splice(mergeIndex - mergeOffset, 2, targetTokenPair);
      mergeOffset += 1;
    }

    // Recompute the tokens in this word
    this.indexWordTokens(targetWordIndex);
  }

  // Merge token pairs in corpus
  mergeTokenPair(targetTokenPair: string) {
    const targetWordIndexes = this.pairIndex.get(targetTokenPair);
    if (!targetWordIndexes || targetWordIndexes.size === 0) return;
    for (const targetWordIndex of targetWordIndexes) {
      this.mergeWordTokenPair(targetTokenPair, targetWordIndex);
    }
  }

  // Merge all token pairs
  mergeAllTokenPairs(vocabularySize: number, logInterval: number = 0.01): void {
    const progress = new ProgressBar(vocabularySize);
    let totalTokens = this.wordToToken.size;
    const progressInterval = Math.floor(totalTokens * logInterval);

    while (totalTokens < vocabularySize) {
      const [mostCommonPair, tokenCount] = this.findMostCommonTokenPair();
      if (!mostCommonPair) break;

      this.tokenCounts.set(totalTokens, tokenCount);
      this.mergeTokenPair(mostCommonPair);

      this.wordToToken.set(mostCommonPair, totalTokens);
      this.tokenToWord.set(totalTokens, mostCommonPair);

      totalTokens += 1;

      if (totalTokens % progressInterval === 0) {
        progress.update(totalTokens);
      }
    }
  }

  // Index tokens from a single word
  indexWordTokens(wordIndex: number) {
    // Get the word from the corpus
    const word = this.texts[wordIndex];

    // Iterate over each token in the word
    for (let i = 0; i < word.length; i++) {
      // If we're at the last character, return
      if (i + 1 >= word.length) return;

      // Otherwise form token pair from adjacent characters
      const tokenPair = `${word[i]}${word[i + 1]}`;

      // Add to frequency map
      this.frequencyMap.set(
        tokenPair,
        (this.frequencyMap.get(tokenPair) ?? 0) + 1
      );

      // Add to index
      if (this.pairIndex.has(tokenPair)) {
        this.pairIndex.get(tokenPair)!.add(wordIndex);
      } else {
        this.pairIndex.set(tokenPair, new Set<number>([wordIndex]));
      }
    }
  }

  // Index tokens from all words
  indexAllWordTokens() {
    // Iterate over all words in corpus
    for (let i = 0; i < this.texts.length; i++) {
      // Index tokens for that word
      this.indexWordTokens(i);
    }
  }

  // Find most common adjacent token pair
  findMostCommonTokenPair(): [string, number] {
    // Set max count and tokens
    let maxCount = -Infinity;
    let mostCommonToken: string = '';

    // Iterate over each token pair in the frequency map
    for (const [token, count] of this.frequencyMap) {
      // If greater than previous max, set as new max
      if (count > maxCount) {
        maxCount = count;
        mostCommonToken = token;
      }
    }

    return [mostCommonToken, maxCount];
  }

  // Tokenize corpus
  tokenizeCorpus(applySubsampling?: boolean) {
    this.tokenizedCorpus = this.texts.flatMap((word: string[]) => {
      return word.map((token) => {
        const foundToken = this.wordToToken.get(token) ?? -1;

        // Error handling for tokens not found in vocabulary
        if (foundToken < 0) {
          throw new Error(`Token '${token}' not recognized`);
        } else {
          return foundToken;
        }
      });
    });

    if (applySubsampling) {
      this.applySubsampling();
    }
  }

  applySubsampling() {
    const probabilities = new Map<number, number>();
    const t = 1e-3;
    const totalTokens = this.tokenizedCorpus.length;

    this.tokenizedCorpus = this.tokenizedCorpus.filter((token) => {
      if (!probabilities.has(token)) {
        const frequency = this.tokenCounts.get(token) ?? 1;
        // Mikolov subsampling formula
        const probability = 1 - Math.sqrt(t / (frequency / totalTokens));
        probabilities.set(token, probability);
      }
      const rand = Math.random();
      return rand > probabilities.get(token)!;
    });
  }

  filterLowFrequency(minCount: number = 5) {
    this.tokenizedCorpus = this.tokenizedCorpus.filter((token) => {
      return (this.tokenCounts.get(token) ?? 0) >= minCount;
    });
  }

  getTokenizedCorpus(): number[] {
    return this.tokenizedCorpus;
  }

  getVocabulary() {
    return this.wordToToken;
  }

  getTokenFromWord(word: string): number {
    if (!this.wordToToken.has(word)) {
      console.warn(`Skipping word '${word}', does not exist in vocabulary`);
      return -1;
    }

    return this.wordToToken.get(word)!;
  }

  getWordFromToken(token: number): string {
    if (!this.tokenToWord.has(token)) {
      console.warn(`Skipping token #${token}, does not exist in vocabulary`);
      return '';
    }
    return this.tokenToWord.get(token)!;
  }

  getMergedText() {
    return this.texts;
  }

  getTokenCount(token: number): number {
    return this.tokenCounts.get(token) || 0;
  }
}
