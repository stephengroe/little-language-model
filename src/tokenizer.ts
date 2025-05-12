export class Tokenizer {
  private wordToToken: Map<string, number>;
  // Set up corpus for merging
  private texts: string[][];
  // Set up index (to store locations of pairs for merging)
  private pairIndex: Map<string, Set<number>>;
  // Set up frequency map
  private frequencyMap: Map<string, number>;
  private tokenizedCorpus: number[];
  private tokenToWord: Map<number, string>;

  constructor(texts: string[][]) {
    this.texts = JSON.parse(JSON.stringify(texts));
    this.pairIndex = new Map<string, Set<number>>();
    this.frequencyMap = new Map<string, number>();
    this.tokenizedCorpus = [];

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
    // Isolate word in corpus
    let targetWord = this.texts[targetWordIndex];
    // Store positions of future merges
    let mergeIndexes: number[] = [];

    // Sliding window across all token pairs
    for (let i = 0; i < targetWord.length; i++) {
      // If we're at the last character
      if (i + 1 >= targetWord.length) continue;
      // Isolate and name the current pair
      const currentTokenPair = `${targetWord[i]}${targetWord[i + 1]}`;

      // Decrement from frequency map (we'll add later if relevant)
      const frequency = this.frequencyMap.get(currentTokenPair) || 1;
      if (frequency === 1) {
        // If last entry in frequency map, delete entry
        this.frequencyMap.delete(currentTokenPair);
      } else {
        this.frequencyMap.set(currentTokenPair, frequency - 1);
      }

      // Remove word from index (we'll add later if relevant)
      this.pairIndex.get(targetTokenPair)?.delete(targetWordIndex);

      // If we've found a pair to merge
      if (currentTokenPair === targetTokenPair) {
        // Save in list of indexes
        mergeIndexes.push(i);
      }
    }

    // Perform merges
    let mergeOffset = 0; // Adjust index for words with multiple merges
    for (const mergeIndex of mergeIndexes) {
      // Merge tokens into one
      targetWord.splice(mergeIndex - mergeOffset, 2, targetTokenPair);
      mergeOffset += 1;
    }

    // Recompute the tokens in this word
    this.indexWordTokens(targetWordIndex);
  }

  // Merge token pairs in corpus
  mergeTokenPair(targetTokenPair: string) {
    // Get positions of all words with target token pair
    const targetWordIndexes = this.pairIndex.get(targetTokenPair);

    // Return if undefined or no indexed words
    if (!targetWordIndexes || targetWordIndexes.size === 0) return;

    // Iterate over target words
    for (const targetWordIndex of targetWordIndexes) {
      this.mergeWordTokenPair(targetTokenPair, targetWordIndex);
    }
  }

  // Merge all token pairs
  mergeAllTokenPairs(vocabularySize: number): void {
    // Start count of all tokens
    let totalTokens = this.wordToToken.size;

    // While we still have merges left, continue
    while (totalTokens < vocabularySize) {
      // Find most common pair
      const mostCommonPair = this.findMostCommonTokenPair();
      // Error handling to prevent infinite loop
      if (!mostCommonPair) break;
      // Merge that pair
      this.mergeTokenPair(mostCommonPair);
      // Add to vocabulary
      this.wordToToken.set(mostCommonPair, totalTokens);
      this.tokenToWord.set(totalTokens, mostCommonPair);
      // Increment merged tokens
      totalTokens += 1;

      if (totalTokens % 1000 === 0) {
        console.log(`Merged ${totalTokens}/${vocabularySize} tokens`);
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
  findMostCommonTokenPair(): string {
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

    return mostCommonToken;
  }

  // Tokenize corpus
  tokenizeCorpus() {
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
  }

  getTokenizedCorpus(): number[] {
    return this.tokenizedCorpus;
  }

  getVocabulary() {
    return this.wordToToken;
  }

  getTokenFromWord(word: string) {
    if (!this.wordToToken.has(word)) {
      throw new Error(`Word '${word}' does not exist in vocabulary`);
    }
    return this.wordToToken.get(word);
  }

  getWordFromToken(token: number) {
    if (!this.tokenToWord.has(token)) {
      throw new Error(`Token #${token} does not exist in vocabulary`);
    }
    return this.tokenToWord.get(token);
  }

  getMergedText() {
    return this.texts;
  }
}
