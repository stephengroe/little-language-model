// Constants for tokenization boundaries
const wordBoundaryToken = '</w>';
const textSeparatorToken = '<|sep|>';

// Type for texts
export type CorpusTexts = string[][];

export class Corpus {
  // List of words compiled from training
  private texts: CorpusTexts;

  constructor() {
    this.texts = [];
  }

  // Build corpus from training data
  addText(content: string): void {
    // Throw error if passed an empty string
    if (!content.trim()) {
      throw new Error('Content for corpus cannot be an empty string');
    }
    // Divide into words
    const contentWords = content.split(/\s+/);

    // Iterate over each word
    for (const word of contentWords) {
      // Split word into characters
      const splitWord = word.split('');
      // Add end of word token
      splitWord.push(wordBoundaryToken);
      // Add to text
      this.texts.push(splitWord);
    }

    // Add text separator
    this.texts.push([textSeparatorToken]);
  }

  // Return texts
  getTexts(): CorpusTexts {
    return this.texts;
  }
}
