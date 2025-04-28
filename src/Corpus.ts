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
    // Divide into words
    const contentWords = content.split(/\s+/);
    // Divide into characters with word boundary at the end
    const wordList = contentWords.map((word) => [
      ...word.split(''),
      wordBoundaryToken,
    ]);
    // Add to texts
    this.texts.push(...wordList);
    // Add text separator
    this.texts.push([textSeparatorToken]);
  }

  // Return texts
  getTexts(): CorpusTexts {
    return this.texts;
  }
}
