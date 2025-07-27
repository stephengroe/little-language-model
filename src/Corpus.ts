import { loadFile } from './utils';

// Constants for tokenization boundaries
const wordBoundaryToken = '</w>';
const textSeparatorToken = '<|sep|>';

export class Corpus {
  // List of words compiled from training
  private texts: string[][];

  constructor() {
    this.texts = [];
  }

  // Build corpus from training data
  addText(content: string): void {
    // Throw error if passed an empty string
    if (!content.trim()) {
      throw new Error('Content for corpus cannot be an empty string');
    }

    // Remove duplicate spaces
    let revisedContent: string[] = [];
    let prevChar = '';

    for (let char of content) {
      if (char === ' ' && prevChar === ' ') return;
      if (char === '\n') {
        if (prevChar === ' ') return;
        else char = ' ';
      }

      revisedContent.push(char);
    }

    const contentWords = content.match(/ ?\p{L}+| ?\p{N}+| ?\p{P}+/gu) || [];

    // Divide into words
    // const contentWords = content.split(/\s+/);

    // Iterate over each word
    for (const word of contentWords!) {
      // Split word into characters
      const splitWord = word.split('');
      // Add end of word token
      // splitWord.push(wordBoundaryToken);
      // Add to text
      this.texts.push(splitWord);
    }

    // Add text separator
    this.texts.push([textSeparatorToken]);
  }

  // Add text in bulk
  async importTexts(fileNames: string[], filePath: string): Promise<number> {
    for (const fileName of fileNames) {
      const content = await loadFile(filePath, fileName);
      const text = content.toString();
      this.addText(text);
    }

    // Return number of words imported
    return this.texts.length;
  }

  // Return texts
  getTexts(): string[][] {
    return this.texts;
  }
}
