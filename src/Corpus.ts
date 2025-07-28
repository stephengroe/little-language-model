import { loadFile } from './utils';

// Constants for tokenization boundaries
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

    // Remove duplicate spaces
    for (let char of content) {
      if (char === ' ' && prevChar === ' ') {
        prevChar = char;
        continue;
      }
      revisedContent.push(char);
      prevChar = char;
    }

    const contentWords =
      revisedContent.join('').match(/ ?\p{L}+| ?\p{N}+| ?\p{P}+/gu) || [];

    for (const word of contentWords!) {
      const splitWord = word.split('');
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
