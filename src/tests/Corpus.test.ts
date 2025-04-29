import { Corpus } from '../Corpus';

describe('Corpus module', () => {
  test('initializes with empty array', () => {
    const corpus = new Corpus();
    expect(corpus.getTexts()).toEqual([]);
  });

  test('accepts and splits single text', () => {
    const corpus = new Corpus();
    corpus.addText('hello world');
    expect(corpus.getTexts()).toEqual([
      ['h', 'e', 'l', 'l', 'o', '</w>'],
      ['w', 'o', 'r', 'l', 'd', '</w>'],
      ['<|sep|>'],
    ]);
  });

  test('accepts and splits multiple texts with separator tokens', () => {
    const corpus = new Corpus();
    corpus.addText('hello world');
    corpus.addText('goodbye');
    expect(corpus.getTexts()).toEqual([
      ['h', 'e', 'l', 'l', 'o', '</w>'],
      ['w', 'o', 'r', 'l', 'd', '</w>'],
      ['<|sep|>'],
      ['g', 'o', 'o', 'd', 'b', 'y', 'e', '</w>'],
      ['<|sep|>'],
    ]);
  });
});
