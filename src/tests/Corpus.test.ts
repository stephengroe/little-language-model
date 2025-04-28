import { Corpus } from '../Corpus';

describe('Corpus module', () => {
  it('initializes with empty array', () => {
    const corpus = new Corpus();
    expect(corpus.getTexts()).toEqual([]);
  });

  it('accepts and splits single text correctly', () => {
    const corpus = new Corpus();
    corpus.addText('hello world');
    expect(corpus.getTexts()).toEqual([
      ['h', 'e', 'l', 'l', 'o', '</w>'],
      ['w', 'o', 'r', 'l', 'd', '</w>'],
      ['<|sep|>'],
    ]);
  });

  it('accepts and splits multiple texts correctly with separator tokens', () => {
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
