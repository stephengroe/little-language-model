import { Tokenizer } from '../Tokenizer';

describe('Tokenizer module', () => {
  test('initializes with cloned data', () => {
    const corpus = [['t', 'e', 's', 't', '</w>']];
    const tokenizer = new Tokenizer(corpus);

    expect(tokenizer.getMergedText()).toEqual(corpus);
    expect(tokenizer.getVocabulary().size).toBe(4);
  });

  test('merges token pairs', () => {
    const corpus = [['t', 'e', 's', 't', '</w>']];
    const vocab = new Map([
      ['t', 2],
      ['e', 1],
      ['s', 1],
      ['</w>', 1],
    ]);
    const tokenizer = new Tokenizer(corpus);
    tokenizer.mergeAllTokenPairs(1);
    expect(tokenizer.getMergedText()).toEqual([['te', 's', 't', '</w>']]);
  });

  test('merges multiple token pairs in the same word', () => {
    const corpus = [['t', 'e', 's', 't', 'e', 'd', '</w>']];
    const vocab = new Map([
      ['t', 2],
      ['e', 2],
      ['s', 1],
      ['d', 1],
      ['</w>', 1],
    ]);
    const tokenizer = new Tokenizer(corpus);
    tokenizer.mergeAllTokenPairs(1);
    expect(tokenizer.getMergedText()).toEqual([['te', 's', 'te', 'd', '</w>']]);
  });
});
