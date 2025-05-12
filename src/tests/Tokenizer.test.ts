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
    const tokenizer = new Tokenizer(corpus);
    tokenizer.mergeAllTokenPairs(5);
    expect(tokenizer.getMergedText()).toEqual([['te', 's', 't', '</w>']]);
  });

  test('merges multiple token pairs in the same word', () => {
    const corpus = [['t', 'e', 's', 't', 'e', 'd', '</w>']];
    const tokenizer = new Tokenizer(corpus);
    tokenizer.mergeAllTokenPairs(6);
    expect(tokenizer.getMergedText()).toEqual([['te', 's', 'te', 'd', '</w>']]);
  });

  test('gets token from word', () => {
    const corpus = [['t', 'e', 's', 't', 'e', 'd', '</w>']];
    const tokenizer = new Tokenizer(corpus);
    tokenizer.mergeAllTokenPairs(6);
    expect(tokenizer.getTokenFromWord('s')).toEqual(2);
  });

  test('gets word from token', () => {
    const corpus = [['t', 'e', 's', 't', 'e', 'd', '</w>']];
    const tokenizer = new Tokenizer(corpus);
    tokenizer.mergeAllTokenPairs(6);
    expect(tokenizer.getWordFromToken(2)).toEqual('s');
  });
});
