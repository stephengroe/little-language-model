import { Vocabulary } from '../Vocabulary';

describe('Vocabulary module', () => {
  test('initializes with empty Map', () => {
    const vocab = new Vocabulary();
    expect(vocab.getVocab()).toBeInstanceOf(Map);
    expect(vocab.getVocab().size).toBe(0);
  });

  test('buildFromCorpus adds tokens to vocabulary', () => {
    const vocab = new Vocabulary();
    const corpus = [
      ['h', 'e', 'l', 'l', 'o', '</w>'],
      ['w', 'o', 'r', 'l', 'd', '</w>'],
    ];
    vocab.buildFromCorpus(corpus);
    expect(vocab.getVocab().has('w')).toBe(true);
    expect(vocab.getVocab().has('</w>')).toBe(true);
  });
});
