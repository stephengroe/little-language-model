import { Embedder } from '../Embedder';

describe('Embedder module', () => {
  test('initializes embeddings with correct random dimensions', () => {
    const vocabulary = new Map<string, number>([
      ['a', 0],
      ['b', 1],
      ['c', 2],
    ]);
    const embedder = new Embedder(vocabulary, 64);
    expect([...embedder.getEmbeddings()][0][1]).toHaveLength(64);
  });
});
