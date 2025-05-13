export const trainingConfig = {
  outputFolder: './output/embedding',
  embedding: {
    vocabularySize: 5_000,
    vectorSize: 64,
    epochs: 1,
    learningRate: 0.01,
    batchSize: 100,
    sampleEmbeddings: 10, // Number of embeddings to sample in console
    nearestNeighbors: 10,
    contextWindow: 2, // Word2Vec context words to include
    inputTextDirectory: './data/dickens/',
    inputTexts: [
      'a-christmas-carol.txt',
      // 'american-notes.txt',
      // 'bleak-house.txt',
      // 'david-copperfield.txt',
      // 'hard-times.txt',
      // 'little-dorrit.txt',
      // 'nicholas-nickleby.txt',
      // 'our-mutual-friend.txt',
      // 'the-old-curiosity-shop.txt',
      // 'the-pickwick-papers.txt',
    ],
  },
};
