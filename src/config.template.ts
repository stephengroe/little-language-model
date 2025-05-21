export const trainingConfig = {
  outputFolder: './output/embedding-test',
  embedding: {
    logInterval: 0.1,
    vocabularySize: 5_000,
    vectorSize: 64,
    epochs: 25,
    learningRate: 0.01,
    batchSize: 100,
    sampledEmbeddings: ['Marley</w>', 'Scrooge</w>', 'Christmas</w>', 'I</w>'],
    randomSamples: 10, // Number of embeddings to sample in console
    nearestNeighbors: 10,
    contextWindow: 2, // Word2Vec context words to include
    inputTextDirectory: './data/dickens/',
    inputTexts: [
      'a-christmas-carol.txt',
      'american-notes.txt',
      'bleak-house.txt',
      'david-copperfield.txt',
      'hard-times.txt',
      'little-dorrit.txt',
      'nicholas-nickleby.txt',
      'our-mutual-friend.txt',
      'the-old-curiosity-shop.txt',
      'the-pickwick-papers.txt',
    ],
  },
};
