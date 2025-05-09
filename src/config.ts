export const trainingConfig = {
  outputFolder: './output/embedding',
  embedding: {
    vocabularySize: 5_000,
    hiddenSize: 64,
    epochs: 5,
    learningRate: 0.01,
    batchSize: 32,
    contextWindow: 2, // Word2Vec context words to include
    inputTextDirectory: './data/dickens/',
    inputTexts: [
      'a-christmas-carol',
      // 'american-notes',
      // 'bleak-house',
      // 'david-copperfield',
      // 'hard-times',
      // 'little-dorrit',
      // 'nicholas-nickleby',
      // 'our-mutual-friend',
      // 'the-old-curiosity-shop',
      // 'the-pickwick-papers',
    ],
  },
};
