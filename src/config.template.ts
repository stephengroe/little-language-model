export const trainingConfig = {
  outputFolder: './output/embedding-test',
  embedding: {
    logInterval: 0.1,
    vocabularySize: 5_000,
    vectorSize: 64,
    epochs: 3,
    learningRate: 0.05,
    batchSize: 32,
    sampledEmbeddings: [' man', ' house', ' walk', ' I'],
    randomSamples: 10, // Number of embeddings to sample in console
    nearestNeighbors: 10,
    contextWindow: 2, // Word2Vec context words to include
    inputTextDirectory: './data/dickens/',
    inputTexts: [
      'a-christmas-carol.txt',
      'american-notes.txt',
      'barnaby-rudge.txt',
      'bleak-house.txt',
      'david-copperfield.txt',
      'dombey-and-son.txt',
      'hard-times.txt',
      'little-dorrit.txt',
      'martin-chuzzlewit.txt',
      'nicholas-nickleby.txt',
      'oliver-twist.txt',
      'our-mutual-friend.txt',
      'the-mystery-of-edwin-drood.txt',
      'the-old-curiosity-shop.txt',
      'the-pickwick-papers.txt',
    ],
  },
};
