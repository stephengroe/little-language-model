import { mkdir } from 'fs/promises';
import { saveFile, loadFile, formatTimestampAsISO } from './utils';
import { Corpus } from './Corpus';
import { Tokenizer } from './Tokenizer';
import { trainingConfig } from './config';
import { Embedder } from './Embedder';

// Run all functions
async function main() {
  // Start timer
  console.time(`Generate project`);
  // Create timestamp to save output
  const timestamp = `${formatTimestampAsISO(new Date())}`;
  const filePath = `${trainingConfig.outputFolder}-${timestamp}/`;

  // Create new folder to save all output
  try {
    await mkdir(filePath);
  } catch (err) {
    console.error(`Unable to create directory ${filePath}: ${err}`);
    return;
  }

  // Save config
  console.log(`Saving config...`);
  await saveFile(
    filePath,
    `config.json`,
    JSON.stringify(trainingConfig, null, 2)
  );

  // Build corpus
  console.log(
    `Building corpus from ${trainingConfig.embedding.inputTexts.length} files...`
  );
  const corpus = new Corpus();
  const importedWords = await corpus.importTexts(
    trainingConfig.embedding.inputTexts,
    trainingConfig.embedding.inputTextDirectory
  );
  console.log(`Imported ${importedWords.toLocaleString()} words!`);

  // Generate tokens
  console.log(`Generating tokens...`);
  const tokenizer = new Tokenizer(corpus.getTexts());
  tokenizer.mergeAllTokenPairs(trainingConfig.embedding.vocabularySize);
  tokenizer.tokenizeCorpus();
  await saveFile(
    filePath,
    `tokens.json`,
    JSON.stringify(Object.fromEntries(tokenizer.getVocabulary()), null, 2)
  );

  // Generate embeddings
  console.log(`Initializing embeddings...`);
  const embedder = new Embedder(
    tokenizer.getVocabulary().size,
    trainingConfig.embedding.vectorSize,
    tokenizer.getTokenizedCorpus()
  );

  /* FOR TRAINING EMBEDDINGS */
  console.log(`Training embeddings...`);
  console.time(`Training embeddings`);
  const embeddingModel = embedder.train(
    trainingConfig.embedding.batchSize,
    trainingConfig.embedding.contextWindow,
    trainingConfig.embedding.learningRate,
    trainingConfig.embedding.epochs,
    trainingConfig.embedding.logInterval
  );
  console.timeEnd(`Training embeddings`);
  await saveFile(
    filePath,
    `model-state.json`,
    JSON.stringify(embeddingModel, null, 2)
  );
  const embeddings = embedder.buildEmbeddings();

  /* FOR GENERATING FROM SAVED MODEL
  const savedModel = await loadFile(filePath, `model-state.json`);
  embedder.buildFromSavedModel(
    JSON.parse(savedModel),
    trainingConfig.embedding.vocabularySize
  );
  */

  // Print nearest neighbors of sample embeddings
  console.log(`Sampling embeddings...`);

  const sampleTokens: number[] = trainingConfig.embedding.sampledEmbeddings
    .map((word) => {
      return tokenizer.getTokenFromWord(word);
    })
    .filter((n) => n >= 0); // remove invalid tokens

  // Sample random tokens from corpus
  const uniqueWords = Array.from(
    new Set<number>(tokenizer.getTokenizedCorpus())
  );
  for (let i = 0; i < trainingConfig.embedding.randomSamples; i++) {
    const randIndex = Math.floor(Math.random() * uniqueWords.length);
    const randToken = uniqueWords[randIndex];
    sampleTokens.push(randToken);
  }

  for (const sampleToken of sampleTokens) {
    console.log(` Nearest to '${tokenizer.getWordFromToken(sampleToken)}'`);
    console.log(`   Frequency: ${tokenizer.getTokenCount(sampleToken)}`);
    const nearestNeighbors = embedder.findNearest(
      sampleToken,
      trainingConfig.embedding.nearestNeighbors
    );
    nearestNeighbors.forEach((token, index) => {
      const word = tokenizer.getWordFromToken(token);
      console.log(`   ${index + 1}) ${word}`);
    });
  }

  console.log(`Saving embeddings...`);
  await saveFile(
    filePath,
    `embeddings.json`,
    JSON.stringify(embedder.getEmbeddings(), null)
  );

  // Log success
  console.log(`Steps complete!`);
  console.timeEnd(`Generate project`);
}

main().catch((err) => {
  console.error(`Fatal error: `, err);
});
