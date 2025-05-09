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
  }

  // Save config
  console.log(`Saving config...`);
  await saveFile(filePath, `config.json`, JSON.stringify(trainingConfig));

  // Build corpus
  console.log(`Creating corpus...`);
  const corpus = new Corpus();
  await corpus.importTexts(
    trainingConfig.embedding.inputTexts,
    trainingConfig.embedding.inputTextDirectory
  );

  // Generate tokens
  console.log(`Generating tokens...`);
  const tokenizer = new Tokenizer(corpus.getTexts());
  await tokenizer.mergeAllTokenPairs(trainingConfig.embedding.vocabularySize);
  await tokenizer.tokenizeCorpus();
  await saveFile(
    filePath,
    `tokens.json`,
    JSON.stringify(Object.fromEntries(tokenizer.getVocabulary()))
  );

  // Generate embeddings
  console.log(`Generating embeddings...`);
  const embedder = new Embedder(
    tokenizer.getVocabulary().size,
    trainingConfig.embedding.vectorSize,
    tokenizer.getTokenizedCorpus()
  );
  const embeddingModel = embedder.train(
    trainingConfig.embedding.epochs,
    trainingConfig.embedding.learningRate,
    trainingConfig.embedding.contextWindow
  );
  await saveFile(
    filePath,
    `embeddings.json`,
    JSON.stringify(Object.fromEntries(embedder.getEmbeddings()))
  );
  await saveFile(filePath, `model-state.json`, JSON.stringify(embeddingModel));

  // Log success
  console.log(`Steps complete!`);
  console.timeEnd(`Generate project`);
}

main();
