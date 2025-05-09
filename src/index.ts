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
  await saveFile(filePath, `config.json`, JSON.stringify(trainingConfig));

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
  await tokenizer.mergeAllTokenPairs(trainingConfig.embedding.vocabularySize);
  await tokenizer.tokenizeCorpus();
  await saveFile(
    filePath,
    `tokens.json`,
    JSON.stringify(Object.fromEntries(tokenizer.getVocabulary()))
  );

  // Generate embeddings
  console.log(`Generating embeddings...`);
  const embedder = await new Embedder(
    tokenizer.getVocabulary().size,
    trainingConfig.embedding.vectorSize,
    tokenizer.getTokenizedCorpus()
  );
  const embeddingModel = await embedder.train(
    trainingConfig.embedding.batchSize,
    trainingConfig.embedding.contextWindow,
    trainingConfig.embedding.learningRate,
    trainingConfig.embedding.epochs
  );
  await saveFile(filePath, `model-state.json`, JSON.stringify(embeddingModel));

  // Log success
  console.log(`Steps complete!`);
  console.timeEnd(`Generate project`);
}

main().catch((err) => {
  console.error(`Fatal error: `, err);
});
