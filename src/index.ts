import { mkdir } from 'fs/promises';
import { saveFile, loadFile, formatTimestampAsISO } from './utils';
import { CorpusTexts, Corpus } from './Corpus';
import { Vocabulary } from './Vocabulary';
import { Tokenizer } from './tokenizer';
import { trainingConfig } from './config';
import { Embedder } from './Embedder';
import { NeuralNetwork, ModelState } from './NeuralNetwork';

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

  // Add books to corpus
  for (const bookTitle of trainingConfig.embedding.inputTexts) {
    // Load book file
    const content = await loadFile(
      trainingConfig.embedding.inputTextDirectory,
      `${bookTitle}.txt`
    );
    // Build corpus from document
    corpus.addText(content);
  }

  // Save raw corpus
  await saveFile(filePath, 'corpus-raw.txt', JSON.stringify(corpus.getTexts()));

  // Build vocabulary
  console.log(`Generating initial vocabulary...`);
  const vocab = new Vocabulary();
  vocab.buildFromCorpus(corpus.getTexts());
  await saveFile(
    filePath,
    `vocabulary.json`,
    JSON.stringify(Object.fromEntries(vocab.getVocab()))
  );

  // Tokenizing
  console.log(`Generating tokens from corpus...`);
  const tokenizer = new Tokenizer(corpus.getTexts(), vocab.getVocab());
  tokenizer.mergeAllTokenPairs(trainingConfig.embedding.vocabularySize);
  await saveFile(
    filePath,
    `corpus-merged.txt`,
    JSON.stringify(tokenizer.getMergedText())
  );
  await saveFile(
    filePath,
    `vocabulary-merged.json`,
    JSON.stringify(Object.fromEntries(tokenizer.getVocabulary()))
  );

  // Tokenize corpus
  console.log(`Tokenizing corpus...`);
  await tokenizer.tokenizeCorpus();
  const tokenizedCorpus = tokenizer.getTokenizedCorpus();
  await saveFile(
    filePath,
    `corpus-tokenized.json`,
    JSON.stringify(tokenizedCorpus)
  );

  // Generate embeddings
  console.log(`Initializing embeddings...`);
  const embedder = new Embedder(
    tokenizer.getVocabulary(),
    trainingConfig.embedding.hiddenSize
  );
  const embeddingLayerSizes = [
    trainingConfig.embedding.vocabularySize,
    trainingConfig.embedding.hiddenSize,
    trainingConfig.embedding.vocabularySize,
  ];
  const neuralNet = new NeuralNetwork(embeddingLayerSizes);

  const trainingData = await embedder.generateTrainingData(
    tokenizedCorpus,
    trainingConfig.embedding.contextWindow
  );
  await saveFile(filePath, `training-data.json`, JSON.stringify(trainingData));

  // Iterate over training data in batches
  console.log(`Training on data in batches...`);
  let batchIndex = 0;
  do {
    console.log(`Training batch ${batchIndex}-${batchIndex + 100}`);
    const vectorizedTrainingData = await embedder.vectorizeBatch(
      trainingData.slice(batchIndex, batchIndex + 100),
      trainingConfig.embedding.vocabularySize
    );

    neuralNet.trainOnBatch(
      vectorizedTrainingData,
      trainingConfig.embedding.epochs,
      trainingConfig.embedding.learningRate
    );

    batchIndex += 100;
  } while (batchIndex < trainingData.length);

  await saveFile(
    filePath,
    `embeddings.json`,
    JSON.stringify(Object.fromEntries(embedder.getEmbeddings()))
  );

  // Save model state
  console.log(`Saving model state...`);
  const modelState: ModelState = await neuralNet.getModelState();
  await saveFile(filePath, `model-state.json`, JSON.stringify(modelState));

  // Log success
  console.log(`Steps complete!`);
  console.timeEnd(`Generate project`);
}

main();
