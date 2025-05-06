import { mkdir } from 'fs/promises';
import { saveFile, loadFile, formatTimestampAsISO } from './utils';
import { CorpusTexts, Corpus } from './Corpus';
import { Vocabulary } from './Vocabulary';
import { Tokenizer } from './Tokenizer';
import { trainingConfig } from './config';
import { Embedder } from './Embedder';
import { NeuralNetwork } from './NeuralNetwork';

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

  // Build corpus
  console.log(`Creating corpus...`);
  const corpus = new Corpus();

  // Add books to corpus
  for (const bookTitle of trainingConfig.inputTexts) {
    // Load book file
    const content = await loadFile(
      trainingConfig.inputTextDirectory,
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
  tokenizer.mergeAllTokenPairs(trainingConfig.vocabularySize);
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
  const tokenizedCorpus = tokenizeCorpus(
    tokenizer.getMergedText(),
    tokenizer.getVocabulary()
  );
  await saveFile(
    filePath,
    `corpus-tokenized.json`,
    JSON.stringify(tokenizedCorpus)
  );

  // Generate embeddings
  console.log(`Initializing embeddings...`);
  const embedder = new Embedder(
    tokenizer.getVocabulary(),
    trainingConfig.embeddingDimensions
  );

  await saveFile(
    filePath,
    `embeddings.json`,
    JSON.stringify(Object.fromEntries(embedder.getEmbeddings()))
  );

  // Run neural net
  const neuralNet = new NeuralNetwork(5, 10);
  const data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
  const answerKey = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
  neuralNet.train(
    data,
    answerKey,
    trainingConfig.epochs,
    trainingConfig.learningRate
  );

  // Log success
  console.log(`Steps complete!`);
  console.timeEnd(`Generate project`);
}

// Tokenize corpus
export function tokenizeCorpus(
  corpus: CorpusTexts,
  vocabulary: Map<string, number>
): number[] {
  // Map over each word
  let tokenizedCorpus: number[][] = corpus.map((word: string[]) => {
    // Map over each token
    return word.map((token) => {
      // Replace with token ID or -1 if not found
      return vocabulary.get(token) ?? -1;
    });
  });
  // Remove word-level subarrays, convert to simplified list of tokens
  const flattenedCorpus = tokenizedCorpus.flat(1);
  return flattenedCorpus;
}

main();
