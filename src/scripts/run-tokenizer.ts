import { mkdir } from 'fs/promises';
import { saveFile, loadFile, formatTimestampAsISO } from '../utils';
import { CorpusTexts, Corpus } from '../Corpus';
import { Vocabulary } from '../Vocabulary';
import { Tokenizer } from '../tokenizer';

// List of Dickens books for source data
export const bookTitles = [
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
];

// Set vocabulary size
const vocabularySize = 1_000;

// Run all functions
async function main() {
  // Create timestamp to save output
  const timestamp = `DELETE-ME-${formatTimestampAsISO(new Date())}`;

  // Create new folder to save all output
  try {
    await mkdir(`./output/${timestamp}`);
  } catch (err) {
    console.error(`Unable to create directory ${timestamp}: ${err}`);
  }
  
  // Build corpus
  console.log(`Creating corpus...`);
  const corpus = new Corpus();

  // Add books to corpus
  for (const bookTitle of bookTitles) {
    // Log
    console.log(`Adding ${bookTitle}...`);
    // Load book file
    const content = await loadFile('./data/dickens/', `${bookTitle}.txt`);
    // Build corpus from document
    corpus.addText(content);
  }

  // Save raw corpus
  await saveFile(`./output/${timestamp}/`, 'corpus-raw.txt', JSON.stringify(corpus.getTexts()));

  // Build vocabulary
  console.log(`Building vocabulary...`);
  const vocab = new Vocabulary();
  vocab.buildFromCorpus(corpus.getTexts());
  await saveFile(`./output/${timestamp}/`, `vocabulary.json`, JSON.stringify(Object.fromEntries(vocab.getVocab())));
  
  // Build list of tokens
  console.log(`Building tokenizer...`);
  const tokenizer = new Tokenizer(corpus.getTexts(), vocab.getVocab());
  
  console.log(`Replacing token pairs for ${vocabularySize} tokens...`);
  console.time(`Token merging`);

  tokenizer.mergeAllTokenPairs(vocabularySize);
  console.timeEnd(`Token merging`);
  await saveFile(`./output/${timestamp}/`, `corpus-merged.txt`, JSON.stringify(tokenizer.getMergedText()));
  await saveFile(`./output/${timestamp}/`, `vocabulary-merged.json`, JSON.stringify(Object.fromEntries(tokenizer.getVocabulary())));
  
  // Tokenize corpus
  console.log(`Tokenizing corpus...`);
  const tokenizedCorpus = tokenizeCorpus(tokenizer.getMergedText(), tokenizer.getVocabulary());
  await saveFile(`./output/${timestamp}/`, `corpus-tokenized.json`, JSON.stringify(tokenizedCorpus));

  // Log success
  console.log(`Tokenization complete!`);
}

// Tokenize corpus
export function tokenizeCorpus(corpus: CorpusTexts, vocabulary: Map<string, number>): number[] {
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
