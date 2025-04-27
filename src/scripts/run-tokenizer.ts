import { mkdir } from 'fs/promises';
import { saveFile, formatTimestampAsISO } from '../utils';
import { bookTitles, Corpus, buildCompleteCorpus, mergeAllTokenPairs, tokenizeCorpus } from '../tokenizer';

// Run all functions
(async () => {
  // Create timestamp to save output
  const timestamp = formatTimestampAsISO(new Date());

  // Create new folder to save all output
  try {
    await mkdir(`./output/${timestamp}`);
  } catch (err) {
    console.error(`Unable to create directory ${timestamp}: ${err}`);
  }
  
  // Build corpus
  console.log(`Building corpus...`);
  const corpus: Corpus = await buildCompleteCorpus(bookTitles);
  await saveFile(`./output/${timestamp}/`, 'corpus-raw.txt', JSON.stringify(corpus));

  // Build list of tokens
  const vocabularySize = 5;
  console.log(`Replacing token pairs for ${vocabularySize} tokens...`)
  console.time(`Token merging`)
  const [mergedCorpus, vocabulary] = mergeAllTokenPairs(corpus, vocabularySize);
  console.timeEnd(`Token merging`);
  await saveFile(`./output/${timestamp}/`, `corpus-merged.txt`, JSON.stringify(mergedCorpus));
  await saveFile(`./output/${timestamp}/`, `vocabulary.json`, JSON.stringify(vocabulary));
  
  // Tokenize corpus
  console.log(`Tokenizing corpus...`);
  const tokenizedCorpus = tokenizeCorpus(mergedCorpus, vocabulary);
  await saveFile(`./output/${timestamp}/`, `corpus-tokenized.json`, JSON.stringify(tokenizedCorpus));

  // Log success
  console.log(`Tokenization complete!`);
})();