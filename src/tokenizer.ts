import { readFile, writeFile } from 'fs/promises';
import { Corpus } from './corpus';

// Load corpus text into memory
async function loadCorpus(): Promise<Corpus> {
  const data = await readFile('./models/corpus.txt');
  const corpus: Corpus = JSON.parse(data);
  return corpus;
}

// Create list of all unique characters
async function generateCharList(corpus: Corpus) {
  // Flatten array of the corpus
  const flattenedCorpus = corpus.flat(Infinity);
  // Create Set with only unique characters
  const uniqueCharacters = new Set(flattenedCorpus);
  // Conver this to an array, then stringified JSON
  const uniqueCharactersString = JSON.stringify([...uniqueCharacters]);
  
  // Save to disk
  await writeFile('./output/charlist.txt', uniqueCharactersString);
}

loadCorpus().then(corpus => {
  generateCharList(corpus);
});