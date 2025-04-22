import { readFile, writeFile } from 'fs/promises';
import { Corpus } from './corpus';

// Load corpus text into memory
async function loadCorpus(): Promise<Corpus> {
  // Define data
  let data: Buffer;  

  // Read from file
  try {
    data = await readFile('./models/corpus.txt');
  } catch (err) {
    throw new Error(`Cannot read corpus.txt: ${err}`);
  }

  // Parse corpus
  const corpus: Corpus = JSON.parse(data.toString());
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
  try {
    await writeFile('./output/charlist.json', uniqueCharactersString);
  } catch (err) {
    throw new Error(`Could not write file charlist.json: ${err}`);
  }
}

(async () => {
  const corpus = await loadCorpus();
  await generateCharList(corpus);
})();
