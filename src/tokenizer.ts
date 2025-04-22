import { readFile, writeFile } from 'fs/promises';
import { Corpus } from './corpus';
import { write } from 'fs';

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

// Load character list into memory
async function loadCharList(): Promise<string[]> {
  // Define data
  let data: Buffer;  

  // Read from file
  try {
    data = await readFile('./output/charlist.json');
  } catch (err) {
    throw new Error(`Cannot read charlist.json: ${err}`);
  }

  // Parse charlist
  const charlist: string[] = JSON.parse(data.toString());
  return charlist;
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

// Create frequency map of byte pairs
function buildFrequencyMap(corpus: Corpus, charlist: string[]): Map<string, number> {
  // Create new Map to store frequency of byte pairs
  const frequencyMap = new Map<string, number>();

  // Iterate over all words in corpus
  for (const word of corpus) {
    // Sliding window across all letters in each word
    for (let i=0; i<word.length; i++) {
      // If we're at the last character, move to the next word
      if (i+1 >= word.length) continue;

      // Otherwise form byte pair from adjacent characters
      const bytePair = `${word[i]}${word[i+1]}`;

      // Add to frequency map or increment
      if (frequencyMap.has(bytePair)) {
        // Get current tally
        const tally = frequencyMap.get(bytePair) ?? 0; // In case undefined for TS
        // Increment by one
        frequencyMap.set(bytePair, tally + 1);
      } else {
        // Otherwise we'll create a new entry
        frequencyMap.set(bytePair, 1);
      }
    }
  }

  return frequencyMap;
}

// Save byte pairs to disk
async function writeBytePairs(bytePairs: Map<string, number> ) {
  const bytePairJSON = JSON.stringify(Object.fromEntries(bytePairs));
  try {
    await writeFile('./output/bytepairs.json', bytePairJSON);
  } catch (err) {
    throw new Error(`Could not write to file: ${err}`);
  }
}


(async () => {
  const corpus = await loadCorpus();
  await generateCharList(corpus);
  
  const charlist = await loadCharList();

  const frequencyMap = buildFrequencyMap(corpus, charlist);
  writeBytePairs(frequencyMap);
})();
