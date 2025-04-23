import { readFile, writeFile } from 'fs/promises';
import { Corpus } from './corpus';

// Utility function to read from disk
async function loadFile(filePath: string, fileName: string): Promise<string> {
  let data: Buffer;

  try {
    data = await readFile(`${filePath}${fileName}`);
  } catch (err) {
    throw new Error(`Cannot read ${filePath}${fileName}: ${err}`);
  }

  const convertedData = data.toString();
  return convertedData;
}

// Utility function to write to disk
async function saveFile(filePath: string, fileName: string, data: string) {
  try {
    writeFile(`${filePath}${fileName}`, data);
  } catch (err) {
    throw new Error(`Unable to save ${filePath}${fileName}: ${err}`);
  }
}

// Create list of all unique characters
async function generateCharList(corpus: Corpus): Promise<string[]> {
  // Flatten array of the corpus
  const flattenedCorpus: string[] = corpus.flat(Infinity) as string[];
  // Create Set with only unique characters
  const uniqueCharacters = new Set(flattenedCorpus);
  // Conver this to an array
  const uniqueCharacterArray = Array.from(uniqueCharacters);

  return uniqueCharacterArray;
}

// Create frequency map of byte pairs
function buildFrequencyMap(corpus: Corpus): Map<string, number> {
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

      // Add new entry to frequency map or increment existing
      frequencyMap.set(bytePair, (frequencyMap.get(bytePair) ?? 0) + 1);
    }
  }

  return frequencyMap;
}

(async () => {
  const data = await loadFile('./models/', 'corpus.txt');
  const corpus: Corpus = JSON.parse(data);
  
  // Build character list
  const charlist = await generateCharList(corpus);
  await saveFile('./output/', 'charlist.txt', JSON.stringify(charlist));

  // Build frequency list
  const frequencyMap = buildFrequencyMap(corpus);
  // Convert Map to object, then object to JSON
  const frequencyMapString = JSON.stringify(Object.fromEntries(frequencyMap));
  await saveFile('./output/', 'bytepairs.json', frequencyMapString);
})();
