import { readFile, writeFile } from 'fs/promises';

// Data structure for individual words for training
export type Corpus = string[][];

// List of Dickens books for source data
const bookTitles = [
  'a-christmas-carol',
  'american-notes',
  'bleak-house',
  'david-copperfield',
  'hard-times',
  'little-dorrit',
  'nicholas-nickleby',
  'our-mutual-friend',
  'the-old-curiosity-shop',
  'the-pickwick-papers',
];

// Constants for tokenization boundaries
const wordBoundaryToken = '</w>';
const bookSeparatorToken = '<|sep|>';

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

// Build corpus from training data
async function buildCorpus(content: string): Promise<Corpus> {
    // Divide into words
    const words = content.split(/\s+/);
    // Divide into characters with word boundary at the end
    let corpus = words.map(word => [...word.split(''), wordBoundaryToken]);
    // Add file separator token
    corpus.push([bookSeparatorToken]);
    return corpus;
}

// Compile entire corpus
async function buildCompleteCorpus(bookTitles: string[]): Promise<Corpus> {
    // Define corpus array
    let corpus: Corpus = [];

    // Iterate over all books
    for (const bookTitle of bookTitles) {
      // Load book file
      const content = await loadFile('./data/dickens/', `${bookTitle}.txt`);
      // Build corpus from document
      const bookCorpus = await buildCorpus(content);
      // Append to corpus array
      corpus = corpus.concat(bookCorpus);
    }

    return corpus;
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

// Run all functions
(async () => {  
  // Build corpus
  console.log(`Building corpus...`);
  const corpus: Corpus = await buildCompleteCorpus(bookTitles);
  await saveFile('./output/', 'corpus.txt', JSON.stringify(corpus));
  
  // Build character list
  console.log(`Building character list...`);
  const charlist = await generateCharList(corpus);
  await saveFile('./output/', 'charlist.txt', JSON.stringify(charlist));

  // Build frequency map
  console.log(`Building frequency map...`);
  const frequencyMap = buildFrequencyMap(corpus);
  const frequencyMapString = JSON.stringify(Object.fromEntries(frequencyMap));
  await saveFile('./output/', 'bytepairs.json', frequencyMapString);

  // Log success
  console.log(`Tokenization complete!`)
})();
