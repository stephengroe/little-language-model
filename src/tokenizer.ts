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

// Create frequency map of token pairs
function buildFrequencyMap(corpus: Corpus): Map<string, number> {
  // Create new Map to store frequency of token pairs
  const frequencyMap = new Map<string, number>();

  // Iterate over all words in corpus
  for (const word of corpus) {
    // Sliding window across all letters in each word
    for (let i=0; i<word.length; i++) {
      // If we're at the last character, move to the next word
      if (i+1 >= word.length) continue;

      // Otherwise form token pair from adjacent characters
      const tokenPair = `${word[i]}${word[i+1]}`;

      // Add new entry to frequency map or increment existing
      frequencyMap.set(tokenPair, (frequencyMap.get(tokenPair) ?? 0) + 1);
    }
  }

  return frequencyMap;
}

// Find most common adjacent token pair
function findMostCommonPair(corpus: Corpus): string {
  // Build frequency map from the corpus
  const frequencyMap = buildFrequencyMap(corpus);

  // Set max count and tokens
  let maxCount = -Infinity;
  let maxToken: string = '';

  // Iterate over each token pair in the frequency map
  for (const [token, count] of frequencyMap) {
    // If greater than previous max, set as new max
    if (count > maxCount) {
      maxCount = count;
      maxToken = token;
    }
  }

  return maxToken;
}

// Merge token pairs in corpus
function mergeTokenPair(corpus: Corpus, targetTokenPair: string): Corpus {
  // Create working copy of corpus (deep copy)
  let mergedCorpus = corpus.map(word => [...word]);

  // Iterate over all words in corpus
  for (let w = 0; w < mergedCorpus.length; w++) {
    // Reference word directly to allow mutations
    let word = mergedCorpus[w];
    // Sliding window across all letters in each word
    for (let i=0; i<word.length; i++) {
      // If we're at the last character, move to the next word
      if (i+1 >= word.length) continue;

      // If token pair matches target token pair
      if (`${word[i]}${word[i+1]}` === targetTokenPair) {
        // Merge tokens into one
        word.splice(i, 2, targetTokenPair);
        // Skip the merged token
        i += 1;
      }
    }
  }

  return mergedCorpus;
}

// Merge all token pairs
function mergeAllTokenPairs(corpus: Corpus, vocabularySize: number): Corpus {
  // Start count of merged tokens
  let mergedTokens = 0;
  // Create working version of corpus (deep copy)
  let mergedCorpus = corpus.map(word => [...word]);
  // Create vocabulary list
  let vocabulary = new Map<string, number>();

  // While we still have merges left, continue
  while (mergedTokens < vocabularySize) {
    // Find most common pair
    const mostCommonPair = findMostCommonPair(mergedCorpus);
    // Merge that pair
    mergedCorpus = mergeTokenPair(mergedCorpus, mostCommonPair);
    // Add to vocabulary
    vocabulary.set(mostCommonPair, mergedTokens);
    // Increment merged tokens
    mergedTokens += 1;
    // Log progress
    console.log(`Merged token ${mergedTokens}/${vocabularySize}: ${mostCommonPair}`);
  }

  // Save vocabulary to disk
  const vocabularyString = JSON.stringify(Object.fromEntries(vocabulary));
  (async () => await saveFile('./output/', 'vocabulary.txt', vocabularyString))();

  return mergedCorpus;
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

  // Replace token pairs
  const vocabularySize = 10;
  const startTime = Date.now();
  console.log(`Replacing token pairs for ${vocabularySize} tokens...`)
  const mergedCorpus = mergeAllTokenPairs(corpus, vocabularySize);
  console.log(`Finished in ${Math.round((Date.now() - startTime) / 1000 / 60 * 10) / 10} minutes`);
  await saveFile('./output/', 'merged-corpus.txt', JSON.stringify(mergedCorpus));

  // Log success
  console.log(`Tokenization complete!`)
})();
