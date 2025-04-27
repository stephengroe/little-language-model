import { readFile, writeFile, mkdir } from 'fs/promises';

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
  const fullPath = `${filePath}${fileName}`;
  try {
    await writeFile(fullPath, data);
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
function generateCharList(corpus: Corpus): string[] {
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
// NOTE: This mutates in place for performance reasons
function mergeTokenPairInPlace(corpus: Corpus, targetTokenPair: string) {
  // Iterate over all words in corpus
  for (let w = 0; w < corpus.length; w++) {
    // Reference word directly to allow mutations
    let word = corpus[w];
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
}

// Merge all token pairs
function mergeAllTokenPairs(corpus: Corpus, vocabularySize: number): [Corpus, Map<string, number>] {
  // Start count of all tokens
  let totalTokens = 0;
  // Create working version of corpus (deep copy)
  let mergedCorpus = corpus.map(word => [...word]);
  // Create vocabulary list
  let vocabulary = new Map<string, number>();
  
  // Get all unique charaters in corpus
  let uniqueCharacters: string[];
  uniqueCharacters = generateCharList(corpus);
  // Populate vocabulary with unique characters first
  for (const char of uniqueCharacters) {
    vocabulary.set(char, totalTokens);
    totalTokens++;
  }

  // While we still have merges left, continue
  while (totalTokens < vocabularySize) {
    // Find most common pair
    const mostCommonPair = findMostCommonPair(mergedCorpus);
    // Error handling to prevent infinite loop
    if (!mostCommonPair) break;
    // Merge that pair
    mergeTokenPairInPlace(mergedCorpus, mostCommonPair);
    // Add to vocabulary
    vocabulary.set(mostCommonPair, totalTokens);
    // Increment merged tokens
    totalTokens += 1;
    // Log progress every 100 tokens
    if (totalTokens % 100 === 0) {
      console.log(`Merged token ${totalTokens}/${vocabularySize}`);
    }
  }

  return [mergedCorpus, vocabulary];
}

// Tokenize corpus
function tokenizeCorpus(corpus: Corpus, vocabulary: Map<string, number>): number[] {
  // Map over each word
  let tokenizedCorpus: number[][] = corpus.map(word => {
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

function formatTimestampAsISO(date: Date) {
  const year = date.getFullYear();
  const month = formatAsTwoDigits(date.getMonth() + 1)
  const day = formatAsTwoDigits(date.getDate());
  const hour = formatAsTwoDigits(date.getHours());
  const minute = formatAsTwoDigits(date.getMinutes());
  const second = formatAsTwoDigits(date.getSeconds());

  function formatAsTwoDigits(number: number) {
    return number.toString().padStart(2, '0');
  }

  return [year, month, day, hour, minute, second].join("");
}

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