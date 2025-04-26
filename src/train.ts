import { readFile, writeFile } from 'fs/promises';

// Data structure for individual words for training
type Corpus = string[][];

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

// Read file and log result
async function openFile(bookTitle: string): Promise<string> {
    // Log progress
    console.log(`Loading ${bookTitle}...`)

    try {
        const data = await readFile(`./data/dickens/${bookTitle}.txt`);
        // Convert book text to string and return
        const content = data.toString();
        return content;
    } catch (err) {
        // Error handling if document can't be read
        throw new Error(`Error reading ${bookTitle}.txt: ${err}`);
    }
}

// Build corpus from file
function buildCorpus(content: string): Corpus {
    // Divide into words
    const words = content.split(/\s+/);
    // Divide into characters with word boundary at the end
    let corpus = words.map(word => [...word.split(''), wordBoundaryToken]);
    // Add file separator token
    corpus.push([bookSeparatorToken]);
    // Log progress
    console.log(`Built corpus`);
    return corpus;
}

// Build entire corpus
async function buildCompleteCorpus(bookTitles: string[]) {
    // Define corpus array
    let corpus: Corpus = [];

    // Iterate over all books
    for (const bookTitle of bookTitles) {
        // Load book file
        const content = await openFile(bookTitle);
        // Build corpus from document
        const bookCorpus = buildCorpus(content);
        // Append to corpus array
        corpus = corpus.concat(bookCorpus);
    }

    // Convert to string
    const corpusString = JSON.stringify(corpus);

    // If corpus.txt exists, clear it to prevent writing twice
    await writeFile('./models/corpus.txt', '');
    // Write corpus to disk
    await writeFile('./models/corpus.txt', corpusString);
    // Log success
    console.log(`Saved corpus to disk`)
}

buildCompleteCorpus(bookTitles);