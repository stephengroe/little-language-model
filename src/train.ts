import { readFile, writeFile, appendFile } from 'fs/promises';

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

// Read file and log result
function openFile(bookTitle: string): string {
    // Log progress
    console.log(`Loading ${bookTitle}...`)

    try {
        const data = readFile(`./data/dickens/${bookTitle}.txt`);
        // Convert book text to string and return
        const content = data.toString();
        return content;
    } catch (err) {
        // Error handling if document can't be read
        throw new Error(`Error reading ${bookTitle}.txt: ${err}`);
    }
}

// Append to corpus document
function appendBookToCorpus(content: string) {
    // Log progress
    console.log(`Appending to corpus...`);

    try {
        appendFile(`./models/corpus.txt`, content);
    } catch (err) {
        throw new Error(`Error writing to corpus: ${err}`)
    }
}

// If corpus.txt exists, clear it to prevent writing twice
await writeFile('./models/corpus.txt', '');

// Iterate over all books
for (const bookTitle of bookTitles) {
    // Load book file
    const content = openFile(bookTitle);
    // Build corpus from document
    const corpus: Corpus = buildCorpus(content);
    // Convert to string
    const corpusString = corpus.toString();
    // Write to corpus
    appendBookToCorpus(corpusString);
}
