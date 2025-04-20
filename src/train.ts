import fs from 'fs';

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
const bookBoundaryToken = '<|sep|>';

// Build corpus from file
function buildCorpus(content: string): Corpus {
    // Divide into words
    const words = content.split(/\s+/);
    // Divide into characters with word boundary at the end
    let corpus = words.map(word => [...word.split(''), wordBoundaryToken]);
    // Add file separator token
    corpus.push([bookBoundaryToken]);
    // Log progress
    console.log(`Built corpus`);
    return corpus;
}

// If corpus.txt exists, clear it to prevent writing twice
fs.writeFileSync('./models/corpus.txt', '');

// Iterate over all books
for (const bookTitle of bookTitles) {
    // Load book file
    fs.readFile(`./data/dickens/${bookTitle}.txt`, (err, data) => {
        if (err) {
            // Error handling if document can't be read
            console.error(`Error reading ${bookTitle}.txt: ${err}`);
            return;
        } else {
            // Log progress
            console.log(`Loading ${bookTitle}...`)
            // Convert book text to string
            const content = data.toString();
            // Build corpus from document
            const corpus: Corpus = buildCorpus(content);
            // Convert datatype of corpus
            const corpusText = JSON.stringify(corpus);

            // Save corpus to file
            fs.appendFile(`./models/corpus.txt`, corpusText, (err) => {
                if (err) {
                    console.error(`Error writing corpus: ${err}`);
                } else {
                    // Log success
                    console.log(`Saved file ${bookTitle}`);
                }
            })
        }
    });
}
