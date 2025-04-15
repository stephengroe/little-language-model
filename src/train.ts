import fs from 'fs';

// List of Dickens books for source data
const books = [
    'a-christmas-carol.txt',
    'american-notes.txt',
    'bleak-house.txt',
    'david-copperfield.txt',
    'hard-times.txt',
    'little-dorrit.txt',
    'nicholas-nickleby.txt',
    'our-mutual-friend.txt',
    'the-old-curiosity-shop.txt',
    'the-pickwick-papers.txt',
];

// Get content from text file
function getWordMap(documentTitle: string, callback: Function) {
    fs.readFile(documentTitle, (err, content) => {
        if (err) {
            return callback(err);
        }
        else {
            // Replace line breaks with spaces, break into words and punctuation
            const words = content.toString().split('\r\n').join(' ').split(/\b(?!\s)/);
            const wordMap = buildWordMap(words);
            callback(null, wordMap);
        }
    });
}

type wordMap = Record<string, Record<string, number>>;

function buildWordMap(wordList: string[]) {
    let wordMap: wordMap = {};
    for (let i = 0; i < wordList.length; i++) {
        const currentWord = wordList[i];
        const nextWord = wordList[i + 1];

        // Add current word to map
        if (!wordMap[currentWord]) {
            wordMap[currentWord] = {};
        }
        // Add next word probability
        const nextLikelyWord = wordMap[currentWord];
        if (!nextLikelyWord[nextWord]) {
            nextLikelyWord[nextWord] = 1;
        }
        else {
            nextLikelyWord[nextWord] = nextLikelyWord[nextWord] + 1;
        }
    }
    return wordMap;
}

// Import model, or create new one
console.log(`Generating...`);

for (const book of books) {
    getWordMap(`./data/dickens/${book}`, (err: unknown, result: wordMap) => {
        if (err) {
            console.log(`Error: ${err}`);
        }
        else {
            console.log(`Saving ${book}...`);
            const fileContent = `module.exports = ${JSON.stringify(result)}`;
            fs.writeFile('./models/wordMap.js', fileContent, (fsErr) => {
                if (fsErr) {
                    console.log(`File write error: ${fsErr}`);
                }
                else {
                    console.log(`Saved`);
                }
            });
        }
    });
}
