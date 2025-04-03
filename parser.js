const fs = require('fs');
const wordMap = require('./wordMap');

function getWordMap(documentTitle, callback) {
  fs.readFile(documentTitle, (err, content) => {
    if (err) {
      return callback(err);
    } else {
      // Replace line breaks with spaces, break into words and punctuation
      const words = content.toString().split('\r\n').join(' ').split(/\b(?!\s)/);
      const wordMap = buildWordMap(words);

      callback(null, wordMap);
    }
  });
}

function buildWordMap(wordList) {

  for (let i=0; i<wordList.length; i++) {
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
    } else {
      nextLikelyWord[nextWord] = nextLikelyWord[nextWord] + 1;
    }
  }

  return wordMap;
}

console.log(`Generating...`)

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

for (const book of books) {
  getWordMap(`./dickens/${book}`, (err, result) => {
    if (err) {
      console.log(`Error: ${err}`);
    } else {
      console.log(`Saving ${book}...`)
      const fileContent = `module.exports = ${JSON.stringify(result)}`;
      fs.writeFile('./wordMap.js', fileContent, (fsErr) => {
        if (fsErr) {
          console.log(`File write error: ${fsErr}`);
        } else {
          console.log(`Saved`);
        }
      });
    }
  });
}



