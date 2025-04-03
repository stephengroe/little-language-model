const wordMap = require('./wordMap');
const readline = require('readline').createInterface({
  input: process.stdin,
  output: process.stdout
});

function getNextWord(word, wordMap) {
  const nextLikelyWords = Object.entries(wordMap[word]);

  // Get weight of all words and random number
  const totalWeight = nextLikelyWords.map(entry => entry[1]).reduce((acc, cur) => acc += cur);
  let randomNumber = Math.random() * totalWeight;

  // Select number based on random weight
  for (let [word, weight] of nextLikelyWords) {
    randomNumber -= weight;
    if (randomNumber <= 0) {
      return word;
    }
  }
}

function generateParagraph(initialWord, wordMap, max) {
  // Ensure word exists
  if (!wordMap[initialWord]) {
    console.log(`The word '${initialWord}' doesn't appear in our database. Using a placeholder instead.`);
    initialWord = 'Marley';
  }

  let numberGenerated = 0;
  let generatedWords = [initialWord];
  let word = initialWord;
  let done = false;

  while (done === false) {
    const newWord = getNextWord(word, wordMap);
    const stopCharacters = new Set([". ", "! ", "? "]);
    generatedWords.push(newWord);
    numberGenerated += 1;
    word = newWord;

    // So we finish on the end of a sentence
    if (numberGenerated >= max && stopCharacters.has(newWord)) {
      done = true;
    }
  }

  return generatedWords.join('');
}

function generateText() {
    readline.question('First word to generate:', word => {
      const generatedText = generateParagraph(word, wordMap, 100);
      console.log(`Your paragraph:
        ${generatedText}`);
      readline.close();
    });
}

generateText();
