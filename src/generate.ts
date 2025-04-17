import readline from 'readline';

type wordMap = Record<string, Record<string, number>>;
const wordMap: wordMap = require('../models/wordMap');

const userInput = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function getNextWord(word:string, wordMap: wordMap) {
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

function generateParagraph(initialWord: string, wordMap: wordMap, max: number) {
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
        const newWord = getNextWord(word, wordMap) ?? ' ';
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

// Softmax function
function applySoftmax(weightedArray: number[], temperature: number = 1): number[] {
    const max = Math.max(...weightedArray);

    // Return argmax if temperature is zero (prevent divide-by-zero error)
    if (temperature === 0) {
        return weightedArray.map(logit => (logit === max ? 1 : 0));
    }

    // Subtracting max value for numerical stability
    const expNumbers = weightedArray.map(logit => Math.exp((logit - max) / temperature));
    const denominator = expNumbers.reduce((acc, curr) => acc += curr, 0);
    const softMaxValues = expNumbers.map(expNum => expNum / denominator);
    return softMaxValues;
}

function generateText() {
    userInput.question('First word to generate:', word => {
        const generatedText = generateParagraph(word, wordMap, 100);
        console.log(`Your paragraph:
        ${generatedText}`);
        userInput.close();
    });
}
generateText();