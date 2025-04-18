import readline from 'readline';

type wordMap = Record<string, Record<string, number>>;
type token = {
    token: string,
    weight: number,
};

const wordMap: wordMap = require('../models/wordMap');

const userInput = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

function getNextWord(word:string, wordMap: wordMap) {
    // Convert words to token data type
    const nextLikelyWords: token[] = Object.entries(wordMap[word]).map(word => {
        const newToken: token = {
            token: word[0],
            weight: word[1],
        }
        return newToken;
    });

    const softmaxTokenWeights = applySoftmax(nextLikelyWords);

    // Choose entry
    let randomNumber = Math.random();

    // Select number based on random weight
    for (let {token, weight} of softmaxTokenWeights) {
        randomNumber -= weight;
        if (randomNumber <= 0) {
            return token;
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
function applySoftmax(tokenArray: token[], temperature: number = 1): token[] {
    const max = Math.max(...tokenArray.map(token => token.weight));
    const safeTemp = Math.max(temperature, 1e-6); // Prevent divide by zero errors

    // Subtracting max value for numerical stability
    const weights: number[] = tokenArray.map(logit => logit.weight);
    const expNumbers = weights.map(weight => Math.exp((weight - max) / safeTemp));
    const denominator = expNumbers.reduce((acc, curr) => acc += curr, 0);
    const softMaxValues = expNumbers.map(expNum => expNum / denominator);

    const weightedTokens: token[] = tokenArray.map((token, i) => {
        return {
            token: token.token,
            weight: softMaxValues[i],
        }
    })

    return weightedTokens;
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