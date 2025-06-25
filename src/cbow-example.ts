import {
  random,
  matrix,
  multiply,
  transpose,
  add,
  subtract,
  dotMultiply,
  reshape,
} from 'mathjs';

// Parameters
const vocab = ['i', 'like', 'cats', 'dogs']; // toy vocabulary
const vocabSize = vocab.length;
const embeddingSize = 5;
const learningRate = 0.05;

// One-hot encoding helper
function oneHot(index: number, size: number): number[] {
  return Array.from({ length: size }, (_, i) => (i === index ? 1 : 0));
}

// Word to index
const wordToIndex = Object.fromEntries(vocab.map((word, idx) => [word, idx]));

// Index to word
const indexToWord = Object.fromEntries(vocab.map((word, idx) => [idx, word]));

// Initialize weights
let W1 = matrix(
  Array.from({ length: vocabSize }, () =>
    Array.from({ length: embeddingSize }, () => random(-1, 1))
  )
);
let W2 = matrix(
  Array.from({ length: embeddingSize }, () =>
    Array.from({ length: vocabSize }, () => random(-1, 1))
  )
);

// Softmax function
function softmax(vec: number[]): number[] {
  const max = Math.max(...vec);
  const exp = vec.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

// Training example
const context = ['i', 'cats']; // input
const target = 'like'; // label

// Convert context to input vector (average of one-hots)
const contextIndices = context.map((word) => wordToIndex[word]);
const contextVectors = contextIndices.map((idx) => oneHot(idx, vocabSize));

const x = contextVectors
  .reduce((a, b) => a.map((val, i) => val + b[i]))
  .map((val) => val / context.length); // average

// Forward pass
const hidden = multiply(x, W1); // x • W1 => hidden layer
const output = multiply(hidden, W2); // hidden • W2 => scores
const yPred = softmax(output.valueOf() as number[]);

// Ground truth
const yTrue = oneHot(wordToIndex[target], vocabSize);

// Loss (cross-entropy)
const loss = -Math.log(yPred[wordToIndex[target]]);
console.log('Loss:', loss);

// Backpropagationx
const e = yPred.map((yp, i) => yp - yTrue[i]); // error
const eMatrix = [e]; // shape [1 x vocabSize]

const hiddenT = transpose([hidden.valueOf() as number[]]);
const dW2 = multiply(hiddenT, eMatrix); // [embedding x 1] • [1 x vocab]

const W2T = transpose(W2);
const dHidden = multiply(e, W2T); // [1 x embedding]

const xT = transpose([x]);
const dW1 = multiply(xT, [dHidden]); // [vocab x 1] • [1 x embedding]

// Gradient descent update
W1 = subtract(W1, multiply(dW1, learningRate));
W2 = subtract(W2, multiply(dW2, learningRate));

// Output updated embeddings
console.log('\nWord embeddings (W1):');
vocab.forEach((word, i) => {
  console.log(`${word}: ${(W1.valueOf() as number[][])[i]}`);
});
