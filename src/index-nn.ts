import { shuffleArray, round } from './utils';
import { NeuralNetwork } from './NeuralNetwork';

type TrainingData = {
  input: number[];
  expected: number[];
}[];

// Parameters
const layerDepth = 16;
const learningRate = 0.1;
const epochs = 100;

const identityTraining: TrainingData = [
  {
    input: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    expected: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  },
  {
    input: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    expected: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
  },
  {
    input: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    expected: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
  },
  {
    input: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    expected: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
  },
  {
    input: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    expected: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
  },
  {
    input: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    expected: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
  },
  {
    input: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    expected: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
  },
  {
    input: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    expected: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
  },
  {
    input: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    expected: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
  },
  {
    input: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    expected: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  },
];

const XORTraining: TrainingData = [
  { input: [0, 0], expected: [1, 0] },
  { input: [1, 0], expected: [0, 1] },
  { input: [0, 1], expected: [0, 1] },
  { input: [1, 1], expected: [1, 0] },
];

// Instantiate neural net
console.log(`Building neural net...`);
const layers = [
  identityTraining[0].input.length,
  layerDepth,
  identityTraining[0].expected.length,
];
const neuralNet = new NeuralNetwork(layers);

console.log(`Training neural net...`);
let finishedEpochs = 0;

while (finishedEpochs < epochs) {
  const shuffledTrainingData = shuffleArray(identityTraining);
  finishedEpochs += 1;

  let loss = 0;

  for (const { input, expected } of shuffledTrainingData) {
    loss += neuralNet.train(input, expected, learningRate);
  }

  console.log(
    `Epoch ${finishedEpochs}: Av Loss: ${round(loss / shuffledTrainingData.length)}`
  );
}

console.log(`Finished training!`);
