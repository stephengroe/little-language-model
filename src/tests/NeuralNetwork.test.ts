import { NeuralNetwork } from '../NeuralNetwork';

describe('NeuralNetwork class', () => {
  test('generates array of layers, minus input layer', () => {
    const neuralNet = new NeuralNetwork([5, 10, 5]);
    expect(neuralNet.getLayers().length).toBe(2);
  });

  test('loss gradient is correct shape', () => {
    const neuralNet = new NeuralNetwork([5, 10, 5]);
    const predicted = [1, 0, 0];
    expect(neuralNet.getLossGradient(predicted, predicted)).toHaveLength(3);
  });

  test('weights change with learning', () => {
    const neuralNet = new NeuralNetwork([3, 10, 3]);
    const prevWeights = neuralNet.getLayers()[1].getWeights().flat();
    const loss = neuralNet.train([1, 0, 0], [1, 0, 0], 0.1);
    const newWeights = neuralNet.getLayers()[1].getWeights().flat();

    expect(newWeights).not.toEqual(prevWeights);
  });

  test('biases change with learning', () => {
    const neuralNet = new NeuralNetwork([3, 10, 3]);
    const prevBiases = neuralNet.getLayers()[1].getBiases();
    neuralNet.train([1, 0, 0], [1, 0, 0], 0.1);
    const newBiases = neuralNet.getLayers()[1].getBiases();

    expect(newBiases).not.toEqual(prevBiases);
  });

  test('cross-entropy loss works', () => {
    const neuralNet = new NeuralNetwork([5, 10, 5]);
    const a = neuralNet.loss([0.9, 0.1], 0);
    const b = neuralNet.loss([0.1, 0.9], 0);

    expect(a < b).toBe(true);
  });
});
