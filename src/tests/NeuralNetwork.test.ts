import { NeuralNetwork } from '../NeuralNetwork';

describe('NeuralNetwork class', () => {
  test('generates array of layers, minus input layer', () => {
    const neuralNet = new NeuralNetwork([5, 10, 5]);
    expect(neuralNet.getLayers().length).toBe(2);
  });
});
