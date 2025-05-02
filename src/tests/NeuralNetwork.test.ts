import { NeuralNetwork } from '../NeuralNetwork';

describe('NeuralNetwork class', () => {
  test('generates array of layers', () => {
    const neuralNet = new NeuralNetwork(2, 10);
    expect(neuralNet.getLayers().length).toBe(2);
  });
});
