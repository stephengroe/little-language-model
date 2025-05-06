import { Node } from '../Node';

describe('Neuron class', () => {
  test('initializes weight based on input', () => {
    const neuron = new Node(5);
    expect(neuron.getWeights().length).toBe(5);
  });

  test('generates bias', () => {
    const neuron = new Node(5);
    expect(neuron.getBias()).toBeGreaterThanOrEqual(0);
    expect(neuron.getBias()).toBeLessThan(1);
  });
});
