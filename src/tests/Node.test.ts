import { Node } from '../Node';
import { Identity } from '../ActivationFunction/Identity';

describe('Node class', () => {
  test('initializes weight based on input', () => {
    const node = new Node(5, new Identity());
    expect(node.getWeights().length).toBe(5);
  });

  test('generates bias', () => {
    const node = new Node(5, new Identity());
    expect(node.getBias()).toBeGreaterThanOrEqual(0);
    expect(node.getBias()).toBeLessThan(1);
  });

  test('handles forward computation', () => {
    const node = new Node(3, new Identity());
    node.setBias(1);
    node.setWeights([0.5, 0.25, 0.75]);
    const inputs = [1, 2, 3];

    expect(node.forward(inputs)).toBe(4.25);
  });

  test('handles backpropogation', () => {
    const node = new Node(3, new Identity());
    node.setBias(1);
    node.setWeights([0.5, 0.25, 0.75]);
    const inputs = [1, 2, 3];
    const lossGradient = 0.5;
    const learningRate = 0.1;
    node.forward(inputs);
    const result = node.backward(lossGradient, learningRate);

    expect(result).toEqual([0.225, 0.075, 0.3]);
    expect(node.getWeights()).toEqual([0.45, 0.15, 0.6]);
    expect(node.getBias()).toBe(0.95);
  });
});
