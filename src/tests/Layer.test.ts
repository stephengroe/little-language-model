import { Layer } from '../Layer';
import { Identity } from '../ActivationFunction/Identity';

describe('Layer module', () => {
  test('initializes array of nodes', () => {
    const layer = new Layer(5, 10, new Identity());
    expect(layer.getNodes().length).toBe(5);
  });

  test('throws error for input length of zero', () => {
    expect(() => new Layer(5, 0, new Identity())).toThrow(Error);
  });

  test('initializes nodes with weights from prior layer', () => {
    const layer = new Layer(5, 10, new Identity());
    expect(layer.getNodes()[0].getWeights().length).toBe(10);
  });

  test('generates output vector', () => {
    const layer = new Layer(5, 10, new Identity());
    const input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    expect(layer.forward(input).length).toBe(5);
  });

  test('returns node weights', () => {
    const layer = new Layer(5, 10, new Identity());
    expect(layer.getWeights()).toHaveLength(5);
  });

  test('returns node biases', () => {
    const layer = new Layer(5, 10, new Identity());
    expect(layer.getBiases()).toHaveLength(5);
  });
});
