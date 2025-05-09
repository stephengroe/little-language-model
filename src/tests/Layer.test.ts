import { Layer } from '../Layer';

describe('Layer module', () => {
  test('layer initializes array of nodes', () => {
    const layer = new Layer(5, 10);
    expect(layer.getNodes().length).toBe(5);
  });

  test('layer initializes nodes with weights from prior layer', () => {
    const layer = new Layer(5, 10);
    expect(layer.getNodes()[0].getWeights().length).toBe(10);
  });

  test('layer generates output vector', () => {
    const layer = new Layer(5, 10);
    const input = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    expect(layer.forward(input).length).toBe(5);
  });
});
