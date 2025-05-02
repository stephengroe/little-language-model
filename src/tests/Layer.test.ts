import { Layer } from '../Layer';

describe('Layer module', () => {
  test('layer initializes array of neurons', () => {
    const layer = new Layer(5);
    expect(layer.getNeurons().length).toBe(5);
  });

  test('layer generates output vector', () => {
    const layer = new Layer(5);
    const input = [0, 1, 2, 3, 4];
    expect(layer.forward(input).length).toBe(5);
  });
});
