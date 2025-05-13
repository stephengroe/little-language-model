import { ActivationFunction } from './ActivationFunction';

// Identity function for leaving values unchanged
// (Particularly useful for output layer)
export class Identity implements ActivationFunction {
  apply(input: number): number {
    return input;
  }

  derivative(input: number): number {
    return input;
  }
}
