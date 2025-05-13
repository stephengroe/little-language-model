import { ActivationFunction } from './ActivationFunction';

// Rectified linear unit (ReLU)
export class ReLU implements ActivationFunction {
  apply(input: number): number {
    return Math.max(0, input);
  }

  derivative(input: number): number {
    return input > 0 ? 1 : 0;
  }
}
