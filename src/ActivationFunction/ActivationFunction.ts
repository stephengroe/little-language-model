export interface ActivationFunction {
  apply(input: number): number;
  derivative(input: number): number;
}
