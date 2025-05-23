export type TrainingData = {
  input: number | number[];
  target: number | number[];
};

export type TrainingBatch = {
  batchInputs: (number | number[])[];
  batchTargets: (number | number[])[];
};

export class DataLoader {
  private data: TrainingData[];

  constructor(data: TrainingData[]) {
    this.data = data;
  }

  // Shuffle array with Fisher-Yates algorithm
  shuffle(): void {
    for (let i = this.data.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.data[i], this.data[j]] = [this.data[j], this.data[i]];
    }
  }

  batch(batchSize: number, shuffle?: boolean): TrainingBatch[] {
    if (shuffle) {
      this.shuffle();
    }

    const batches: TrainingBatch[] = [];
    for (let i = 0; i < this.data.length; i += batchSize) {
      const dataSlice = this.data.slice(i, i + batchSize);
      const newBatch: TrainingBatch = { batchInputs: [], batchTargets: [] };

      dataSlice.forEach(({ input, target }) => {
        newBatch.batchInputs.push(input);
        newBatch.batchTargets.push(target);
      });
      batches.push(newBatch);
    }
    return batches;
  }
}
