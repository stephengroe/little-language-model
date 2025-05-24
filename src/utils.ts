import { readFile, writeFile } from 'fs/promises';
import { prependListener } from 'process';

// Read from disk
export async function loadFile(
  filePath: string,
  fileName: string
): Promise<string> {
  let data: Buffer;

  try {
    data = await readFile(`${filePath}${fileName}`);
  } catch (err) {
    throw new Error(`Cannot read ${filePath}${fileName}: ${err}`);
  }

  const convertedData = data.toString();
  return convertedData;
}

// Write to disk
export async function saveFile(
  filePath: string,
  fileName: string,
  data: string | Buffer
) {
  const fullPath = `${filePath}${fileName}`;
  try {
    await writeFile(fullPath, data);
  } catch (err) {
    throw new Error(`Unable to save ${filePath}${fileName}: ${err}`);
  }
}

// Format timestamps
export function formatTimestampAsISO(date: Date) {
  const year = date.getFullYear();
  const month = formatAsTwoDigits(date.getMonth() + 1);
  const day = formatAsTwoDigits(date.getDate());
  const hour = formatAsTwoDigits(date.getHours());
  const minute = formatAsTwoDigits(date.getMinutes());
  const second = formatAsTwoDigits(date.getSeconds());

  function formatAsTwoDigits(number: number) {
    return number.toString().padStart(2, '0');
  }

  return [year, month, day, hour, minute, second].join('');
}

// Shuffle array with Fisher-Yates algorithm
export function shuffleArray<T>(arr: T[]): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export function softmax(input: number[], temperature: number = 1): number[] {
  // Prevent divide by zero errors
  const safeTemp = Math.max(temperature, 0e-6);
  // Subtract max for numerical stability
  const max = Math.max(...input);

  const adjustedInput = input.map((weight) =>
    Math.exp((weight - max) / safeTemp)
  );
  const denominator = adjustedInput.reduce((acc, curr) => (acc += curr), 0);

  return adjustedInput.map((inputNum) => inputNum / denominator);
}

// Get cosine similarity
export function getCosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error(
      `Vectors must be of same length (received ${a.length} and ${b.length})`
    );
  }

  const dotProduct = a.reduce((acc, cur, index) => {
    return acc + cur * b[index];
  }, 0);

  const sizeA = Math.sqrt(
    a.reduce((acc, cur) => {
      return acc + Math.pow(cur, 2);
    }, 0)
  );

  const sizeB = Math.sqrt(
    b.reduce((acc, cur) => {
      return acc + Math.pow(cur, 2);
    }, 0)
  );

  return dotProduct / (sizeA * sizeB);
}

// Round a large decimal
export function round(input: number, places: number = 2): number {
  return Math.round(input * 10 ** places) / 10 ** places;
}

// Euclidian norm of vector
export function norm(vector: number[]): number {
  return Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
}

// Progress loading bar
export class ProgressBar {
  private total: number;
  private progress: number;
  private barWidth = 50;

  constructor(total: number) {
    this.total = total;
    this.progress = 0;

    this.update(0);
  }

  update(newProgress: number) {
    if (newProgress > 0) {
      this.progress = newProgress;
    }
    const percent = Math.round((this.progress / this.total) * 100);
    this.render(Math.min(percent, 100)); // Prevent 100+ percent

    if (this.progress >= this.total) {
      this.done();
    }
  }

  render(percent: number) {
    process.stdout.clearLine(1);
    process.stdout.cursorTo(0);
    process.stdout.write(
      `${percent}% ${this.drawBar()} [${this.progress}/${this.total}]`
    );
  }

  drawBar() {
    const done = Math.max(
      Math.floor(this.progress / (this.total / this.barWidth)),
      0
    );
    const undone = this.barWidth - done;

    return `${'█'.repeat(done)}${'▒'.repeat(undone)}`;
  }

  done() {
    process.stdout.write(`\n`);
  }
}

// Convert index to one-hot
export function toOneHot(
  index: number | number[],
  vectorSize: number
): number[] {
  const oneHot = Array.from({ length: vectorSize }, () => 0);

  if (Array.isArray(index)) {
    index.forEach((i) => {
      oneHot[i] += 1 / index.length; // return averaged one-hot
    });
  } else {
    oneHot[index] = 1;
  }

  return oneHot;
}
