import { readFile, writeFile } from 'fs/promises';

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
  private barWidth = 25;

  constructor(total: number) {
    this.total = total;
    this.progress = 0;

    this.update(0);
  }

  update(newProgress: number, message?: string) {
    if (newProgress > 0) {
      this.progress = newProgress;
    }
    const percent = Math.round((this.progress / this.total) * 100);

    const text = message ? ` ${message}` : '';

    this.render(Math.min(percent, 100), text); // Prevent 100+ percent

    if (this.progress >= this.total) {
      this.done();
    }
  }

  render(percent: number, message: string) {
    if (process.stdout.isTTY) {
      process.stdout.clearLine(1);
      process.stdout.cursorTo(0);
      process.stdout.write(
        `${percent}% ${this.drawBar()} [${this.progress}/${this.total}]${message}`
      );
    }
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
    if (process.stdout.isTTY) {
      process.stdout.write(`\n`);
    }
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

// Matrix-to-matrix multiplication
export function matrixMultiply(x: number[][], y: number[][]): number[][] {
  const rowsX = x.length;
  const colsX = x[0].length;
  const rowsY = y.length;
  const colsY = y[0].length;

  if (colsX !== rowsY) {
    throw new Error(
      `Mismatched matrix sizes: X columns (${colsX}) =/= Y rows (${rowsY})`
    );
  }

  const result: number[][] = Array.from({ length: rowsX }, () => {
    return new Array(colsY).fill(0);
  });

  for (let i = 0; i < rowsX; i++) {
    for (let j = 0; j < colsY; j++) {
      let sum = 0;

      for (let k = 0; k < colsX; k++) {
        sum += x[i][k] * y[k][j];
      }
      result[i][j] = sum;
    }
  }

  return result;
}

// Matrix-to-vector multiplication
export function vectorMatrixMultiply(vec: number[], mat: number[][]): number[] {
  const rows = mat.length;
  const cols = mat[0].length;

  if (rows !== vec.length) {
    throw new Error(
      `Mixmatched matrix/vector: rows (${rows}) =/= vector length (${vec.length})`
    );
  }

  const result = new Array(cols).fill(0);

  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      result[i] += mat[j][i] * vec[j];
    }
  }

  return result;
}

export function outerProduct(vecA: number[], vecB: number[]): number[][] {
  return vecA.map((a) => vecB.map((b) => a * b));
}

export function transpose(matrix: number[][]): number[][] {
  return matrix[0].map((_, columnIndex) =>
    matrix.map((row) => row[columnIndex])
  );
}
