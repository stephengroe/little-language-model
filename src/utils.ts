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
  data: string
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
