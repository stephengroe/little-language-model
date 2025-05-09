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
  for (let i = result.length; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}
