export const tokenizerConfig = {
  vocabularySize: 10_000,
  inputTextDirectory: './data/dickens/',
  inputTexts: [
    'a-christmas-carol',
    'american-notes',
    'bleak-house',
    'david-copperfield',
    'hard-times',
    'little-dorrit',
    'nicholas-nickleby',
    'our-mutual-friend',
    'the-old-curiosity-shop',
    'the-pickwick-papers',
  ],
  outputFolder: './output/tokenizing-benchmarks',
};
