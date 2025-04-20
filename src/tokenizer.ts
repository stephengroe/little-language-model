import fs from 'fs';

type TokenList = Record<string, number>;

function buildCorpus(content: string, filename: string) {

  // Get unique characters
  const characterSet = new Set<string>(content); 

  // Build corpus of all words
  const words = content.split(/\s/);
  const corpus: string[][] = words.map(word => [...word.split(''), '</w>']);

  const stringifiedResult = `module.exports = ${JSON.stringify(corpus)}`;
  writeCorpusToFile(stringifiedResult, `${filename}-corpus`);
}

function openTokenListOrCreateNew() {
  fs.readFile(`./models/tokens.js`, (err, data) => {
    if (err) {
      console.log(`No existing token list, creating new...`);
      let tokenList: TokenList = {};

      return tokenList;
    } else {

    }
  })
}


function writeCorpusToFile(content: string, filename: string) {
  fs.writeFile(`./models/${filename}.js`, content, (err) => {
      if (err) {
          console.log(`File write error: ${err}`);
      }
      else {
          console.log(`Saved token list to file`);
      }
  });
}

export default buildTokenList;