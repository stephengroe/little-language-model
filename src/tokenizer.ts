import fs from 'fs';

type TokenList = Record<string, number>;

function buildTokenList(content: string, filename: string) {
  let tokenList: TokenList = {};

  for (const char of content) {
    if (tokenList[char]) {
      tokenList[char] += 1;
    } else {
      tokenList[char] = 1;
    }
  }

  const stringifiedResult = `module.exports = ${JSON.stringify(tokenList)}`;
  writeTokenListToFile(stringifiedResult, `${filename}-tokens`);
}

function writeTokenListToFile(content: string, filename: string) {
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