# Little Language Model

> **A GPT-style language model written from scratch in TypeScript—no ML libraries.**

I’m building a small language model from first principles. Currently, it functions as a tokenization tool and CBOW-style word embedding model.

## Current features

- Tokenization
  - Converts a corpus into tokens using Byte-Pair Encoding (BPE) with controls over vocabulary size
- Word embeddings
  - Generates word embeddings using Word2Vec-style Continuous-Bag-of-Words (CBOW) architecture
  - Handles subsampling of frequent tokens and filtering of rare words
  - Trains on a neural matrix architecture using batched gradient descent, all built in pure TypeScript with no ML libraries
  - Allows nearest-neighbor querying of embeddings

## Upcoming features

- Attention model
- Simple transformer architecture
- Browser-based inference

## How to Run

### 1. Install dependencies

`npm install`

### 2. Download input data

This model expects plain text `.txt` files in `./data/`. In the current model, I'm using the complete writings of Charles Dickens, downloaded from [Project Gutenberg](https://www.gutenberg.org/) and lightly modified to remove boilerplate text.

### 3. Configure training

Update `config.ts` to set vocabulary size, embedding dimensions, learning rate, CBOW context window, and input texts.

It currently also generates a list of nearest neighbors for selected tokens. You can add specific tokens, as well as the number of total embeddings to sample.

### 4. Train the model

`npm run train`

The model will log nearest neighbors to specified and randomly sampled tokens to the console. It saves all generated embeddings as a raw binary .bin file (Float32Array), where each token's vector is stored in a fixed-length row.

Training across the full Dickens corpus with `vocabularySize=10_000` and `vectorSize=64` takes ~15 hours on a MacBook Air (M1, 16GB RAM).

## Sample output

Neighbors suggest semantic clustering is working well. After training on two epochs of the works of Dickens, these are nearest neighbors to `"man</w>"` (sorted by cosine similarity):

```
  1. gentleman</w> (0.83)
  2. man,</w> (0.77)
  3. woman</w> (0.72)
  4. lady</w> (0.71)
  5. person</w> (0.68)
```

## Loss curve

Loss decreases consistently, showing convergence on meaningful embeddings. Shown is average loss per epoch, trained on _A Christmas Carol_ with `vocabularySize=5_000` and `vectorSize=64`:

```
Epoch 1: loss = 6.2372
Epoch 2: loss = 5.8554
Epoch 3: loss = 5.4417
Epoch 4: loss = 4.935
Epoch 5: loss = 4.2933
Epoch 6: loss = 3.5365
```

## License

MIT License. Input text should be sourced from public domain content.

---

📫 **I'm actively exploring engineering opportunities** where I can bring this kind of work into production. Feel free to [reach out](stephen@stephengroe.com) or connect on [LinkedIn](https://www.linkedin.com/in/stephengroe).
