# Word2Vec from Scratch (CBOW)

Implementation of Continuous Bag-of-Words using only NumPy.

## What it does

- Builds vocabulary from a text corpus
- Generates training data (center word + context word pairs)
- Two-layer neural network (embedding matrix → softmax)
- Trains via backpropagation and gradient descent
- After 50 epochs, plots the training loss curve
- Shows the most similar word to "learning" and the embedding for "machine"

## Run it

```bash
pip install -r requirements.txt
python word2vec.py
```

You'll see:
- A loss curve plot window
- Console output: most similar word to "learning" and the embedding vector for "machine"

## Requirements

```
numpy
matplotlib
```

Based on CS50AI coursework.
