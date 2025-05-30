# Sentiment Analysis using Bag-of-Words (BoW) Model

IMDB movie review sentiment classifier implemented with PyTorch using Bag-of-Words and embedding approaches.

## Features

- **Text Preprocessing Pipeline**:
  - HTML cleaning, non-ASCII removal, number replacement
  - Tokenization, punctuation/whitespace removal
  - Lemmatization (word + verb) and stopword removal
  - Custom text normalization function

- **Vectorization**:
  - Implemented BoW from scratch:
    - Vocabulary construction
    - Word-to-index mapping with `<PAD>` and `<UNK>` tokens
    - Custom vectorization functions
  - Scikit-learn's CountVectorizer implementation

- **Models**:
  ```python
  1. BoWClassifier (PyTorch):
     - 3-layer NN with ReLU activations
     - Sigmoid output for binary classification

  2. EmbeddingClassifier (PyTorch):
     - Embedding layer + flattened features
     - 3 fully-connected layers
