# Fake News Detection with LSTM

A deep learning project for detecting fake news using LSTM (Long Short-Term Memory) neural networks. This binary classification model analyzes news articles and predicts whether they are authentic or fake.

## Project Overview

This project implements a fake news detection system using natural language processing (NLP) and deep learning techniques. The model is trained on a dataset containing both fake and true news articles, learning to distinguish between them based on textual patterns and features.

**Key Features:**
- Binary classification of news articles (Fake/True)
- LSTM-based deep learning architecture
- Text preprocessing and cleaning pipeline
- Model evaluation with comprehensive metrics
- Prediction function for new articles

## Requirements

### Dependencies

```python
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow>=2.0
```

### Installation

Install the required packages using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
```

## Dataset

The project uses two CSV files:
- `Fake.csv` - Contains fake news articles
- `True.csv` - Contains authentic news articles

Each dataset includes the following columns:
- `title` - Article headline
- `text` - Article content
- `subject` - Article category
- `date` - Publication date

## Preprocessing Pipeline

### 1. Text Cleaning

The preprocessing function performs the following operations:

```python
def preprocess_text(text):
    - Convert text to lowercase
    - Remove URLs, emails, and numbers
    - Remove special characters and punctuation
    - Tokenization and word splitting
```

### 2. Feature Engineering

- **Content Creation**: Combines title and text for better context
- **Tokenization**: Converts text to numerical sequences
- **Vocabulary Size**: 10,000 most frequent words
- **Sequence Length**: Fixed at 200 tokens with padding

### 3. Data Preparation

- Train-test split: 80-20 ratio
- Stratified sampling to maintain class balance
- Padding sequences for uniform input length

## Model Architecture

### LSTM Network Structure

```
Sequential Model:
├── Embedding Layer (5000 words, 64 dimensions)
├── SpatialDropout1D (0.2)
├── LSTM Layer (64 units, dropout=0.2, recurrent_dropout=0.2)
├── Dense Layer (32 units, ReLU activation)
├── Dropout (0.3)
└── Dense Layer (1 unit, Sigmoid activation)
```

### Model Configuration

- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy
- **Total Parameters**: ~387K trainable parameters

## Training Setup

### Hyperparameters

| Parameter           | Value |
| ------------------- | ----- |
| Batch Size          | 128   |
| Epochs              | 20    |
| Validation Split    | 10%   |
| Max Sequence Length | 200   |
| Embedding Dimension | 64    |
| Vocabulary Size     | 10,000|

### Callbacks

- **Early Stopping**:
  - Monitor: Validation loss
  - Patience: 3 epochs
  - Restores best weights

- **ReduceLROnPlateau**:
  - Reduces learning rate when validation loss plateaus
  - Factor: 0.5
  - Patience: 2 epochs
  - Min LR: 0.00001

## Results

### Model Performance

The model achieves high accuracy on the test set with the following metrics:

- **Test Accuracy**: ~99%+
- **Precision**: High for both classes
- **Recall**: High for both classes
- **F1-Score**: Balanced performance

### Visualizations

The project includes:
1. **Training History Plots**:
   - Training vs Validation Accuracy
   - Training vs Validation Loss

2. **Confusion Matrix**:
   - Visual representation of classification results
   - True Positives, True Negatives, False Positives, False Negatives

### Classification Report

```
              precision    recall  f1-score   support

        Fake       0.99      0.99      0.99      ~4500
        True       0.99      0.99      0.99      ~4500

    accuracy                           0.99      ~9000
```

## Usage

### Training the Model

```python
# Load and preprocess data
df['content'] = df['title'] + ' ' + df['text']
df['cleaned_content'] = df['content'].apply(preprocess_text)

# Tokenize and pad sequences
tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_content'])
sequences = tokenizer.texts_to_sequences(df['cleaned_content'])
X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Train the model
history = model.fit(X_train, y_train,
                    batch_size=128,
                    epochs=5,
                    validation_split=0.1)
```

### Making Predictions

```python
def predict_news(text):
    """Predict whether a news article is fake or true"""
    cleaned = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        return f"TRUE news (confidence: {prediction:.2%})"
    else:
        return f"FAKE news (confidence: {(1-prediction):.2%})"

# Example
text = "Breaking news: Scientists discover new planet in solar system"
print(predict_news(text))
```

## Project Structure

```
├── final.ipynb          # Main Jupyter notebook with complete pipeline
├── README.md            # Project documentation
└── datasets/
    ├── Fake.csv         # Fake news dataset
    └── True.csv         # True news dataset
```
