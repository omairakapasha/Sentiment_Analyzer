# Sentiment Analysis System

A comprehensive sentiment analysis tool built with NLTK, spaCy, and scikit-learn that analyzes the sentiment of user input text.

## Project Structure

```
Sentiment Analysis/
├── main.py                 # Main application script
├── app.py                  # For Frontend to Backend Connection and apis
├── training_data.py        # Training dataset for directly running main.py in cli
├── training_data.csv, sample_data_large.csv        # Training dataset for different modals in GUI
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── models folde    # Trained models
```

## Features

- **Multiple NLP Libraries**: Combines NLTK, spaCy, and scikit-learn for robust text processing
- **Text Preprocessing**: Advanced preprocessing using both NLTK and spaCy methods
- **Machine Learning Models**: Supports Logistic Regression and Naive Bayes classifiers
- **Interactive Interface**: Real-time sentiment analysis on user input
- **Model Persistence**: Save and load trained models
- **Confidence Scores**: Provides probability scores for predictions
- **Entity Recognition**: Uses spaCy for additional text analysis

## Installation

## Important Note use python 3.11.5 for better compatibility

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Download the spaCy language model (this will be done automatically on first run, but you can also do it manually):

```bash
python -m spacy download en_core_web_sm
```

## Usage

Run the server:

```bash
python app.py
```

The server will:
1. Be accessible at 127.0.0.1:5000
2. We can train modal by uploading csv files from here and can select any trained modal
3. Enter text and hit analyze for result
4. Type Ctrl - C in powershell to terminate the server

## Example

```
Enter text to analyze: I absolutely love this product! It's amazing!

────────────────────────────────────────────────────────────
Input Text: I absolutely love this product! It's amazing!
────────────────────────────────────────────────────────────
Predicted Sentiment: POSITIVE
Confidence: 89.23%
────────────────────────────────────────────────────────────
```

## Components

### NLTK
- Text preprocessing and tokenization
- Stopword removal
- WordNet lemmatization

### spaCy
- Advanced tokenization and lemmatization
- Named Entity Recognition
- Part-of-speech tagging
- Sentence segmentation

### scikit-learn
- TF-IDF vectorization
- Machine learning classifiers (Logistic Regression, Naive Bayes)
- Model evaluation and metrics

## Customization

### Using Your Own Dataset

Edit `training_data.py` to add your own training examples:

```python
# In training_data.py, add your examples to the respective functions:

def get_positive_examples():
    return [
        "Your positive example 1",
        "Your positive example 2",
        # ... more examples
    ]

def get_negative_examples():
    return [
        "Your negative example 1",
        "Your negative example 2",
        # ... more examples
    ]

def get_neutral_examples():
    return [
        "Your neutral example 1",
        "Your neutral example 2",
        # ... more examples
    ]
```

The current dataset includes **600 examples** (200 positive, 200 negative, 200 neutral).

### Changing the Model Type

Initialize the analyzer with a different model:

```python
# For Naive Bayes
analyzer = SentimentAnalyzer(model_type='naive_bayes')

# For Logistic Regression (default)
analyzer = SentimentAnalyzer(model_type='logistic')
```

### Preprocessing Methods

Choose between NLTK and spaCy preprocessing:

```python
# Using spaCy (default)
analyzer.train(texts, labels, preprocess_method='spacy')

# Using NLTK
analyzer.train(texts, labels, preprocess_method='nltk')
```
