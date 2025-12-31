"""
Sentiment Analysis Script using NLTK, spaCy, and scikit-learn
This script performs sentiment analysis on user input text.
"""

import re
import pickle
import os
import sys
import subprocess
from pathlib import Path

import pandas as pd

# NLTK imports
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# spaCy imports
import spacy

# scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Import training data
from training_data import get_sample_data


def load_training_dataset():
    """
    Load labeled text data from CSV if available; otherwise fall back to the
    small built-in sample set. Accepts either the existing misspelled file name
    or the correctly spelled version so users do not need to rename files.
    """
    csv_candidates = [
        Path("uploads") / "traning_data.csv",  # existing file name in repo
        Path("uploads") / "training_data.csv",
    ]

    for csv_path in csv_candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if {"text", "label"}.issubset(df.columns):
                texts = df["text"].astype(str).tolist()
                labels = df["label"].astype(str).tolist()
                print(f"Loaded {len(texts)} rows from {csv_path}")
                return texts, labels

            print(
                f"Found {csv_path} but it does not have required columns 'text' and 'label'; "
                "falling back to sample data."
            )

    print("No CSV training data found; using built-in sample set.")
    return get_sample_data()


def ensure_spacy_model(model_name: str = "en_core_web_sm", model_version: str = "3.7.1"):
    """
    Make sure the requested spaCy model is present. Falls back to a direct wheel
    install if the standard downloader cannot resolve a version (which otherwise
    results in a 404 like the one seen in the PowerShell output).
    """
    try:
        return spacy.load(model_name)
    except OSError:
        print(f"Downloading spaCy language model '{model_name}'...")
        try:
            from spacy.cli import download

            download(model_name)
        except Exception as download_err:
            print("Standard download failed, trying direct wheel installation...")
            wheel_url = (
                f"https://github.com/explosion/spacy-models/releases/download/"
                f"{model_version}/{model_name}-{model_version}-py3-none-any.whl"
            )
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", wheel_url])
            except Exception as install_err:
                raise RuntimeError(
                    f"Unable to install spaCy model '{model_name}'. "
                    f"Download error: {download_err} | Direct install error: {install_err}"
                ) from install_err

        return spacy.load(model_name)

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')


class SentimentAnalyzer:
    
    def __init__(self, model_type='logistic'):
        
        self.model_type = model_type
        self.pipeline = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load spaCy model with resilient installer
        self.nlp = ensure_spacy_model('en_core_web_sm')
    
    def preprocess_nltk(self, text):
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = text.split()
        
        # Negation words to preserve
        negations = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 
                     'nowhere', 'hardly', 'scarcely', 'barely', 'dont', 
                     'doesnt', 'didnt', 'wont', 'wouldnt', 'shouldnt', 'cant', 'couldnt'}
        
        # Remove stopwords but keep negations, then lemmatize
        tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens 
            if word not in self.stop_words or word in negations
        ]
        
        return ' '.join(tokens)
    
    def preprocess_spacy(self, text):
        
        # Process text with spaCy
        doc = self.nlp(text.lower())
        
        # Negation words to preserve
        negations = {'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 
                     'nowhere', 'hardly', 'scarcely', 'barely', "n't", 'dont', 
                     'doesnt', 'didnt', 'wont', 'wouldnt', 'shouldnt', 'cant', 'couldnt'}
        
        tokens = []
        for token in doc:
            # Keep negation words
            if token.text in negations or token.lemma_ in negations:
                tokens.append(token.lemma_)
            # Keep other meaningful words (not stopwords, punctuation)
            elif not token.is_stop and not token.is_punct and token.is_alpha:
                tokens.append(token.lemma_)
        
        return ' '.join(tokens)
    
    def preprocess_text(self, text, method='spacy'):
        
        if method == 'nltk':
            return self.preprocess_nltk(text)
        else:
            return self.preprocess_spacy(text)
    
    def train(self, texts, labels, test_size=0.15, preprocess_method='spacy'):
        
        print(f"Preprocessing {len(texts)} documents using {preprocess_method}...")
        processed_texts = [self.preprocess_text(text, preprocess_method) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size, random_state=42
        )
        
        # Create pipeline
        if self.model_type == 'naive_bayes':
            classifier = MultinomialNB(alpha=0.1)
        else:
            classifier = LogisticRegression(
                max_iter=2000, 
                random_state=42,
                C=2.0, 
                solver='lbfgs',
                class_weight='balanced' 
            )
        
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=8000, 
                ngram_range=(1, 3), 
                min_df=2,  
                sublinear_tf=True 
            )),
            ('classifier', classifier)
        ])
        
        # Train model
        print(f"Training {self.model_type} model...")
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    def predict(self, text, preprocess_method='spacy'):
        
        if self.pipeline is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        processed_text = self.preprocess_text(text, preprocess_method)
        prediction = self.pipeline.predict([processed_text])[0]
        
        # Predicted probability
        proba = self.pipeline.predict_proba([processed_text])[0]
        confidence = max(proba)
        
        return prediction, confidence
    
    def save_model(self, filepath=Path("models") / "sentiment_model.pkl"):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.pipeline, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath=Path("models") / "sentiment_model.pkl"):
        filepath = Path(filepath)
        # Some legacy pickles reference the private module name `numpy._core`.
        # Modern NumPy exposes `numpy.core`, so we register a compatibility
        # alias to keep pickle.load from failing with ModuleNotFoundError.
        if "numpy._core" not in sys.modules:
            import numpy.core as _np_core
            sys.modules["numpy._core"] = _np_core
        with open(filepath, 'rb') as f:
            self.pipeline = pickle.load(f)
        print(f"Model loaded from {filepath}")


def main():
    print("=" * 60)
    print("Sentiment Analysis System")
    print("Using NLTK, spaCy, and scikit-learn")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(model_type='logistic')
    
    # Check if model exists in models directory
    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "sentiment_model.pkl"
    
    if os.path.exists(model_path):
        print(f"\nFound existing model at {model_path}")
        load_choice = input("Do you want to load the existing model? (y/n): ").lower()
        
        if load_choice == 'y':
            analyzer.load_model(model_path)
        else:
            # Train new model
            print("\nTraining new model with sample data...")
            texts, labels = load_training_dataset()
            analyzer.train(texts, labels, preprocess_method='spacy')
            
            save_choice = input("\nDo you want to save this model? (y/n): ").lower()
            if save_choice == 'y':
                analyzer.save_model(model_path)
    else:
        # Train new model
        print("\nNo existing model found. Training new model...")
        texts, labels = load_training_dataset()
        analyzer.train(texts, labels, preprocess_method='spacy')
        
        save_choice = input("\nDo you want to save this model? (y/n): ").lower()
        if save_choice == 'y':
            analyzer.save_model(model_path)
    
    # Interactive sentiment analysis
    print("\n" + "=" * 60)
    print("Ready for sentiment analysis!")
    print("Enter 'quit' or 'exit' to stop")
    print("=" * 60)
    
    while True:
        print("\n")
        user_input = input("Enter text to analyze: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Sentiment Analysis System!")
            break
        
        if not user_input:
            print("Please enter some text to analyze.")
            continue
        
        try:
            # Predict sentiment
            sentiment, confidence = analyzer.predict(user_input)
            
            print(f"\n{'─' * 60}")
            print(f"Input Text: {user_input}")
            print(f"{'─' * 60}")
            print(f"Predicted Sentiment: {sentiment.upper()}")
            print(f"Confidence: {confidence:.2%}")
            print(f"{'─' * 60}")
            
            # Additional analysis using spaCy
            doc = analyzer.nlp(user_input)
            print(f"\nAdditional Analysis (spaCy):")
            print(f"  - Number of tokens: {len(doc)}")
            print(f"  - Number of sentences: {len(list(doc.sents))}")
            
            # Extract named entities
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            if entities:
                print(f"  - Named Entities: {entities}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
