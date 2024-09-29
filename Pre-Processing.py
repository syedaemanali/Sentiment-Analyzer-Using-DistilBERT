import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd  # Assuming reviews is a DataFrame or similar structure
import json

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load stopwords from nltk
nltk_stopwords = set(stopwords.words('english'))
nltk_stopwords.append(['Amazon','buy','amazon','year','review','people','think','will','thing','first','even'])

# List of punctuation marks
punct_words = set(string.punctuation)


# Function to remove numbers
def remove_numbers(text):
    return re.sub(r'\d+', '', text)


# Function to tokenize text
def tokenize_text(text):
    return word_tokenize(text.lower())


# Function to check which field contains the reviews (for CSV/JSON)
def extract_review_column(data):
    # Look for common review column names
    possible_columns = ['review', 'feedback', 'comment', 'text']
    for column in data.columns:
        if column.lower() in possible_columns:
            return column
    return None  # If no column found, return None

# Function to remove stopwords and punctuation, and lemmatize the tokens
def remove_stopwords_and_punctuation(tokens):
    return [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stopwords and word not in punct_words
    ]


# Function to clean a single review text
def clean_review(text):
    text = remove_numbers(text)  # Step 1: Remove numbers
    tokens = tokenize_text(text)  # Step 2: Tokenize
    clean_tokens = remove_stopwords_and_punctuation(tokens)  # Step 3: Remove stopwords, punctuation and lemmatize
    return ' '.join(clean_tokens)  # Return the cleaned text as a single string


# Preprocess the entire DataFrame/series
def preprocess_text(reviews):
    reviews = reviews.apply(clean_review)
    return reviews
