
import re
import html

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

stop_words = set(ENGLISH_STOP_WORDS)


def clean_text(text):

    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove HTML entities
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove special characters and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenization
    tokens = text.split()

    # Stopword removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)


def preprocess_dataframe(df):

    df = df.copy()

    df["clean_text"] = df["document"].apply(clean_text)

    print("Text Preprocessing Completed!")

    return df
