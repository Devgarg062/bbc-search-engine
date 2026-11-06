
# Intelligent BBC News Search Engine

## Project Overview

An Information Retrieval system designed to search BBC news articles using four different ranking algorithms.

## Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- BM25
- Sentence Transformers
- Matplotlib
- NLTK

## Retrieval Models

1. TF-IDF
2. BM25
3. Sentence Transformer
4. Hybrid Reciprocal Rank Fusion

## Evaluation Metrics

- Precision@10
- Recall@10
- Mean Average Precision
- Normalized Discounted Cumulative Gain

## Project Execution

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

Run model evaluation:

python evaluation/run_evaluation.py

## Dataset

BBC News dataset containing article text and category labels.

Place the original CSV file inside the data folder.

## Hybrid Ranking

RRF(d) = Sum(1 / (k + rank(d)))

The hybrid model combines rankings from TF-IDF, BM25 and Sentence Transformers.

## Expected Outputs

- Interactive news search interface
- Four retrieval models
- Search result rankings
- Performance comparison table
- Precision-Recall graph
- Evaluation metrics
