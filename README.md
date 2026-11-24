# Movie Recommendation System

A content-based Movie Recommendation System built using Python and Machine Learning. It recommends similar movies based on movie descriptions, genres, keywords, cast, and directors.

## Features
- Content-based movie recommendations
- TF-IDF Vectorization
- Cosine Similarity
- Movie search and selection
- Interactive Streamlit web application

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Cosine Similarity
- Streamlit
- Jupyter Notebook

## Dataset
TMDB 5000 Movie Dataset:
- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

## Project Structure
```text
Movies Recommendation/
├── app/
│   └── app.py
├── models/
│   ├── tfidf_vectorizer.pkl
│   └── tfidf_matrix.pkl
├── processed/
│   └── movies_processed.csv
├── README.md
└── requirements.txt
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app/app.py
```

## Machine Learning Approach
1. Data cleaning and preprocessing
2. Feature extraction from movie metadata
3. Text combination into movie tags
4. TF-IDF vectorization
5. Cosine similarity calculation
6. Movie recommendations based on similarity scores

## Author
Lokesh

