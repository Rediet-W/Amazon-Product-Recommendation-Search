import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
from nltk.corpus import wordnet  # For query expansion
import nltk


nltk.download('wordnet')

# Function to preprocess text
def preprocess_text(text):
    if pd.isna(text):  # Handle missing values
        return ''
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    return text.strip()

# Query expansion function
def expand_query(query):
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):  # Find synonyms using WordNet
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
    expanded_query = " ".join(synonyms)
    return query + " " + expanded_query  # Combine original query with expanded terms

# Load and preprocess the dataset
@st.cache_data  # Cache the dataset to avoid reloading it on every app interaction
def load_data():
    # Load the CSV file
    file_path = "amazon.csv"  # Replace with your actual file path
    dataset = pd.read_csv(file_path)

    # Preprocess relevant columns
    dataset['processed_name'] = dataset['product_name'].apply(preprocess_text)
    dataset['processed_about_product'] = dataset['about_product'].apply(preprocess_text)
    dataset['processed_review'] = (
        dataset['review_title'].fillna('') + " " +
        dataset['review_content'].fillna('')
    ).apply(preprocess_text)

    # Combine fields into a single column for semantic search
    dataset['search_text'] = (
        dataset['processed_name'] + " " +
        dataset['processed_about_product'] + " " +
        dataset['processed_review']
    )

    # Convert the 'rating' column to numeric
    dataset['rating'] = pd.to_numeric(dataset['rating'], errors='coerce').fillna(0)

    # Convert 'discounted_price' to numeric
    dataset['discounted_price'] = (
        dataset['discounted_price']
        .str.replace(r'[^\d.]', '', regex=True)  # Remove non-numeric characters like ₹ or commas
        .astype(float)
    )

    return dataset

# Load the pre-trained model
@st.cache_resource  # Cache the model to avoid reloading it on every app interaction
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Semantic search function
def search_products(query, dataset, model, top_n=10):
    # Preprocess and expand the query
    expanded_query = expand_query(preprocess_text(query))

    # Encode the expanded query
    query_embedding = model.encode(expanded_query).reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, np.vstack(dataset['embeddings']))

    # Rank products by similarity
    dataset['similarity'] = similarities[0]
    ranked_products = dataset.sort_values(by='similarity', ascending=False).head(top_n)
    return ranked_products[['product_name', 'category', 'discounted_price', 'rating', 'similarity']]

# Filtered search function
def search_with_filters(query, dataset, model, category=None, min_rating=None, max_price=None):
    # Perform semantic search
    results = search_products(query, dataset, model)

    # Apply filters
    if category and category != "All":
        results = results[results['category'] == category]
    if min_rating:
        results = results[results['rating'] >= min_rating]
    if max_price:
        results = results[results['discounted_price'] <= max_price]
    return results

# Generate embeddings for the dataset (cached)
@st.cache_data
def generate_embeddings(dataset, _model):
    dataset['embeddings'] = list(_model.encode(dataset['search_text']))
    return dataset

# Main Streamlit app
st.title("Amazon Product Recommendation Search")

# Load resources
model = load_model()
dataset = load_data()

# Generate and cache embeddings
dataset = generate_embeddings(dataset, model)

# User inputs
query = st.text_input("Enter your search query:")
category = st.selectbox("Select a category (optional):", ["All"] + list(dataset['category'].unique()))
min_rating = st.slider("Minimum Rating:", 0.0, 5.0, 3.0)
max_price = st.number_input("Maximum Price (optional):", value=5000.0)

# Perform search and display results
if query:
    results = search_with_filters(query, dataset, model,
                                  category=category,
                                  min_rating=min_rating,
                                  max_price=max_price)
    for _, row in results.iterrows():
        st.write(f"**{row['product_name']}** - ${row['discounted_price']:.2f}")
        st.write(f"Rating: {row['rating']}")
        st.write(f"Category: {row['category']}")
        st.write("---")
