#!/usr/bin/env python3
"""
Lab | Search by Meaning, by Hand

This script implements a local meaning-based search system.
It loads a knowledge base, embeds the text using sentence-transformers,
and computes the cosine similarity against query vectors using NumPy by hand.
"""

# Workaround for multiple libiomp5md.dll initialization error on Windows / Anaconda
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import numpy as np
from sentence_transformers import SentenceTransformer

def load_knowledge_base(filepath="knowledge_base.json"):
    """Loads knowledge base documents from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def manual_cosine_similarity(query_vec, doc_vecs):
    """
    Computes cosine similarity between a query vector and a matrix of document vectors by hand.
    
    Formula:
        similarity = (A . B) / (||A|| * ||B||)
        
    Args:
        query_vec: 1D NumPy array of shape (D,) representing the query embedding.
        doc_vecs: 2D NumPy array of shape (N, D) representing the document embeddings.
        
    Returns:
        1D NumPy array of shape (N,) containing similarity scores in range [-1, 1].
    """
    # Dot product of query with each document vector: shape (N,)
    dot_products = np.dot(doc_vecs, query_vec)
    
    # L2 Norms of document vectors: shape (N,)
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    
    # L2 Norm of query vector: scalar
    query_norm = np.linalg.norm(query_vec)
    
    # Compute product of norms
    norm_products = doc_norms * query_norm
    
    # Guard against division by zero if norm is 0
    norm_products = np.where(norm_products == 0.0, 1.0, norm_products)
    
    # Cosine Similarity
    similarities = dot_products / norm_products
    return similarities

def main():
    # 1. Load the knowledge base
    print("Loading knowledge base...")
    kb = load_knowledge_base()
    print(f"Loaded {len(kb)} passages from knowledge base.\n")
    
    # 2. Initialize embedding model
    # We use 'all-MiniLM-L6-v2', a fast and robust local model
    print("Loading embedding model 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # 3. Embed all passages
    print("Embedding passages...")
    passages = [doc["text"] for doc in kb]
    doc_vectors = model.encode(passages, show_progress_bar=False)
    
    # Convert to numpy array
    doc_vectors = np.array(doc_vectors)
    print(f"Passage embeddings shape: {doc_vectors.shape}\n")
    
    # 4. Define test queries
    queries = [
        "my laptop won't switch on",
        "how do I stop being billed every month?",
        "access denied error when saving a file",
        "where do I leave my car in the evening?"
    ]
    
    # 5. Execute search for each query
    print("=" * 80)
    print("RUNNING SEARCH FOR TEST QUERIES")
    print("=" * 80)
    for query in queries:
        print(f"\nQuery: '{query}'")
        # Embed query
        query_vec = model.encode(query)
        
        # Calculate manual cosine similarity
        similarities = manual_cosine_similarity(query_vec, doc_vectors)
        
        # Rank passages: sort in descending order of similarity
        top_indices = np.argsort(similarities)[::-1][:3]
        
        print(f"Top 3 matching passages (manual cosine similarity):")
        for rank, idx in enumerate(top_indices, 1):
            doc = kb[idx]
            score = similarities[idx]
            print(f"  {rank}. [{doc['id']}] (Source: {doc['source']}) [Score: {score:.4f}]")
            print(f"     Text: \"{doc['text']}\"")
            
    # 6. Optional Stretch: Query not covered by KB
    stretch_query = "what's the wifi password?"
    print("\n" + "=" * 80)
    print("OPTIONAL STRETCH: QUERY NOT IN KNOWLEDGE BASE")
    print("=" * 80)
    print(f"Query: '{stretch_query}'")
    
    stretch_vec = model.encode(stretch_query)
    stretch_similarities = manual_cosine_similarity(stretch_vec, doc_vectors)
    best_idx = np.argmax(stretch_similarities)
    best_doc = kb[best_idx]
    best_score = stretch_similarities[best_idx]
    
    print(f"Best matching passage:")
    print(f"  [{best_doc['id']}] (Source: {best_doc['source']}) [Score: {best_score:.4f}]")
    print(f"  Text: \"{best_doc['text']}\"")
    print(f"\nStretch Reflection:")
    print("The highest score is low (about 0.20-0.30 depending on the model, compared to 0.50-0.70 for relevant ones).")
    print("We can use a similarity threshold (e.g., threshold = 0.40). If the best match score is below this threshold,")
    print("we can return a friendly default response like 'Sorry, I couldn't find an answer to your question in the knowledge base.'")

    # 7. Reflections
    print("\n" + "=" * 80)
    print("REFLECTIONS & WORD OVERLAP ANALYSIS")
    print("=" * 80)
    reflections = (
        "1. 'my laptop won't switch on'\n"
        "   - Best match: kb-02 (\"To power up a device that won't turn on, hold the power button...\")\n"
        "   - Word overlap: Shares almost no key words. 'laptop' and 'switch on' do not appear in kb-02.\n"
        "   - Why it matched: The embedding captured the semantic equivalence between 'switch on' / 'laptop' / 'device' / 'turn on'.\n\n"
        "2. 'how do I stop being billed every month?'\n"
        "   - Best match: kb-05 (\"To cancel your subscription, open Account Settings and choose End Plan...\")\n"
        "   - Word overlap: Shares zero words with the query!\n"
        "   - Why it matched: The model successfully connected 'stop being billed every month' to the concepts of 'cancel subscription' and 'End Plan'.\n\n"
        "3. 'access denied error when saving a file'\n"
        "   - Best match: kb-08 (\"The error code 0x80070005 means 'access denied'. Run the application as administrator...\")\n"
        "   - Word overlap: Only 'access denied' overlaps. It doesn't share 'error', 'saving', or 'file' with the text (which mentions 'error code' and 'write permission to the target folder').\n"
        "   - Why it matched: The embedding models mapped concepts of file permissions ('saving a file', 'write permission') and access restrictions close together.\n\n"
        "4. 'where do I leave my car in the evening?'\n"
        "   - Best match: kb-01 (\"Employees may park in lot B after 6pm on weekdays...\")\n"
        "   - Word overlap: Shares zero words!\n"
        "   - Why it matched: The embedding model linked 'leave my car' to 'park' and 'in the evening' to 'after 6pm'.\n\n"
        "Conclusion: This proves that dense vector embeddings capture latent semantic space and conceptual meaning (context, synonyms, related concepts) rather than just exact keyword occurrences, which allows retrieving contextually relevant passages even with zero vocabulary overlap."
    )
    print(reflections)

if __name__ == "__main__":
    main()
