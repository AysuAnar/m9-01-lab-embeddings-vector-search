![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Search by Meaning, by Hand

## Overview

Today's lesson made one promise: retrieval is, at its core, just comparing arrows. In this lab you'll prove it. You'll turn a small knowledge base into embeddings, turn a few questions into embeddings, and find the best-matching passages **by computing cosine similarity yourself with NumPy** — no vector store, no magic. By the end you'll have watched the right passage rise to the top using nothing but a dot product.

This is the first of three connected labs. The `knowledge_base.json` you use today is the **same corpus** you'll index with a real vector store tomorrow and improve the day after. Get comfortable with it now.

## Learning Goals

- Turn text into embeddings using a real embedding model
- Compute cosine similarity between a query and each document by hand
- Rank passages by meaning and confirm the right ones surface — even when they share no words with the query

## Setup

Fork this repo, clone it, and work on a branch. You'll reuse the free **Gemini** API key from Unit 8. If you'd rather stay fully local and keyless, the local fallback works too.

```bash
pip install -r requirements.txt
```

Set your key in the environment (never commit it):

```bash
export GOOGLE_API_KEY="your-free-gemini-key"
```

Embeddings come from Gemini's `gemini-embedding-001`. The local fallback is `sentence-transformers`, which downloads a small model and needs no key — your choice.

## Your Task

Work in a notebook or a script (`search_by_hand.ipynb` or `.py`). The corpus is in `knowledge_base.json` — a list of passages, each with an `id`, `source`, and `text`.

**Build a meaning-based search, computing the similarity yourself.**

1. Load the knowledge base and **embed every passage's `text`** once. Keep the vectors alongside their `id` and `source`.
2. Take this set of test queries — chosen because each shares **few or no words** with its best-matching passage:
   - `"my laptop won't switch on"`
   - `"how do I stop being billed every month?"`
   - `"access denied error when saving a file"`
   - `"where do I leave my car in the evening?"`
3. For each query: embed it, then **compute cosine similarity against every passage vector using NumPy** (not a library's search function — write the cosine formula yourself). Print the **top 3** passages with their scores.
4. In a short markdown cell or comment, answer: for each query, did the best match share any words with the query? What does that tell you about what the embedding captured?

The point isn't a fancy result — it's *seeing* that a hand-written dot product retrieves by meaning.

### Optional stretch

Add one query that is genuinely **not covered** by the knowledge base (e.g. `"what's the wifi password?"`). Look at the top score. Is it high or low? Write a sentence on how you might use a similarity threshold to decide "we don't actually have an answer for this."

## Submission

Commit your notebook or script and push to your branch. Open a Pull Request and paste its link into the submission box. Your PR should let a reader see, for each query, the top-3 passages and your short reflection.

## Quality Bar

- The same embedding model is used for both passages and queries
- Cosine similarity is computed **by hand** (NumPy), not via a vector store's built-in search
- Each test query returns sensible top matches, and your reflection notes the word-overlap observation
- No API key is committed to the repo
