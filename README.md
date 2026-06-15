# Egyptian Arabic Sentiment Analysis

A natural language processing project that analyzes sentiment in **Egyptian dialect Arabic** restaurant reviews using pretrained transformer models.

## Overview

This project classifies Arabic text as **positive** or **negative**, with a focus on Egyptian colloquial dialect — a harder problem than Modern Standard Arabic, since most Arabic NLP models are trained primarily on formal Arabic.

## What it does

- Classifies sentiment in **English** (DistilBERT) and **Egyptian Arabic** (CAMeLBERT-DA, a dialect-aware Arabic model)
- Runs batch analysis on multiple reviews and **visualizes** the sentiment breakdown
- Evaluates model performance on a **custom dataset of real Egyptian reviews** I collected and annotated myself

## The dataset

Since no labeled Egyptian-dialect restaurant-review sentiment dataset was readily available, I **collected and annotated my own**: 17 real reviews of B Laban (a popular Egyptian dessert chain) from Google Maps, hand-labeled as positive or negative. This reflects real-world ML practice, where data collection and annotation are often the first and most important steps.

## Results

On the custom Egyptian review set, the CAMeLBERT-DA model achieved **100% accuracy** (precision, recall, and F1 all 1.00 across both classes).

**Interpretation:** While encouraging, I treat this result with caution. The test set is small (17 reviews) and the reviews carried mostly clear, unambiguous sentiment. A more rigorous evaluation would use a larger sample with more borderline cases — for example, mixed reviews like "good but pricey," where I observed the model's confidence drop to around 0.5, reflecting genuine ambiguity. This is a known limitation of single-label sentiment models.

## Tech stack

- **Python**
- **Hugging Face Transformers** (DistilBERT, CAMeLBERT-DA)
- **PyTorch** (model backend)
- **scikit-learn** (evaluation: accuracy, precision, recall, F1, confusion matrix)
- **matplotlib** (visualization)

## Files

- `sentiment.py` — English sentiment analysis
- `arabic_sentiment.py` — Egyptian Arabic sentiment analysis
- `sentiment_dashboard.py` — batch analysis with visualization
- `balaban_reviews.py` — custom annotated Egyptian review dataset
- `evaluate_balaban.py` — model evaluation against ground-truth labels

## Future work

- Expand the dataset with more reviews, including ambiguous/mixed sentiment
- Add a neutral class (currently binary positive/negative)
- Explore aspect-based sentiment (e.g., separating food vs. service vs. price)
- Fine-tune a model on Egyptian-specific data

## Author

**Andrew Fahmy** — Software Engineering graduate specializing in Data Science and AI