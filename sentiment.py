from transformers import pipeline

# Load a pretrained sentiment analysis model
# (downloads once, then cached on your machine)
print("Loading model... (first run downloads it, please wait)")
classifier = pipeline("sentiment-analysis")

# Test it on some example sentences
texts = [
    "I love this product, it works perfectly!",
    "This is the worst experience I've ever had.",
    "The service was okay, nothing special.",
]

# Run the model on each text and print the result
for text in texts:
    result = classifier(text)[0]
    print(f"\nText: {text}")
    print(f"Sentiment: {result['label']} (confidence: {result['score']:.2f})")