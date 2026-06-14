from transformers import pipeline

# Load an Arabic sentiment model (CAMeLBERT - trained on Arabic incl. dialects)
print("Loading Arabic model... (first run downloads it, please wait)")
classifier = pipeline(
    "sentiment-analysis",
    model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
)

# Egyptian dialect test sentences
texts = [
    "المكان جميل جدا والأكل لذيذ هرجع تاني",      # nice place, tasty food, will return
    "الخدمة وحشة والانتظار طويل مش هرجع",          # bad service, long wait, won't return
    "المنتج كويس بس غالي شوية",                    # product is good but a bit pricey
    "تجربة رائعة بجد استمتعت جدا",                  # wonderful experience, really enjoyed
]

# Run the model on each text
for text in texts:
    result = classifier(text)[0]
    print(f"\nText: {text}")
    print(f"Sentiment: {result['label']} (confidence: {result['score']:.2f})")