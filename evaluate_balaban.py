from balaban_reviews import reviews
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load your Egyptian Arabic sentiment model
print("Loading model...")
classifier = pipeline(
    "sentiment-analysis",
    model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
)

# Run the model on each review, compare to your label
true_labels = []
predicted_labels = []

print("\n--- Predictions ---")
for r in reviews:
    text = r["text"]
    true = r["label"]
    pred = classifier(text, truncation=True, max_length=512)[0]["label"].lower()

    # model may output "positive"/"negative"/"neutral" — map neutral to negative for 2-class
    if pred not in ["positive", "negative"]:
        pred = "negative"

    true_labels.append(true)
    predicted_labels.append(pred)

    match = "✓" if pred == true else "✗"
    print(f"{match} predicted={pred:8s} actual={true:8s} | {text[:45]}...")

# Measure performance against YOUR labels
print("\n" + "="*50)
print("RESULTS — model vs. my annotations")
print("="*50)
acc = accuracy_score(true_labels, predicted_labels)
print(f"\nAccuracy: {acc:.1%}")
print("\nDetailed report:")
print(classification_report(true_labels, predicted_labels,
                            labels=["negative", "positive"],
                            target_names=["negative", "positive"],
                            zero_division=0))
print("Confusion matrix (rows=actual, cols=predicted):")
print("             pred_neg  pred_pos")
cm = confusion_matrix(true_labels, predicted_labels, labels=["negative", "positive"])
print(f"actual_neg     {cm[0][0]:4d}     {cm[0][1]:4d}")
print(f"actual_pos     {cm[1][0]:4d}     {cm[1][1]:4d}")