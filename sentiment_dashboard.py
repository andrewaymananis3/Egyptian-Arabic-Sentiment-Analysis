from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter

# Load the Egyptian Arabic sentiment model
print("Loading Arabic model...")
classifier = pipeline(
    "sentiment-analysis",
    model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
)

# A batch of Egyptian dialect reviews (imagine these are restaurant reviews)
reviews = [
    "المكان جميل جدا والأكل لذيذ هرجع تاني",
    "الخدمة وحشة والانتظار طويل مش هرجع",
    "تجربة رائعة بجد استمتعت جدا",
    "الأكل بايخ والاسعار غالية",
    "المكان نضيف والمعاملة محترمة",
    "زحمة جدا والصوت عالي مكانش مريح",
    "أحلى مطعم زرته في حياتي",
    "الطلب اتأخر ساعة وبرد، تجربة سيئة",
    "الديكور شيك والقهوة ممتازة",
    "غالي على الفاضي مفيش حاجة تستاهل",
]

# Analyze each review and collect results
results = []
for review in reviews:
    prediction = classifier(review)[0]
    results.append({
        "text": review,
        "sentiment": prediction["label"],
        "confidence": prediction["score"]
    })
    print(f"{prediction['label']:10s} ({prediction['score']:.2f})  -  {review}")

# Count how many of each sentiment
sentiment_counts = Counter(r["sentiment"] for r in results)
print("\nSummary:", dict(sentiment_counts))

# --- VISUALIZATION ---
labels = list(sentiment_counts.keys())
counts = list(sentiment_counts.values())
colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6"}
bar_colors = [colors.get(label, "#3498db") for label in labels]

plt.figure(figsize=(8, 5))
plt.bar(labels, counts, color=bar_colors)
plt.title("Egyptian Arabic Review Sentiment Analysis", fontsize=14, fontweight="bold")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.tight_layout()

# Save the chart as an image (for your README / portfolio)
plt.savefig("sentiment_results.png", dpi=150)
print("\nChart saved as sentiment_results.png")
plt.show()