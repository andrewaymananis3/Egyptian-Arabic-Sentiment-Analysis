import gradio as gr
from transformers import pipeline

# Load the Egyptian Arabic sentiment model (loads once when app starts)
print("Loading model...")
classifier = pipeline(
    "sentiment-analysis",
    model="CAMeL-Lab/bert-base-arabic-camelbert-da-sentiment"
)

# The function that runs when a user submits text
def analyze_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text."
    result = classifier(text, truncation=True, max_length=512)[0]
    label = result["label"]
    confidence = result["score"]
    # Build a friendly output with an emoji
    emoji = "😊" if label.lower() == "positive" else "😞" if label.lower() == "negative" else "😐"
    return f"{emoji}  Sentiment: {label.upper()}  |  Confidence: {confidence:.0%}"

# Example reviews users can click to try (real Egyptian dialect)
examples = [
    "الايس كريم تحفة وطعمه جامد جدا هرجع تاني اكيد",
    "الخدمة بطيئة جدا والطلب اتأخر، تجربة سيئة",
    "مكان جميل والأسعار مناسبة بس الزحمة كتير",
]

# Build the web interface
demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(
        lines=3,
        placeholder="اكتب رأيك هنا... (Type an Arabic review here)",
        label="Review Text",
    ),
    outputs=gr.Textbox(label="Result"),
    title="🇪🇬 Egyptian Arabic Sentiment Analyzer",
    description="Analyzes sentiment in Egyptian dialect Arabic using a dialect-aware transformer model (CAMeLBERT-DA). Type a review or click an example below.",
    examples=examples,
    theme=gr.themes.Soft(),
)

# Launch the app
if __name__ == "__main__":
    demo.launch()