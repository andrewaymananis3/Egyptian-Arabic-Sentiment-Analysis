from datasets import load_dataset

# Load a real Arabic sentiment dataset (confirmed on the Hub)
print("Loading dataset... (downloads once)")
dataset = load_dataset("arbml/Arabic_Sentiment_Twitter_Corpus", split="train")

print("\nNumber of reviews:", len(dataset))
print("\nColumns:", dataset.column_names)
print("\n--- First 3 examples ---")
for i in range(3):
    print(f"\nExample {i+1}:")
    print(dataset[i])