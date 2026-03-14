import pandas as pd
from transformers import pipeline

# Load Excel file
df = pd.read_excel("Discrimination_Exercise_Job_posts.xlsx")

# Load HuggingFace model
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Discrimination categories
labels = [
    "salary ",
    "mobility ",
    "origin ",
    "standing ",
    "gender ",
    "age ",
    "no "
]

# Analyze job posts
for text in df["Job post text"]:

    result = classifier(
        str(text),
        candidate_labels=labels,
        multi_label=True
    )

    print("\nJob Post:")
    print(text)

    print("Predicted discrimination type:")
    print(result["labels"][0])