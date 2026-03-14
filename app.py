import pandas as pd
from transformers import pipeline
import os

# 1. INITIALIZE A LIGHTWEIGHT MODEL
# 'valhalla/distilbart-mnli-12-1' is a smaller, faster version of the BART model
print("--- Initializing Fast AI System (Optimized for Speed) ---")
classifier = pipeline(
    "zero-shot-classification", 
    model="valhalla/distilbart-mnli-12-1"
)

# 2. LOAD DATA
file_path = "data.csv"

if not os.path.exists(file_path):
    print(f"ERROR: File '{file_path}' not found.")
else:
    try:
        # Load the Excel file (even if named .csv)
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        # We take only the first 10-20 rows if you want an instant result for the demo
        # df = df.head(20) 
        
        job_posts = df["Job post text"].astype(str).tolist()
        teacher_labels = df["Discrimination"].astype(str).tolist()
    except Exception as e:
        print(f"Loading Error: {e}")
        exit()

    # 3. SET CATEGORIES
    candidate_labels = ["salary", "mobility", "origin", "standing", "gender", "age", "none"]

    print(f"Processing {len(job_posts)} posts... Printing results immediately:")

    ai_predictions = []
    correct_count = 0

    # 4. FAST LOOP WITH INSTANT PRINT
    print("\n" + "="*70)
    print(f"{'ID':<4} | {'AI PREDICTION':<15} | {'TEACHER LABEL':<15} | {'STATUS'}")
    print("-" * 70)

    for i in range(len(job_posts)):
        # We limit the text length to 200 characters to make it super fast
        text_snippet = job_posts[i][:200] 
        
        result = classifier(text_snippet, candidate_labels=candidate_labels)
        predicted = result["labels"][0]
        ai_predictions.append(predicted)
        
        actual = teacher_labels[i].lower()
        
        # Check for matches
        is_correct = predicted in actual or (predicted == "none" and ("no" in actual or "nan" in actual))
        status = "✅ MATCH" if is_correct else "❌ DIFF"
        if is_correct: correct_count += 1

        # This prints EACH row as soon as the AI finishes it
        print(f"{i+1:<4} | {predicted:<15} | {actual[:15]:<15} | {status}")

    # 5. FINAL PERFORMANCE
    accuracy = (correct_count / len(job_posts)) * 100
    print("="*70)
    print(f"TOTAL ANALYZED: {len(job_posts)}")
    print(f"FINAL ACCURACY: {accuracy:.2f}%")
    print("="*70)

    # Export results
    df["AI_Result"] = ai_predictions
    df.to_excel("Fast_Analysis_Report.xlsx", index=False)