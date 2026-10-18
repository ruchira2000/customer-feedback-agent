import sqlite3
import pandas as pd

df = pd.read_csv("../data/reviews.csv", index_col=0)
print("Original columns:", df.columns.tolist())

df = df.rename(columns={
    "Clothing ID": "clothing_id",
    "Age": "age",
    "Title": "title",
    "Review Text": "review_text",
    "Rating": "rating",
    "Recommended IND": "recommended",
    "Positive Feedback Count": "positive_feedback",
    "Division Name": "division",
    "Department Name": "department",
    "Class Name": "class_name",
})

print(f"\nRows loaded: {len(df)}")
print("\nMissing values per column:")
print(df.isna().sum())

df = df.dropna(subset=["review_text"]).reset_index(drop=True)
df.insert(0, "review_id", df.index + 1)

print(f"\nRows with review text: {len(df)}")
print("\nRating counts:")
print(df["rating"].value_counts().sort_index())
print(f"\nRecommended rate: {df['recommended'].mean() * 100:.1f}%")

conn = sqlite3.connect("../data/reviews.db")
df.to_sql("reviews", conn, if_exists="replace", index=False)
conn.close()
print("\nSaved to data/reviews.db")