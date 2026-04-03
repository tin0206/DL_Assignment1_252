import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from collections import Counter
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
from PIL import Image
from tqdm import tqdm

# Fix for DecompressionBomb and Windows ResourceTracker errors
Image.MAX_IMAGE_PIXELS = None 

# 1. LOAD DATASET
print("Loading MM-IMDb dataset...")
multiprocessing.set_start_method("spawn", force=True)
ds = load_dataset("sxj1215/mmimdb", split='train', num_proc=1)
df = pd.DataFrame(ds)

# 2. DATA PARSING
def parse_content(messages):
    try:
        user_msg = messages[0]['content']
        plot = user_msg.split("Plot: ")[1].split("\nNote")[0].strip()
        genres = messages[1]['content'].split(", ")
        return plot, genres
    except:
        return None, None

print("Parsing all genres and plots...")
df[['plot_clean', 'genres_clean']] = df['messages'].apply(lambda x: pd.Series(parse_content(x)))
df = df.dropna(subset=['plot_clean', 'genres_clean'])

# --- 3. ALL GENRES DISTRIBUTION ---
all_genres = [g for sublist in df['genres_clean'] for g in sublist]
genre_counts = Counter(all_genres)
genre_df = pd.DataFrame(genre_counts.items(), columns=['Genre', 'Count']).sort_values('Count', ascending=False)

plt.figure(figsize=(12, 10))
sns.barplot(data=genre_df, x='Count', y='Genre', hue='Genre', palette='viridis', legend=False)
plt.title('Figure 1: Full Genre Distribution (All 26+ Classes)')
plt.xlabel('Number of Samples')
plt.show()

# --- 4. MULTI-LABEL ANALYSIS ---
df['num_genres'] = df['genres_clean'].apply(len)
plt.figure(figsize=(8, 5))
sns.countplot(x='num_genres', data=df, palette='magma')
plt.title('Figure 2: Number of Genres per Movie')
plt.xlabel('Label Count')
plt.ylabel('Frequency')
plt.show()

# --- 5. CO-OCCURRENCE HEATMAP (Degree of Difficulty) ---
mlb = MultiLabelBinarizer()
genre_bin = mlb.fit_transform(df['genres_clean'])
genre_bin_df = pd.DataFrame(genre_bin, columns=mlb.classes_)

plt.figure(figsize=(14, 10))
sns.heatmap(genre_bin_df.corr(), annot=False, cmap='coolwarm', center=0)
plt.title('Figure 3: Genre Correlation Matrix (Co-occurrence)')
plt.show()

# --- 6. TEXT LENGTH ANALYSIS ---
df['plot_word_count'] = df['plot_clean'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 5))
sns.histplot(df['plot_word_count'], bins=50, kde=True, color='teal')
plt.title('Figure 4: Plot Word Count Distribution')
plt.xlabel('Number of Words')
plt.show()

print(f"\n✅ Total Rows: {len(df)}")
print(f"✅ Unique Genres: {len(mlb.classes_)}")
print(f"✅ Avg Genres per Movie: {df['num_genres'].mean():.2f}")