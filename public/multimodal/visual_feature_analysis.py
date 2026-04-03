import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
SAVE_DIR = "data/images"
os.makedirs(SAVE_DIR, exist_ok=True)
# Fix for DecompressionBomb and Windows ResourceTracker errors
Image.MAX_IMAGE_PIXELS = None 

# 1. LOAD DATASET
print("Loading MM-IMDb dataset...")
multiprocessing.set_start_method("spawn", force=True)
ds = load_dataset("sxj1215/mmimdb", split='train', num_proc=1)
df = pd.DataFrame(ds)

data_list = []

# 2. Loop to save images and extract text features
for i, item in enumerate(tqdm(ds, desc="Saving images and metadata")):
    try:
        # Define image filename and path
        img_filename = f"movie_{i:05d}.jpg"
        img_path = os.path.join(SAVE_DIR, img_filename)
        
        # Save image (if it doesn't already exist)
        if not os.path.exists(img_path):
            # Extract image object (handle list or single object)
            img_obj = item['images'][0] if isinstance(item['images'], list) else item['images']
            
            # Resize slightly to save disk space while maintaining ViT-compatible quality
            img_obj.convert('RGB').save(img_path, quality=90)
        
        # Extract Plot and Genres from the 'messages' attribute
        # Extracting everything after "Plot: "
        plot_text = item['messages'][0]['content'].split("Plot: ")[-1]
        
        # Cleaning and splitting genre strings
        genres_text = item['messages'][1]['content']
        genres_list = [g.strip() for g in genres_text.split(',')]
        
        data_list.append({
            'plot': plot_text,
            'image_path': img_path,
            'genres': "|".join(genres_list) # Save as pipe-separated string for easier CSV handling
        })
    except Exception as e:
        print(f"Error at item {i}: {e}")
        continue

# 3. Save to CSV for persistent use
df = pd.DataFrame(data_list)
df.to_csv("mmimdb_metadata.csv", index=False)
print(f"\n>>> Success! Saved {len(df)} samples to mmimdb_metadata.csv")

# 4. DATA PARSING
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

# 5. ADVANCED METADATA & QUALITY EXTRACTION ---
print("Extracting metadata from 'data/images/movie_xxxxx.jpg'...")

def get_advanced_metadata(idx):
    file_path = os.path.join('data', 'images', f'movie_{idx:05d}.jpg')
    if not os.path.exists(file_path):
        return [None] * 9
    
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            mode = img.mode
            ratio = width / height
            f_size = os.path.getsize(file_path) / 1024
            
            # ALWAYS CONVERT TO RGB BEFORE OPENCV ---
            img_rgb = img.convert('RGB')
            img_rgb.thumbnail((256, 256)) # Resize for speed
            img_np = np.array(img_rgb)
            
            # Sharpness & Contrast
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            contrast = np.std(gray)
            avg_rgb = np.mean(img_np, axis=(0, 1)) 
            
            return width, height, ratio, mode, f_size, sharpness, contrast, avg_rgb[0], avg_rgb[1]
    except Exception as e:
        return [None] * 9

df = df.reset_index(drop=True)
meta_cols = ['width', 'height', 'aspect_ratio', 'color_mode', 'file_size_kb', 'sharpness', 'contrast', 'r_avg', 'g_avg']

# Use a simple list comprehension, but added a check
results = []
for i in tqdm(range(len(df)), desc="Processing Images"):
    results.append(get_advanced_metadata(i))

df[meta_cols] = pd.DataFrame(results, columns=meta_cols)
df = df.dropna(subset=['width'])

# 6. SIZE MARGINAL DISTRIBUTION ---
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='width', y='height', alpha=0.3)

plt.title('Image Size Distribution')
plt.xlabel('Width (pixels)')
plt.ylabel('Height (pixels)')
plt.grid(True)
plt.show()

# 7. FILE SIZE & ASPECT RATIO DISTRIBUTIONS ---
# File Size
plt.figure(figsize=(10, 6))
sns.histplot(df['file_size_kb'], bins=50, kde=True, color='skyblue', edgecolor='white')
plt.title('File Size Distribution', fontsize=15)
plt.xlabel('File Size (KB)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.show()

# Aspect Ratio with standard movie ratio markers
plt.figure(figsize=(10, 6))
sns.histplot(df['aspect_ratio'], bins=50, kde=True, color='orange', edgecolor='white')

# Standard movie ratio markers
ratios = {0.75: '3:4', 1.0: '1:1', 1.33: '4:3', 1.5: '3:2', 2.39: '2.39:1'}
for r, label in ratios.items():
    plt.axvline(r, color='red', linestyle='--', alpha=0.5)
    plt.text(r, plt.gca().get_ylim()[1]*0.9, label, color='red', ha='center', fontweight='bold')

plt.title('Aspect Ratio Distributions', fontsize=15)
plt.xlabel('Aspect Ratio', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.show()
# 8. RGB COLOR SPACE DISTRIBUTION (3D) ---
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# We use the average R and G values, and calculate a dummy B for visualization
scatter = ax.scatter(df['r_avg'], df['g_avg'], df['width']/df['width'].max()*255, 
                     c=df['r_avg'], cmap='viridis', alpha=0.6)
ax.set_title('RGB Color Space Distribution (3D)')
ax.set_xlabel('Red')
ax.set_ylabel('Green')
ax.set_zlabel('Blue (Proxy)')
plt.colorbar(scatter, label='Brightness Proxy')
plt.show()

# 9. IMAGE QUALITY METRICS ---
plt.figure(figsize=(10, 6))
# Filter out extreme outliers for better visualization
q_df = df[df['sharpness'] < df['sharpness'].quantile(0.95)] 

sns.scatterplot(data=q_df, x='sharpness', y='contrast', alpha=0.4, color='tomato')
plt.title('Image Quality Metrics')
plt.xlabel('Sharpness (Laplacian Variance)')
plt.ylabel('Contrast (Std Deviation)')
plt.grid(True, alpha=0.3)
plt.show()