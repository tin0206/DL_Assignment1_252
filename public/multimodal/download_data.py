import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# --- CONFIGURATION ---
SAVE_DIR = "data/images"
os.makedirs(SAVE_DIR, exist_ok=True)
# Allow processing of large image files
Image.MAX_IMAGE_PIXELS = None 

# 1. Load Dataset
print(">>> Loading dataset from Hugging Face...")
# Loading the MMIMDb dataset (train split)
ds = load_dataset("sxj1215/mmimdb", split='train')

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