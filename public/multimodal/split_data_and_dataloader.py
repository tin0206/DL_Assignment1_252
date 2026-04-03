import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoImageProcessor

# 1. Custom Dataset Definition
class MMIMDbLocalDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, max_len=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- Text Processing ---
        text_enc = self.tokenizer(
            str(row['plot']), 
            truncation=True, 
            max_length=self.max_len, 
            padding='max_length', 
            return_tensors='pt'
        )
        
        # --- Image Processing (Loading from local file path) ---
        try:
            image = Image.open(row['image_path']).convert('RGB')
            # ViT requires 224x224 input size
            image = image.resize((224, 224))
            image_enc = self.image_processor(images=image, return_tensors="pt")
        except Exception as e:
            # Fallback if the image file is corrupted or missing
            image = Image.new('RGB', (224, 224))
            image_enc = self.image_processor(images=image, return_tensors="pt")
        
        # --- Label Handling ---
        labels = torch.tensor(row['labels_binary'], dtype=torch.float)
        
        return {
            'input_ids': text_enc['input_ids'].squeeze(0),
            'attention_mask': text_enc['attention_mask'].squeeze(0),
            'pixel_values': image_enc['pixel_values'].squeeze(0),
            'labels': labels
        }

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 2. Metadata Loading and Data Splitting
    print(">>> Reading metadata...")
    df = pd.read_csv("mmimdb_metadata.csv")
    
    # Split genre strings into lists
    df['genres'] = df['genres'].apply(lambda x: x.split("|") if isinstance(x, str) else [])

    # Binarize labels (transforming movie genres into binary vectors of 0s and 1s)
    mlb = MultiLabelBinarizer()
    labels_binary = mlb.fit_transform(df['genres'])
    df['labels_binary'] = list(labels_binary)
    num_classes = len(mlb.classes_)

    # 80/10/10 Data Split (Train/Val/Test)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    print(f"Train samples: {len(train_df)} - Val: {len(val_df)} - Test: {len(test_df)}")
    print(f"Number of target classes: {num_classes}")

    # 3. Processor Initialization
    # Using AutoImageProcessor instead of FeatureExtractor for compatibility with ViT
    print(">>> Initializing Tokenizer and ImageProcessor...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    # 4. DataLoader Setup
    train_dataset = MMIMDbLocalDataset(train_df, tokenizer, image_processor)
    val_dataset = MMIMDbLocalDataset(val_df, tokenizer, image_processor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f">>> DataLoaders ready! ({len(train_loader)} batches total)")

    # Verification of the first batch
    first_batch = next(iter(train_loader))
    print("\nFirst Batch Verification:")
    print(f"- Input IDs shape: {first_batch['input_ids'].shape}")
    print(f"- Pixel Values shape: {first_batch['pixel_values'].shape}")
    print(f"- Labels shape: {first_batch['labels'].shape}")