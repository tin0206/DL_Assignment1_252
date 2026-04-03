import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, accuracy_score, hamming_loss
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import MultiLabelBinarizer

# --- 1. SETUP & MODEL LOADING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Load CLIP with safetensors to avoid security errors on older Torch versions
model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(device)
processor = CLIPProcessor.from_pretrained(model_id)

# --- 2. DATA PREPARATION ---
df = pd.read_csv("mmimdb_metadata.csv")

# Ensure genres are lists (converts "Drama|Action" -> ["Drama", "Action"])
df['genres_list'] = df['genres'].apply(lambda x: x.split("|") if isinstance(x, str) else [])

# IMPORTANT: Fit MLB on the ENTIRE dataset first to lock the 26-column format
mlb = MultiLabelBinarizer()
mlb.fit(df['genres_list'])
class_names = mlb.classes_
num_classes = len(class_names)

print(f"✅ Total unique genres identified: {num_classes}")
print(f"✅ Genres: {', '.join(class_names)}")

# Prepare the 26 natural language prompts
prompts = [f"A movie poster of the genre {g}" for g in class_names]

# --- 3. INFERENCE ENGINE ---
def run_zero_shot(mode="multimodal", sample_size=500):
    all_labels = []
    all_preds = []
    
    print(f"\nEvaluation Mode: {mode.upper()}")
    model.eval()
    
    # Use a fixed random_state so all 3 modes test on the EXACT same movies
    test_subset = df.sample(min(sample_size, len(df)), random_state=42)
    
    with torch.no_grad():
        for _, row in tqdm(test_subset.iterrows(), total=len(test_subset)):
            # A. Prepare Text (Plot) - Truncate to stay within CLIP's 77-token limit
            plot_snippet = str(row['plot'])[:250] 
            
            # B. Prepare Image
            try:
                image = Image.open(row['image_path']).convert('RGB')
            except Exception:
                # Fallback for missing images
                image = Image.new('RGB', (224, 224), color=0)

            # C. CLIP Forward Pass
            # We process 26 genre prompts + 1 movie plot + 1 image in one go
            inputs = processor(
                text=prompts + [plot_snippet], 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(device)
            
            outputs = model(**inputs)
            
            # --- IMAGE-TO-GENRE BRANCH ---
            # Similarity between Image and the 26 Prompts
            probs_img = outputs.logits_per_image[:, :num_classes].softmax(dim=-1).cpu().numpy()[0]

            # --- TEXT-TO-GENRE BRANCH ---
            # Compare Plot Embedding against the 26 Genre Embeddings
            text_embeds = outputs.text_embeds      # Shape: [27, 512]
            genre_embeds = text_embeds[:-1]        # First 26 (The Genres)
            plot_embed = text_embeds[-1:]          # Last 1 (The Plot)
            
            # Manual Cosine Similarity Calculation for text-only (Plot vs Genres)
            logits_text = torch.matmul(plot_embed, genre_embeds.t()) * model.logit_scale.exp()
            probs_text = logits_text.softmax(dim=-1).cpu().numpy()[0]

            # --- FUSION LOGIC ---
            if mode == "image_only":
                final_probs = probs_img
            elif mode == "text_only":
                final_probs = probs_text
            else: # Multimodal (Mean Fusion of Image and Text probabilities)
                final_probs = (probs_img + probs_text) / 2

            # Multi-label thresholding (0.1 is standard for MM-IMDb zero-shot)
            pred_binary = (final_probs > 0.1).astype(int)
            
            # Transform ground truth to binary vector [0, 1, 0...]
            true_binary = mlb.transform([row['genres_list']])[0]
            
            # Safety check: ensure arrays match the expected 26 classes
            if len(pred_binary) == num_classes:
                all_labels.append(true_binary)
                all_preds.append(pred_binary)

    # Use vstack to ensure a clean 2D matrix for sklearn [Samples x Classes]
    return np.vstack(all_labels), np.vstack(all_preds)

# --- 4. EXECUTION & RESULTS ---
final_results = {}

for m in ["image_only", "text_only", "multimodal"]:
    y_true, y_pred = run_zero_shot(mode=m)
    
    # Calculate Micro F1 (standard for multi-label tasks)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Optional: Calculate Subset Accuracy (strict match, not ideal for multi-label but informative)
    subset_acc = accuracy_score(y_true, y_pred)

    hamming_acc = 1 - hamming_loss(y_true, y_pred)
    final_results[m] = {
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "subset_acc": subset_acc,
        "hamming_acc": hamming_acc
    }

# --- 5. FINAL COMPARISON TABLE ---
print("\n" + "="*75)
print(f"{'MODE':<20} | {'Micro-F1':<10} | {'Macro-F1':<10} | {'Subset Acc':<12} | {'Hamming Acc':<12}")
print("-" * 75)

for mode, scores in final_results.items():
    print(f"{mode:<20} | {scores['micro_f1']:.4f}     | {scores['macro_f1']:.4f}     | {scores['subset_acc']:.4f}      | {scores['hamming_acc']:.4f}")

print("="*75)

# Optional: Print detailed report for the best performing mode (usually Multimodal)
print("\nDetailed Report for Multimodal Fusion:")
labels_final, preds_final = run_zero_shot(mode="multimodal", sample_size=300)
print(classification_report(labels_final, preds_final, target_names=class_names, zero_division=0))