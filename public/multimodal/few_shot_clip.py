import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, accuracy_score, hamming_loss
from transformers import CLIPProcessor, CLIPModel

# =========================
# 1. DATASET
# =========================
class CLIPMultimodalDataset(Dataset):
    def __init__(self, df, processor):
        self.df = df.reset_index(drop=True)
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Text (truncate for CLIP)
        plot_text = str(row['plot'])[:150]

        # Image
        try:
            image = Image.open(row['image_path']).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), color=0)

        inputs = self.processor(
            text=[plot_text],
            images=image,
            return_tensors="pt",
            padding='max_length',
            truncation=True
        )

        labels = torch.tensor(row['labels_binary'], dtype=torch.float)

        return (
            inputs['pixel_values'].squeeze(0),
            inputs['input_ids'].squeeze(0),
            inputs['attention_mask'].squeeze(0),
            labels
        )

# =========================
# 2. MODEL
# =========================
class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model, num_classes, mode="multimodal"):
        super().__init__()
        self.mode = mode
        self.clip = clip_model

        # Freeze CLIP
        for param in self.clip.parameters():
            param.requires_grad = False

        v_dim = self.clip.vision_model.config.hidden_size
        t_dim = self.clip.text_model.config.hidden_size

        if mode == "image_only":
            input_dim = v_dim
        elif mode == "text_only":
            input_dim = t_dim
        else:
            input_dim = v_dim + t_dim

        # Improved classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        if self.mode in ["image_only", "multimodal"]:
            v_outputs = self.clip.vision_model(pixel_values=pixel_values)
            v_out = v_outputs.pooler_output

        if self.mode in ["text_only", "multimodal"]:
            t_outputs = self.clip.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            t_out = t_outputs.pooler_output

        if self.mode == "image_only":
            features = v_out

        elif self.mode == "text_only":
            features = t_out

        else:  # multimodal
            features = torch.cat((v_out, t_out), dim=1)

        return self.classifier(features)

# =========================
# 3. TRAIN + EVAL
# =========================
def run_few_shot_experiment(mode, train_loader, val_loader, num_classes, device, clip_model):

    print(f"\nTraining FEW-SHOT: {mode.upper()}")

    model = CLIPLinearProbe(clip_model, num_classes, mode=mode).to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # ===== TRAIN =====
    for epoch in range(5):
        model.train()
        total_loss = 0

        for v, tid, am, labels in train_loader:
            v, tid, am, labels = v.to(device), tid.to(device), am.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(v, tid, am)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/5 - Loss: {total_loss/len(train_loader):.4f}")

    # ===== EVAL =====
    model.eval()
    all_true, all_pred = [], []

    with torch.no_grad():
        for v, tid, am, labels in val_loader:
            v, tid, am = v.to(device), tid.to(device), am.to(device)

            logits = model(v, tid, am)
            probs = torch.sigmoid(logits)

            preds = (probs > 0.3).cpu().numpy()

            all_true.append(labels.numpy())
            all_pred.append(preds)

    y_true = np.vstack(all_true)
    y_pred = np.vstack(all_pred)

    # ===== METRICS =====
    micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    subset_acc = accuracy_score(y_true, y_pred)
    hamming_acc = 1 - hamming_loss(y_true, y_pred)

    return {
        "Micro-F1": micro,
        "Macro-F1": macro,
        "Subset Acc": subset_acc,
        "Hamming Acc": hamming_acc
    }

# =========================
# 4. MAIN
# =========================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(">>> Loading dataset...")
    df = pd.read_csv("mmimdb_metadata.csv")

    df['genres_list'] = df['genres'].apply(lambda x: x.split("|"))

    mlb = MultiLabelBinarizer()
    df['labels_binary'] = list(mlb.fit_transform(df['genres_list']))

    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Few-shot (5%)
    few_shot_df = train_df.sample(frac=0.05, random_state=42)
    print(f">>> Few-shot samples: {len(few_shot_df)}")

    # Load CLIP
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        use_safetensors=True
    ).to(device)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # DataLoader
    train_loader = DataLoader(CLIPMultimodalDataset(few_shot_df, processor), batch_size=32, shuffle=True)
    val_loader = DataLoader(CLIPMultimodalDataset(val_df, processor), batch_size=32)

    # ===== RUN EXPERIMENTS =====
    results = {}

    for mode in ["image_only", "text_only", "multimodal"]:
        results[mode] = run_few_shot_experiment(
            mode,
            train_loader,
            val_loader,
            len(mlb.classes_),
            device,
            clip_model
        )

    # ===== PRINT TABLE =====
    print("\n" + "="*75)
    print(f"{'MODE':<15} | {'Micro-F1':<10} | {'Macro-F1':<10} | {'Subset Acc':<12} | {'Hamming Acc':<12}")
    print("-"*75)

    for mode, m in results.items():
        print(f"{mode:<15} | {m['Micro-F1']:.4f}     | {m['Macro-F1']:.4f}     | {m['Subset Acc']:.4f}      | {m['Hamming Acc']:.4f}")

    print("="*75)