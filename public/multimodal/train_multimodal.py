import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel, ViTModel

from split_data_and_dataloader import MMIMDbLocalDataset


# =========================
# 1. MODELS
# =========================

class TextOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask, pixel_values=None):
        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        feat = out.last_hidden_state[:, 0, :]
        return self.classifier(feat)


class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        out = self.image_model(pixel_values=pixel_values)
        feat = out.last_hidden_state[:, 0, :]
        return self.classifier(feat)


class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        self.image_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.classifier = nn.Sequential(
            nn.Linear(768 + 768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0, :]

        img_out = self.image_model(pixel_values=pixel_values)
        img_feat = img_out.last_hidden_state[:, 0, :]

        combined = torch.cat((text_feat, img_feat), dim=1)
        return self.classifier(combined)


# =========================
# 2. EVALUATE
# =========================

def evaluate(model, dataloader, criterion, device):
    model.eval()
    all_labels, all_preds = [], []
    val_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)

    all_labels = np.vstack(all_labels)
    all_preds = np.vstack(all_preds)

    return (
        val_loss / len(dataloader),
        f1_score(all_labels, all_preds, average='micro'),
        f1_score(all_labels, all_preds, average='macro'),
        accuracy_score(all_labels, all_preds)
    )


# =========================
# 3. TRAIN FUNCTION
# =========================

def train_model(model, train_loader, val_loader, device, epochs=5):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    best_f1 = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        v_loss, micro, macro, acc = evaluate(model, val_loader, criterion, device)

        print(f"\n[Val] Loss: {v_loss:.4f} | Micro-F1: {micro:.4f} | Macro-F1: {macro:.4f} | Acc: {acc:.4f}")

        if micro > best_f1:
            best_f1 = micro

    return best_f1, macro, acc


# =========================
# 4. MAIN
# =========================

if __name__ == "__main__":
    print(">>> Loading dataset...")
    df = pd.read_csv("mmimdb_metadata.csv")

    df['genres'] = df['genres'].apply(lambda x: x.split("|") if isinstance(x, str) else [])
    mlb = MultiLabelBinarizer()
    labels_binary = mlb.fit_transform(df['genres'])
    df['labels_binary'] = list(labels_binary)
    num_classes = len(mlb.classes_)

    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, _ = train_test_split(temp_df, test_size=0.5, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

    train_loader = DataLoader(MMIMDbLocalDataset(train_df, tokenizer, image_processor),
                              batch_size=32, shuffle=True)

    val_loader = DataLoader(MMIMDbLocalDataset(val_df, tokenizer, image_processor),
                            batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modes = {
        "text_only": TextOnlyModel(num_classes),
        "image_only": ImageOnlyModel(num_classes),
        "multimodal": MultimodalModel(num_classes)
    }

    results = {}

    for mode, model in modes.items():
        print(f"\n🚀 Training: {mode.upper()}")
        micro, macro, acc = train_model(model, train_loader, val_loader, device)
        results[mode] = (micro, macro, acc)

    # =========================
    # 5. FINAL RESULTS
    # =========================
    print("\n" + "="*70)
    print("MODE            | Micro-F1   | Macro-F1   | Accuracy")
    print("-"*70)
    for mode, (micro, macro, acc) in results.items():
        print(f"{mode:<15} | {micro:.4f}     | {macro:.4f}     | {acc:.4f}")
    print("="*70)