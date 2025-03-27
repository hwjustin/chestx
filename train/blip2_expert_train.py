import argparse
import random
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model  # Import LoRA modules

# Define the categories and their corresponding character tokens
CATEGORIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged-Cardiomediastinum", "Fracture", "Lung-Lesion", 
    "Lung-Opacity", "No-Finding", "Pleural-Effusion", 
    "Pleural_Other", "Pneumonia", "Pneumothorax", "Support-Devices"
]

CHARACTER_MAPPING = {category: chr(97 + i) for i, category in enumerate(CATEGORIES)}  # Maps to 'a', 'b', 'c', ...

def get_character_token_ids(tokenizer):
    # Map each character to a token ID
    return {char: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(char)[0]) for char in CHARACTER_MAPPING.values()}

class ChestXDataset(Dataset):
    def __init__(self, csv_file, image_folder, report_file, tokenizer, processor, max_length=512):
        self.data = self.load_data(csv_file, report_file)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def load_data(self, csv_file, report_file):
        # Load image IDs from the train_csv
        image_ids = []
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_ids.append(row['id'])

        # Load reports and ground truth from proc/train.csv
        data = {}
        with open(report_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_id = row['id']
                if image_id in image_ids:
                    report = row['report']
                    ground_truth = {category: int(row[category]) for category in CATEGORIES}
                    data[image_id] = {'report': report, 'ground_truth': ground_truth}

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = list(self.data.keys())[idx]
        entry = self.data[image_id]
        image_path = os.path.join(self.image_folder, image_id)
        
        # Load the original image
        image = Image.open(image_path)

        # Process the image
        image = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)
        
        report = entry['report']
        # Update the prompt to use character mapping
        category_mapping_str = ", ".join([f"{char}: {category}" for category, char in CHARACTER_MAPPING.items()])
        prompt = (
            f"You are a doctor helping to predict the medical condition of a patient. "
            f"You are now given the chest x-ray image and the following report: {report}. "
            f"Please determine if the user has certain medical conditions. Use the following mapping: {category_mapping_str}. "
            f"Answer only with the corresponding character."
        )
        text_encoding = self.tokenizer(
            prompt, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        # Convert ground truth to character labels
        label = torch.tensor([entry['ground_truth'][category] for category in CATEGORIES], dtype=torch.float)

        return {
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "label": label,
            "id": image_id,
        }

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_category_token_ids(tokenizer):
    # Map each category to a token ID
    category_token_ids = {}
    for category in CATEGORIES:
        tokens = tokenizer.tokenize(category)
        print(f"Category: {category}, Tokens: {tokens}")
        category_token_ids[category] = tokenizer.convert_tokens_to_ids(tokens[0])
    return category_token_ids

def evaluate(tokenizer, model, dataloader, device):
    model.eval()
    all_labels = []
    all_predictions = []
    character_token_ids = get_character_token_ids(tokenizer)

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            logits = outputs.logits[:, -1, :]
            character_logits = torch.stack(
                [logits[:, token_id] for token_id in character_token_ids.values()],
                dim=-1,
            )
            probs = torch.sigmoid(character_logits)
            predictions = (probs > 0.5).float()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_labels = torch.tensor(all_labels)
    all_predictions = torch.tensor(all_predictions)

    # Compute multi-label classification metrics
    f1 = f1_score(all_labels, all_predictions, average="macro")
    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average="macro")
    return f1, precision, recall

def train(model, train_dataloader, val_dataloader, tokenizer, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    character_token_ids = get_character_token_ids(tokenizer)

    best_f1 = -1
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            logits = outputs.logits[:, -1, :]
            character_logits = torch.stack(
                [logits[:, token_id] for token_id in character_token_ids.values()],
                dim=-1,
            )
            # labels = labels.argmax(dim=-1)  # Convert one-hot to class indices
            loss = criterion(character_logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        f1, precision, recall = evaluate(tokenizer, model, val_dataloader, device)
        print(f"Epoch {epoch + 1}")
        print(f"Validation F1 Score: {f1:.4f}")
        print(f"Validation Precision: {precision:.4f}")
        print(f"Validation Recall: {recall:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            print("Model achieves better F1 score!")

        # Save the model at the end of each epoch
        epoch_save_path = os.path.join(args.save_path, f"epoch_{epoch+1}")  # Ensure epoch starts from 1
        os.makedirs(epoch_save_path, exist_ok=True)  # Create directory if it doesn't exist
        model.save_pretrained(epoch_save_path)
        print(f"Model saved at {epoch_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BLIP-2 model on the chestx dataset")
    parser.add_argument("--train_csv", type=str, default="chestx_split_results.csv", help="Path to the CSV file with image IDs")
    parser.add_argument("--report_csv", type=str, default="proc/train.csv", help="Path to the CSV file with reports and ground truth")
    parser.add_argument("--image_folder", type=str, default="proc/images", help="Path to the image folder")
    parser.add_argument("--save_path", type=str, default="./blip2_chestx_model", help="Path to save the model")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for initialization")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for tokenized sequences")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=float, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")

    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

    # Apply LoRA to the model
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj"],
    )
    model = get_peft_model(model, config)

    model.to(device)

    train_dataset = ChestXDataset(args.train_csv, args.image_folder, args.report_csv, tokenizer, processor, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)  # For simplicity, using the same dataset for validation

    train(model, train_dataloader, val_dataloader, tokenizer, device, args)
