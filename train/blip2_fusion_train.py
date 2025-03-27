import argparse
import random
import os
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration, Blip2Processor
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model  

CATEGORIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged-Cardiomediastinum", "Fracture", "Lung-Lesion", 
    "Lung-Opacity", "No-Finding", "Pleural-Effusion", 
    "Pleural_Other", "Pneumonia", "Pneumothorax", "Support-Devices"
]

CHARACTER_MAPPING = {category: chr(97 + i) for i, category in enumerate(CATEGORIES)}  # Maps to 'a', 'b', 'c', ...

def get_character_token_ids(tokenizer):
    return {char: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(char)[0]) for char in CHARACTER_MAPPING.values()}

class ChestXDataset(Dataset):
    def __init__(self, csv_file, image_folder, report_file, tokenizer, processor, max_length=512):
        self.data = self.load_data(csv_file, report_file)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def load_data(self, csv_file, report_file):
        data = {}
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_id = row['id']
                category = row['category']
                if category == "R":
                    label = "R"
                elif category == "U_text":
                    label = "T"
                elif category == "U_image":
                    label = "I"
                else:
                    label = "S"
                data[image_id] = {'label': label}

        with open(report_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_id = row['id']
                if image_id in data:
                    report = row['report']
                    data[image_id]['report'] = report

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id = list(self.data.keys())[idx]
        entry = self.data[image_id]
        image_path = os.path.join(self.image_folder, image_id)

        image = Image.open(image_path)
        image = self.processor(image, return_tensors="pt").pixel_values.squeeze(0)

        report = entry['report']
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
        label_map = {"R": 0, "T": 1, "I": 2, "S": 3}
        label = torch.tensor(label_map[entry['label']], dtype=torch.long)

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
    category_token_ids = {}
    for category in CATEGORIES:
        tokens = tokenizer.tokenize(category)
        print(f"Category: {category}, Tokens: {tokens}")
        category_token_ids[category] = tokenizer.convert_tokens_to_ids(tokens[0])
    return category_token_ids

def evaluate(tokenizer, model, dataloader, device):
    model.eval()
    total_correct, total = 0, 0
    all_labels, all_predictions = [], []
    token_ids = {token: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))[0] for token in ["R", "T", "I", "S"]}

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids, attention_mask, images, labels = (
            batch[key].to(device) for key in ["input_ids", "attention_mask", "image", "label"]
        )
        ids = batch["id"]

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images).logits[:, -1, :]
            rti_logits_batch = torch.stack([logits[:, token_ids[token]] for token in ["R", "T", "I", "S"]], dim=-1)
            predictions = torch.argmax(rti_logits_batch, dim=-1)

            total_correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    return {
        "accuracy": total_correct / total,
        "f1": f1_score(all_labels, all_predictions, average="macro"),
        "precision": precision_score(all_labels, all_predictions, average="macro"),
        "recall": recall_score(all_labels, all_predictions, average="macro"),
    }

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        else:
            BCE_loss = F.cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def train(model, train_dataloader, val_dataloader, tokenizer, device, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = FocalLoss(alpha=1, gamma=2)
    token_ids = {token: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))[0] for token in ["R", "T", "I", "S"]}
    best_f1 = -1

    for epoch in range(args.epochs):
        total_loss = 0
        model.train()

        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", leave=False)):
            input_ids, attention_mask, images, labels = (
                batch[key].to(device) for key in ["input_ids", "attention_mask", "image", "label"]
            )

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images).logits[:, -1, :]
            rti_logits_batch = torch.stack([logits[:, token_ids[token]] for token in ["R", "T", "I", "S"]], dim=-1)
            loss = criterion(rti_logits_batch, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        metrics = evaluate(tokenizer, model, val_dataloader, device)
        print(f"Epoch {epoch + 1}")
        print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
        print(f"Validation F1 Score: {metrics['f1']:.4f}")
        print(f"Validation Precision: {metrics['precision']:.4f}")
        print(f"Validation Recall: {metrics['recall']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            print("Model achieves better F1 score!")


        epoch_save_path = os.path.join(args.save_path, f"epoch_{epoch+1}") 
        os.makedirs(epoch_save_path, exist_ok=True)  
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
    parser.add_argument("--eval_steps", type=int, default=100, help="Number of steps between evaluations")

    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

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

    val_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False) 

    train(model, train_dataloader, val_dataloader, tokenizer, device, args)
