import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

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

def evaluate_and_save_predictions(tokenizer, model, dataloader, device, output_csv):
    model.eval()
    character_token_ids = get_character_token_ids(tokenizer)
    predictions_list = []
    all_labels = []
    all_predictions = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        image_ids = batch["id"]
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

        for image_id, logit in zip(image_ids, character_logits):
            predictions_list.append({
                "id": image_id,
                "logits": logit.cpu().tolist()  # Convert logits to a list
            })

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    # Calculate metrics
    # accuracy = sum([pred == label for pred, label in zip(all_predictions, all_labels)]) / len(all_labels)
    all_labels = torch.tensor(all_labels)
    all_predictions = torch.tensor(all_predictions)
    
    f1 = f1_score(all_labels, all_predictions, average='macro')
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')

    # print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Ensure the directory for the output CSV exists
    output_dir = os.path.dirname(output_csv)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save predictions to CSV
    with open(output_csv, mode='w', newline='') as file:
        fieldnames = ["id", "logits"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference with BLIP-2 model on the chestx test dataset")
    parser.add_argument("--test_csv", type=str, default="proc/test.csv", help="Path to the CSV file with test data")
    parser.add_argument("--image_folder", type=str, default="proc/images", help="Path to the image folder")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="Path to save the predictions")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum length for tokenized sequences")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("Salesforce/blip2-opt-2.7b")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path)
    model.to(device)

    test_dataset = ChestXDataset(args.test_csv, args.image_folder, args.test_csv, tokenizer, processor, args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    evaluate_and_save_predictions(tokenizer, model, test_dataloader, device, args.output_csv)
