import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoTokenizer, Blip2ForConditionalGeneration
from tqdm import tqdm
from PIL import Image

CATEGORIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged-Cardiomediastinum", "Fracture", "Lung-Lesion", 
    "Lung-Opacity", "No-Finding", "Pleural-Effusion", 
    "Pleural_Other", "Pneumonia", "Pneumothorax", "Support-Devices"
]

CHARACTER_MAPPING = {category: chr(97 + i) for i, category in enumerate(CATEGORIES)}

class ChestXInferenceDataset(Dataset):
    def __init__(self, csv_file, image_folder, tokenizer, processor, max_length=512):
        self.data = self.load_data(csv_file)
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length

    def load_data(self, csv_file):
        data = []
        with open(csv_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_id = row['id']
                report = row['report']
                data.append({'id': image_id, 'report': report})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_id = entry['id']
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

        return {
            "input_ids": text_encoding["input_ids"].squeeze(),
            "attention_mask": text_encoding["attention_mask"].squeeze(),
            "image": image,
            "id": image_id,
        }

def evaluate_and_save_predictions(tokenizer, model, dataloader, device, output_csv):
    model.eval()
    token_ids = {token: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))[0] for token in ["R", "T", "I", "S"]}
    predictions_list = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        image_ids = batch["id"]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=images)
            logits = outputs.logits[:, -1, :]
            character_logits = torch.stack(
                [logits[:, token_id] for token_id in token_ids.values()],
                dim=-1,
            )
            predictions = torch.argmax(character_logits, dim=-1)

        for image_id, prediction, logit in zip(image_ids, predictions, character_logits):
            predictions_list.append({
                "image_id": image_id,
                "prediction": prediction.item(),
                "logits": logit.cpu().tolist()  # Convert logits to a list
            })

    # Save predictions to CSV
    with open(output_csv, mode='w', newline='') as file:
        fieldnames = ["image_id", "prediction", "logits"]
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

    test_dataset = ChestXInferenceDataset(args.test_csv, args.image_folder, tokenizer, processor, args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    evaluate_and_save_predictions(tokenizer, model, test_dataloader, device, args.output_csv)
