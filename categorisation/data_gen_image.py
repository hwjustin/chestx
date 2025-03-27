# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import csv
import torch
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
from tqdm import tqdm
from PIL import Image

# Model configuration
MODEL_PATH = "deepseek-ai/deepseek-vl2-tiny"
DEVICE = "cuda"
BATCH_SIZE = 1  # Process one image at a time for simplicity

# File paths
CSV_FILE = "data/chestx/proc/train.csv"
OUTPUT_CSV = "data/chestx/split/train/unimodal_image_train.csv"
IMAGE_FOLDER = "data/chestx/proc/images/"

def load_image_paths(csv_file, limit=None):
    """Load image paths from a CSV file, with an optional limit."""
    image_paths = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            image_id = row['id']
            image_path = os.path.join(IMAGE_FOLDER, image_id)
            image_paths.append(image_path)
    
    return image_paths

def process_images(image_paths, vl_chat_processor, tokenizer, vl_gpt, csv_file):
    """Process each image and generate responses."""
    results = []
    categories = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
        "Enlarged-Cardiomediastinum", "Fracture", "Lung-Lesion", 
        "Lung-Opacity", "No-Finding", "Pleural-Effusion", 
        "Pleural_Other", "Pneumonia", "Pneumothorax", "Support-Devices"
    ]

    # Load ground truth from CSV
    ground_truths = {}
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_id = row['id']
            ground_truths[image_id] = {category: int(row[category]) for category in categories}

    for image_path in tqdm(image_paths, desc="Processing Images"):
        # Prepare the conversation
        #   "If you believe there is no condition in the image, you could answer 'No-Finding'."
        conversation = [
            {
                "role": "<|User|>",
                "content": (
                    "<image>\n<|grounding|>This image is a chest x-ray image of a patient. "
                    "Now, please act as a radiologist to analyze the image and predict medical conditions of the patient. "
                    "You could select conditions only from this list: "
                    "[Atelectasis,Cardiomegaly,Consolidation,Edema,Enlarged-Cardiomediastinum,Fracture,Lung-Lesion,Lung-Opacity,No-Finding,Pleural-Effusion,Pleural_Other,Pneumonia,Pneumothorax,Support-Devices] "
                    "In most case, you should only select one condition in the list, but if you really believe there are multiple conditions in the image, you could answer all of them separated by commas."
                    "If you can find any tiny clue of an anomaly from the list, you could select it even you are not sure"
                    "You could answer 'No-Finding' only if you cannot find anything in the image."
                    "Please do not add any other text in your answer, only conditions from the list are permitted."                
                    ),
                "images": [image_path],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # Load images and prepare inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(DEVICE)

        # Generate embeddings and response
        with torch.no_grad():
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = vl_gpt.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask.to(DEVICE),
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

            # Decode the response
            answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            image_id = os.path.basename(image_path)

        # Store the answer and image_id for later processing
        results.append({"id": image_id, "answer": answer})

    # Parse responses and add ground truth outside the loop
    for result in results:
        answer = result["answer"]
        image_id = result["id"]

        # Parse the model's response
        response_categories = {category: 0 for category in categories}
        for category in answer.split(','):
            category = category.strip()
            if category in response_categories:
                response_categories[category] = 1

        # Get ground truth
        ground_truth = ground_truths.get(image_id, {category: 0 for category in categories})

        # Prepare result entry
        result_entry = {"id": image_id, "answer": answer}
        for category in categories:
            result_entry[f"{category}_pred"] = response_categories[category]
            result_entry[f"{category}_true"] = ground_truth[category]

        # Update the result with parsed data
        result.update(result_entry)

    return results

def save_results_to_csv(results, output_csv):
    """Save the results to a CSV file."""
    categories = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
        "Enlarged-Cardiomediastinum", "Fracture", "Lung-Lesion", 
        "Lung-Opacity", "No-Finding", "Pleural-Effusion", 
        "Pleural_Other", "Pneumonia", "Pneumothorax", "Support-Devices"
    ]
    fieldnames = ["id", "answer"] + \
                 [f"{category}_pred" for category in categories] + \
                 [f"{category}_true" for category in categories]

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    # Load model and processor
    dtype = torch.bfloat16
    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(MODEL_PATH)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=dtype
    )
    vl_gpt = vl_gpt.cuda().eval()

    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load image paths with an optional limit
    image_paths = load_image_paths(CSV_FILE, limit=None)

    # Process images and get results
    results = process_images(image_paths, vl_chat_processor, tokenizer, vl_gpt, CSV_FILE)

    # Save results to CSV
    save_results_to_csv(results, OUTPUT_CSV)

if __name__ == "__main__":
    main()
