import os
import csv
import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm

# Model configuration
MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DEVICE = "cuda"
BATCH_SIZE = 1  # Process one report at a time for simplicity

# File paths
CSV_FILE = "data/chestx/proc/train.csv"
OUTPUT_CSV = "data/chestx/split/unimodal_text_train.csv"

def load_reports(csv_file, limit=None):
    """Load reports from a CSV file, with an optional limit."""
    reports = []
    with open(csv_file, mode='r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            if limit is not None and i >= limit:
                break
            report = row['report']
            image_id = row['id']
            reports.append((image_id, report))
    
    return reports

def process_reports(reports, model, csv_file):
    """Process each report and generate responses."""
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

    sampling_params = SamplingParams(temperature=0.5, top_p=0.95, max_tokens=5000)  # Define sampling parameters

    for image_id, report in tqdm(reports, desc="Processing Reports"):
        # Prepare the prompt
        # f"If you can find any tiny clue of an anomaly from the list, you could select it even if you are not sure. "
        prompt = (
            f"This is a medical report of a patient. "
            f"Please act as a radiologist to analyze the report and predict medical conditions of the patient. "
            f"Report: {report}"
            f"Your answer should be the most likely conditions from the list: "
            f"[{','.join(categories)}] "
            f"In most cases, you should only select one condition in the list, but if you really believe there are multiple conditions in the report, you could answer all of them separated by commas. "
            f"You could answer 'No-Finding' only if you cannot find anything in the report. "
            f"You could answer 'Support-Devices' only if you see some devices in the report. "
            f"Your final answer should be only category names from the list, just the words, do not repeat category, do not add any other text, sentence, markdown, explanation. or analysis.\n"
        )

        # Generate response
        outputs = model.generate([prompt], sampling_params)
        
        # Decode the response
        full_answer = outputs[0].outputs[0].text.strip()
        # Extract the answer after </think>
        if '</think>' in full_answer:
            answer = full_answer.split('</think>')[-1].strip()
        else:
            answer = ''
        # answer = outputs[0].outputs[0].text.strip()

        # Store the answer and image_id for later processing
        results.append({"image_id": image_id, "answer": answer})

    # Parse responses and add ground truth outside the loop
    for result in results:
        answer = result["answer"]
        image_id = result["image_id"]

        # Parse the model's response
        response_categories = {category: 0 for category in categories}
        for category in answer.split(','):
            category = category.strip()
            if category in response_categories:
                response_categories[category] = 1

        # Get ground truth
        ground_truth = ground_truths.get(image_id, {category: 0 for category in categories})

        # Prepare result entry
        result_entry = {"image_id": image_id, "answer": answer}
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
    fieldnames = ["image_id", "answer"] + \
                 [f"{category}_pred" for category in categories] + \
                 [f"{category}_true" for category in categories]

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

def main():
    # Load model using vllm
    model = LLM(model=MODEL_PATH)

    # Ensure the output directory exists
    output_dir = os.path.dirname(OUTPUT_CSV)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load reports with an optional limit
    reports = load_reports(CSV_FILE, limit=None)

    # Process reports and get results
    results = process_reports(reports, model, CSV_FILE)

    # Save results to CSV
    save_results_to_csv(results, OUTPUT_CSV)

if __name__ == "__main__":
    main()
