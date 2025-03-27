import csv
from collections import Counter

def read_predictions(file_path):
    predictions = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_id = row['image_id']
            predictions[image_id] = {key: int(value) for key, value in row.items() if '_pred' in key}
    return predictions

def read_ground_truth(file_path):
    ground_truths = {}
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            image_id = row['image_id']
            ground_truths[image_id] = {key: int(value) for key, value in row.items() if '_true' in key}
    return ground_truths

def select_subset_ids(text_preds, image_preds, ground_truths):
    R_ids, U_text_ids, U_image_ids, AS_ids = [], [], [], []
    for image_id, gth in ground_truths.items():
        text_pred = text_preds.get(image_id, {})
        image_pred = image_preds.get(image_id, {})
        
        if not text_pred or not image_pred:
            continue
        
        text_correct = any(text_pred.get(key.replace('_true', '_pred'), 0) == 1 and value == 1 for key, value in gth.items())
        image_correct = any(image_pred.get(key.replace('_true', '_pred'), 0) == 1 and value == 1 for key, value in gth.items())
        
        if text_correct and image_correct:
            R_ids.append(image_id)
        elif text_correct:
            U_text_ids.append(image_id)
        elif image_correct:
            U_image_ids.append(image_id)
        else:
            AS_ids.append(image_id)
    
    return R_ids, U_text_ids, U_image_ids, AS_ids

def save_results_to_csv(file_path, R_ids, U_text_ids, U_image_ids, AS_ids):
    """Save the categorized results to a CSV file."""
    # Save combined results
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'category'])
        for image_id in R_ids:
            writer.writerow([image_id, 'R'])
        for image_id in U_text_ids:
            writer.writerow([image_id, 'U_text'])
        for image_id in U_image_ids:
            writer.writerow([image_id, 'U_image'])
        for image_id in AS_ids:
            writer.writerow([image_id, 'AS'])

    # Save R category results
    with open('R_category_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'category'])
        for image_id in R_ids:
            writer.writerow([image_id, 'R'])

    # Save U category results (combining U_text and U_image)
    with open('U_category_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'category'])
        for image_id in U_text_ids:
            writer.writerow([image_id, 'U_text'])
        for image_id in U_image_ids:
            writer.writerow([image_id, 'U_image'])

    # Save AS category results
    with open('AS_category_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_id', 'category'])
        for image_id in AS_ids:
            writer.writerow([image_id, 'AS'])

def main():
    text_preds = read_predictions('results_text.csv')
    image_preds = read_predictions('results_image.csv')
    ground_truths = read_ground_truth('results_image.csv')  # Assuming ground truth is the same in both files

    R_ids, U_text_ids, U_image_ids, AS_ids = select_subset_ids(text_preds, image_preds, ground_truths)

    print("R_ids:", len(R_ids))
    print("U_text_ids:", len(U_text_ids))
    print("U_image_ids:", len(U_image_ids))
    print("AS_ids:", len(AS_ids))

    save_results_to_csv('chestx_split_results.csv', R_ids, U_text_ids, U_image_ids, AS_ids)

if __name__ == "__main__":
    main()
