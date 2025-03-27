import json
import os
from collections import defaultdict

import csv
# import jsonlines
import numpy as np
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, roc_auc_score)

CATEGORIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", 
    "Enlarged-Cardiomediastinum", "Fracture", "Lung-Lesion", 
    "Lung-Opacity", "No-Finding", "Pleural-Effusion", 
    "Pleural_Other", "Pneumonia", "Pneumothorax", "Support-Devices"
]


def load_weights(weights_file):
    with open(weights_file, "r") as f:
        reader = csv.DictReader(f)
        return {row["id"]: eval(row["logits"]) for row in reader}


def load_and_transform_data(test_file, file_dir, subset_names, weights=None):
    results = defaultdict(
        lambda: {"logits": defaultdict(list), "target": None, "weights": {}}
    )
    with open(test_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data_id = row["id"]
            # Directly use the values from the CSV row
            results[data_id]["target"] = [int(row[category]) for category in CATEGORIES]


    for name in subset_names:
        file_path = os.path.join(file_dir, f"predictions_{name}.csv")   
        with open(file_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_id = row["id"]
                results[data_id]["logits"][name] = eval(row["logits"])
                if weights and data_id in weights:
                    model_indices = {"R": 0, "T": 1, "I": 2, "AS": 3}
                    if name == "U":
                        results[data_id]["weights"]["T"] = weights[data_id][model_indices["T"]]
                        results[data_id]["weights"]["I"] = weights[data_id][model_indices["I"]]
                    else:
                        results[data_id]["weights"][name] = weights[data_id][model_indices[name]]
    return results


def calculate_metrics(gths, preds, probabilities):
    # Convert gths and preds to NumPy arrays if they aren't already
    gths = np.array(gths)
    preds = np.array(preds)
    probabilities = np.array(probabilities)
    print(gths.shape)

    auc = roc_auc_score(gths, probabilities)

    return {
        "f1": f1_score(gths, preds, average="macro", zero_division=0),
        "precision": precision_score(gths, preds, average="macro", zero_division=0),
        "recall": recall_score(gths, preds, average="macro", zero_division=0),
        "auc": auc
    }


def get_predictions(results, fusion_strategy, *args):
    gths, preds, probs = [], [], []
    for data in results.values():
        predictions, probabilities = fusion_strategy(data["logits"], data["weights"], threshold=0.1, *args)
        gths.append(np.array(data["target"]))  # Convert to NumPy array
        preds.append(np.array(predictions))    # Convert to NumPy array
        probs.append(np.array(probabilities))
    # print(gths)
    return calculate_metrics(gths, preds, probs)


def baseline_fusion(logits, weights, threshold=0.5, *args):
    # Convert logits to a NumPy array if it's not already
    baseline_logits = np.array(logits["baseline"])
    
    # Calculate probabilities using the sigmoid function
    probabilities = 1 / (1 + np.exp(-baseline_logits))
    
    # Apply threshold to determine predictions
    predictions = (probabilities > threshold).astype(int)
    
    return predictions, probabilities

def weighted_sigmoid_rtis_fusion(logits, weights, threshold=0.5, *args):
    softmax_weights = {
        "R": np.exp(weights["R"])
        / (np.exp(weights["R"]) + np.exp(weights["T"]) + np.exp(weights["I"]) + np.exp(weights["AS"])),
        "T": np.exp(weights["T"])
        / (np.exp(weights["R"]) + np.exp(weights["T"]) + np.exp(weights["I"]) + np.exp(weights["AS"])),
        "I": np.exp(weights["I"])
        / (np.exp(weights["R"]) + np.exp(weights["T"]) + np.exp(weights["I"]) + np.exp(weights["AS"])),
        "AS": np.exp(weights["AS"])
        / (np.exp(weights["R"]) + np.exp(weights["T"]) + np.exp(weights["I"]) + np.exp(weights["AS"])),
    }
   
    weighted_logits = sum(
        softmax_weights[name] * np.array(logit)  # Directly use raw logits
        for name, logit in logits.items()
    )
    
    probabilities = 1 / (1 + np.exp(-weighted_logits))
    predictions = (probabilities > threshold).astype(int)
    return predictions, probabilities

def weighted_sigmoid_rus_fusion(logits, weights, threshold=0.5, *args):
    # Calculate softmax weights
    softmax_weights = {
        "R": np.exp(weights["R"])
        / (np.exp(weights["R"]) + np.exp(weights["T"]) + np.exp(weights["I"]) + np.exp(weights["AS"])),
        "U": (np.exp(weights["T"]) + np.exp(weights["I"]))
        / (np.exp(weights["R"]) + np.exp(weights["T"]) + np.exp(weights["I"]) + np.exp(weights["AS"])),
        "AS": np.exp(weights["AS"])
        / (np.exp(weights["R"]) + np.exp(weights["T"]) + np.exp(weights["I"]) + np.exp(weights["AS"])),
    }

    # Calculate softmax logits
    weighted_logits = sum(
        softmax_weights[name] * np.array(logit)  # Directly use raw logits
        for name, logit in logits.items()
    )
    
    
    # Apply threshold to determine multi-label predictions
    probabilities = 1 / (1 + np.exp(-weighted_logits))
    predictions = (probabilities > threshold).astype(int)
    return predictions, probabilities


def simple_average(logits, weights, threshold=0.5, *args):
    avg_logits = np.mean([logits[name] for name in logits], axis=0)

    probabilities = 1 / (1 + np.exp(-avg_logits))
    predictions = (probabilities > threshold).astype(int)
    return predictions, probabilities


def weighted_average(logits, weights, threshold=0.5, *args):
    weighted_logits = sum(weights[name] * np.array(logits[name]) for name in logits)

    probabilities = 1 / (1 + np.exp(-weighted_logits))
    predictions = (probabilities > threshold).astype(int)
    return predictions, probabilities


def max_fusion(logits, weights, threshold=0.5, *args):
    max_logits = np.max([logits[name] for name in logits], axis=0)

    probabilities = 1 / (1 + np.exp(-max_logits))
    predictions = (probabilities > threshold).astype(int)
    return predictions, probabilities


def softmax_fusion(logits, weights, threshold=0.5, *args):
    softmaxed_probs = np.mean(
        [np.exp(logits[name]) / np.sum(np.exp(logits[name])) for name in logits], axis=0
    )

    probabilities = 1 / (1 + np.exp(-softmaxed_probs))
    predictions = (probabilities > threshold).astype(int)
    return predictions, probabilities


def cascaded_fusion(logits, threshold, *args):
    softmaxed_probs = {
        name: np.exp(logit) / np.sum(np.exp(logit)) for name, logit in logits.items()
    }
    if max(softmaxed_probs["R"]) > threshold and max(softmaxed_probs["U"]) > threshold:
        return np.argmax(
            softmaxed_probs["R"]
            if max(softmaxed_probs["R"]) > max(softmaxed_probs["U"])
            else softmaxed_probs["U"]
        )
    return np.argmax(softmaxed_probs["AS"])


def main():
    AS_test_data_ids = []
    with open('data/chestx/split/AS_category_results.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            AS_test_data_ids.append(row['id'])

    R_test_data_ids = []
    with open('data/chestx/split/R_category_results.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            R_test_data_ids.append(row['id'])

    T_test_data_ids = []
    with open('data/chestx/split/T_category_results.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            T_test_data_ids.append(row['id'])

    I_test_data_ids = []
    with open('data/chestx/split/I_category_results.csv', mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            I_test_data_ids.append(row['id'])

    test_file = f"./data/chestx/proc/test.csv"
    file_dir = f"./predictions"
    weights_file = f"./predictions/predictions_fusion.csv"

    weights = load_weights(weights_file) if os.path.exists(weights_file) else None

    rtis_results = load_and_transform_data(test_file, file_dir, ["R", "T", "I", "AS"], weights)
    # rus_results = load_and_transform_data(test_file, file_dir, ["R", "U", "AS"], weights)
    baseline_results = load_and_transform_data(test_file, file_dir, ["baseline"], {})

    print(
        "Baseline Interaction Type Accuracy:",
        get_predictions(baseline_results, baseline_fusion),
    )

    if weights:
        print("RTIS Fusion:", get_predictions(rtis_results, weighted_sigmoid_rtis_fusion))
        # print("RUS Fusion:", get_predictions(rus_results, weighted_sigmoid_rus_fusion))

    print("Simple Average Fusion:", get_predictions(rtis_results, simple_average))
    print("Max Fusion:", get_predictions(rtis_results, max_fusion))
    print("Softmax Fusion:", get_predictions(rtis_results, softmax_fusion))

    subpart_results = {"AS": {}, "R": {}, "T": {}, "I": {}}
    subpart_baseline_results = {"AS": {}, "R": {}, "T": {}, "I": {}}
    for result in rtis_results:
        if result in AS_test_data_ids:
            subpart_results["AS"][result] = rtis_results[result]
            subpart_baseline_results["AS"][result] = baseline_results[result]
        elif result in R_test_data_ids:
            subpart_results["R"][result] = rtis_results[result]
            subpart_baseline_results["R"][result] = baseline_results[result]
        elif result in T_test_data_ids:
            subpart_results["T"][result] = rtis_results[result]
            subpart_baseline_results["T"][result] = baseline_results[result]
        elif result in I_test_data_ids:
            subpart_results["I"][result] = rtis_results[result]
            subpart_baseline_results["I"][result] = baseline_results[result]

    for interaction_type in ["AS", "R", "T", "I"]:
        print(
            f"{interaction_type} expert results on the {interaction_type} test set:",
            get_predictions(
                subpart_results[interaction_type],
                lambda logits, weights, threshold=0.5, *args: (
                    (1 / (1 + np.exp(-np.array(logits[interaction_type]))) > 0.2).astype(int), 
                    (1 / (1 + np.exp(-np.array(logits[interaction_type]))))
                ),
            ),
        )
        print(
            f"Baseline results on the {interaction_type} test set:",
            get_predictions(
                subpart_baseline_results[interaction_type],
                lambda logits, weights, threshold=0.5, *args: (
                    (1 / (1 + np.exp(-np.array(logits["baseline"]))) > threshold).astype(int), 
                    (1 / (1 + np.exp(-np.array(logits["baseline"]))))
                ),
            ),
        )
        print("=" * 10)


if __name__ == "__main__":
    main()
