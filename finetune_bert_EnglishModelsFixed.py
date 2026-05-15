import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, DistilBertForTokenClassification, BertForTokenClassification, AutoModelForMaskedLM, DistilBertForMaskedLM, AutoModelForTokenClassification, RobertaForTokenClassification, AdamW
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
import itertools
import random
import numpy as np

# Configuration
DATA_DIR = "./multicardioner/track1/"
TRAIN_DIR = os.path.join(DATA_DIR, "train/brat")
DEV_DIR = os.path.join(DATA_DIR, "dev/brat")
TEST_DIR = os.path.join(DATA_DIR, "test/brat")
#MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
#MODEL_NAME = "medicalai/ClinicalBERT
#MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
#MODEL_NAME = "llange/xlm-roberta-large-spanish-clinical"
MODEL_NAME = "./ClinicalBERT"  #Here I'm using local downloaded repository instead of github repository version
MAX_LEN = 512
#SAVE_MODEL_BASE_PATH = "./fine_tuned_SpanishClinicaROBERT"
#SAVE_MODEL_BASE_PATH = "./fine_tuned_SpanishBERT"
SAVE_MODEL_BASE_PATH = "./fine_tuned_ClinicalDistilBERT"
#SAVE_MODEL_BASE_PATH = "./fine_tuned_BioClinicalBERT"

SEED = 42

# --- Seed for Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Hyperparameter sweep space
param_grid = {
    "batch_size": [8, 16, 32],
    "epochs": [3, 5, 7],
    "learning_rate": [2e-5, 3e-5, 5e-5],
}

def read_brat_data(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            txt_path = os.path.join(directory, filename)
            ann_path = os.path.join(directory, filename[:-4] + ".ann")

            with open(txt_path, "r", encoding="utf-8") as f_txt:
                text = f_txt.read()

            annotations = []
            if os.path.exists(ann_path):
                with open(ann_path, "r", encoding="utf-8") as f_ann:
                    for line in f_ann:
                        parts = line.strip().split("\t")
                        if len(parts) == 3 and parts[0].startswith("T"):
                            tag_parts = parts[1].split()
                            label = tag_parts[0]
                            start = int(tag_parts[1])
                            end = int(tag_parts[-1])
                            annotations.append((start, end, label))
            data.append({"text": text, "annotations": annotations})
    return data

def convert_to_iob2(data, tokenizer):
    dataset = []
    for item in data:
        text = item["text"]
        annotations = item["annotations"]
        encoding = tokenizer(text, return_offsets_mapping=True, truncation=True, padding="max_length", max_length=MAX_LEN)
        labels = ["O"] * len(encoding["input_ids"])

        for ann_start, ann_end, label in annotations:
            for i, (start, end) in enumerate(encoding["offset_mapping"]):
                if start is None or end is None:
                    continue
                if start >= ann_start and end <= ann_end:
                    labels[i] = f"B-{label}" if start == ann_start else f"I-{label}"
        
        dataset.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": labels,
        })
    return dataset

class MedicalDataset(Dataset):
    def __init__(self, data, label_map):
        self.data = data
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids = torch.tensor(item["input_ids"])
        attention_mask = torch.tensor(item["attention_mask"])
        labels = torch.tensor([self.label_map[label] for label in item["labels"]])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def train(model, train_dataloader, dev_dataloader, optimizer, epochs, label_map, device):
    best_dev_f1 = 0.0
    best_model = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            model.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = total_loss / len(train_dataloader)
        dev_f1 = evaluate(model, dev_dataloader, label_map, device)
        
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_model = model
    return best_model, best_dev_f1

def evaluate(model, dataloader, label_map, device):
    model.eval()
    y_true, y_pred = [], []
    label_map_inv = {v: k for k, v in label_map.items()}
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=2)
            
            for i in range(labels.shape[0]):
                true_labels = [label_map_inv[label.item()] for label in labels[i] if label.item() in label_map_inv]
                pred_labels = [label_map_inv[pred.item()] for pred in predictions[i] if pred.item() in label_map_inv]
                y_true.append(true_labels)
                y_pred.append(pred_labels)
    
    f1 = f1_score(y_true, y_pred, mode='strict', scheme=IOB2)
    print(classification_report(y_true, y_pred, mode='strict', scheme=IOB2))
    return f1

if __name__ == "__main__":
    #For reproducibility
    set_seed(SEED)
    # Load tokenizer (model loaded inside the loop)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load and preprocess data (only once) -> output is a list of dictionaries: [{'text': text_1, 'annotations': annotations_1},...,{'text': text_n, 'annotations': annotations_n}] where each annotation is a list of tuples [(strart_1, end_1, label_1), ... , (start_m, end_m, label_m)]
    train_data = read_brat_data(TRAIN_DIR)
    dev_data = read_brat_data(DEV_DIR)
    test_data = read_brat_data(TEST_DIR)
    
    #Tokenize and convert data to IOB2 format -> output is a list of dictionaries [{'input_ids': [12, 34, 14,..., 0,0,...], 'attention_mask': [1,1,1,1...,0,0,...], 'labels': [0, B_enfermdedad, I-enfermedad, 0, B_Enfermedad, I-Enfermedad, ..., 0, 0,...]},...,{'input_ids': [56, 4, 33,..., 0,0,...], 'attention_mask': [1,1,1,1...,0,0,...], 'labels': [0, O-Enfermedad, B-Enfermedad, O-Enfermedad, 0, B-Enfermedad, ..., 0, 0,...]}] where each key in a dictionary is a list
    train_processed = convert_to_iob2(train_data, tokenizer)
    dev_processed = convert_to_iob2(dev_data, tokenizer)
    test_processed = convert_to_iob2(test_data, tokenizer)
    
    # Create label map (only once)
    #all_labels = {label for item in train_processed + dev_processed + test_processed for label in item["labels"]}
    #label_map = {label: i for i, label in enumerate(all_labels)}
    all_labels = sorted({label for item in train_processed + dev_processed + test_processed for label in item["labels"]})
    label_map = {label: i for i, label in enumerate(all_labels)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate all combinations of hyperparameters
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_f1_overall = 0.0
    best_model_overall = None
    best_params_overall = None
    
    for params in param_combinations:
        print(f"Training with hyperparameters: {params}")
        #model = DistilBertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map)).to(device)
        #model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, num_labels=len(label_map)).to(device)  
        #model = DistilBertForMaskedLM.from_pretrained(MODEL_NAME, num_labels=len(label_map)).to(device)
        #model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map)).to(device)     #Good for BERT models but not for distriBERT models like CLINICAL
        #model = RobertaForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map)).to(device)
        #model = AutoModel.from_pretrained(MODEL_NAME, num_labels=len(label_map)).to(device)      #Good for BERT and distriBERT models, but cannot handle labels
        model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_map)).to(device)  
        optimizer = AdamW(model.parameters(), lr=params["learning_rate"])
        train_dataset = MedicalDataset(train_processed, label_map)
        dev_dataset = MedicalDataset(dev_processed, label_map)
        train_dataloader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        dev_dataloader = DataLoader(dev_dataset, batch_size=params["batch_size"], shuffle=False)
        
        # Train the model for the current hyperparameter combination
        best_model, best_dev_f1 = train(
            model, train_dataloader, dev_dataloader, optimizer, params["epochs"], label_map, device
        )

        if not best_model:      # If model output is 0, which does not return a best_model we skip the model save, so it does not 
            continue 

        model_save_path = os.path.join(
            SAVE_MODEL_BASE_PATH,
            f"model_lr_{params['learning_rate']}_bs_{params['batch_size']}_epochs_{params['epochs']}",
        )
        best_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        with open(os.path.join(model_save_path, "label_map.json"), "w") as f:
            json.dump(label_map, f)

        # Update best overall model if needed
        if best_dev_f1 > best_f1_overall:
            best_f1_overall = best_dev_f1
            best_model_overall = best_model
            best_params_overall = params

    # Evaluate the overall best model on the test set
    print(f"Best hyperparameters: {best_params_overall}")
    print(f"Best F1 score on dev set: {best_f1_overall:.4f}")

    
    test_dataset = MedicalDataset(test_processed, label_map)
    test_dataloader = DataLoader(test_dataset, batch_size=best_params_overall["batch_size"], shuffle=False)
    
    print("Evaluating best model on the test set:")
    test_f1 = evaluate(best_model_overall, test_dataloader, label_map, device)
    print(f"Test F1 score: {test_f1:.4f}")

    # Save the overall best model
    best_model_save_path = os.path.join(SAVE_MODEL_BASE_PATH, "best_model")
    best_model_overall.save_pretrained(best_model_save_path)
    tokenizer.save_pretrained(best_model_save_path)
    with open(os.path.join(best_model_save_path, "label_map.json"), "w") as f:
        json.dump(label_map, f)
    with open(os.path.join(best_model_save_path, "best_params.json"), "w") as f:
        json.dump(best_params_overall, f)

    print(f"Best model saved to {best_model_save_path}")
