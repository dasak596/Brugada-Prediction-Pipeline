# --- Imports ---
import os
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tqdm import tqdm
import itertools
from pathlib import Path
from collections import Counter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import classification_report, confusion_matrix
import copy
from copy import deepcopy
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model_path", type=str, required=True, 
                        help="Base path to save the trained models")
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Path to the BERT model to use")
    return parser.parse_args()

# --- Configuration ---
args = parse_args()

CONFIG = {
    "SAVE_MODEL_BASE_PATH": args.save_model_path,
    "CSV_FILE_PATH": "./sintomaticos.csv",
    "DATA_DIR": "brugada_new",
    "NUM_CLASSES": 2,
    "MAX_SEQ_LENGTH": 512,
    "SEED": 42,
    "TRUE_LABEL_WEIGHT_MULTIPLIER": 2.0,
    "BERT_MODEL": args.bert_model,
    "PARAM_GRID": {
        "learning_rate": [2e-5, 1e-5, 3e-5],
        "batch_size": [16, 32, 64],
        "num_epochs": [3, 5, 7],
        "filter_sizes": [[10,20,30]], #[[14, 15, 16], [10,15,20], [10,20,30]],  # CNN filter sizes
        "num_filters": [100, 200, 300], # CNN filter size
    },
}


# --- Device and Distributed Init ---
def init_distributed_mode():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        is_distributed = True
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_distributed = False
        local_rank = 0
    return device, is_distributed, local_rank

DEVICE, IS_DISTRIBUTED, LOCAL_RANK = init_distributed_mode()



def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --- Dataset ---
class ClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=False,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# --- Model ---
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_sizes, num_filters, num_classes, pad_token_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        conved = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(c, dim=2)[0] for c in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


# --- Training ---
def train_model(model, train_loader, val_loader, params, device, log, class_weights, tokenizer, distributed=False):
    model.to(device)
    if distributed and dist.is_initialized():
        model = DDP(model, device_ids=[device.index])

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=params["learning_rate"])
    num_training_steps = len(train_loader) * params["num_epochs"] // params.get("gradient_accumulation_steps", 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, num_training_steps)
    scaler = GradScaler()

    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(params["num_epochs"]):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with autocast():
                logits = model(input_ids)
                loss = criterion(logits, labels)
                loss = loss / params.get("gradient_accumulation_steps", 1)

            scaler.scale(loss).backward()

            if (step + 1) % params.get("gradient_accumulation_steps", 1) == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()

        log.info(f"Epoch {epoch+1} - Loss: {total_loss / len(train_loader):.4f}")

        val_f1 = evaluate(model, val_loader, device)
        log.info(f"Epoch {epoch+1} - Val F1: {val_f1:.4f}")

        is_main_process = not distributed or (dist.is_initialized() and dist.get_rank() == 0)
        if val_f1 > best_val_f1 and is_main_process:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.module.state_dict() if distributed else model.state_dict())

    if distributed and dist.is_initialized():
        if dist.get_rank() == 0:
            log.info(f"Finished training. Best F1: {best_val_f1:.4f}")
    else:
        log.info(f"Finished training. Best F1: {best_val_f1:.4f}")

    return best_val_f1, best_model_state



def evaluate(model, val_loader, device):
    model.eval()
    preds, labels_all = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                logits = model(input_ids)
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())
    return f1_score(labels_all, preds, average="weighted")


# --- Data Loading ---
def load_and_prepare_data(csv_path, tokenizer, max_seq_length, batch_size, data_dir, distributed=False):
    df = pd.read_csv(csv_path)
    positive_files = set(df["filename"].tolist())
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    texts, labels = [], []
    for file in all_files:
        with open(os.path.join(data_dir, file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
        labels.append(1 if file in positive_files else 0)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=CONFIG["SEED"], stratify=labels
    )

    train_counts = Counter(train_labels)
    class_weights = torch.tensor([1.0 / train_counts[i] for i in range(CONFIG["NUM_CLASSES"])], dtype=torch.float).to(DEVICE)
    class_weights[1] *= CONFIG["TRUE_LABEL_WEIGHT_MULTIPLIER"]

    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_seq_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_seq_length)

    if distributed and dist.is_initialized():
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader, class_weights


def final_evaluate_model(model_class, model_params, state_dict, data_loader, device, tokenizer, log):
    embedding_dim = 768  # same as in config
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id

    model = model_class(vocab_size, embedding_dim, model_params["filter_sizes"], model_params["num_filters"], CONFIG["NUM_CLASSES"], pad_token_id)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    preds, labels_all = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            with autocast():
                logits = model(input_ids)
            predictions = torch.argmax(logits, dim=-1)
            preds.extend(predictions.cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    f1 = f1_score(labels_all, preds, average="weighted")
    report = classification_report(labels_all, preds)
    print(report)
    log.info(f"Final evaluation F1: {f1:.4f}")
    return f1

if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    param_combinations = list(itertools.product(*CONFIG["PARAM_GRID"].values()))
    param_keys = list(CONFIG["PARAM_GRID"].keys())

    overall_best_f1 = 0.0
    overall_best_state = None
    overall_best_params = None

    all_results = []  

    for param_values in param_combinations:
        params = dict(zip(param_keys, param_values))
        #output_dir = os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"], f"run_{'_'.join(f'{k}_{v}' for k, v in params.items())}")
        #Path(output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"]))
        log = logging.getLogger()

        log.info(f"Starting training with hyperparameters: {params}")
        print(f"Starting training with hyperparameters: {params}")

        tokenizer = AutoTokenizer.from_pretrained(CONFIG["BERT_MODEL"])
        embedding_dim = 768
        vocab_size = tokenizer.vocab_size
        pad_token_id = tokenizer.pad_token_id

        model = CNNClassifier(vocab_size, embedding_dim, params["filter_sizes"], params["num_filters"], CONFIG["NUM_CLASSES"], pad_token_id)
        train_loader, val_loader, class_weights = load_and_prepare_data(
            CONFIG["CSV_FILE_PATH"],
            tokenizer,
            CONFIG["MAX_SEQ_LENGTH"],
            params["batch_size"],
            CONFIG["DATA_DIR"],
            distributed=IS_DISTRIBUTED
        )

        val_f1, best_model_state = train_model(model, train_loader, val_loader, params, DEVICE, log, class_weights, tokenizer, distributed=IS_DISTRIBUTED)

        all_results.append({"params": params, "val_f1": val_f1})  

        is_main_process = not IS_DISTRIBUTED or (dist.is_initialized() and dist.get_rank() == 0)
        if is_main_process:
            #with open(os.path.join(output_dir, "params_metrics.json"), 'w') as f:
                #json.dump({**params, "val_f1": val_f1}, f, indent=4)

            if val_f1 > overall_best_f1:
                overall_best_f1 = val_f1
                overall_best_state = best_model_state
                overall_best_params = params

    # Imprimir o guardar la lista completa al final
    if not IS_DISTRIBUTED or (dist.is_initialized() and dist.get_rank() == 0):
        print("\nSummary of all validation F1 scores:")
        for res in all_results:
            print(f"F1: {res['val_f1']:.4f} - Params: {res['params']}")

        # También guardarlo en un archivo JSON
        with open(os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"], "all_val_f1_results.json"), "w") as f:
            json.dump(all_results, f, indent=4)

    if not IS_DISTRIBUTED or (dist.is_initialized() and dist.get_rank() == 0):
        final_model_path = os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"], "final_best_model.pth")
        torch.save(overall_best_state, final_model_path)

        tokenizer = AutoTokenizer.from_pretrained(CONFIG["BERT_MODEL"])
        _, val_loader, _ = load_and_prepare_data(
            CONFIG["CSV_FILE_PATH"],
            tokenizer,
            CONFIG["MAX_SEQ_LENGTH"],
            overall_best_params["batch_size"],
            CONFIG["DATA_DIR"],
            distributed=IS_DISTRIBUTED
        )

        final_f1 = final_evaluate_model(
            CNNClassifier,
            overall_best_params,
            overall_best_state,
            val_loader,
            DEVICE,
            tokenizer,
            logging.getLogger()
        )

        
        with open(os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"], "final_evaluation_metrics.json"), 'w') as f:
            json.dump({"final_val_f1": final_f1, "best_params": overall_best_params}, f, indent=4)

        logging.info(f"Final evaluation completed. F1: {final_f1:.4f} with best parameters: {overall_best_params}")
        print(f"Final evaluation completed. F1: {final_f1:.4f} with best parameters: {overall_best_params}")

    if IS_DISTRIBUTED and dist.is_initialized():
        dist.destroy_process_group()