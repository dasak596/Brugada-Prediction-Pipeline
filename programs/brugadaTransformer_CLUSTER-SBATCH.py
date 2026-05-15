import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'   ###If script do not work, remove this part, the last I added
import json
import logging
import random
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score
from tqdm import tqdm
import itertools
from pathlib import Path
from collections import Counter
import argparse
from sklearn.metrics import classification_report, confusion_matrix
from copy import deepcopy

class CustomTransformerModel(nn.Module):
    def __init__(self, model_name_or_path, num_classes, unfreeze_layers=0):
        super().__init__()
        # Cargar el modelo base
        self.base_model = AutoModel.from_pretrained(model_name_or_path)
        self.num_classes = num_classes
        
        # Congelar todas las capas primero
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Descongelar las capas seleccionadas
        model_type = self.base_model.config.model_type.lower()
        if "distilbert" in model_type:
            layers = self.base_model.transformer.layer
        elif "roberta" in model_type or "bert" in model_type:
            layers = self.base_model.encoder.layer
        else:
            raise ValueError(f"Model type {model_type} not supported")
            
        for i, layer in enumerate(layers):
            if i < unfreeze_layers:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Capa de clasificación
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Usar el token [CLS] para clasificación
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_model_base_path", type=str, required=True,
                        help="Base path where models will be saved")
    return parser.parse_args()

# --- Configuration ---
args = parse_args()

# --- Configuration ---
CONFIG = {
    "SAVE_MODEL_BASE_PATH": args.save_model_base_path,
    "CSV_FILE_PATH": "./sintomaticos.csv",
    "DATA_DIR": "brugada_new",
    "NUM_CLASSES": 2,
    "MAX_SEQ_LENGTH": 512,
    "SEED": 42,
    "TRUE_LABEL_WEIGHT_MULTIPLIER": 2.0,
    "GRADIENT_ACCUMULATION_STEPS": 4,
    "PARAM_GRID": {
        "learning_rate": [2e-5, 1e-5, 3e-5],
        "batch_size": [16, 32, 64],
        "num_epochs": [3, 5, 7],
        "unfreeze_layers": [0, 2, 4],
    },
}


# Initialize distributed training if available
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        gpu = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(gpu)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        return True, rank, world_size, gpu
    return False, 0, 1, 0

is_distributed, RANK, WORLD_SIZE, LOCAL_RANK = setup_distributed()
DEVICE = torch.device(f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu")

# --- Logging setup ---
def setup_logging(output_dir):
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO if RANK in [-1, 0] else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()] if RANK in [-1, 0] else [logging.FileHandler(log_file)],
    )

# --- Seed for Reproducibility ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Data Loading and Preprocessing ---
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
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

def load_and_prepare_data(csv_path, tokenizer, max_seq_length, batch_size, data_dir):
    df = pd.read_csv(csv_path)
    positive_files = set(df["filename"].tolist())

    all_files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    texts = []
    labels = []

    for file in all_files:
        file_path = os.path.join(data_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        texts.append(text)
        labels.append(1 if file in positive_files else 0)

    # Split data into train and test
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=CONFIG["SEED"], stratify=labels
    )

    train_counts = Counter(train_labels)
    class_weights = torch.tensor([1.0 / train_counts[i] for i in range(CONFIG["NUM_CLASSES"])], dtype=torch.float).to(DEVICE)
    class_weights[1] *= CONFIG["TRUE_LABEL_WEIGHT_MULTIPLIER"]

    train_dataset = ClassificationDataset(train_texts, train_labels, tokenizer, max_seq_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_seq_length)

    # Use DistributedSampler if running in distributed mode
    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, class_weights

def load_and_modify_model(model_path, num_classes, unfreeze_layers):
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_path, "original_model"))
    model = CustomTransformerModel(
        os.path.join(model_path, "original_model"),
        num_classes=num_classes,
        unfreeze_layers=unfreeze_layers
    )
    return model, tokenizer

def load_best_model_for_eval(best_output_dir, config, device):
    """
    Loads the best fine-tuned model using its saved state_dict and parameters for evaluation.

    Args:
        best_output_dir (str): Path to the directory containing the best model, params, and tokenizer.
        config (dict): Configuration dictionary with relevant keys.
        device (torch.device): The device to load the model onto.

    Returns:
        model (CustomTransformerModel): The loaded and ready-for-evaluation model.
        tokenizer (AutoTokenizer): The tokenizer used for this model.
    """
    import json
    import torch
    from transformers import AutoTokenizer

    # Load best params
    with open(os.path.join(best_output_dir, "best_params.json")) as f:
        best_info = json.load(f)

    best_params = best_info["params"]

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(config["SAVE_MODEL_BASE_PATH"], "original_model"))

    # Instantiate model architecture
    model = CustomTransformerModel(
        model_name_or_path=os.path.join(config["SAVE_MODEL_BASE_PATH"], "original_model"),
        num_classes=config["NUM_CLASSES"],
        unfreeze_layers=best_params["unfreeze_layers"]
    )

    # Load weights
    state_dict_path = os.path.join(best_output_dir, "best_fine_tuned_model.pth")
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)
    model.eval()

    return model, tokenizer
# --- Training ---
def train_model(
    model,
    train_dataloader,
    val_dataloader,
    learning_rate,
    num_epochs,
    device,
    log,
    class_weights
):
    # Initialize mixed precision training
    scaler = GradScaler()
    
    # Initialize distributed training if available
    if is_distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[LOCAL_RANK],
            output_device=LOCAL_RANK
        )
    elif torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs for parallel training")
        model = nn.DataParallel(model)
    
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Adjust for gradient accumulation
    effective_batch_size = train_dataloader.batch_size * max(1, WORLD_SIZE) * CONFIG["GRADIENT_ACCUMULATION_STEPS"]
    log.info(f"Effective batch size: {effective_batch_size}")

    num_training_steps = len(train_dataloader) * num_epochs // CONFIG["GRADIENT_ACCUMULATION_STEPS"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    
    best_val_f1 = 0.0
    best_model_state = None
    best_model = None
    best_epoch = -1 

    for epoch in range(num_epochs):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0
        train_preds = []
        train_labels_all = []
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{num_epochs} - Training",
            disable=not (log.isEnabledFor(logging.INFO) or RANK not in [-1, 0]
        ))):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels) / CONFIG["GRADIENT_ACCUMULATION_STEPS"]
            
            scaler.scale(loss).backward()

            if (step + 1) % CONFIG["GRADIENT_ACCUMULATION_STEPS"] == 0 or (step + 1) == len(train_dataloader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * CONFIG["GRADIENT_ACCUMULATION_STEPS"]

            with torch.no_grad():
                predictions = torch.argmax(logits, dim=-1)
                train_preds.extend(predictions.cpu().numpy())
                train_labels_all.extend(labels.cpu().numpy())

        avg_train_loss = total_loss / len(train_dataloader)
        if RANK in [-1, 0]:
            log.info(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f}")
            train_f1 = f1_score(train_labels_all, train_preds, average='weighted')
            log.info(f"Epoch {epoch+1}/{num_epochs} - Train F1: {train_f1:.4f}")

        # --- Validation ---
        if RANK in [-1, 0]:
            if is_distributed:
                dist.barrier()
            model.eval()
            val_preds = []
            val_labels_all = []
            with torch.no_grad():
                for batch in val_dataloader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(logits, dim=-1)
                    val_preds.extend(predictions.cpu().numpy())
                    val_labels_all.extend(labels.cpu().numpy())

            val_f1 = f1_score(val_labels_all, val_preds, average='weighted')
            log.info(f"Epoch {epoch+1}/{num_epochs} - Val F1: {val_f1:.4f}")

            # Update best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_model_state = deepcopy(model.module.state_dict() if hasattr(model, 'module') else model.state_dict())

    if RANK in [-1, 0]:
        log.info("Finished Fine-Tuning")
    # Then return just the state_dict
    return best_val_f1, best_model_state, None

def evaluate_and_report(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].cpu().numpy()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    report = classification_report(all_labels, all_preds)
    print(report)
    
# --- Main Script ---
if __name__ == "__main__":
    args = parse_args()
    set_seed(CONFIG["SEED"])
    best_overall_f1 = 0.0
    best_overall_params = None
    best_model_state = None
    best_tokenizer = None
    all_results = []
    # Generate all combinations of parameters
    param_keys = list(CONFIG["PARAM_GRID"].keys())
    param_values_combinations = list(itertools.product(*CONFIG["PARAM_GRID"].values()))

    for param_values in param_values_combinations:
        torch.cuda.empty_cache()  # Clear cache before starting a new trial
        params = dict(zip(param_keys, param_values))
        #output_dir = os.path.join(config["SAVE_MODEL_BASE_PATH"], f"run_{'_'.join([f'{k}_{v}' for k,v in params.items()])}")
        #Path(output_dir).mkdir(parents=True, exist_ok=True)
        setup_logging(os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"]))
        log = logging.getLogger()

        if RANK in [-1, 0]:
            log.info("---------------------------------")
            log.info(f"Running with params: {params}")

        try:
            # 1. Load and Prepare Data
            model_path = os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"])
            model, tokenizer = load_and_modify_model(model_path, CONFIG["NUM_CLASSES"], params["unfreeze_layers"])
            train_dataloader, val_dataloader, class_weights = load_and_prepare_data(
                CONFIG["CSV_FILE_PATH"], tokenizer, CONFIG["MAX_SEQ_LENGTH"], params["batch_size"], CONFIG["DATA_DIR"]
            )

            # 2. Train the model
            val_f1, model_state, original_model = train_model(
                model,
                train_dataloader,
                val_dataloader,
                params["learning_rate"],
                params["num_epochs"],
                DEVICE,
                log,
                class_weights
            )

            if RANK in [-1, 0]:
                all_results.append({**params, "val_f1": val_f1})

                # Update best overall model if this is the best so far
                if val_f1 > best_overall_f1 and model_state is not None:
                    best_overall_f1 = val_f1
                    best_overall_params = params
                    best_model_state = model_state
                    best_tokenizer = tokenizer
                """
                # Save metrics for this run
                with open(os.path.join(output_dir, "best_params.json"), 'w') as f:
                    json.dump({**params, "val_f1": val_f1}, f, indent=4)
                log.info("---------------------------------")
                """
        except RuntimeError as e:
            if "out of memory" in str(e):
                log.error(f"CUDA Out of Memory error with parameters {params}: {e}")
                torch.cuda.empty_cache()
            else:
                log.error(f"An unexpected error occurred during training with parameters {params}: {e}")

    if is_distributed:
        dist.barrier()
    # After all runs, save the best models (only on main process)
    if RANK in [-1, 0] and best_model_state is not None:
        best_output_dir = os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"], "best_model")
        Path(best_output_dir).mkdir(parents=True, exist_ok=True)
        # Save the complete model
        torch.save(best_model_state, os.path.join(best_output_dir, "best_fine_tuned_model.pth"))
        # Save best metrics 
        with open(os.path.join(best_output_dir, "best_params.json"), 'w') as f:
            json.dump({"params": best_overall_params, "val_f1": best_overall_f1}, f, indent=4)

        
    if is_distributed:
        dist.barrier()
    if RANK in [-1, 0]:
        log.info("---------------------------------")
        log.info("Finished Parameter Sweeping")
        log.info(f"Best validation F1: {best_overall_f1}")
        log.info(f"Best parameters: {best_overall_params}")
        
        #############Classification Report##############
        if is_distributed:
            dist.barrier()
        if RANK in [-1, 0]:
            log.info("---------------------------------")
            log.info("Finished Parameter Sweeping")
            log.info(f"Best validation F1: {best_overall_f1}")
            log.info(f"Best parameters: {best_overall_params}")
            
            # Reload data to create val_dataloader for final evaluation
            _, val_dataloader, _ = load_and_prepare_data(
                CONFIG["CSV_FILE_PATH"], best_tokenizer, CONFIG["MAX_SEQ_LENGTH"], 
                best_overall_params["batch_size"], CONFIG["DATA_DIR"]
            )

            model, tokenizer = load_best_model_for_eval(best_output_dir, CONFIG, DEVICE)
    
            log.info("Evaluating best model on validation set...")
            # Evaluate
            evaluate_and_report(model, val_dataloader)
            ################################################

        log.info("All Results:")
        for result in all_results:
            log.info(result)
    if is_distributed:
        dist.destroy_process_group()