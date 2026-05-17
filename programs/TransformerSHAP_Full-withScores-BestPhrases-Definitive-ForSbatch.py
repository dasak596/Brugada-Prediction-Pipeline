import os
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from transformers import AutoTokenizer, AutoModel
from torch import nn
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.patches as patches


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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


def setup_logging(output_dir, combination_name, overwrite=True):
    """
    Set up fresh logging for each combination
    
    Args:
        output_dir: Directory to store log files
        combination_name: Identifier for this combination
        overwrite: If True, clears previous log content (default: True)
    """
    log_file = os.path.join(output_dir, f'training_{combination_name}.log')
    
    # Clear previous log file if requested
    if overwrite and os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('')  # Truncate file
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Remove all existing handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # Set up new handlers with explicit file mode
    file_handler = logging.FileHandler(log_file, mode='w')  # 'w' for overwrite
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

# Configuration matching your training setup
def get_config(model_path):
    return {
        "SAVE_MODEL_BASE_PATH": model_path,
        "NUM_CLASSES": 2,
        "CSV_FILE_PATH": "./sintomaticos.csv",
        "DATA_DIR": "brugada_new",
        "MAX_SEQ_LENGTH": 512,
        "TRUE_LABEL_WEIGHT_MULTIPLIER": 2.0,
        "SEED": 42
    }
    
    
# ========== Custom Model Class (to match your training setup) ==========
class CustomTransformerModel(nn.Module):
    def __init__(self, model_name_or_path, num_classes, unfreeze_layers=0):
        super().__init__()
        # Load base model
        self.base_model = AutoModel.from_pretrained(model_name_or_path)
        self.num_classes = num_classes
        
        # Freeze all layers first
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Unfreeze selected layers
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
        
        # Classification layer
        self.classifier = nn.Linear(self.base_model.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token for classification
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits



class BatchedHedgeExplainer:
    def __init__(self, model, tokenizer, device, pred_batch_size, max_phrase_len=20, min_score_threshold=0.1, 
                 max_splits_per_span=3, max_seq_length=512, num_shuffle_samples=100):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_phrase_len = max_phrase_len
        self.min_score_threshold = min_score_threshold
        self.max_splits_per_span = max_splits_per_span
        self.max_seq_length = max_seq_length
        self.pred_batch_size = pred_batch_size
        self.num_shuffle_samples = num_shuffle_samples  # For cohesion-score
        self.cache = {}

    def predict_probas(self, texts):
        """Batch prediction with caching"""
        uncached_texts = [t for t in texts if t not in self.cache]
        if uncached_texts:
            encodings = self.tokenizer(
                uncached_texts,
                add_special_tokens=True,
                max_length=self.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(logits, dim=1).cpu().numpy()

            for text, prob in zip(uncached_texts, probs):
                self.cache[text] = prob

        return np.array([self.cache[t] for t in texts])

    def calculate_aopc(self, text, target_label, k_percent=20):
        """Calculate Area Over the Perturbation Curve"""
        words = text.split()
        n_words = len(words)
        k = max(1, int(n_words * k_percent / 100))
        
        # Get original prediction
        original_prob = self.predict_probas([text])[0][target_label]
        
        # Get importance scores for all words
        word_scores = []
        for i in range(n_words):
            masked_text = ' '.join(words[:i] + words[i+1:])
            masked_prob = self.predict_probas([masked_text])[0][target_label]
            word_scores.append((words[i], original_prob - masked_prob))
        
        # Sort words by importance
        word_scores.sort(key=lambda x: x[1], reverse=True)
        top_k_words = [w[0] for w in word_scores[:k]]
        
        # Create perturbed text by removing top k words
        perturbed_text = ' '.join([w for w in words if w not in top_k_words])
        perturbed_prob = self.predict_probas([perturbed_text])[0][target_label]
        
        return original_prob - perturbed_prob

    def calculate_log_odds(self, text, target_label, r_percent=20):
        """Calculate Log-odds score"""
        words = text.split()
        n_words = len(words)
        r = max(1, int(n_words * r_percent / 100))
        
        # Get original prediction
        original_prob = self.predict_probas([text])[0][target_label]
        
        # Get importance scores for all words
        word_scores = []
        for i in range(n_words):
            masked_text = ' '.join(words[:i] + ['[PAD]'] + words[i+1:])
            masked_prob = self.predict_probas([masked_text])[0][target_label]
            word_scores.append((words[i], original_prob - masked_prob))
        
        # Sort words by importance
        word_scores.sort(key=lambda x: x[1], reverse=True)
        top_r_words = [w[0] for w in word_scores[:r]]
        
        # Create perturbed text by masking top r words
        perturbed_text = ' '.join(['[PAD]' if w in top_r_words else w for w in words])
        perturbed_prob = self.predict_probas([perturbed_text])[0][target_label]
        
        return np.log(perturbed_prob / original_prob)

    def calculate_cohesion_score(self, text, target_label, top_span):
        """Calculate cohesion score for a given span"""
        words = text.split()
        original_prob = self.predict_probas([text])[0][target_label]
        
        # Find the span in the original text
        span_words = top_span['span'].split()
        span_start = None
        for i in range(len(words) - len(span_words) + 1):
            if words[i:i+len(span_words)] == span_words:
                span_start = i
                break
        
        if span_start is None:
            return 0.0
        
        span_end = span_start + len(span_words)
        outside_words = words[:span_start] + words[span_end:]
        
        total_diff = 0.0
        for _ in range(self.num_shuffle_samples):
            # Shuffle the span words
            shuffled_span = np.random.permutation(span_words)
            # Insert back at random position
            insert_pos = np.random.randint(0, len(outside_words) + 1)
            shuffled_text = ' '.join(
                outside_words[:insert_pos] + list(shuffled_span) + outside_words[insert_pos:]
            )
            shuffled_prob = self.predict_probas([shuffled_text])[0][target_label]
            total_diff += (original_prob - shuffled_prob)
        
        return total_diff / self.num_shuffle_samples

    def explain(self, text, target_label):
        """Main explanation method with added evaluation scores"""
        words = text.split()
        
        # First build the explanation tree
        tree = self._recursive_split(words, target_label)
        
        # Calculate evaluation metrics
        tree['aopc'] = self.calculate_aopc(text, target_label)
        tree['log_odds'] = self.calculate_log_odds(text, target_label)
        
        # Find the most important span for cohesion score
        top_span = self._find_top_span(tree)
        if top_span:
            tree['cohesion_score'] = self.calculate_cohesion_score(text, target_label, top_span)
        else:
            tree['cohesion_score'] = 0.0
            
        return tree

    def _recursive_split(self, span_words, target_label):
        """Recursive splitting function (same as your original recursive_split)"""
        if len(span_words) == 1:
            prob = self.predict_probas([' '.join(span_words)])[0][target_label]
            return {
                'span': ' '.join(span_words),
                'score': float(prob),
                'interaction_score': None,
                'children': []
            }

        span_len = len(span_words)
        split_points = np.linspace(1, span_len - 1, num=min(self.max_splits_per_span, span_len - 1), dtype=int)
        candidates = []
        candidate_texts = []

        for j in split_points:
            left, right = span_words[:j], span_words[j:]
            left_text = ' '.join(left)
            right_text = ' '.join(right)

            left_prob = self.cache.get(left_text)
            right_prob = self.cache.get(right_text)

            if left_prob is not None and left_prob[target_label] < self.min_score_threshold:
                continue
            if right_prob is not None and right_prob[target_label] < self.min_score_threshold:
                continue

            candidates.append((j, left, right))
            candidate_texts.extend([left_text, right_text])

        if not candidates:
            prob = self.predict_probas([' '.join(span_words)])[0][target_label]
            return {
                'span': ' '.join(span_words),
                'score': float(prob),
                'interaction_score': None,
                'children': []
            }

        batch_probs = self.predict_probas(candidate_texts)

        interaction_scores = []
        for idx, (split_point, left, right) in enumerate(candidates):
            left_prob = batch_probs[2 * idx][target_label]
            right_prob = batch_probs[2 * idx + 1][target_label]
            full_prob = self.predict_probas([' '.join(span_words)])[0][target_label]

            score = abs(full_prob - (left_prob + right_prob))
            interaction_scores.append((idx, score))

        best_idx, best_score = min(interaction_scores, key=lambda x: x[1])
        split_point, left_words, right_words = candidates[best_idx]

        left_node = self._recursive_split(left_words, target_label)
        right_node = self._recursive_split(right_words, target_label)

        full_prob = self.predict_probas([' '.join(span_words)])[0][target_label]

        return {
            'span': ' '.join(span_words),
            'score': float(full_prob),
            'interaction_score': best_score,
            'children': [left_node, right_node]
        }

    def _find_top_span(self, tree):
        """Find the span with highest score in the tree"""
        top_span = tree
        max_score = tree['score']
        
        for child in tree.get('children', []):
            child_top = self._find_top_span(child)
            if child_top['score'] > max_score:
                top_span = child_top
                max_score = child_top['score']
                
        return top_span

    def explain_batch(self, input_ids_tensor, target_labels_tensor):
        """Batch explanation with evaluation metrics"""
        batch_size = input_ids_tensor.size(0)
        texts = [
            self.tokenizer.decode(
                input_ids_tensor[i], skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for i in range(batch_size)
        ]

        target_labels_list = [int(t.item()) for t in target_labels_tensor]

        results = []
        for text, label in tqdm(zip(texts, target_labels_list), total=batch_size, desc="Explaining documents"):
            explanation = self.explain(text, label)
            results.append(explanation)

        return results
                                           
# ========== Convert Tree to Levels Dict ==========
def hedge_tree_to_levels(tree, words, level=0, levels_dict=None, word_to_idx=None):
    if levels_dict is None:
        levels_dict = dict()
    if word_to_idx is None:
        word_to_idx = {word: idx for idx, word in enumerate(words)}
    span_words = tree['span'].split()
    span_positions = [word_to_idx[word] for word in span_words]
    if level not in levels_dict:
        levels_dict[level] = []
    levels_dict[level].append((span_words, span_positions, tree['score']))
    for child in tree.get('children', []):
        hedge_tree_to_levels(child, words, level+1, levels_dict, word_to_idx)
    return levels_dict

# ========== Barplot Style Visualizer ==========
def visualize_levels_tree_by_words(levels_dict, pred_label, rank, save_directory, fontsize=10, tag=''):
    pos_per_word = 1.5

    levels = max(levels_dict.keys())

    vals = np.array([score for level in levels_dict.values() for _, _, score in level])
    min_val = np.min(vals)
    max_val = np.max(vals)
    
    #Coloring scale and other coloring settings
    cnorm = mpl.colors.Normalize(vmin=min_val, vmax=max_val, clip=False)
    cmap_name = 'RdYlBu' if pred_label == 1 else 'RdYlBu_r'
    cmapper = mpl.cm.ScalarMappable(norm=cnorm, cmap=cmap_name)

    # Obtener todas las posiciones
    all_positions = []
    for level_spans in levels_dict.values():
        for _, pos_list, _ in level_spans:
            all_positions.extend(pos_list)

    if not all_positions:
        print("No hay posiciones para graficar.")
        return

    min_pos = min(all_positions)
    max_pos = max(all_positions)

    fig, ax = plt.subplots(figsize=((max_pos - min_pos + 1) * pos_per_word, levels + 2))

    ax.xaxis.set_visible(False)
    ylabels = ['Level ' + str(idx) for idx in range(levels + 1)]
    ax.set_yticks(list(range(0, levels + 1)))
    ax.set_yticklabels(ylabels)
    ax.set_ylim(levels + 0.5, -0.5)

    # Ajustar l�mites X normalizados
    ax.set_xlim(0, (max_pos - min_pos + 1) * pos_per_word)

    height = 0.8

    for key in range(levels + 1):
        for span_words, span_positions, score in levels_dict.get(key, []):
            # Normalizar las posiciones restando min_pos
            norm_positions = [p - min_pos for p in span_positions]

            start = min(norm_positions) * pos_per_word
            end = (max(norm_positions) + 1) * pos_per_word
            width = end - start

            fea_color = cmapper.to_rgba(score)
            r, g, b, _ = fea_color

            rect = patches.Rectangle((start, key - height / 2), width, height,
                                     linewidth=1.5, edgecolor='black', facecolor=fea_color, alpha=0.6)
            ax.add_patch(rect)

            for i, word in enumerate(span_words):
                word_pos = start + i * pos_per_word + pos_per_word / 2
                text_color = 'white' if (r * g * b) < 0.3 else 'black'
                ax.text(word_pos, key, word, ha='center', va='center', fontsize=fontsize, color=text_color)

    fig.colorbar(cmapper, ax=ax)
    plt.title("HEDGE Explanation (por palabras) " + tag, fontsize=fontsize + 2)
    plt.tight_layout()
    plt.savefig( os.path.join(save_directory, f"Span_Hierarchal_Structure_rank={rank}.png"))

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

    
def load_and_prepare_data(csv_path, tokenizer, max_seq_length, batch_size, data_dir, seed):
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
        texts, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    full_dataset = ClassificationDataset(texts, labels, tokenizer, max_seq_length)
    val_dataset = ClassificationDataset(val_texts, val_labels, tokenizer, max_seq_length)

    # Use DistributedSampler if running in distributed mode
    full_sampler = DistributedSampler(full_dataset) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None

    full_dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=(full_sampler is None),
        sampler=full_sampler,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        pin_memory=True
    )

    return full_dataloader, val_dataloader

# ========== Load Best Model Function ==========
def load_best_model_for_eval(best_output_dir, config, device):
    """
    Loads the best fine-tuned model using its saved state_dict and parameters for evaluation.
    """
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
    

def find_top_subtrees(full_tree, val_dataloader, pred_labels, combination, target_tokens=20, top_k=10):
    top_subtrees = []

    # Extract all true labels from val_dataloader (assuming order matches full_tree)
    all_labels = []
    for batch in val_dataloader:
        all_labels.extend(batch["labels"].tolist())

    def _recurse(node, target_tokens=17, tolerance=2):
        found = []
        span_tokens = node['span'].split()
        token_count = len(span_tokens)
        if abs(token_count - target_tokens) <= tolerance:
            found.append(node)
        for child in node.get('children', []):
            found.extend(_recurse(child, target_tokens, tolerance))
        return found

    for idx, tree in enumerate(full_tree):
        # Check true label and predicted label
        if all_labels[idx] == combination[0] and pred_labels[idx] == combination[1]:
            candidates = _recurse(tree)
            for candidate in candidates:
                top_subtrees.append((idx, candidate, tree))  # Now including the full tree

    # Sort by subtree score descending
    top_subtrees = sorted(top_subtrees, key=lambda x: x[1]['score'], reverse=True)

    # Return top_k results with tree indices and full trees
    return top_subtrees[:top_k]

def spans_to_levels_dict_from_tree(tree, words, level=0, levels_dict=None, search_range=None):
    """
    Convierte un arbol de spans a un dict con niveles, heredando las posiciones de los padres.

    Args:
      tree: dict con keys 'span', 'score', 'children'
      words: lista de tokens del texto completo
      level: nivel actual en el arbol
      levels_dict: dict acumulador de niveles
      search_range: tupla (start_idx, end_idx) delimitando donde buscar en words para este span

    Retorna:
      levels_dict con entradas:
      { nivel: [ (lista_de_palabras_span, lista_de_posiciones_en_words, score), ... ] }
    """

    if levels_dict is None:
        levels_dict = dict()
    if search_range is None:
        search_range = (0, len(words))

    span_words = tree['span'].split()

    # Buscar posiciones consecutivas de span_words dentro de words en el rango search_range
    start_search, end_search = search_range

    # Encontrar indice donde comienza span_words dentro de words[start_search:end_search]
    # Buscamos secuencia completa de span_words en words dentro del rango
    def find_sublist_in_list(sublist, mainlist):
        """Busca la primera aparicion de sublist en mainlist y retorna indice o -1."""
        n, m = len(mainlist), len(sublist)
        for i in range(n - m + 1):
            if mainlist[i:i+m] == sublist:
                return i
        return -1

    # Indice relativo a search_range donde aparece span_words
    rel_start = find_sublist_in_list(span_words, words[start_search:end_search])
    if rel_start == -1:
        print(f"Span '{tree['span']}' no encontrado en el rango {search_range}.")
        # Como fallback, no asignamos posiciones
        span_positions = []
        child_search_range = (start_search, end_search)
    else:
        abs_start = start_search + rel_start
        abs_end = abs_start + len(span_words)
        span_positions = list(range(abs_start, abs_end))
        child_search_range = (abs_start, abs_end)

    if level not in levels_dict:
        levels_dict[level] = []
    levels_dict[level].append((span_words, span_positions, tree['score']))

    # Recursi�n en hijos, con rango restringido al span actual para mantener coherencia
    for child in tree.get('children', []):
        spans_to_levels_dict_from_tree(child, words, level + 1, levels_dict, child_search_range)

    return levels_dict


def propagate_leaves_to_lower_levels(levels_dict):
    """
    Propagates leaf nodes (nodes without children) to all lower levels in the hierarchy.
    
    Args:
        levels_dict: Dictionary produced by spans_to_levels_dict_from_tree
        
    Returns:
        Modified levels_dict where leaf nodes are copied to all lower levels
    """
    if not levels_dict:
        return levels_dict
    
    # First find all leaf nodes (nodes that don't appear as parents in the next level)
    max_level = max(levels_dict.keys())
    
    # Create a new dictionary to store the modified levels
    new_levels_dict = {level: [] for level in levels_dict}
    
    # For each level except the last one
    for level in sorted(levels_dict.keys()):
        current_nodes = levels_dict[level]
        next_level_nodes = levels_dict.get(level + 1, [])
        
        
        # Find all parent spans in this level by looking at next level's positions
        parent_positions = set()
        for _, positions, _ in next_level_nodes:
            if positions:
                parent_start = positions[0]
                parent_end = positions[-1]
                parent_positions.add((parent_start, parent_end))
        
        # Classify nodes as leaves or parents
        for node in current_nodes:
            words, positions, score = node
            if not positions:
                # If no positions, treat as leaf
                new_levels_dict[level].append(node)
                continue
                
            node_start = positions[0]
            node_end = positions[-1]
            is_parent = any(start <= node_start and node_end <= end 
                            for (start, end) in parent_positions)
            
            if is_parent:
                # It's a parent node, just add to current level
                new_levels_dict[level].append(node)
            else:
                # It's a leaf node, add to current and all lower levels
                for l in range(level, max_level + 1):
                    new_levels_dict[l].append(node)
                    
    
    return new_levels_dict

def get_pred_labels(model, val_dataloader, device):
    model.eval()
    pred_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)  # logits output

            preds = torch.argmax(logits, dim=-1)  # Get predicted class index

            pred_labels.extend(preds.cpu().tolist())
    return pred_labels

def json_default(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Unserializable object {obj} of type {type(obj)}")


# ========== Main ==========
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model explanation with specified model path')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the directory containing the trained model')
    args = parser.parse_args()
    
    # Get config with the provided model path
    CONFIG = get_config(args.model_path)
    print(CONFIG)
    set_seed(CONFIG["SEED"])
    best_output_dir = os.path.join(CONFIG["SAVE_MODEL_BASE_PATH"], "best_model")
    
    with open(os.path.join(best_output_dir, "best_params.json"), "r") as f:
        metrics = json.load(f)
    params = metrics["params"]
    batch_size = int(params["batch_size"])
    unfreeze_layers = int(params["unfreeze_layers"])
    
    # Load model and tokenizer
    model, tokenizer = load_best_model_for_eval(best_output_dir, CONFIG, DEVICE)
    
    # Initialize explainer
    explainer_full = BatchedHedgeExplainer(model, tokenizer, DEVICE, batch_size)

    # Load data
    full_dataloader, val_dataloader = load_and_prepare_data(
        CONFIG["CSV_FILE_PATH"], tokenizer, CONFIG["MAX_SEQ_LENGTH"], batch_size, CONFIG["DATA_DIR"], CONFIG["SEED"]
    )
    
    # Evaluate and report
    evaluate_and_report(model, val_dataloader)
    
    ####COMMENT THIS PART IF LOADING####
    # Process data batch by batch
    all_trees = []
    all_input_ids = []
    all_labels = []

    # Initialize accumulators for weighted scores
    total_aopc = 0.0
    total_log_odds = 0.0
    total_cohesion = 0.0
    total_instances = 0
    
    for batch_idx, batch in enumerate(full_dataloader):
        # Get predictions for the batch
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        batch_size = input_ids.size(0)
        
        # Explain the batch
        batch_trees = explainer_full.explain_batch(input_ids, labels)
        all_trees.extend(batch_trees)
        all_input_ids.append(input_ids)
        all_labels.append(labels)
        
        # Calculate batch metrics
        batch_aopc = 0.0
        batch_log_odds = 0.0
        batch_cohesion = 0.0
        
        for explanation in batch_trees:
            batch_aopc += explanation['aopc']
            batch_log_odds += explanation['log_odds']
            batch_cohesion += explanation['cohesion_score']
        
        # Update weighted sums
        total_aopc += batch_aopc
        total_log_odds += batch_log_odds
        total_cohesion += batch_cohesion
        total_instances += batch_size
        
        # Print batch statistics
        print(f"\nBatch {batch_idx + 1} Statistics:")
        print(f"  Instances: {batch_size}")
        print(f"  Avg AOPC: {batch_aopc / batch_size:.4f}")
        print(f"  Avg Log-odds: {batch_log_odds / batch_size:.4f}")
        print(f"  Avg Cohesion-score: {batch_cohesion / batch_size:.4f}")
    
    # Calculate final weighted averages
    final_aopc = total_aopc / total_instances
    final_log_odds = total_log_odds / total_instances
    final_cohesion = total_cohesion / total_instances
    
    print("\nFinal Weighted Averages Across All Batches:")
    print(f"  Total Instances: {total_instances}")
    print(f"  Weighted Avg AOPC: {final_aopc:.4f}")
    print(f"  Weighted Avg Log-odds: {final_log_odds:.4f}")
    print(f"  Weighted Avg Cohesion-score: {final_cohesion:.4f}")
    
    # Combine all input_ids and labels
    all_input_ids = torch.cat(all_input_ids)
    all_labels = torch.cat(all_labels)
    
    # Save the full tree
    with open("tree_full.json", "w") as f:
        json.dump(all_trees, f, indent=4, default=json_default)
    #############################################################
    """
    with open("tree_full.json", "r") as f:
        all_trees = json.load(f)
    
    all_input_ids = []
    all_labels = []
    for batch in full_dataloader:
        all_input_ids.append(batch["input_ids"].to(DEVICE))
        all_labels.append(batch["labels"].to(DEVICE))
    all_input_ids = torch.cat(all_input_ids)
    all_labels = torch.cat(all_labels)
    
    """
    # Get predicted labels for validation set
    pred_labels = get_pred_labels(model, full_dataloader, DEVICE)
    # Find top subtrees
    
    combinations = [(1,1), (0,0), (1,0)]     
    titles = ["true_positive", "true_negative", "false_negative"]
    for combination, title in zip(combinations, titles):               #combination[0]=true label and combination[1]=pred_label
        top_10_subtrees = find_top_subtrees(all_trees, full_dataloader, pred_labels, combination, target_tokens=20, top_k=10)
        plots_output_dir = f"{CONFIG['SAVE_MODEL_BASE_PATH']}_plots"
        #plots_output_dir = os.path.join(best_output_dir, title)
        os.makedirs(plots_output_dir, exist_ok=True)
        case_plots_output_dir = os.path.join(plots_output_dir, title)
        os.makedirs(case_plots_output_dir, exist_ok=True)
        setup_logging(plots_output_dir, title)
        log = logging.getLogger()
    # Visualize top subtrees
        for rank, (tree_idx, subtree, full_tree) in enumerate(top_10_subtrees, start=1):
            print(f"\nRank #{rank} - Tree #{tree_idx} - Score: {subtree['score']:.4f}")
            log.info(f"\nRank #{rank} - Tree #{tree_idx} - Score: {subtree['score']:.4f}")
            print(f"Span: {subtree['span']}")
            log.info(f"Span: {subtree['span']}")

            # Get the original text for visualization
            input_ids = all_input_ids[tree_idx]
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            words = text.split()

            # Calculate metrics specifically for this span
            span_text = subtree['span']
            target_label = 1  # Since we're only looking at class 1

            # Calculate AOPC for this span
            aopc = explainer_full.calculate_aopc(span_text, target_label)
            print(f"AOPC for span: {aopc:.4f}")
            log.info(f"AOPC for span: {aopc:.4f}")

            # Calculate Log-odds for this span
            log_odds = explainer_full.calculate_log_odds(span_text, target_label)
            print(f"Log-odds for span: {log_odds:.4f}")
            log.info(f"Log-odds for span: {log_odds:.4f}")

            # Calculate Cohesion-score for this span
            cohesion_score = explainer_full.calculate_cohesion_score(text, target_label, subtree)
            print(f"Cohesion-score for span: {cohesion_score:.4f}")
            log.info(f"Cohesion-score for span: {cohesion_score:.4f}")

            # Create visualization
            levels_dict = spans_to_levels_dict_from_tree(subtree, words)
            levels_dict = propagate_leaves_to_lower_levels(levels_dict)
            visualize_levels_tree_by_words(levels_dict, combination[1], rank, case_plots_output_dir)

if __name__ == "__main__":
    main()
