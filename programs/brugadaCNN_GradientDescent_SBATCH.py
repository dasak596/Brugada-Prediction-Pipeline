import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn.functional as F
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Base path where models will be saved")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to the BERT model to use")
    parser.add_argument("--bert_type", type=str, required=True,
                        help="Path to the BERT model to use")
    return parser.parse_args()

# --- Configuration ---
args = parse_args()
model_name = os.path.basename(args.model_path)

# Configuration
CONFIG = {
    "MODEL_CNN_DIR": args.model_path,
    "MODEL_TOKENIZER_DIR": args.tokenizer,
    "TOKENIZER_TYPE": args.bert_type,
    "CSV_FILE_PATH": "./sintomaticos.csv",
    "DATA_DIR": "brugada_new",
    "MAX_SEQ_LENGTH": 512,
    "NUM_CLASSES": 2,
    "SEED": 42,
    "NORMALIZATION": "perClass",
    "OUTPUT_FILE": "interpretability_report_{}.tex".format(model_name)
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

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
            "raw_text": text
        }

class CNNClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_sizes, num_filters, num_classes, pad_token_id):
        super(CNNClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.convs = torch.nn.ModuleList([
            torch.nn.Conv1d(embedding_dim, num_filters, k) for k in filter_sizes
        ])
        self.fc = torch.nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = torch.nn.Dropout(0.5)
        self.filter_sizes = filter_sizes

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        conved = [torch.relu(conv(x)) for conv in self.convs]
        pooled = [torch.max(conv, axis=2)[0] for conv in conved]
        cat = self.dropout(torch.cat(pooled, axis=1))
        return self.fc(cat)

def load_data(tokenizer, max_len, batch_size):
    df = pd.read_csv(CONFIG["CSV_FILE_PATH"])
    positive_files = set(df["filename"].tolist())
    all_files = [f for f in os.listdir(CONFIG["DATA_DIR"]) if f.endswith(".txt")]

    texts, labels = [], []
    for file in all_files:
        with open(os.path.join(CONFIG["DATA_DIR"], file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
        labels.append(1 if file in positive_files else 0)

    full_dataset = ClassificationDataset(texts, labels, tokenizer, max_len)
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
    return full_loader, labels

def process_span(span, tokenizer_type):
    decoded = []
    current_word = ""
    
    for token in span:
        if token in ["[PAD]", "[CLS]", "[SEP]"]:
            continue
        if tokenizer_type == "roberta":
            if token.startswith("▁"):
                if current_word:
                    decoded.append(current_word)
                current_word = token[1:]
            else:
                current_word += token
        else:
            if token.startswith("##"):
                current_word += token[2:]
            else:
                if current_word:
                    decoded.append(current_word)
                current_word = token
    
    if current_word:
        decoded.append(current_word)
    return " ".join(decoded)

def get_case_type(true_label, pred_label):
    if true_label == 1 and pred_label == 1:
        return "True Positive"
    elif true_label == 0 and pred_label == 1:
        return "False Positive"
    elif true_label == 1 and pred_label == 0:
        return "False Negative"
    else:
        return "True Negative"

def compute_grad_saliency_cnn(model, input_ids, labels, device):
    """
    Computes gradient-based saliency maps for a CNN text classifier.
    
    Args:
        model: trained CNNClassifier
        input_ids: tensor of shape [batch_size, seq_len]
        labels: tensor of shape [batch_size]
        device: 'cuda' or 'cpu'
        
    Returns:
        gradients_per_conv: dict of gradients w.r.t. conv outputs, keyed by filter size
        logits: model predictions before softmax
        loss_value: total loss value
    """
    model.eval()
    #model.zero_grad()

    # Embedding lookup and transpose
    embeddings = model.embedding(input_ids)  # (batch, seq_len, embed_dim)
    embeddings = embeddings.transpose(1, 2)  # (batch, embed_dim, seq_len)

    conved_outputs = []
    for conv_layer in model.convs:
        conv_out = torch.relu(conv_layer(embeddings))  # (batch, num_filters, seq_len)
        conv_out.retain_grad()  # <-- this is crucial for saliency!
        conved_outputs.append(conv_out)

    # Max pooling over sequence length
    pooled = [torch.max(c, dim=2)[0] for c in conved_outputs]  # list of (batch, num_filters)

    # Concatenate pooled outputs
    cat = model.dropout(torch.cat(pooled, dim=1))  # (batch, num_filters * num_filter_sizes)

    # Forward through final FC
    logits = model.fc(cat)  # (batch, num_classes)

    # Compute loss
    loss = F.cross_entropy(logits, labels)
    loss_value = loss.item()

    # Backward pass — gradients flow to retained conved_outputs
    loss.backward()

    # Collect gradients for each conv output
    filter_sizes = model.filter_sizes
    gradients_per_conv = {
        fs: conved_outputs[i].grad.detach().cpu() for i, fs in enumerate(filter_sizes)
    }

    return gradients_per_conv, logits.detach().cpu(), loss_value    
    
    
def extract_interpretability(model, tokenizer, tokenizer_type, normalization, input_ids_batch, 
                             texts, batch_labels, batch_predictions, global_phrase_store=None):
    model.eval()

    # Get saliency gradients and logits from full forward+backward
    gradients_per_conv, logits, _ = compute_grad_saliency_cnn(
        model, input_ids_batch, batch_labels, DEVICE
    )

    if global_phrase_store is None:
        global_phrase_store = {
            fs: {"phrases": [], "gradients": [], "scores": [], "contexts": [], "types": [],
                 "aopc": [], "logodds": [], "cohesion": []}
            for fs in model.filter_sizes
        }

    for conv_idx, kernel_size in enumerate(model.filter_sizes):
        saliency_grads = gradients_per_conv[kernel_size]  # shape (batch_size, num_filters, seq_len)
        seq_len = saliency_grads.shape[2]
        num_filters = saliency_grads.shape[1]

        for sample_idx in range(saliency_grads.size(0)):
            input_id = input_ids_batch[sample_idx].unsqueeze(0)
            true_label = batch_labels[sample_idx].item()
            pred_label = batch_predictions[sample_idx].item()

            # Skip if not correct prediction
            # if true_label != pred_label:
            #     continue

            with torch.no_grad():
                orig_logits = model(input_id)
                if orig_logits.dim() == 1:
                    orig_logits = orig_logits.unsqueeze(0)
                orig_prob = F.softmax(orig_logits, dim=1)[0, pred_label]

            # Compute saliency norms (L2 over filters)
            saliency_norms = torch.norm(saliency_grads[sample_idx], p=2, dim=0)  # shape: [seq_len]

            # Get top-k positions
            topk = min(10, seq_len)
            topk_values, topk_indices = torch.topk(saliency_norms, k=topk)

            input_tokens = tokenizer.convert_ids_to_tokens(input_id[0])

            for score, idx in zip(topk_values, topk_indices):
                idx = idx.item()
                phrase_start = idx
                phrase_end = min(idx + kernel_size, len(input_tokens))

                # Extract phrase and skip special tokens
                target_phrase = input_tokens[phrase_start:phrase_end]
                if any(t in ['[CLS]', '[SEP]', '[PAD]'] for t in target_phrase):
                    continue

                processed_phrase = process_span(target_phrase, tokenizer_type)
                if not processed_phrase:
                    continue

                with torch.no_grad():
                    # AOPC
                    masked_input = input_id.clone()
                    masked_input[0, phrase_start:phrase_end] = tokenizer.pad_token_id
                    masked_prob = F.softmax(model(masked_input), dim=1)[0, pred_label]
                    aopc = orig_prob - masked_prob

                    # Log-odds
                    pad_mask = torch.ones_like(input_id)
                    pad_mask[0, phrase_start:phrase_end] = 0
                    pad_prob = F.softmax(model(input_id * pad_mask), dim=1)[0, pred_label]
                    logodds = torch.log(pad_prob / orig_prob)

                    # Cohesion
                    Q = 20
                    cohesion = 0
                    phrase_words = input_id[0, phrase_start:phrase_end].tolist()
                    for _ in range(Q):
                        shuffled = input_id.clone()
                        np.random.shuffle(phrase_words)
                        shuffled[0, phrase_start:phrase_end] = torch.tensor(phrase_words).to(DEVICE)
                        shuffled_prob = F.softmax(model(shuffled), dim=1)[0, pred_label]
                        cohesion += (orig_prob - shuffled_prob)
                    cohesion /= Q

                # Context window
                ctx_start = max(0, phrase_start - 10)
                ctx_end = min(len(input_tokens), phrase_end + 10)
                context = process_span(input_tokens[ctx_start:ctx_end], tokenizer_type)

                # Store results
                filter_key = kernel_size
                global_phrase_store[filter_key]["phrases"].append(processed_phrase)
                global_phrase_store[filter_key]["gradients"].append(score.item())
                global_phrase_store[filter_key]["scores"].append(orig_prob.item())
                global_phrase_store[filter_key]["contexts"].append(context)
                global_phrase_store[filter_key]["types"].append(get_case_type(true_label, pred_label))
                global_phrase_store[filter_key]["aopc"].append(aopc.item())
                global_phrase_store[filter_key]["logodds"].append(logodds.item())
                global_phrase_store[filter_key]["cohesion"].append(cohesion.item())

    return global_phrase_store


def plot_activation_heatmap(global_phrase_store, model, stat="max", cmap="viridis"):
    """
    Visualiza un heatmap de las activaciones normalizadas por filtro y tipo de caso.

    Parámetros:
    - global_phrase_store: dict con activaciones y tipos de caso.
    - model: modelo CNN con atributo filter_sizes.
    - stat: estadístico a visualizar ("mean", "max", "std").
    - cmap: colormap para el heatmap.

    """
    data = []
    for fs in model.filter_sizes:
        phrases = global_phrase_store[fs]["phrases"]
        scores = global_phrase_store[fs]["scores"]
        types = global_phrase_store[fs]["types"]
        
        for phrase, score, case_type in zip(phrases, scores, types):
            data.append({
                "Filter size": fs,
                "Score": score,
                "Type": case_type
            })

    df = pd.DataFrame(data)

    if stat == "mean":
        pivot_df = df.groupby(["Filter size", "Type"])["Score"].mean().reset_index()
    elif stat == "max":
        pivot_df = df.groupby(["Filter size", "Type"])["Score"].max().reset_index()
    elif stat == "std":
        pivot_df = df.groupby(["Filter size", "Type"])["Score"].std().reset_index()
    else:
        raise ValueError("stat debe ser 'mean', 'max' o 'std'.")

    heatmap_data = pivot_df.pivot(index="Type", columns="Filter size", values="Score")

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap=cmap, fmt=".2f")
    plt.title(f"{stat.capitalize()} de activaciones normalizadas por filtro y tipo de caso")
    plt.ylabel("Tipo de caso")
    plt.xlabel("Tamaño de filtro")
    plt.show()

def normalize_phrase_store(global_phrase_store, model, normalization="perClass"):
    for fs in model.filter_sizes:
        if normalization == "perClass":
            df = pd.DataFrame({
                "phrase": global_phrase_store[fs]["phrases"],
                "score": global_phrase_store[fs]["scores"],
                "type": global_phrase_store[fs]["types"]
            })
            normalized_scores = []
            for case_type in df["type"].unique():
                mask = df["type"] == case_type
                case_scores = df.loc[mask, "score"].values
                mean, std = case_scores.mean(), case_scores.std()
                z_scores = (case_scores - mean) / std
                sigmoid_scores = 1 / (1 + np.exp(-z_scores))
                normalized_scores.extend(sigmoid_scores)
            global_phrase_store[fs]["scores"] = normalized_scores

        else:  # general normalization
            scores = np.array(global_phrase_store[fs]["scores"])
            mean = scores.mean()
            std = scores.std()
            z_scores = (scores - mean) / std
            sigmoid_scores = 1 / (1 + np.exp(-z_scores))
            global_phrase_store[fs]["scores"] = sigmoid_scores.tolist()

    return global_phrase_store

def load_test_data(tokenizer, max_len, batch_size):
    df = pd.read_csv(CONFIG["CSV_FILE_PATH"])
    positive_files = set(df["filename"].tolist())
    all_files = [f for f in os.listdir(CONFIG["DATA_DIR"]) if f.endswith(".txt")]

    texts, labels = [], []
    for file in all_files:
        with open(os.path.join(CONFIG["DATA_DIR"], file), 'r', encoding='utf-8') as f:
            texts.append(f.read())
        labels.append(1 if file in positive_files else 0)

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=CONFIG["SEED"], stratify=labels
    )

    test_dataset = ClassificationDataset(X_test, y_test, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader, y_test

"""
def escape_latex_special_chars(text):
    '''
    Escape LaTeX special characters in a given text.
    '''
    special_chars = {
        '%': r'\%',
        '_': r'\_',
        '$': r'\$',
        '&': r'\&',
        '#': r'\#',
        '{': r'\{',
        '}': r'\}',
        '~': r'\textasciitilde{}',
        '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}'
    }
    for char, escape in special_chars.items():
        text = text.replace(char, escape)
    return text
        
 """

def escape_latex_minimal(text):
    return text.replace('%', r'\%').replace('_', r'\_')

def get_model_name(tokenizer_dir):
    """Extracts the model name after the last underscore in the path"""
    base_name = os.path.basename(os.path.normpath(tokenizer_dir))
    return base_name.split('_')[-1]  # Gets part after last underscore

def generate_latex_report(global_store, test_true_labels,  true_labels, test_preds, predictions, output_file, tokenizer_dir):
    with open(output_file, 'w', encoding='utf-8') as f:
        model_name = get_model_name(tokenizer_dir)
        # Write LaTeX header
        f.write(r"""\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{geometry}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage{array}
\usepackage{ragged2e}

\geometry{a4paper, margin=1in}
\newcolumntype{P}[1]{>{\RaggedRight\arraybackslash}p{#1}}


\title{CNN Interpretability Report: """ + model_name + r"""}

\begin{document}

\maketitle

""")
        
        # Add classification metrics
        report = classification_report(test_true_labels, test_preds, target_names=["Negative", "Positive"], digits=4)
        cm = confusion_matrix(test_true_labels, test_preds)
        cm_df = pd.DataFrame(cm, 
                           index=["True Negative", "True Positive"],
                           columns=["Predicted Negative", "Predicted Positive"])
        
        f.write(r"""
\section*{Classification Metrics}
\begin{verbatim}
""" + report + r"""
\end{verbatim}

\subsection*{Confusion Matrix}
\begin{verbatim}
""" + str(cm_df) + r"""
\end{verbatim}
""")

        
        # Add interpretability results for each filter size
        for kernel_size, data in global_store.items():
        #if kernel_size == 20:  # Only process items where kernel_size is 20. Remove this "if" statement if you prefer loop over all kernez_sizes
            df = pd.DataFrame({
                "phrase": data["phrases"],
                "gradients": data["gradients"],
                "score": data["scores"],
                "context": data["contexts"],
                "case_type": data["types"],
                "aopc": data["aopc"],
                "logodds": data["logodds"],
                "cohesion": data["cohesion"]
            })

            # Filter to only include True Positive and True Negative
            df = df[df["case_type"].isin(["True Positive", "True Negative", "False Negative"])]

            f.write(f"""
\\section{{{{{kernel_size}-gram Filter Results}}}}

""")

            # True Positive Patterns
            if "True Positive" in df["case_type"].unique():
                tp_df = df[df["case_type"] == "True Positive"]
               #tp_phrases = escape_latex_special_chars(tp_df.groupby('phrase')['score'].max().sort_values(ascending=False).head(10))
                tp_phrases = tp_df.groupby('phrase')['gradients'].max().sort_values(ascending=False).head(10)
                f.write(r"""
\subsection*{True Positive Patterns}
\begin{longtable}{|c|c|p{0.2\textwidth}|p{0.4\textwidth}|c|c|c|}
\hline
\multicolumn{7}{|c|}{\textbf{True Positive Patterns}} \\
\hline
\textbf{\#} & \textbf{Classification score} & \textbf{Phrase} & \textbf{Example Context}   \\
\hline
""")
                for rank, (phrase, gradient) in enumerate(tp_phrases.items(), 1):
                    contexts = tp_df[tp_df['phrase'] == phrase]['context'].unique()[:1]
                    scores = tp_df[tp_df['phrase'] == phrase]['score'].unique()[:1]
                    aopcs = tp_df[tp_df['phrase'] == phrase]['aopc'].unique()[:1]
                    logodds = tp_df[tp_df['phrase'] == phrase]['logodds'].unique()[:1]
                    cohesions = tp_df[tp_df['phrase'] == phrase]['cohesion'].unique()[:1]
                    context = contexts[0] if len(contexts) > 0 else ""
                    phrase = escape_latex_minimal(phrase)
                    context = escape_latex_minimal(context)
                    score = f"{scores[0]:.4f}" if len(scores) > 0 else "N/A"
                    aopc = f"{aopcs[0]:.3f}" if len(aopcs) > 0 else "N/A"        
                    logodd = f"{logodds[0]:.3f}" if len(logodds) > 0 else "N/A"
                    cohesion = f"{cohesions[0]:.3f}" if len(cohesions) > 0 else "N/A"
                    f.write(f"{rank} & {score} & {phrase} & {context} \\\\ \\hline\n")
                f.write(r"\end{longtable}" + "\n")

            # True Negative Patterns
            if "True Negative" in df["case_type"].unique():
                tn_df = df[df["case_type"] == "True Negative"]
                tn_phrases = tn_df.groupby('phrase')['gradients'].max().sort_values(ascending=False).head(10)

                f.write(r"""
\subsection*{True Negative Patterns}
\begin{longtable}{|c|c|p{0.2\textwidth}|p{0.4\textwidth}|c|c|c|}
\hline
\multicolumn{7}{|c|}{\textbf{True Negative Patterns}} \\
\hline
\textbf{\#} & \textbf{Classification score} & \textbf{Phrase} & \textbf{Example Context}  \\
\hline
""")
                for rank, (phrase, score) in enumerate(tn_phrases.items(), 1):
                    contexts = tn_df[tn_df['phrase'] == phrase]['context'].unique()[:1]
                    scores = tn_df[tn_df['phrase'] == phrase]['score'].unique()[:1]
                    aopcs = tn_df[tn_df['phrase'] == phrase]['aopc'].unique()[:1]
                    logodds = tn_df[tn_df['phrase'] == phrase]['logodds'].unique()[:1]
                    cohesions = tn_df[tn_df['phrase'] == phrase]['cohesion'].unique()[:1]
                    context = contexts[0] if len(contexts) > 0 else ""
                    phrase = escape_latex_minimal(phrase)
                    context = escape_latex_minimal(context)
                    score = f"{scores[0]:.4f}" if len(scores) > 0 else "N/A"
                    aopc = f"{aopcs[0]:.3f}" if len(aopcs) > 0 else "N/A"
                    logodd = f"{logodds[0]:.3f}" if len(logodds) > 0 else "N/A"
                    cohesion = f"{cohesions[0]:.3f}" if len(cohesions) > 0 else "N/A"
                    f.write(f"{rank} & {score} & {phrase} & {context}\\\\ \\hline\n")
                f.write(r"\end{longtable}" + "\n")

            # False Negative Patterns
            if "False Negative" in df["case_type"].unique():
                fn_df = df[df["case_type"] == "False Negative"]
               #tp_phrases = escape_latex_special_chars(tp_df.groupby('phrase')['score'].max().sort_values(ascending=False).head(10))
                fn_phrases = fn_df.groupby('phrase')['gradients'].max().sort_values(ascending=False).head(10)
                f.write(r"""
\subsection*{False Negative Patterns}
\begin{longtable}{|c|c|p{0.2\textwidth}|p{0.4\textwidth}|c|c|c|}
\hline
\multicolumn{7}{|c|}{\textbf{True Negative Patterns}} \\
\hline
\textbf{\#} & \textbf{Classification score} & \textbf{Phrase} & \textbf{Example Context}  \\
\hline
""")
                for rank, (phrase, score) in enumerate(fn_phrases.items(), 1):
                    contexts = fn_df[fn_df['phrase'] == phrase]['context'].unique()[:1]
                    scores = fn_df[fn_df['phrase'] == phrase]['score'].unique()[:1]
                    aopcs = fn_df[fn_df['phrase'] == phrase]['aopc'].unique()[:1]
                    logodds = fn_df[fn_df['phrase'] == phrase]['logodds'].unique()[:1]
                    cohesions = fn_df[fn_df['phrase'] == phrase]['cohesion'].unique()[:1]
                    context = contexts[0] if len(contexts) > 0 else ""
                    phrase = escape_latex_minimal(phrase)
                    score = f"{scores[0]:.4f}" if len(scores) > 0 else "N/A"
                    context = escape_latex_minimal(context)
                    aopc = f"{aopcs[0]:.3f}" if len(aopcs) > 0 else "N/A"
                    logodd = f"{logodds[0]:.3f}" if len(logodds) > 0 else "N/A"
                    cohesion = f"{cohesions[0]:.3f}" if len(cohesions) > 0 else "N/A"
                    f.write(f"{rank} & {score} & {phrase} & {context}  \\\\ \\hline\n")
                f.write(r"\end{longtable}" + "\n")

        # Write LaTeX footer
        f.write(r"""


\end{document}
""")



if __name__ == "__main__":
    set_seed(CONFIG["SEED"])
    with open(os.path.join(CONFIG["MODEL_CNN_DIR"], "final_evaluation_metrics.json"), "r") as f:
        metrics = json.load(f)
    
    params = metrics["best_params"]
    filter_sizes = eval(str(params["filter_sizes"]))
    num_filters = int(params["num_filters"])
    batch_size = int(params["batch_size"])

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["MODEL_TOKENIZER_DIR"])
    tokenizer_type = CONFIG["TOKENIZER_TYPE"]
    normalization = CONFIG["NORMALIZATION"]
    model = CNNClassifier(
        tokenizer.vocab_size, 768,
        filter_sizes, num_filters,
        CONFIG["NUM_CLASSES"], tokenizer.pad_token_id
    )
    model.load_state_dict(torch.load(os.path.join(CONFIG["MODEL_CNN_DIR"], "final_best_model.pth"), map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    full_loader, true_labels = load_data(tokenizer, CONFIG["MAX_SEQ_LENGTH"], batch_size)
    
    predictions = []
    global_store = None
    
    #with torch.no_grad():
    for batch_idx, batch in enumerate(full_loader):
        input_ids = batch["input_ids"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        logits = model(input_ids)
        batch_preds = torch.argmax(logits, dim=1)
        predictions.extend(batch_preds.cpu().numpy())

        global_store = extract_interpretability(
            model, tokenizer, tokenizer_type, normalization, input_ids,
            batch["raw_text"], labels, batch_preds,
            global_store
        )


    # Debug print after normalization
    print("\nAverage Results:")
    for fs in model.filter_sizes:
        print(f"\nFilter size {fs}:")
        print(f"Total phrases: {len(global_store[fs]['phrases'])}")
        print(f"Sample scores: {global_store[fs]['scores'][:5]}")

    test_loader, test_true_labels = load_test_data(tokenizer, CONFIG["MAX_SEQ_LENGTH"], batch_size)
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            logits = model(input_ids)
            batch_preds = torch.argmax(logits, dim=1)
            test_preds.extend(batch_preds.cpu().numpy())
    
    
    generate_latex_report(global_store, test_true_labels, true_labels, test_preds, predictions, CONFIG["OUTPUT_FILE"], CONFIG["MODEL_TOKENIZER_DIR"])
    print(f"LaTeX report generated at: {CONFIG['OUTPUT_FILE']}")