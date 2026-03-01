#!/usr/bin/env python
"""
NLP training script (text classification / regression).

Uses TF-IDF + Logistic Regression for CPU-based inference, or HuggingFace
Transformers when a GPU is available.

Usage:
    python train_nlp.py \
        --data-path data.csv \
        --target-column label \
        --text-column text \
        --task-type nlp_classification \
        --output-dir /tmp/output
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", required=True)
    p.add_argument("--target-column", required=True)
    p.add_argument("--text-column", default=None, help="Auto-detected if not set")
    p.add_argument("--task-type", default="nlp_classification",
                   choices=["nlp_classification", "nlp_regression"])
    p.add_argument("--model-name", default="distilbert-base-uncased",
                   help="HuggingFace model name for transformer-based training")
    p.add_argument("--output-dir", default=os.environ.get("AIP_MODEL_DIR", "/tmp/output"))
    p.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "experimenter"))
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--max-features", type=int, default=10000,
                   help="TF-IDF max features (CPU path)")
    p.add_argument("--use-transformers", action="store_true", default=False,
                   help="Use HuggingFace Transformers (requires GPU)")
    p.add_argument("--no-wandb", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()

    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    data_path = args.data_path
    if data_path.startswith("gs://"):
        from google.cloud import storage
        bucket, blob_name = data_path[5:].split("/", 1)
        client = storage.Client()
        client.bucket(bucket).blob(blob_name).download_to_filename("/tmp/nlp_data.csv")
        data_path = "/tmp/nlp_data.csv"

    df = pd.read_csv(data_path)
    if args.target_column not in df.columns:
        print(f"ERROR: Target '{args.target_column}' not found", file=sys.stderr)
        sys.exit(1)

    # Auto-detect text column
    text_col = args.text_column
    if not text_col:
        obj_cols = [c for c in df.select_dtypes("object").columns if c != args.target_column]
        if not obj_cols:
            print("ERROR: No text column found", file=sys.stderr)
            sys.exit(1)
        text_col = obj_cols[0]
        print(f"Auto-detected text column: {text_col!r}")

    X_text = df[text_col].fillna("").astype(str)
    y = df[args.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=args.test_size, random_state=42
    )

    if args.use_transformers:
        metrics = _train_transformer(
            X_train, X_test, y_train, y_test, args
        )
    else:
        metrics = _train_tfidf(X_train, X_test, y_train, y_test, args)

    print(f"Metrics: {metrics}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = str(out / "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if not args.no_wandb and os.environ.get("WANDB_API_KEY"):
        try:
            import wandb
            run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
                reinit=True,
            )
            wandb.log(metrics)
            wandb.save(metrics_path)
            print(f"W&B: {run.get_url()}")
            run.finish()
        except Exception as e:
            print(f"W&B failed: {e}", file=sys.stderr)

    print("NLP training complete.")


def _train_tfidf(X_train, X_test, y_train, y_test, args):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
    import numpy as np

    print(f"Training TF-IDF (max_features={args.max_features}) + Logistic Regression...")
    vec = TfidfVectorizer(max_features=args.max_features, ngram_range=(1, 2))
    X_tr = vec.fit_transform(X_train)
    X_te = vec.transform(X_test)

    clf = LogisticRegression(max_iter=500, C=1.0)
    clf.fit(X_tr, y_train)
    y_pred = clf.predict(X_te)

    if args.task_type == "nlp_classification":
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }
    else:
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }


def _train_transformer(X_train, X_test, y_train, y_test, args):
    """Fine-tune a HuggingFace transformer (requires GPU)."""
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments,
        )
        import torch

        print(f"Fine-tuning {args.model_name} on GPU...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        num_labels = len(set(y_train))

        encodings_train = tokenizer(
            list(X_train), truncation=True, padding=True, max_length=128
        )
        encodings_test = tokenizer(
            list(X_test), truncation=True, padding=True, max_length=128
        )

        from torch.utils.data import Dataset

        class TextDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = list(labels)

            def __getitem__(self, idx):
                item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
                item["labels"] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)

        le_map = {v: i for i, v in enumerate(sorted(set(y_train)))}
        y_train_enc = [le_map[v] for v in y_train]
        y_test_enc = [le_map.get(v, 0) for v in y_test]

        train_ds = TextDataset(encodings_train, y_train_enc)
        test_ds = TextDataset(encodings_test, y_test_enc)

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=num_labels
        )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            report_to="wandb" if (not args.no_wandb and os.environ.get("WANDB_API_KEY")) else "none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
        )
        trainer.train()

        preds = trainer.predict(test_ds).predictions.argmax(axis=1)
        from sklearn.metrics import accuracy_score, f1_score
        return {
            "accuracy": float(accuracy_score(y_test_enc, preds)),
            "f1": float(f1_score(y_test_enc, preds, average="weighted", zero_division=0)),
        }

    except ImportError:
        print("Transformers/torch not available — falling back to TF-IDF", file=sys.stderr)
        return _train_tfidf(X_train, X_test, y_train, y_test, args)


if __name__ == "__main__":
    main()
