# train_bert.py
import argparse
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    BertTokenizerFast, BertForSequenceClassification,
    TrainingArguments, Trainer
)
from sklearn.preprocessing import LabelEncoder
from utils_text import detect_columns, clean_text_series

def main(args):
    df = pd.read_csv(args.data)
    text_col, label_col = detect_columns(df)
    df[text_col] = clean_text_series(df[text_col])

    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df[label_col].astype(str))
    num_labels = len(le.classes_)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    ds = Dataset.from_pandas(df[[text_col, "label_id"]].rename(columns={text_col: "text"}))

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)
    ds = ds.map(tok, batched=True)
    ds = ds.train_test_split(test_size=args.test_size, seed=42)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import accuracy_score, f1_score
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        return {"accuracy": acc, "f1_macro": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)
    # Save label encoder
    import joblib
    joblib.dump(le, os.path.join(args.model_dir, "label_encoder.joblib"))
    print("✅ BERT model saved to:", args.model_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/Labeled_Laptop_Reviews.csv", type=str)
    parser.add_argument("--out_dir", default="outputs/bert_training", type=str)
    parser.add_argument("--model_dir", default="models/bert", type=str)
    parser.add_argument("--test_size", default=0.2, type=float)
    args = parser.parse_args()
    main(args)
