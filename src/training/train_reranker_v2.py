import json

import torch
from datasets import Dataset

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss


MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

TRAIN_FILE = "data/training_v2/train.jsonl"
VALIDATION_FILE = "data/training_v2/validation.jsonl"

OUTPUT_DIR = "checkpoints/private_rank_v2"
FINAL_MODEL_DIR = "models/private_rank_v2"


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

    return rows


def to_dataset(rows):
    return Dataset.from_list(
        [
            {
                "query": row["query"],
                "document": row["document"],
                "label": float(row["label"]),
            }
            for row in rows
        ]
    )


def evaluate_pairwise(model, rows):
    grouped = {}

    for row in rows:
        key = (row["case_id"], row["query"])
        grouped.setdefault(key, []).append(row)

    correct = 0
    total = 0

    for items in grouped.values():

        positive_items = [
            item for item in items
            if item["label"] == 1
        ]

        negative_items = [
            item for item in items
            if item["label"] == 0
        ]

        if not positive_items or not negative_items:
            continue

        positive = positive_items[0]
        negative = negative_items[0]

        query = positive["query"]

        scores = model.predict(
            [
                (query, positive["document"]),
                (query, negative["document"]),
            ]
        )

        if float(scores[0]) > float(scores[1]):
            correct += 1

        total += 1

    return correct / total if total else 0


def main():

    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_rows = load_jsonl(TRAIN_FILE)
    validation_rows = load_jsonl(VALIDATION_FILE)

    train_dataset = to_dataset(train_rows)
    validation_dataset = to_dataset(validation_rows)

    print("Train examples:", len(train_dataset))
    print("Validation examples:", len(validation_dataset))

    print("\nLoading base Qwen reranker...")

    model = CrossEncoder(
        MODEL_NAME,
        max_length=128,
        device="cuda",
        model_kwargs={
            "torch_dtype": torch.bfloat16
        },
    )

    print("\nValidation BEFORE training...")

    before_accuracy = evaluate_pairwise(
        model,
        validation_rows
    )

    print(
        "Pairwise validation accuracy:",
        f"{before_accuracy:.2%}"
    )

    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,

        # Start conservatively.
        num_train_epochs=1,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,

        gradient_accumulation_steps=4,

        learning_rate=5e-6,

        bf16=True,
        fp16=False,

        gradient_checkpointing=False,

        optim="adafactor",

        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        logging_steps=25,
        logging_first_step=True,

        report_to="none",

        seed=42,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        loss=loss,
    )

    print("\nStarting PrivateRank V2 training...\n")

    trainer.train()

    print("\nValidation AFTER training...")

    after_accuracy = evaluate_pairwise(
        model,
        validation_rows
    )

    print(
        "Pairwise validation accuracy:",
        f"{after_accuracy:.2%}"
    )

    print("\nSaving PrivateRank V2...")

    model.save_pretrained(FINAL_MODEL_DIR)

    print("Model saved to:", FINAL_MODEL_DIR)

    print(
        "Peak GPU memory:",
        round(
            torch.cuda.max_memory_allocated()
            / 1024**3,
            2
        ),
        "GB"
    )


if __name__ == "__main__":
    main()