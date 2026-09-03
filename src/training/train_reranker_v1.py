import json

import torch
from datasets import Dataset

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import (
    BinaryCrossEntropyLoss,
)


MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

TRAIN_FILE = "data/training_v1/train.jsonl"

OUTPUT_DIR = "checkpoints/private_rank_v1_smoke"


def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))

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


def main():

    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = load_jsonl(TRAIN_FILE)

    print("Training examples:", len(train_dataset))

    print("\nLoading reranker...")

    model = CrossEncoder(
        MODEL_NAME,
        max_length=128,
        device="cuda",
        model_kwargs={
            "torch_dtype": torch.bfloat16
        },
    )

    loss = BinaryCrossEntropyLoss(model)

    args = CrossEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,

        # SMOKE TEST ONLY
        max_steps=10,

        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,

        learning_rate=2e-5,

        fp16=False,
        bf16=True,

        gradient_checkpointing=False,

        optim="adafactor",

        logging_steps=1,
        logging_first_step=True,

        save_strategy="no",
        eval_strategy="no",

        report_to="none",

        seed=42,
    )

    trainer = CrossEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    print("\nStarting PrivateRank V1 smoke training...\n")

    trainer.train()

    print("\nSmoke training completed successfully.")

    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )

    print(
        "Peak GPU memory:",
        round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "GB",
    )


if __name__ == "__main__":
    main()
