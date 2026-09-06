import gc
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder import (
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import RankNetLoss


MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

TRAIN_FILE = Path("data/training_v4_1/train.jsonl")
VALIDATION_FILE = Path("data/training_v4_1/validation.jsonl")

OUTPUT_DIR = "checkpoints/private_rank_v4_1"
FINAL_MODEL_DIR = "models/private_rank_v4_1"

MAX_LENGTH = 256


# Keep this identical to the production/evaluation instruction.
RERANKER_PROMPT = (
    "Rank enterprise passages by the exact user intent. "
    "Respect dates, policy versions, employee status, region, "
    "entity IDs, numeric thresholds, and business context. "
    "Temporal rules are strict: if the query asks for a rule BEFORE, "
    "PRIOR TO, OLD, PREVIOUS, HISTORICAL, RETIRED, or SUPERSEDED "
    "relative to a date, prefer the passage explicitly valid before "
    "that date. A passage whose rule starts FROM, ON, or AFTER the "
    "cutoff date is not the answer to a before-date query. "
    "For CURRENT, ACTIVE, or LATEST queries, prefer the active rule "
    "and reject retired or superseded rules."
)


def clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def to_dataset(rows):
    return Dataset.from_list(
        [
            {
                "query": row["query"],
                "docs": row["docs"],
                "labels": [float(value) for value in row["labels"]],
            }
            for row in rows
        ]
    )


def evaluate_pairwise(model, rows, label):
    correct = 0
    total = 0
    category_stats = {}

    for row in rows:
        query = row["query"]
        docs = row["docs"]
        labels = row["labels"]

        positive_index = labels.index(1.0)
        negative_index = labels.index(0.0)

        scores = model.predict(
            [
                (query, docs[positive_index]),
                (query, docs[negative_index]),
            ],
            show_progress_bar=False,
        )

        is_correct = float(scores[0]) > float(scores[1])
        correct += int(is_correct)
        total += 1

        category = row["category"]
        stats = category_stats.setdefault(category, {"correct": 0, "total": 0})
        stats["total"] += 1
        stats["correct"] += int(is_correct)

    accuracy = correct / total if total else 0.0

    print(f"\n{label}: {accuracy:.2%} ({correct}/{total})")
    for category in sorted(category_stats):
        stats = category_stats[category]
        cat_accuracy = stats["correct"] / stats["total"]
        print(
            f"  {category}: {cat_accuracy:.2%} "
            f"({stats['correct']}/{stats['total']})"
        )

    return accuracy


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count

    percent = (100 * trainable / total) if total else 0.0

    print("Trainable parameters:", f"{trainable:,}")
    print("Total parameters:", f"{total:,}")
    print("Trainable percent:", f"{percent:.4f}%")


def build_model():
    model = CrossEncoder(
        MODEL_NAME,
        max_length=MAX_LENGTH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_kwargs={
            "torch_dtype": torch.bfloat16
        } if torch.cuda.is_available() else {},
        prompts={"query": RERANKER_PROMPT},
        default_prompt_name="query",
    )

    # Qwen3-Reranker-0.6B is a causal-LM reranker in Sentence Transformers,
    # so the PEFT task type must be CAUSAL_LM rather than SEQ_CLS.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    model.add_adapter(lora_config)
    return model


def main():
    print("CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    train_rows = load_jsonl(TRAIN_FILE)
    validation_rows = load_jsonl(VALIDATION_FILE)

    print("Train ranking rows:", len(train_rows))
    print("Validation ranking rows:", len(validation_rows))

    train_dataset = to_dataset(train_rows)
    validation_dataset = to_dataset(validation_rows)

    print("\nLoading fresh base Qwen reranker + LoRA...")
    model = build_model()
    print_trainable_parameters(model)

    print("\nValidation BEFORE LoRA training...")
    before_accuracy = evaluate_pairwise(
        model,
        validation_rows,
        "Pairwise validation before",
    )

    # RankNet directly optimizes positive-vs-hard-negative ordering.
    loss = RankNetLoss(model, mini_batch_size=1)

    args = CrossEncoderTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_available(),
        fp16=False,
        gradient_checkpointing=False,
        optim="adamw_torch",
        eval_strategy="steps",
        eval_steps=0.25,
        save_strategy="steps",
        save_steps=0.25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
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

    print("\nStarting PrivateRank V4.1 LoRA training...\n")
    trainer.train()

    print("\nValidation AFTER LoRA training...")
    after_accuracy = evaluate_pairwise(
        model,
        validation_rows,
        "Pairwise validation after",
    )

    print("\nValidation change:", f"{after_accuracy - before_accuracy:+.2%}")

    print("\nSaving PrivateRank V4.1...")
    model.save_pretrained(FINAL_MODEL_DIR)
    print("Model saved to:", FINAL_MODEL_DIR)

    if torch.cuda.is_available():
        print(
            "Peak GPU memory:",
            round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "GB",
        )

    # Immediate reload smoke test: catches adapter-saving/loading issues now,
    # before the production search script uses this model.
    print("\nReload smoke test...")
    del trainer
    del model
    clear_gpu()

    reloaded = CrossEncoder(
        FINAL_MODEL_DIR,
        max_length=MAX_LENGTH,
        device="cuda" if torch.cuda.is_available() else "cpu",
        prompts={"query": RERANKER_PROMPT},
        default_prompt_name="query",
    )

    reloaded_accuracy = evaluate_pairwise(
        reloaded,
        validation_rows,
        "Reloaded V4.1 validation",
    )
    print("Reload smoke test complete:", f"{reloaded_accuracy:.2%}")


if __name__ == "__main__":
    main()
