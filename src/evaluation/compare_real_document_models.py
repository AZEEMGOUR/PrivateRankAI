import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from sentence_transformers import (
    CrossEncoder,
    SentenceTransformer,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]


EMBEDDING_MODEL = (
    "Qwen/Qwen3-Embedding-0.6B"
)

BASE_RERANKER = (
    "Qwen/Qwen3-Reranker-0.6B"
)

PRIVATE_RANK_V3 = (
    PROJECT_ROOT
    / "models"
    / "private_rank_v3"
)

INDEX_DIR = (
    PROJECT_ROOT
    / "data"
    / "indexes"
)

CHUNKS_FILE = (
    INDEX_DIR
    / "chunks.json"
)

EMBEDDINGS_FILE = (
    INDEX_DIR
    / "embeddings.pt"
)

QUERY_FILE = (
    PROJECT_ROOT
    / "data"
    / "evaluation"
    / "real_document_v1"
    / "queries.json"
)


RETRIEVAL_TOP_K = 10


RERANKER_PROMPT = (
    "Rank enterprise passages by the exact user intent. "
    "Respect dates, policy versions, current versus historical rules, "
    "employee status, region, entity IDs, numeric thresholds, "
    "and business context."
)


def clear_gpu():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_expected(chunk, item):
    if (
        chunk["page"]
        != item["expected_page"]
    ):
        return False

    expected_text = (
        item["expected_contains"]
        .lower()
    )

    return (
        expected_text
        in chunk["text"].lower()
    )


def load_data(device):
    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        chunks = json.load(file)

    document_embeddings = torch.load(
        EMBEDDINGS_FILE,
        map_location=device,
        weights_only=True,
    )

    with open(
        QUERY_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        queries = json.load(file)

    return (
        chunks,
        document_embeddings,
        queries,
    )


def build_retrieval_candidates(
    chunks,
    document_embeddings,
    queries,
    device,
):
    print(
        "\nLoading embedding model..."
    )

    embedding_model = (
        SentenceTransformer(
            EMBEDDING_MODEL,
            device=device,
            model_kwargs={
                "torch_dtype":
                torch.float16
            }
            if device == "cuda"
            else {},
        )
    )

    query_texts = [
        item["query"]
        for item in queries
    ]

    print(
        "Creating query embeddings..."
    )

    query_embeddings = (
        embedding_model.encode(
            query_texts,
            prompt_name="query",
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=True,
        )
    )

    all_candidates = []

    retrieval_top1 = 0
    retrieval_top3 = 0
    retrieval_top10 = 0

    for item, query_embedding in zip(
        queries,
        query_embeddings,
    ):
        scores = (
            query_embedding
            @ document_embeddings.T
        )

        top_k = min(
            RETRIEVAL_TOP_K,
            len(chunks),
        )

        indices = torch.topk(
            scores,
            k=top_k,
        ).indices.tolist()

        candidates = []

        for index in indices:
            candidate = dict(
                chunks[index]
            )

            candidate[
                "retrieval_score"
            ] = float(
                scores[index].item()
            )

            candidates.append(
                candidate
            )

        if is_expected(
            candidates[0],
            item,
        ):
            retrieval_top1 += 1

        if any(
            is_expected(
                candidate,
                item,
            )
            for candidate
            in candidates[:3]
        ):
            retrieval_top3 += 1

        if any(
            is_expected(
                candidate,
                item,
            )
            for candidate
            in candidates
        ):
            retrieval_top10 += 1

        all_candidates.append(
            candidates
        )

    del query_embeddings
    del embedding_model

    clear_gpu()

    total = len(queries)

    print("\nRetrieval:")
    print(
        "Recall@1:",
        f"{retrieval_top1 / total:.2%}",
    )
    print(
        "Recall@3:",
        f"{retrieval_top3 / total:.2%}",
    )
    print(
        "Recall@10:",
        f"{retrieval_top10 / total:.2%}",
    )

    return all_candidates


def evaluate_reranker(
    model_name,
    label,
    queries,
    all_candidates,
    device,
):
    print("\n")
    print("=" * 70)
    print(
        f"EVALUATING: {label}"
    )
    print("=" * 70)

    model_path = Path(
        str(model_name)
    )

    if model_path.exists():
        model_source = str(
            model_path
        )
    else:
        model_source = str(
            model_name
        )

    reranker = CrossEncoder(
        model_source,
        device=device,
        max_length=256,
        prompts={
            "query":
            RERANKER_PROMPT
        },
        default_prompt_name="query",
    )

    results = []

    correct = 0
    reciprocal_ranks = []
    latencies = []

    category_stats = defaultdict(
        lambda: {
            "correct": 0,
            "total": 0,
        }
    )

    for number, (
        item,
        candidates,
    ) in enumerate(
        zip(
            queries,
            all_candidates,
        ),
        1,
    ):
        start = time.perf_counter()

        pairs = [
            (
                item["query"],
                candidate["text"],
            )
            for candidate in candidates
        ]

        scores = reranker.predict(
            pairs
        )

        ranked = sorted(
            zip(
                candidates,
                scores,
            ),
            key=lambda pair:
            float(pair[1]),
            reverse=True,
        )

        ranked_chunks = [
            candidate
            for candidate, _
            in ranked
        ]

        expected_ranks = [
            rank
            for rank, candidate
            in enumerate(
                ranked_chunks,
                1,
            )
            if is_expected(
                candidate,
                item,
            )
        ]

        if expected_ranks:
            best_rank = min(
                expected_ranks
            )
            reciprocal_ranks.append(
                1 / best_rank
            )
        else:
            best_rank = None
            reciprocal_ranks.append(
                0
            )

        top1_correct = (
            best_rank == 1
        )

        if top1_correct:
            correct += 1

        category = item[
            "category"
        ]

        category_stats[
            category
        ]["total"] += 1

        if top1_correct:
            category_stats[
                category
            ]["correct"] += 1

        latency = (
            time.perf_counter()
            - start
        ) * 1000

        latencies.append(
            latency
        )

        results.append(
            {
                "query":
                item["query"],
                "category":
                category,
                "correct":
                top1_correct,
                "rank":
                best_rank,
                "predicted_page":
                ranked_chunks[0][
                    "page"
                ],
                "predicted_text":
                ranked_chunks[0][
                    "text"
                ],
                "expected_page":
                item[
                    "expected_page"
                ],
                "expected_contains":
                item[
                    "expected_contains"
                ],
            }
        )

        status = (
            "PASS"
            if top1_correct
            else "FAIL"
        )

        print(
            f"{number:02d}. "
            f"[{status}] "
            f"{item['query']}"
        )

        if not top1_correct:
            print(
                "    Expected page:",
                item[
                    "expected_page"
                ],
            )
            print(
                "    Expected:",
                item[
                    "expected_contains"
                ],
            )
            print(
                "    Predicted page:",
                ranked_chunks[0][
                    "page"
                ],
            )
            print(
                "    Correct rank:",
                best_rank,
            )

    total = len(
        queries
    )

    accuracy = (
        correct / total
    )

    mrr = (
        sum(
            reciprocal_ranks
        )
        / total
    )

    avg_latency = (
        sum(latencies)
        / total
    )

    print("\n")
    print("-" * 70)
    print(
        f"{label} SUMMARY"
    )
    print("-" * 70)

    print(
        "Top-1 accuracy:",
        f"{accuracy:.2%}",
        f"({correct}/{total})",
    )

    print(
        "MRR:",
        round(
            mrr,
            4,
        ),
    )

    print(
        "Average reranker latency:",
        round(
            avg_latency,
            2,
        ),
        "ms",
    )

    print(
        "\nCategory accuracy:"
    )

    for category in sorted(
        category_stats
    ):
        stats = (
            category_stats[
                category
            ]
        )

        category_accuracy = (
            stats["correct"]
            / stats["total"]
        )

        print(
            f"  {category}: "
            f"{category_accuracy:.2%} "
            f"({stats['correct']}/"
            f"{stats['total']})"
        )

    del reranker

    clear_gpu()

    return {
        "label": label,
        "accuracy": accuracy,
        "mrr": mrr,
        "avg_latency": avg_latency,
        "results": results,
    }


def compare_models(
    base_result,
    private_result,
):
    improvements = []
    regressions = []
    both_correct = []
    both_wrong = []

    for base, private in zip(
        base_result["results"],
        private_result["results"],
    ):
        if (
            not base["correct"]
            and private["correct"]
        ):
            improvements.append(
                private
            )

        elif (
            base["correct"]
            and not private["correct"]
        ):
            regressions.append(
                private
            )

        elif (
            base["correct"]
            and private["correct"]
        ):
            both_correct.append(
                private
            )

        else:
            both_wrong.append(
                private
            )

    print("\n")
    print("=" * 70)
    print(
        "BASE vs PRIVATE RANK V3"
    )
    print("=" * 70)

    print(
        "Base Top-1:",
        f"{base_result['accuracy']:.2%}",
    )

    print(
        "PrivateRank V3 Top-1:",
        f"{private_result['accuracy']:.2%}",
    )

    print(
        "Accuracy change:",
        f"{private_result['accuracy'] - base_result['accuracy']:+.2%}",
    )

    print(
        "\nBase wrong -> V3 correct:",
        len(improvements),
    )

    print(
        "Base correct -> V3 wrong:",
        len(regressions),
    )

    print(
        "Both correct:",
        len(both_correct),
    )

    print(
        "Both wrong:",
        len(both_wrong),
    )

    if improvements:
        print(
            "\nIMPROVEMENTS"
        )

        for item in improvements:
            print(
                " +",
                item["query"],
            )

    if regressions:
        print(
            "\nREGRESSIONS"
        )

        for item in regressions:
            print(
                " -",
                item["query"],
            )

    if both_wrong:
        print(
            "\nBOTH MODELS WRONG"
        )

        for item in both_wrong:
            print(
                " !",
                item["query"],
            )


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

    chunks, document_embeddings, queries = (
        load_data(
            device
        )
    )

    print(
        "Indexed chunks:",
        len(chunks),
    )

    print(
        "Evaluation queries:",
        len(queries),
    )

    all_candidates = (
        build_retrieval_candidates(
            chunks,
            document_embeddings,
            queries,
            device,
        )
    )

    base_result = (
        evaluate_reranker(
            BASE_RERANKER,
            "Base Qwen",
            queries,
            all_candidates,
            device,
        )
    )

    private_result = (
        evaluate_reranker(
            PRIVATE_RANK_V3,
            "PrivateRank V3",
            queries,
            all_candidates,
            device,
        )
    )

    compare_models(
        base_result,
        private_result,
    )


if __name__ == "__main__":
    main()
    