# PrivateRankAI Model Progress

## Model Progress Summary

| Model | Development Benchmark V2 Recall@1 | MRR |
|---|---:|---:|
| Base Qwen3 Reranker | 80% | 0.900 |
| PrivateRank V1 | 85% | 0.925 |
| PrivateRank V2 | 85% | 0.925 |
| PrivateRank V3 | 90% | 0.950 |

## Locked Holdout V3

| Model | Reranker Recall@1 | MRR |
|---|---:|---:|
| Base Qwen3 Reranker | 100% | 1.000 |
| PrivateRank V3 | 100% | 1.000 |

## Key Findings

- Fine-tuning improved development benchmark Recall@1 from 80% to 90%.
- PrivateRank V3 showed no regression on the locked holdout benchmark.
- The current V3 holdout is too easy to differentiate the base model from PrivateRank V3.
- The next benchmark should be substantially harder and larger.
- Future evaluation should include longer documents, near-duplicate policies, conflicting versions, numerical distinctions, multilingual queries, and more difficult hard negatives.

## Current Best Model

PrivateRank V3

Model path:

`models/private_rank_v3`

## Next Goal

Build Benchmark V4 as a harder evaluation suite with at least 100 queries and more realistic enterprise ambiguity.