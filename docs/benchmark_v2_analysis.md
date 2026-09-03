# Benchmark V2 Analysis

## Results

| Metric | Score |
|---|---:|
| Retrieval Recall@1 | 70.00% |
| Retrieval Recall@3 | 100.00% |
| Reranker Recall@1 | 80.00% |
| MRR | 0.90 |
| Average Query Latency | 117.75 ms |

## Key Observation

The correct document appeared within the top 3 retrieval candidates for every evaluation query.

This indicates that the embedding retrieval stage is performing well as a candidate generator.

The main improvement opportunity is currently the reranking stage.

## Reranker Failure Categories

### 1. Policy Version Awareness

The model confused current policies with superseded policies.

Examples:

- Current annual leave policy vs old annual leave policy
- Current password requirement vs previous password requirement

### 2. Priority / Severity Understanding

The model confused:

- Critical support incidents
- High-priority support incidents

### 3. Business Process Intent

The model confused:

- Returning a laptop during employee exit
- Replacing a damaged laptop

## Next Goal

Create a dedicated training dataset containing hard negative examples for:

- Current vs old policies
- Active vs superseded rules
- Critical vs high vs normal priority
- Employee lifecycle events
- Similar business processes with different intent
- English and Hinglish queries

Benchmark V2 will remain an unseen evaluation set and will not be used as training data.