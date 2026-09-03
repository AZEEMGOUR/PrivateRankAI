import torch
from sentence_transformers import CrossEncoder

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

print("\nLoading PrivateRank base model...")

model = CrossEncoder(
    "Qwen/Qwen3-Reranker-0.6B",
    device="cuda"
)

query = "How many paid leaves can an employee take?"

documents = [
    "Employees receive 24 paid leaves every calendar year.",
    "Travel reimbursement must be submitted within 30 days.",
    "Employees receive 24 unpaid leaves every calendar year.",
    "The company provides health insurance to all permanent employees."
]

pairs = [(query, document) for document in documents]

scores = model.predict(pairs)

results = list(zip(documents, scores))
results.sort(key=lambda x: x[1], reverse=True)

print("\nQUERY:")
print(query)

print("\nRANKING:")

for rank, (document, score) in enumerate(results, start=1):
    print(f"\n#{rank}")
    print("Score:", float(score))
    print("Document:", document)