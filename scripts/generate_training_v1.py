import json
import random
from pathlib import Path


OUTPUT = Path("data/training_v1")
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)


cases = [
    {
        "category": "policy_version",
        "positive": "The current employee travel allowance is 1800 rupees per day. This policy became effective in July 2026.",
        "negative": "The previous employee travel allowance was 1500 rupees per day. This policy has been replaced.",
        "queries": [
            "What is the current daily travel allowance?",
            "Abhi employee ko daily travel allowance kitna milta hai?",
            "What allowance is active now for business travel?",
            "Under the latest policy, what is the travel allowance?"
        ]
    },
    {
        "category": "policy_version",
        "positive": "The current meal reimbursement limit is 1200 rupees per day.",
        "negative": "Before April 2026, the meal reimbursement limit was 900 rupees per day.",
        "queries": [
            "What is the current meal reimbursement limit?",
            "Latest meal allowance kitna hai?",
            "How much can employees claim for meals now?",
            "What is the active meal expense limit?"
        ]
    },
    {
        "category": "priority",
        "positive": "Severity-one production outages require an engineering response within 20 minutes.",
        "negative": "Severity-two production incidents require an engineering response within two hours.",
        "queries": [
            "How quickly must engineers respond to a severity-one outage?",
            "Critical production outage ka response time kya hai?",
            "What is the response target for the most severe outage?",
            "How fast should engineering react to a severity-one problem?"
        ]
    },
    {
        "category": "priority",
        "positive": "Urgent payroll failures must be investigated within 30 minutes.",
        "negative": "Normal payroll questions are reviewed within one business day.",
        "queries": [
            "How quickly must an urgent payroll failure be investigated?",
            "Urgent payroll issue kitni der me check hona chahiye?",
            "What is the investigation target for a payroll failure?",
            "How fast is an urgent payroll incident handled?"
        ]
    },
    {
        "category": "employee_status",
        "positive": "Confirmed employees may work remotely four days per month.",
        "negative": "Employees under probation may work remotely only one day per month.",
        "queries": [
            "How often can a confirmed employee work remotely?",
            "Confirmed staff ko month me kitne remote days milte hain?",
            "What is the remote-work allowance after confirmation?",
            "How many work-from-home days are available to permanent confirmed staff?"
        ]
    },
    {
        "category": "employee_status",
        "positive": "Employees under probation must provide ten days of resignation notice.",
        "negative": "Confirmed employees must provide forty-five days of resignation notice.",
        "queries": [
            "How much notice is required during probation?",
            "Probation employee resign kare to notice kitna hai?",
            "What resignation notice applies before confirmation?",
            "How many notice days does a probationary employee need?"
        ]
    },
    {
        "category": "payment_terms",
        "positive": "Standard retail customers must pay invoices within 21 days.",
        "negative": "Authorized distributors receive a 45-day invoice payment period.",
        "queries": [
            "When must a normal retail customer pay an invoice?",
            "Normal retail customer ka payment term kitna hai?",
            "How many days does a standard retail customer have to pay?",
            "What is the invoice term for regular retail customers?"
        ]
    },
    {
        "category": "payment_terms",
        "positive": "International suppliers are normally paid within 60 days after invoice approval.",
        "negative": "Domestic suppliers are normally paid within 30 days after invoice approval.",
        "queries": [
            "How long does payment take for an international supplier?",
            "International vendor ko payment kitne din me hoti hai?",
            "What payment period applies to overseas suppliers?",
            "After approval, when is an international supplier normally paid?"
        ]
    },
    {
        "category": "business_intent",
        "positive": "Departing employees must return their company phone before final settlement.",
        "negative": "Employees with a damaged company phone may request a replacement from IT.",
        "queries": [
            "What happens to the company phone when an employee leaves?",
            "Employee company chhod raha hai to company phone ka kya karega?",
            "Must a departing employee return the company phone?",
            "What is required for the phone during employee exit?"
        ]
    },
    {
        "category": "business_intent",
        "positive": "Cancelled customer orders require inventory reservation to be released immediately.",
        "negative": "Delayed customer orders keep their inventory reservation until the revised shipping date.",
        "queries": [
            "What happens to reserved inventory when an order is cancelled?",
            "Order cancel hone par reserved stock ka kya hoga?",
            "Should inventory remain reserved after cancellation?",
            "What action is required for stock on a cancelled order?"
        ]
    },
    {
        "category": "access_control",
        "positive": "Database administrator access requires approval from both the department manager and security team.",
        "negative": "Standard reporting access requires approval only from the department manager.",
        "queries": [
            "Who must approve database administrator access?",
            "DB admin access ke liye kis kis ki approval chahiye?",
            "What approvals are needed for administrator database access?",
            "Does privileged database access require security approval?"
        ]
    },
    {
        "category": "access_control",
        "positive": "Production server access requires security approval and a valid change ticket.",
        "negative": "Development server access may be approved by the engineering manager alone.",
        "queries": [
            "What is required to access a production server?",
            "Production server access lene ke liye kya approval chahiye?",
            "Does production access require a change ticket?",
            "What controls apply to production server access?"
        ]
    }
]


rows = []

for case_id, case in enumerate(cases, start=1):

    for query in case["queries"]:

        rows.append({
            "case_id": case_id,
            "category": case["category"],
            "query": query,
            "document": case["positive"],
            "label": 1
        })

        rows.append({
            "case_id": case_id,
            "category": case["category"],
            "query": query,
            "document": case["negative"],
            "label": 0
        })


# Split by case, not by individual rows.
# This reduces leakage between train and validation.
case_ids = list(range(1, len(cases) + 1))
random.shuffle(case_ids)

validation_case_ids = set(case_ids[:2])

train_rows = [
    row for row in rows
    if row["case_id"] not in validation_case_ids
]

validation_rows = [
    row for row in rows
    if row["case_id"] in validation_case_ids
]


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as file:
        for row in data:
            file.write(
                json.dumps(row, ensure_ascii=False) + "\n"
            )


write_jsonl(
    OUTPUT / "train.jsonl",
    train_rows
)

write_jsonl(
    OUTPUT / "validation.jsonl",
    validation_rows
)


print("Training V1 generated.")
print("Total pairs:", len(rows))
print("Train pairs:", len(train_rows))
print("Validation pairs:", len(validation_rows))
print("Validation case IDs:", sorted(validation_case_ids))