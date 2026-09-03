import json
import random
from pathlib import Path


OUTPUT_DIR = Path("data/training_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


domains = [
    {
        "category": "policy_version",
        "subject": "annual leave",
        "current": "The current annual leave allowance is {current} days per year.",
        "old": "The previous annual leave allowance was {old} days per year and is no longer active.",
        "current_values": [(24, 20), (28, 22), (25, 18)],
    },
    {
        "category": "password_policy",
        "subject": "password length",
        "current": "The current password policy requires at least {current} characters.",
        "old": "The old password policy required at least {old} characters and has been replaced.",
        "current_values": [(14, 12), (16, 10), (15, 8)],
    },
    {
        "category": "support_priority",
        "subject": "support response",
        "current": "Critical incidents require a response within {current} minutes.",
        "old": "High-priority incidents require a response within {old} minutes.",
        "current_values": [(30, 120), (20, 90), (15, 60)],
    },
    {
        "category": "notice_period",
        "subject": "resignation notice",
        "current": "Permanent employees must provide {current} days of resignation notice.",
        "old": "Probation employees must provide {old} days of resignation notice.",
        "current_values": [(60, 15), (45, 10), (30, 7)],
    },
    {
        "category": "payment_terms",
        "subject": "invoice payment",
        "current": "Standard customers must pay invoices within {current} days.",
        "old": "Enterprise customers with negotiated terms may pay within {old} days.",
        "current_values": [(30, 60), (21, 45), (15, 90)],
    },
    {
        "category": "vendor_payment",
        "subject": "vendor payment",
        "current": "Domestic suppliers are normally paid within {current} days.",
        "old": "International suppliers are normally paid within {old} days.",
        "current_values": [(30, 60), (21, 45), (15, 90)],
    },
    {
        "category": "employee_exit",
        "subject": "company laptop",
        "current": "Employees leaving the company must return their company laptop before final settlement.",
        "old": "Employees with a damaged company laptop may request a replacement from IT.",
        "current_values": [(1, 1)],
    },
    {
        "category": "employee_exit",
        "subject": "company phone",
        "current": "Employees leaving the company must return their company phone before final settlement.",
        "old": "Employees with a damaged company phone may request a replacement device.",
        "current_values": [(1, 1)],
    },
    {
        "category": "access_control",
        "subject": "administrator access",
        "current": "Administrator access requires approval from both the manager and security team.",
        "old": "Standard user access requires approval only from the reporting manager.",
        "current_values": [(1, 1)],
    },
    {
        "category": "remote_work",
        "subject": "remote work",
        "current": "Confirmed employees may work remotely {current} days per month.",
        "old": "Probation employees may work remotely only {old} days per month.",
        "current_values": [(4, 1), (6, 2), (8, 2)],
    },
]


query_templates = [
    "What is the current rule for {subject}?",
    "What is the latest policy for {subject}?",
    "Abhi {subject} ka current rule kya hai?",
    "Latest {subject} policy kya kehti hai?",
    "What rule is active now for {subject}?",
    "Please tell me the current {subject} requirement.",
    "{subject} ka present rule kya hai?",
    "What is currently applicable for {subject}?",
]


old_query_templates = [
    "What was the previous rule for {subject}?",
    "What did the old policy say about {subject}?",
    "Pehle {subject} ka rule kya tha?",
    "What requirement applied earlier for {subject}?",
]


rows = []
case_id = 1


def add_pair(query, positive, negative, category):
    global case_id

    rows.append({
        "case_id": case_id,
        "category": category,
        "query": query,
        "document": positive,
        "label": 1
    })

    rows.append({
        "case_id": case_id,
        "category": category,
        "query": query,
        "document": negative,
        "label": 0
    })

    case_id += 1


for domain in domains:

    for current_value, old_value in domain["current_values"]:

        positive = domain["current"].format(
            current=current_value,
            old=old_value
        )

        negative = domain["old"].format(
            current=current_value,
            old=old_value
        )

        for template in query_templates:

            query = template.format(
                subject=domain["subject"]
            )

            add_pair(
                query,
                positive,
                negative,
                domain["category"]
            )

        # Reverse task:
        # old policy becomes correct answer
        for template in old_query_templates:

            query = template.format(
                subject=domain["subject"]
            )

            add_pair(
                query,
                negative,
                positive,
                domain["category"]
            )


# Add paraphrase variations
base_rows = list(rows)

prefixes = [
    "",
    "Please check: ",
    "I need to know: ",
    "Company policy question: ",
]

expanded_rows = []

for row in base_rows:

    expanded_rows.append(row)

    for prefix in prefixes[1:]:

        copy_row = row.copy()

        copy_row["query"] = (
            prefix + copy_row["query"]
        )

        expanded_rows.append(copy_row)


rows = expanded_rows

random.shuffle(rows)


# Split by case id to reduce leakage
case_ids = sorted(
    set(row["case_id"] for row in rows)
)

random.shuffle(case_ids)

validation_count = max(
    1,
    int(len(case_ids) * 0.15)
)

validation_ids = set(
    case_ids[:validation_count]
)


train_rows = [
    row for row in rows
    if row["case_id"] not in validation_ids
]

validation_rows = [
    row for row in rows
    if row["case_id"] in validation_ids
]


def write_jsonl(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as file:

        for row in data:

            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )


write_jsonl(
    OUTPUT_DIR / "train.jsonl",
    train_rows
)

write_jsonl(
    OUTPUT_DIR / "validation.jsonl",
    validation_rows
)


print("Training V2 generated.")
print("Total rows:", len(rows))
print("Train rows:", len(train_rows))
print("Validation rows:", len(validation_rows))
print("Unique cases:", len(case_ids))