import json
import random
from collections import Counter
from pathlib import Path


OUTPUT_DIR = Path("data/training_v4_1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG = random.Random(42)


def make_rank_row(scenario_id, category, query, positive, negative, split):
    docs = [positive, negative]
    labels = [1.0, 0.0]

    # Randomize document order so the model cannot learn a position shortcut.
    if RNG.random() < 0.5:
        docs.reverse()
        labels.reverse()

    return {
        "scenario_id": scenario_id,
        "category": category,
        "split": split,
        "query": query,
        "docs": docs,
        "labels": labels,
    }


TEMPORAL_FAMILIES = [
    {
        "subject": "mobile reimbursement",
        "category": "temporal_version",
        "split": "train",
        "unit": "rupees per month",
        "variants": [
            ("1 April 2026", 2200, 1500),
            ("1 May 2026", 2600, 1800),
            ("1 June 2026", 3000, 2000),
            ("1 July 2026", 3400, 2200),
        ],
    },
    {
        "subject": "internet allowance",
        "category": "temporal_version",
        "split": "train",
        "unit": "rupees per month",
        "variants": [
            ("1 March 2026", 1800, 1200),
            ("1 April 2026", 2100, 1400),
            ("1 May 2026", 2400, 1600),
            ("1 June 2026", 2800, 1800),
        ],
    },
    {
        "subject": "domestic meal cap",
        "category": "temporal_version",
        "split": "train",
        "unit": "rupees per day",
        "variants": [
            ("1 February 2026", 1800, 1200),
            ("1 March 2026", 2000, 1400),
            ("1 April 2026", 2200, 1500),
            ("1 May 2026", 2500, 1700),
        ],
    },
    {
        "subject": "customer database backup interval",
        "category": "temporal_version",
        "split": "train",
        "unit": "hours",
        "variants": [
            ("1 January 2026", 4, 12),
            ("1 March 2026", 3, 10),
            ("1 May 2026", 2, 8),
            ("1 July 2026", 1, 6),
        ],
    },
    {
        "subject": "contractor notice period",
        "category": "temporal_version",
        "split": "train",
        "unit": "calendar days",
        "variants": [
            ("1 March 2026", 30, 15),
            ("1 April 2026", 45, 20),
            ("1 June 2026", 60, 30),
            ("1 August 2026", 75, 45),
        ],
    },
    {
        "subject": "support ticket retention",
        "category": "temporal_version",
        "split": "train",
        "unit": "months",
        "variants": [
            ("1 February 2026", 24, 12),
            ("1 April 2026", 30, 18),
            ("1 June 2026", 36, 24),
            ("1 August 2026", 48, 30),
        ],
    },
    {
        "subject": "regional supplier payment term",
        "category": "temporal_version",
        "split": "train",
        "unit": "calendar days",
        "variants": [
            ("1 January 2026", 25, 45),
            ("1 March 2026", 30, 50),
            ("1 May 2026", 35, 55),
            ("1 July 2026", 40, 60),
        ],
    },
    {
        "subject": "remote work allowance",
        "category": "temporal_version",
        "split": "train",
        "unit": "days per week",
        "variants": [
            ("1 February 2026", 2, 1),
            ("1 April 2026", 3, 1),
            ("1 June 2026", 3, 2),
            ("1 August 2026", 4, 2),
        ],
    },
    # Entire subjects held out from training for transfer validation.
    {
        "subject": "security token rotation interval",
        "category": "temporal_version",
        "split": "validation",
        "unit": "days",
        "variants": [
            ("1 March 2026", 45, 90),
            ("1 May 2026", 30, 75),
            ("1 July 2026", 28, 60),
            ("1 September 2026", 21, 45),
        ],
    },
    {
        "subject": "purchase approval ceiling",
        "category": "temporal_version",
        "split": "validation",
        "unit": "thousand rupees",
        "variants": [
            ("1 February 2026", 80, 50),
            ("1 April 2026", 100, 60),
            ("1 June 2026", 120, 75),
            ("1 August 2026", 150, 100),
        ],
    },
]


PRESERVATION_FAMILIES = [
    {
        "category": "employee_status",
        "variants": [
            (
                "Confirmed employees receive an annual learning budget of 50000 rupees after probation is completed.",
                "Employees still on probation receive an annual learning budget of 12000 rupees.",
                [
                    "Confirmed employee ka annual learning budget kitna hai?",
                    "What learning budget applies after probation is completed?",
                    "How much training budget does a confirmed employee receive?",
                ],
            ),
            (
                "Confirmed remote employees may claim an equipment allowance of 30000 rupees.",
                "Probationary remote employees may claim an equipment allowance of 10000 rupees.",
                [
                    "Confirmed remote employee ko equipment allowance kitna milta hai?",
                    "What equipment allowance applies after confirmation?",
                    "How much can confirmed remote staff claim for equipment?",
                ],
            ),
        ],
    },
    {
        "category": "priority",
        "variants": [
            (
                "Severity 1 is critical. The response team must acknowledge a Severity 1 incident within 15 minutes.",
                "Severity 2 is high priority. The response team must acknowledge a Severity 2 incident within 90 minutes.",
                [
                    "Severity 1 critical incident ko kitni der me acknowledge karna hai?",
                    "What is the acknowledgement target for the critical incident?",
                    "Highest severity incident ka response target kya hai?",
                ],
            ),
            (
                "Priority P1 production failure requires investigation within 20 minutes.",
                "Priority P2 production degradation requires investigation within 120 minutes.",
                [
                    "P1 production failure ko kitni jaldi investigate karna hai?",
                    "What investigation target applies to Priority P1?",
                    "How fast must the highest priority production failure be checked?",
                ],
            ),
        ],
    },
    {
        "category": "payment_context",
        "variants": [
            (
                "Approved India suppliers are normally paid within 25 calendar days from final invoice approval.",
                "Approved Gulf suppliers are normally paid within 35 calendar days from final invoice approval.",
                [
                    "Approved India supplier ko payment kitne din me milta hai?",
                    "What is the normal payment period for an India supplier?",
                    "How long does an approved domestic supplier wait for payment?",
                ],
            ),
            (
                "Standard business customers must pay invoices within 30 calendar days from invoice date.",
                "Authorized resellers may pay invoices within 60 calendar days from invoice date.",
                [
                    "Normal business customer ka invoice term kya hai?",
                    "How soon must a standard customer pay?",
                    "What payment period applies to a non-reseller customer?",
                ],
            ),
        ],
    },
    {
        "category": "business_intent",
        "variants": [
            (
                "Employee exit rule: departing employees must return the assigned laptop before final exit clearance.",
                "Repair rule: an active employee with a damaged laptop must open an IT support ticket for repair.",
                [
                    "Employee company chhod raha hai to assigned laptop ka kya hoga?",
                    "What happens to the laptop during employee exit?",
                    "Departing employee ko company laptop ke saath kya karna hai?",
                ],
            ),
            (
                "Employee exit rule: departing employees must return their company access card before clearance.",
                "Replacement rule: an active employee with a damaged access card may request a replacement from Facilities.",
                [
                    "Employee resign kare to access card ka kya karega?",
                    "What is required for the access card during exit?",
                    "Departing staff ke access card ka process kya hai?",
                ],
            ),
        ],
    },
    {
        "category": "access_control",
        "variants": [
            (
                "Finance administrator access requires approval from both the finance director and the security team.",
                "Finance read-only access requires approval from the employee's manager.",
                [
                    "Who must approve finance administrator access?",
                    "Finance admin access ke liye kiski approval chahiye?",
                    "What approvals are required for privileged finance access?",
                ],
            ),
            (
                "HR administrator access requires approval from the HR director and information security.",
                "Standard HR employee access requires approval only from the reporting manager.",
                [
                    "Who approves HR administrator access?",
                    "HR admin access ke liye security approval chahiye?",
                    "What approval path applies to elevated HR access?",
                ],
            ),
        ],
    },
    {
        "category": "numeric_threshold",
        "variants": [
            (
                "Purchases up to INR 50000 may be approved by a department manager.",
                "Purchases above INR 50000 and up to INR 250000 require department head approval.",
                [
                    "Who can approve a purchase request of INR 40000?",
                    "40,000 rupees ki purchase kis level par approve hogi?",
                    "What approval applies to a purchase below INR 50000?",
                ],
            ),
            (
                "Expenses above INR 25000 and up to INR 100000 require department head approval.",
                "Expenses above INR 100000 require finance controller approval.",
                [
                    "Who approves a business expense of INR 80000?",
                    "80,000 rupees ka expense kis approval level par jayega?",
                    "What approval applies to an expense between 25000 and 100000 rupees?",
                ],
            ),
        ],
    },
    {
        "category": "retention",
        "variants": [
            (
                "Tax and statutory finance records should normally be retained for 8 years.",
                "Routine customer support tickets should normally be retained for 3 years after closure.",
                [
                    "Tax records normally kitne saal retain karne hain?",
                    "How long are statutory finance records kept?",
                    "What is the retention period for tax records?",
                ],
            ),
            (
                "Critical-system security event logs should be retained for at least 18 months.",
                "Routine application debug logs should be retained for 90 days.",
                [
                    "Critical security logs minimum kitne time retain hone chahiye?",
                    "How long are critical-system security event logs kept?",
                    "What retention period applies to critical security logs?",
                ],
            ),
        ],
    },
]


rows = []
scenario_id = 1

# Bidirectional temporal training: every rule pair teaches both CURRENT and BEFORE directions.
for family in TEMPORAL_FAMILIES:
    subject = family["subject"]
    unit = family["unit"]
    split = family["split"]

    for effective_date, current_value, old_value in family["variants"]:
        current_doc = (
            f"Current {subject} policy. From {effective_date}, the active {subject} is "
            f"{current_value} {unit}. This rule is currently in force."
        )
        old_doc = (
            f"Old {subject} policy - retired. Before {effective_date}, the {subject} was "
            f"{old_value} {unit}. This older rule is retained only for historical reference."
        )

        current_queries = [
            f"What is the current {subject}?",
            f"Abhi active {subject} kya hai?",
            f"From {effective_date}, which {subject} applies?",
            f"Latest {subject} rule kya hai?",
        ]
        historical_queries = [
            f"What was the {subject} before {effective_date}?",
            f"Before {effective_date} {subject} kya tha?",
            f"Prior to {effective_date}, which {subject} applied?",
            f"{effective_date} se pehle old {subject} rule kya tha?",
        ]

        for query in current_queries:
            rows.append(
                make_rank_row(
                    scenario_id,
                    family["category"],
                    query,
                    current_doc,
                    old_doc,
                    split,
                )
            )
        for query in historical_queries:
            rows.append(
                make_rank_row(
                    scenario_id,
                    family["category"],
                    query,
                    old_doc,
                    current_doc,
                    split,
                )
            )

        scenario_id += 1

# Replay examples protect capabilities that the base model already handles well.
for family_index, family in enumerate(PRESERVATION_FAMILIES):
    for variant_index, (positive, negative, queries) in enumerate(family["variants"]):
        # Second variant of each replay family is validation-only.
        split = "validation" if variant_index == 1 else "train"

        for query in queries:
            repeat_count = 4 if split == "train" else 1

            for _ in range(repeat_count):
                rows.append(
                    make_rank_row(
                        scenario_id,
                        family["category"],
                        query,
                        positive,
                        negative,
                        split,
                    )
                )

        scenario_id += 1

train_rows = [row for row in rows if row["split"] == "train"]
validation_rows = [row for row in rows if row["split"] == "validation"]

RNG.shuffle(train_rows)
RNG.shuffle(validation_rows)


def write_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as file:
        for row in data:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


write_jsonl(OUTPUT_DIR / "train.jsonl", train_rows)
write_jsonl(OUTPUT_DIR / "validation.jsonl", validation_rows)

manifest = {
    "version": "PrivateRank V4.1",
    "format": "RankNet labeled-list: query + two docs + labels",
    "seed": 42,
    "train_rows": len(train_rows),
    "validation_rows": len(validation_rows),
    "train_categories": dict(Counter(row["category"] for row in train_rows)),
    "validation_categories": dict(Counter(row["category"] for row in validation_rows)),
    "notes": [
        "Temporal examples are bidirectional: current and historical.",
        "Two entire temporal subjects are held out for validation.",
        "Replay examples preserve non-temporal enterprise ranking skills and are weighted 4x in training.",
        "The known real password failure text is not copied into training data.",
    ],
}

with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as file:
    json.dump(manifest, file, indent=2, ensure_ascii=False)

print("PrivateRank V4.1 training data generated.")
print("Train rows:", len(train_rows))
print("Validation rows:", len(validation_rows))
print("Train categories:")
for category, count in sorted(manifest["train_categories"].items()):
    print(f"  {category}: {count}")
print("Validation categories:")
for category, count in sorted(manifest["validation_categories"].items()):
    print(f"  {category}: {count}")
print("Output:", OUTPUT_DIR)
