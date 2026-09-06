import json
from pathlib import Path


OUTPUT_DIR = Path(
    "data/evaluation/real_document_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


QUERIES = [
    {
        "query": "How many paid annual leave days do permanent employees currently receive?",
        "expected_page": 3,
        "expected_contains": "24 working days",
        "category": "policy_version",
    },
    {
        "query": "What was the annual leave allowance before January 2026?",
        "expected_page": 3,
        "expected_contains": "21 working days",
        "category": "policy_version",
    },
    {
        "query": "Probation employee confirmation se pehle maximum kitni annual leave use kar sakta hai?",
        "expected_page": 3,
        "expected_contains": "no more than 6",
        "category": "employee_status",
    },
    {
        "query": "Current policy me unused annual leave next year kitni carry forward ho sakti hai?",
        "expected_page": 3,
        "expected_contains": "carried forward up to 5",
        "category": "policy_version",
    },

    {
        "query": "Confirmed employee ko saal me kitni paid sick leave milti hai?",
        "expected_page": 4,
        "expected_contains": "12 working days",
        "category": "employee_status",
    },
    {
        "query": "When can the company ask for a medical certificate for sick leave?",
        "expected_page": 4,
        "expected_contains": "more than two consecutive working days",
        "category": "business_rule",
    },
    {
        "query": "Eligible employee ek week me kitne din remotely work kar sakta hai?",
        "expected_page": 4,
        "expected_contains": "up to two days per week",
        "category": "business_rule",
    },
    {
        "query": "Do contractors receive employee leave benefits under this policy?",
        "expected_page": 4,
        "expected_contains": "Contractors do not receive employee leave entitlements",
        "category": "employee_status",
    },

    {
        "query": "Who can approve a business expense of INR 20,000?",
        "expected_page": 5,
        "expected_contains": "up to INR 25,000",
        "category": "numeric_threshold",
    },
    {
        "query": "30,000 rupees ka business expense kis approval level par jayega?",
        "expected_page": 5,
        "expected_contains": "above INR 25,000 require finance-controller approval",
        "category": "numeric_threshold",
    },
    {
        "query": "What approval is required for international business travel?",
        "expected_page": 5,
        "expected_contains": "International travel requires documented pre-approval",
        "category": "business_rule",
    },

    {
        "query": "What is the current minimum password length for privileged accounts?",
        "expected_page": 6,
        "expected_contains": "at least 16 characters",
        "category": "policy_version",
    },
    {
        "query": "Before February 2026 privileged account ka minimum password length kya tha?",
        "expected_page": 6,
        "expected_contains": "at least 12 characters",
        "category": "policy_version",
    },
    {
        "query": "Who may approve read-only access to a standard internal system?",
        "expected_page": 6,
        "expected_contains": "Read-only access",
        "category": "access_control",
    },

    {
        "query": "Severity 1 critical incident ko kitni der me acknowledge karna hai?",
        "expected_page": 7,
        "expected_contains": "within 15 minutes",
        "category": "priority",
    },
    {
        "query": "How quickly must a Severity 2 incident be acknowledged?",
        "expected_page": 7,
        "expected_contains": "within 90 minutes",
        "category": "priority",
    },

    {
        "query": "Employee company chhod raha hai to assigned laptop ka kya hoga?",
        "expected_page": 8,
        "expected_contains": "must be returned before final exit clearance",
        "category": "business_intent",
    },
    {
        "query": "What should an active employee do when assigned equipment is damaged?",
        "expected_page": 8,
        "expected_contains": "report the issue to IT",
        "category": "business_intent",
    },
    {
        "query": "When should system access be removed during employee offboarding?",
        "expected_page": 8,
        "expected_contains": "confirmed last working day",
        "category": "access_control",
    },

    {
        "query": "Approved Gulf supplier ko currently payment kitne din me milta hai?",
        "expected_page": 9,
        "expected_contains": "within 30 days",
        "category": "payment_context",
    },
    {
        "query": "How long is the normal payment period for an approved India supplier?",
        "expected_page": 9,
        "expected_contains": "within 25 days",
        "category": "payment_context",
    },
    {
        "query": "Before March 2026 Gulf supplier invoices were normally paid in how many days?",
        "expected_page": 9,
        "expected_contains": "within 45 days",
        "category": "policy_version",
    },

    {
        "query": "Enterprise customer with negotiated terms ko invoice pay karne ke liye kitne din mil sakte hain?",
        "expected_page": 10,
        "expected_contains": "up to 60 days",
        "category": "payment_context",
    },
    {
        "query": "What is the normal payment deadline for a standard business customer?",
        "expected_page": 10,
        "expected_contains": "within 30 days",
        "category": "payment_context",
    },
    {
        "query": "What happens to a formally disputed customer invoice?",
        "expected_page": 10,
        "expected_contains": "placed on hold",
        "category": "business_intent",
    },

    {
        "query": "Tax and finance records normally kitne saal retain karne hain?",
        "expected_page": 11,
        "expected_contains": "retained for 8 years",
        "category": "retention",
    },
    {
        "query": "How long should routine closed customer support tickets be retained?",
        "expected_page": 11,
        "expected_contains": "retained for 3 years",
        "category": "retention",
    },
    {
        "query": "Critical-system security logs minimum kitne time retain hone chahiye?",
        "expected_page": 11,
        "expected_contains": "at least 18 months",
        "category": "retention",
    },

    {
        "query": "Who can approve a purchase request of INR 40,000?",
        "expected_page": 12,
        "expected_contains": "up to INR 50,000",
        "category": "numeric_threshold",
    },
    {
        "query": "INR 100,000 ki purchase request ko kaun approve karega?",
        "expected_page": 12,
        "expected_contains": "above INR 50,000 and up to INR 250,000",
        "category": "numeric_threshold",
    },
    {
        "query": "Who approves a purchase request above INR 250,000?",
        "expected_page": 12,
        "expected_contains": "above INR 250,000 require finance-controller approval",
        "category": "numeric_threshold",
    },
    {
        "query": "When are at least three competitive quotations normally required?",
        "expected_page": 12,
        "expected_contains": "above INR 100,000",
        "category": "numeric_threshold",
    },

    {
        "query": "Which information can be distributed externally without restriction?",
        "expected_page": 13,
        "expected_contains": "unrestricted external distribution",
        "category": "classification",
    },
    {
        "query": "Internal information is intended for which people?",
        "expected_page": 13,
        "expected_contains": "employees and approved contractors",
        "category": "classification",
    },
    {
        "query": "Sensitive customer data and non-public financial information belong to which classification?",
        "expected_page": 13,
        "expected_contains": "Confidential information includes",
        "category": "classification",
    },
]


OUTPUT_FILE = (
    OUTPUT_DIR / "queries.json"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        QUERIES,
        file,
        indent=2,
        ensure_ascii=False,
    )


print(
    "Real Document Evaluation V1 created."
)
print(
    "Queries:",
    len(QUERIES),
)
print(
    "Output:",
    OUTPUT_FILE,
)
