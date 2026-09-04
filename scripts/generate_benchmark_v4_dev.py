import json
import random
from pathlib import Path


OUTPUT = Path("data/benchmark_v4_dev")
OUTPUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

documents = {}
queries = []

case_id = 1


def add_case(category, query, correct_text, negatives):
    global case_id

    correct_file = f"{category}_{case_id}_correct.txt"

    documents[correct_file] = correct_text

    for index, text in enumerate(negatives, 1):
        filename = f"{category}_{case_id}_negative_{index}.txt"
        documents[filename] = text

    queries.append({
        "query": query,
        "expected_files": [correct_file]
    })

    case_id += 1


# -------------------------------------------------
# 1. CURRENT VS OLD POLICY
# -------------------------------------------------

for i in range(10):

    old_value = 18 + i
    current_value = old_value + 5

    add_case(
        "policy_version",

        f"What is the currently active annual leave allowance for policy group {i + 1}?",

        f"""
Policy Group {i + 1}

Effective from July 2026, confirmed employees receive
{current_value} paid annual leave days each calendar year.

This policy replaces the earlier annual leave rule.
Only this version is currently active.
""",

        [
            f"""
Policy Group {i + 1}

Before July 2026, confirmed employees received
{old_value} paid annual leave days each year.

This document has been superseded by a newer policy.
""",

            f"""
Policy Group {i + 1}

Probationary employees may receive
{max(5, old_value - 8)} paid leave days during probation.

This rule applies only while the employee remains on probation.
"""
        ]
    )


# -------------------------------------------------
# 2. EXPENSE THRESHOLDS
# -------------------------------------------------

for i in range(10):

    threshold = 20000 + (i * 5000)

    add_case(
        "expense_threshold",

        f"Expense group {i + 1}: if an employee submits {threshold + 5000} rupees, whose approval is required?",

        f"""
Expense Group {i + 1}

Business expenses of {threshold} rupees or more require
approval from both the reporting manager and finance department.

Receipts and business justification are mandatory.
""",

        [
            f"""
Expense Group {i + 1}

Business expenses below {threshold} rupees require
approval only from the reporting manager.

Valid receipts must still be attached.
""",

            f"""
Expense Group {i + 1}

Travel advances below {threshold} rupees may be approved
by the employee's department manager.
"""
        ]
    )


# -------------------------------------------------
# 3. INCIDENT SEVERITY
# -------------------------------------------------

for i in range(10):

    critical = 10 + i
    high = 60 + (i * 5)
    normal = 480 + (i * 10)

    add_case(
        "incident_priority",

        f"Service group {i + 1}: critical incident ka first response time kya hai?",

        f"""
Service Group {i + 1}

Severity-one critical incidents require an initial response
within {critical} minutes.

Immediate investigation must begin after acknowledgement.
""",

        [
            f"""
Service Group {i + 1}

Severity-two high-priority incidents require an initial response
within {high} minutes.
""",

            f"""
Service Group {i + 1}

Standard severity-three incidents require an initial response
within {normal} minutes.
"""
        ]
    )


# -------------------------------------------------
# 4. EMPLOYEE STATUS
# -------------------------------------------------

for i in range(10):

    permanent = 5 + i
    probation = 1 + (i % 3)

    add_case(
        "employee_status",

        f"Team {i + 1}: confirmed employee month me kitne remote-work days use kar sakta hai?",

        f"""
Team {i + 1}

Confirmed permanent employees may work remotely
up to {permanent} days each month.

Manager approval is required.
""",

        [
            f"""
Team {i + 1}

Employees currently under probation may work remotely
only {probation} days each month.
""",

            f"""
Team {i + 1}

External contractors are not automatically eligible
for the employee remote-work allowance.
"""
        ]
    )


# -------------------------------------------------
# 5. REGION
# -------------------------------------------------

for i in range(10):

    india = 30 + i
    gulf = 45 + i

    add_case(
        "regional_policy",

        f"Region policy {i + 1}: Gulf supplier ko approved invoice ka payment kitne din me milta hai?",

        f"""
Region Policy {i + 1}

Approved suppliers registered in Gulf-region offices
are normally paid within {gulf} days after invoice approval.

Regional compliance verification must be complete.
""",

        [
            f"""
Region Policy {i + 1}

Approved suppliers registered in India
are normally paid within {india} days after invoice approval.
""",

            f"""
Region Policy {i + 1}

International customer invoices have a payment period
of {gulf + 15} days.

This rule applies to customers, not suppliers.
"""
        ]
    )


# -------------------------------------------------
# 6. CUSTOMER TYPE
# -------------------------------------------------

for i in range(10):

    standard = 20 + i
    enterprise = 50 + i

    add_case(
        "customer_type",

        f"Customer group {i + 1}: enterprise customer with negotiated terms ko payment ke liye kitne din milte hain?",

        f"""
Customer Group {i + 1}

Approved enterprise customers with negotiated credit terms
may pay invoices within {enterprise} calendar days.

The negotiated term overrides the standard customer term.
""",

        [
            f"""
Customer Group {i + 1}

Standard business customers must pay invoices
within {standard} calendar days.
""",

            f"""
Customer Group {i + 1}

Approved suppliers are normally paid
within {standard + 10} days after invoice verification.
"""
        ]
    )


# -------------------------------------------------
# 7. EXIT VS REPAIR
# -------------------------------------------------

assets = [
    "laptop",
    "phone",
    "tablet",
    "access card",
    "security token",
    "monitor",
    "headset",
    "company router",
    "portable drive",
    "test device",
]

for i, asset in enumerate(assets, 1):

    add_case(
        "employee_exit",

        f"Employee company leave kar raha hai. Assigned {asset} ka kya karna hai?",

        f"""
Asset Procedure {i}

When employment ends, the employee must return
the assigned {asset} before final clearance.

Failure to return company property may delay settlement.
""",

        [
            f"""
Asset Procedure {i}

If the assigned {asset} is damaged during employment,
the employee should submit it to IT for repair assessment.
""",

            f"""
Asset Procedure {i}

An employee may request replacement of the assigned
{asset} when the existing equipment is unusable.
"""
        ]
    )


# -------------------------------------------------
# 8. ACCESS LEVEL
# -------------------------------------------------

for i in range(10):

    add_case(
        "access_control",

        f"System {i + 1}: administrator access ke liye kaun kaun approval dega?",

        f"""
System {i + 1}

Administrator-level access requires approval from
the department manager, system owner, and information-security team.

All three approvals are mandatory.
""",

        [
            f"""
System {i + 1}

Read-only access requires approval from
the employee's reporting manager only.
""",

            f"""
System {i + 1}

Standard application access requires approval from
the department manager and application owner.
"""
        ]
    )


# -------------------------------------------------
# 9. NOTICE PERIOD
# -------------------------------------------------

for i in range(10):

    permanent = 45 + i
    probation = 7 + i
    contractor = 15 + i

    add_case(
        "notice_period",

        f"Employment group {i + 1}: employee abhi probation par hai, resignation notice kitna dena hai?",

        f"""
Employment Group {i + 1}

Employees currently serving probation must provide
{probation} calendar days of resignation notice.
""",

        [
            f"""
Employment Group {i + 1}

Confirmed permanent employees must provide
{permanent} calendar days of resignation notice.
""",

            f"""
Employment Group {i + 1}

Fixed-term contractors must normally provide
{contractor} calendar days of notice.
"""
        ]
    )


# -------------------------------------------------
# 10. RETENTION PERIOD
# -------------------------------------------------

for i in range(10):

    tax = 7 + (i % 3)
    support = 2 + (i % 2)
    security = 1 + (i % 2)

    add_case(
        "retention",

        f"Record group {i + 1}: tax records kitne saal preserve karne hain?",

        f"""
Record Group {i + 1}

Tax and statutory financial records must normally
be retained for {tax} years.

Records under legal hold must be retained longer when required.
""",

        [
            f"""
Record Group {i + 1}

Closed customer-support tickets are retained
for {support} years.
""",

            f"""
Record Group {i + 1}

Routine security-event logs are retained
for {security} years unless an investigation requires longer retention.
"""
        ]
    )


# -------------------------------------------------
# WRITE FILES
# -------------------------------------------------

for filename, content in documents.items():

    (OUTPUT / filename).write_text(
        content.strip(),
        encoding="utf-8"
    )


(OUTPUT / "queries.json").write_text(
    json.dumps(
        queries,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


print("Benchmark V4 Dev created.")
print("Documents:", len(documents))
print("Queries:", len(queries))
print("Cases:", case_id - 1)