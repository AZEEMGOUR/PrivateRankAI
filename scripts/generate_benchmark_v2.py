import json
from pathlib import Path


OUTPUT = Path("data/benchmark_v2")
OUTPUT.mkdir(parents=True, exist_ok=True)


documents = {
    "annual_leave_current.txt": """
Effective from January 2026, permanent employees receive
24 days of paid annual leave per calendar year.
This policy replaces all earlier annual leave rules.
""",

    "annual_leave_old.txt": """
Under the policy used before January 2026, permanent employees
received 20 days of paid annual leave each year.
This policy has been superseded.
""",

    "probation_leave.txt": """
Employees who are still completing probation may use
up to 10 paid leave days during the probation period.
Normal annual leave rules apply after confirmation.
""",

    "unpaid_leave.txt": """
Employees may request up to 24 days of unpaid leave.
Unpaid leave is separate from paid annual leave.
""",

    "customer_invoice.txt": """
Standard customer invoices become due 30 days after
the invoice date unless another term is written in the contract.
""",

    "vendor_invoice.txt": """
Approved supplier invoices are normally processed
within 45 days after successful verification.
""",

    "enterprise_invoice.txt": """
Enterprise customers with specially negotiated contracts
may receive a 60-day payment period.
The standard 30-day term does not apply to these contracts.
""",

    "domestic_travel.txt": """
Domestic business travel requires manager approval
before travel arrangements are finalized.
""",

    "international_travel.txt": """
Overseas business trips require department-head approval
before flights or hotels are booked.
""",

    "travel_reimbursement.txt": """
Expense claims related to completed business trips
must be filed within 30 days after returning.
""",

    "password_current.txt": """
The current security standard requires passwords
to contain at least 14 characters.
This requirement became effective in 2026.
""",

    "password_old.txt": """
The previous security policy required passwords
to contain at least 12 characters.
This rule is no longer active.
""",

    "account_lock.txt": """
User accounts are temporarily locked after
five consecutive failed authentication attempts.
Automatic access is restored after 30 minutes.
""",

    "permanent_notice.txt": """
Confirmed permanent employees must normally serve
a 60-day notice period after resignation.
""",

    "probation_notice.txt": """
Employees currently under probation are required
to provide 15 days of resignation notice.
""",

    "contractor_exit.txt": """
External contractors may end an assignment with
seven days of notice unless their contract states otherwise.
""",

    "support_critical.txt": """
A severity-one customer incident requires
an initial support response within one hour.
""",

    "support_high.txt": """
High-priority customer incidents require
an initial response within four hours.
""",

    "support_normal.txt": """
Standard customer support requests normally receive
a response within one business day.
""",

    "financial_backup.txt": """
Critical finance databases are backed up every night.
Backup restoration is tested regularly.
""",

    "development_backup.txt": """
Development repositories are backed up once every week.
Developers should also push important source code frequently.
""",

    "financial_retention.txt": """
Accounting and financial records must normally
be retained for seven years.
""",

    "security_log_retention.txt": """
Security event logs are retained for twelve months
unless an investigation requires longer retention.
""",

    "software_access.txt": """
Ordinary software access requests require approval
from the employee's reporting manager.
""",

    "privileged_access.txt": """
Administrative and privileged system access requires
approval from both the manager and information-security team.
""",

    "standard_refund.txt": """
Normal approved customer refunds are processed
within seven business days.
""",

    "large_refund.txt": """
Customer refunds above 100000 rupees require
additional finance verification before processing.
""",

    "asset_exit.txt": """
Before final settlement, departing employees must return
their laptop, access card and all other company equipment.
""",

    "laptop_replacement.txt": """
Employees may request replacement of a damaged company laptop
after IT verifies the hardware problem.
"""
}


queries = [
    {
        "query": "Under the current policy, how much paid annual leave does a confirmed employee get?",
        "expected_files": ["annual_leave_current.txt"]
    },
    {
        "query": "How much annual leave was provided under the old policy?",
        "expected_files": ["annual_leave_old.txt"]
    },
    {
        "query": "Employee probation me hai, usko kitni paid leave mil sakti hai?",
        "expected_files": ["probation_leave.txt"]
    },
    {
        "query": "Are 24 unpaid days the same as annual paid leave?",
        "expected_files": ["unpaid_leave.txt"]
    },
    {
        "query": "Customer ka normal invoice kitne din me due hota hai?",
        "expected_files": ["customer_invoice.txt"]
    },
    {
        "query": "How long can an enterprise customer with negotiated terms take to pay?",
        "expected_files": ["enterprise_invoice.txt"]
    },
    {
        "query": "Supplier invoice process hone me normally kitna time lagta hai?",
        "expected_files": ["vendor_invoice.txt"]
    },
    {
        "query": "Who needs to approve an overseas trip before booking the flight?",
        "expected_files": ["international_travel.txt"]
    },
    {
        "query": "Trip complete hone ke baad expense claim kab tak submit karna hai?",
        "expected_files": ["travel_reimbursement.txt"]
    },
    {
        "query": "What is the minimum password length under the current security rule?",
        "expected_files": ["password_current.txt"]
    },
    {
        "query": "What was the previous minimum password requirement?",
        "expected_files": ["password_old.txt"]
    },
    {
        "query": "What happens if someone enters the wrong password five times?",
        "expected_files": ["account_lock.txt"]
    },
    {
        "query": "Permanent employee resign kare to notice period kitna hai?",
        "expected_files": ["permanent_notice.txt"]
    },
    {
        "query": "How much notice is required while still on probation?",
        "expected_files": ["probation_notice.txt"]
    },
    {
        "query": "Critical customer issue ka first response kitni der me dena hai?",
        "expected_files": ["support_critical.txt"]
    },
    {
        "query": "What is the response target for a high priority support incident?",
        "expected_files": ["support_high.txt"]
    },
    {
        "query": "How frequently are critical finance databases backed up?",
        "expected_files": ["financial_backup.txt"]
    },
    {
        "query": "For how many years must accounting records be preserved?",
        "expected_files": ["financial_retention.txt"]
    },
    {
        "query": "Who must approve administrator-level system access?",
        "expected_files": ["privileged_access.txt"]
    },
    {
        "query": "Employee company chhod raha hai to laptop ka kya hoga?",
        "expected_files": ["asset_exit.txt"]
    }
]


for filename, text in documents.items():
    (OUTPUT / filename).write_text(
        text.strip(),
        encoding="utf-8"
    )


with open(
    OUTPUT / "queries.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        queries,
        file,
        indent=2,
        ensure_ascii=False
    )


print("Benchmark V2 created.")
print("Documents:", len(documents))
print("Queries:", len(queries))