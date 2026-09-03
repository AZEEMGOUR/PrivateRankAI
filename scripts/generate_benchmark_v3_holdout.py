import hashlib
import json
from pathlib import Path


OUTPUT = Path("data/benchmark_v3_holdout")
OUTPUT.mkdir(parents=True, exist_ok=True)


documents = {
    "medical_leave_confirmed.txt": """
Confirmed employees receive 14 days of paid medical leave each calendar year.
A medical certificate is required when absence exceeds two consecutive working days.
""",

    "medical_leave_probation.txt": """
Employees serving probation receive 7 days of medical leave during the probation period.
Unused probation medical leave does not carry forward after confirmation.
""",

    "annual_leave_contract.txt": """
Fixed-term contract employees receive 18 paid annual leave days each year.
This entitlement applies only to employees hired under fixed-term contracts.
""",

    "annual_leave_permanent.txt": """
Permanent confirmed employees receive 26 paid annual leave days each year.
This rule does not apply to probation or fixed-term contract employees.
""",

    "expense_manager_limit.txt": """
Business expenses below 25000 rupees require approval from the reporting manager.
Receipts must be attached to the expense claim.
""",

    "expense_finance_limit.txt": """
Business expenses of 25000 rupees or more require both manager and finance approval.
Supporting receipts and business justification are mandatory.
""",

    "purchase_director_limit.txt": """
Purchases above 200000 rupees require approval from the department director.
The purchase order must be created before the supplier is instructed to proceed.
""",

    "purchase_manager_limit.txt": """
Purchases up to 200000 rupees may be approved by the department manager.
Normal procurement verification remains required.
""",

    "customer_payment_standard.txt": """
Regular domestic customers have a standard payment period of 30 calendar days.
The period begins from the invoice date.
""",

    "customer_payment_export.txt": """
Approved export customers receive a payment period of 75 calendar days.
This exception applies only to export accounts with approved credit terms.
""",

    "supplier_payment_local.txt": """
Verified local suppliers are normally paid within 30 days of invoice approval.
""",

    "supplier_payment_foreign.txt": """
Verified overseas suppliers are normally paid within 60 days of invoice approval.
Foreign payment compliance checks must be completed first.
""",

    "production_access.txt": """
Production application access requires manager approval, security approval,
and an approved access ticket.
""",

    "staging_access.txt": """
Staging application access requires approval from the engineering manager.
Security approval is not normally required for standard staging access.
""",

    "database_read_access.txt": """
Read-only production database access requires manager and data-owner approval.
Users may view records but cannot modify them.
""",

    "database_admin_access.txt": """
Production database administrator access requires manager, data-owner,
and information-security approval.
Administrator access permits controlled database changes.
""",

    "security_sev1.txt": """
Severity-one security incidents must be acknowledged within 15 minutes.
The incident response team must begin investigation immediately.
""",

    "security_sev2.txt": """
Severity-two security incidents must be acknowledged within 90 minutes.
Investigation begins according to the high-priority response process.
""",

    "security_sev3.txt": """
Severity-three security incidents must receive an initial response
within one business day.
""",

    "remote_india.txt": """
Employees assigned to the India office may work remotely up to six days per month
after confirmation.
""",

    "remote_gulf.txt": """
Employees assigned to Gulf-region offices may work remotely up to three days per month
after confirmation.
""",

    "probation_india.txt": """
New India-office employees normally complete a six-month probation period.
""",

    "probation_gulf.txt": """
New Gulf-region employees normally complete a three-month probation period.
""",

    "notice_permanent.txt": """
Permanent employees must normally provide 60 calendar days of resignation notice.
""",

    "notice_contract.txt": """
Fixed-term contract employees must normally provide 30 calendar days of resignation notice.
""",

    "notice_probation.txt": """
Employees currently under probation must provide 10 calendar days of resignation notice.
""",

    "backup_customer_db.txt": """
The production customer database is backed up every four hours.
Daily copies are retained separately for disaster recovery.
""",

    "backup_analytics.txt": """
Analytics datasets are backed up once every night.
Historical analytical exports are archived weekly.
""",

    "backup_source_code.txt": """
Source-code repositories are backed up every seven days.
Developers are expected to push committed work to the central repository regularly.
""",

    "refund_standard.txt": """
Approved customer refunds below 50000 rupees are normally processed
within five business days.
""",

    "refund_large.txt": """
Customer refunds of 50000 rupees or more require additional finance verification
and may take up to twelve business days.
""",

    "contract_review_standard.txt": """
Standard customer contracts should be reviewed 45 days before expiry.
""",

    "contract_review_strategic.txt": """
Strategic enterprise contracts should be reviewed 90 days before expiry
because commercial renegotiation may require additional approval.
""",

    "laptop_exit.txt": """
Departing employees must return their assigned laptop before final clearance.
Failure to return company equipment may delay final settlement.
""",

    "laptop_upgrade.txt": """
Employees may request a laptop upgrade when their current hardware
does not meet approved job requirements.
""",

    "phone_exit.txt": """
Departing employees must return assigned company mobile phones
before completion of exit clearance.
""",

    "phone_repair.txt": """
Damaged company phones should be submitted to IT for repair or replacement assessment.
""",

    "training_optional.txt": """
Optional professional training may be funded when the manager approves
the development request and budget is available.
""",

    "training_mandatory.txt": """
Mandatory compliance training must be completed by all applicable employees
within 30 days of assignment.
""",

    "records_tax.txt": """
Tax records must normally be retained for eight years.
""",

    "records_support.txt": """
Closed customer support tickets are retained for two years.
""",
}


queries = [
    {
        "query": "Confirmed employee ko saal me kitni paid medical leave milti hai?",
        "expected_files": ["medical_leave_confirmed.txt"]
    },
    {
        "query": "How much annual leave does a fixed-term employee receive?",
        "expected_files": ["annual_leave_contract.txt"]
    },
    {
        "query": "If an expense is 30000 rupees, whose approval is required?",
        "expected_files": ["expense_finance_limit.txt"]
    },
    {
        "query": "Two lakh se zyada purchase approve kaun karega?",
        "expected_files": ["purchase_director_limit.txt"]
    },
    {
        "query": "Export customer ko invoice pay karne ke liye kitne din milte hain?",
        "expected_files": ["customer_payment_export.txt"]
    },
    {
        "query": "How soon is an overseas supplier normally paid after approval?",
        "expected_files": ["supplier_payment_foreign.txt"]
    },
    {
        "query": "Production application access ke liye security approval chahiye?",
        "expected_files": ["production_access.txt"]
    },
    {
        "query": "Who must approve production database administrator access?",
        "expected_files": ["database_admin_access.txt"]
    },
    {
        "query": "How quickly must the team acknowledge the most severe security incident?",
        "expected_files": ["security_sev1.txt"]
    },
    {
        "query": "Severity two incident ka acknowledgement time kya hai?",
        "expected_files": ["security_sev2.txt"]
    },
    {
        "query": "India office confirmed employee month me kitne din ghar se kaam kar sakta hai?",
        "expected_files": ["remote_india.txt"]
    },
    {
        "query": "How long is probation for employees joining a Gulf-region office?",
        "expected_files": ["probation_gulf.txt"]
    },
    {
        "query": "Contract employee resign kare to notice kitna dena padega?",
        "expected_files": ["notice_contract.txt"]
    },
    {
        "query": "Employee abhi probation par hai. Resignation notice kitna hai?",
        "expected_files": ["notice_probation.txt"]
    },
    {
        "query": "How frequently is the live customer database backed up?",
        "expected_files": ["backup_customer_db.txt"]
    },
    {
        "query": "Analytics data ka backup kitni baar hota hai?",
        "expected_files": ["backup_analytics.txt"]
    },
    {
        "query": "A customer refund of 80000 rupees may take how long?",
        "expected_files": ["refund_large.txt"]
    },
    {
        "query": "Normal small approved refund kitne business days me process hota hai?",
        "expected_files": ["refund_standard.txt"]
    },
    {
        "query": "Strategic enterprise contract expiry se kitne din pehle review hona chahiye?",
        "expected_files": ["contract_review_strategic.txt"]
    },
    {
        "query": "What should happen to an assigned laptop when the employee exits?",
        "expected_files": ["laptop_exit.txt"]
    },
    {
        "query": "Damaged company phone ko employee kahan submit kare?",
        "expected_files": ["phone_repair.txt"]
    },
    {
        "query": "Mandatory compliance training kitne din ke andar complete karni hai?",
        "expected_files": ["training_mandatory.txt"]
    },
    {
        "query": "For how many years are tax records normally kept?",
        "expected_files": ["records_tax.txt"]
    },
    {
        "query": "How long are closed customer support tickets retained?",
        "expected_files": ["records_support.txt"]
    },
]


for filename, text in documents.items():
    (OUTPUT / filename).write_text(
        text.strip(),
        encoding="utf-8"
    )


queries_path = OUTPUT / "queries.json"

queries_path.write_text(
    json.dumps(
        queries,
        indent=2,
        ensure_ascii=False
    ),
    encoding="utf-8"
)


# Create a manifest so we can prove the holdout
# dataset has not silently changed later.
manifest = {}

for path in sorted(OUTPUT.glob("*")):
    if path.name == "manifest.json":
        continue

    digest = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    manifest[path.name] = digest


(OUTPUT / "manifest.json").write_text(
    json.dumps(
        manifest,
        indent=2
    ),
    encoding="utf-8"
)


print("Benchmark V3 Holdout created.")
print("Documents:", len(documents))
print("Queries:", len(queries))
print("Manifest files:", len(manifest))