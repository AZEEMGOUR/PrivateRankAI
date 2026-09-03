import json
from pathlib import Path


OUTPUT_DIR = Path("data/benchmark_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


documents = {
    "paid_leave_policy.txt": """
Employees are entitled to 24 paid annual leave days per calendar year.
Paid leave requires approval from the employee's reporting manager.
Unused paid leave may be carried forward subject to company policy.
""",

    "unpaid_leave_policy.txt": """
Employees may request up to 24 unpaid leave days per calendar year.
Unpaid leave is separate from paid annual leave.
Manager approval is mandatory before unpaid leave begins.
""",

    "sick_leave_policy.txt": """
Employees receive 12 sick leave days every year.
A medical certificate is required when sick leave exceeds two consecutive days.
Sick leave cannot normally be converted into paid annual leave.
""",

    "travel_reimbursement.txt": """
Travel reimbursement claims must be submitted within 30 days of completing the trip.
Valid receipts are required for hotel and transportation expenses.
Late claims may be rejected by the finance department.
""",

    "international_travel.txt": """
International business travel requires prior approval from department management.
Employees must submit the travel plan before booking international flights.
Passport and visa requirements remain the employee's responsibility.
""",

    "invoice_payment_terms.txt": """
Standard customer invoices are payable within 30 days from the invoice date.
Different payment terms may apply when specified in a signed customer agreement.
Overdue invoices may attract penalties.
""",

    "vendor_payment_terms.txt": """
Approved vendor invoices are normally processed within 45 days.
Vendor payment schedules depend on successful invoice verification.
Incomplete invoices may be returned to the supplier.
""",

    "expense_policy.txt": """
Employees must submit business expense claims with valid receipts.
Personal expenses are not reimbursable.
Expense reports should be submitted within 15 days.
""",

    "password_policy.txt": """
Company passwords must contain at least 12 characters.
Passwords should include uppercase letters, lowercase letters, numbers, and symbols.
Employees must not share passwords with colleagues.
""",

    "account_lock_policy.txt": """
A user account is temporarily locked after five failed login attempts.
The account may be unlocked automatically after 30 minutes.
IT support can manually unlock verified employee accounts.
""",

    "remote_work_policy.txt": """
Employees may work remotely up to three days per week with manager approval.
Employees must remain available during official working hours.
Confidential company information must be protected while working remotely.
""",

    "office_hours.txt": """
Normal office hours are from 9 AM to 6 PM Monday through Friday.
Employees are expected to complete their required working hours.
Individual departments may operate different approved shifts.
""",

    "notice_period.txt": """
Permanent employees are required to serve a 60-day notice period after resignation.
Management may approve an early release.
Pending company assets must be returned before the final working day.
""",

    "probation_policy.txt": """
New employees normally complete a six-month probation period.
Performance is reviewed before confirmation.
Management may extend probation when additional assessment is required.
""",

    "health_insurance.txt": """
Permanent employees are eligible for company-provided health insurance.
Coverage may also include eligible dependents according to the selected plan.
Insurance benefits begin after completion of enrollment requirements.
""",

    "laptop_policy.txt": """
Company laptops are provided to employees who require them for their work.
Employees are responsible for protecting assigned equipment.
Company laptops must be returned when employment ends.
""",

    "data_backup_policy.txt": """
Critical company data must be backed up every day.
Weekly backup copies are retained separately from primary systems.
Backup restoration tests are conducted periodically.
""",

    "data_retention.txt": """
Financial records must normally be retained for seven years.
Other business records may have different retention periods.
Records must not be deleted while subject to a legal hold.
""",

    "purchase_order_policy.txt": """
Purchases above 50000 rupees require an approved purchase order.
The purchase order must be issued before the supplier provides goods.
Emergency purchases require written management approval.
""",

    "vendor_onboarding.txt": """
New vendors must provide tax registration details and banking information.
Procurement verifies vendor documents before activation.
Incomplete onboarding applications remain pending.
""",

    "refund_policy.txt": """
Approved customer refunds are normally processed within seven business days.
Refunds are returned through the original payment method when possible.
Additional verification may be required for large refunds.
""",

    "customer_support_sla.txt": """
Critical customer support incidents require an initial response within one hour.
High-priority incidents require a response within four hours.
Normal support requests are handled according to the standard support queue.
""",

    "software_access_policy.txt": """
Employees receive software access according to their job responsibilities.
Access requests require approval from the employee's manager.
Access must be removed when it is no longer required.
""",

    "security_incident_policy.txt": """
Suspected security incidents must be reported immediately to the security team.
Employees should not attempt to hide or independently investigate serious incidents.
The security team coordinates investigation and response.
""",

    "overtime_policy.txt": """
Overtime must be approved by the reporting manager before additional hours are worked.
Approved overtime may be compensated according to employment terms.
Unauthorized overtime may not qualify for compensation.
""",

    "performance_review.txt": """
Formal employee performance reviews are conducted twice each year.
Managers evaluate objectives, achievements, and development requirements.
Review outcomes may be used when planning promotions and training.
""",

    "training_policy.txt": """
Employees may attend approved professional training programs.
Training expenses require prior authorization.
Certain sponsored training programs may include a service commitment.
""",

    "asset_return_policy.txt": """
Employees leaving the company must return laptops, access cards, and other company assets.
Asset clearance must be completed before final settlement.
Missing equipment may delay clearance.
""",

    "confidentiality_policy.txt": """
Employees must protect confidential business and customer information.
Confidential data must only be shared with authorized persons.
Confidentiality obligations may continue after employment ends.
""",

    "contract_renewal.txt": """
Customer contracts should be reviewed at least 30 days before their expiry date.
Renewal terms must be approved before a replacement agreement is signed.
Commercial changes should be documented in the renewed contract.
"""
}


queries = [
    {"query": "How many paid annual leave days do employees receive?", "expected_file": "paid_leave_policy.txt"},
    {"query": "How much unpaid leave can an employee request?", "expected_file": "unpaid_leave_policy.txt"},
    {"query": "How many sick days are available each year?", "expected_file": "sick_leave_policy.txt"},
    {"query": "What is the deadline for submitting travel expenses?", "expected_file": "travel_reimbursement.txt"},
    {"query": "Does overseas business travel need approval before booking?", "expected_file": "international_travel.txt"},
    {"query": "When does a customer normally have to pay an invoice?", "expected_file": "invoice_payment_terms.txt"},
    {"query": "How long does the company normally take to process vendor invoices?", "expected_file": "vendor_payment_terms.txt"},
    {"query": "Within how many days should employees file an expense report?", "expected_file": "expense_policy.txt"},
    {"query": "What is the minimum company password length?", "expected_file": "password_policy.txt"},
    {"query": "What happens after five incorrect login attempts?", "expected_file": "account_lock_policy.txt"},
    {"query": "How many days per week can staff work from home?", "expected_file": "remote_work_policy.txt"},
    {"query": "What are the normal working hours?", "expected_file": "office_hours.txt"},
    {"query": "How much notice must a permanent employee give before leaving?", "expected_file": "notice_period.txt"},
    {"query": "How long is the probation period for a new employee?", "expected_file": "probation_policy.txt"},
    {"query": "Are permanent employees provided medical insurance?", "expected_file": "health_insurance.txt"},
    {"query": "What must happen to a company laptop when an employee leaves?", "expected_file": "laptop_policy.txt"},
    {"query": "How frequently should critical information be backed up?", "expected_file": "data_backup_policy.txt"},
    {"query": "For how many years should financial records normally be kept?", "expected_file": "data_retention.txt"},
    {"query": "At what purchase value is a purchase order required?", "expected_file": "purchase_order_policy.txt"},
    {"query": "What information does a new supplier need to provide?", "expected_file": "vendor_onboarding.txt"},
    {"query": "How quickly are approved customer refunds normally processed?", "expected_file": "refund_policy.txt"},
    {"query": "How fast should support respond to a critical customer incident?", "expected_file": "customer_support_sla.txt"},
    {"query": "Who must approve an employee software access request?", "expected_file": "software_access_policy.txt"},
    {"query": "Who should employees contact when they suspect a cyber security incident?", "expected_file": "security_incident_policy.txt"},
    {"query": "Does overtime need approval before working extra hours?", "expected_file": "overtime_policy.txt"},
    {"query": "How often are formal employee performance reviews conducted?", "expected_file": "performance_review.txt"},
    {"query": "Can the company pay for approved professional training?", "expected_file": "training_policy.txt"},
    {"query": "Which company assets must be returned during employee exit?", "expected_file": "asset_return_policy.txt"},
    {"query": "Can employees share confidential customer information with anyone?", "expected_file": "confidentiality_policy.txt"},
    {"query": "How early should an expiring customer contract be reviewed?", "expected_file": "contract_renewal.txt"}
]


for filename, content in documents.items():
    path = OUTPUT_DIR / filename

    path.write_text(
        content.strip(),
        encoding="utf-8"
    )


with open(
    OUTPUT_DIR / "queries.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        queries,
        file,
        indent=2,
        ensure_ascii=False
    )


print("Benchmark V1 created.")
print("Documents:", len(documents))
print("Queries:", len(queries))