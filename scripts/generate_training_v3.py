import json
import random
from pathlib import Path


OUTPUT_DIR = Path("data/training_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)


SCENARIOS = [
    # --------------------------------------------------
    # CURRENT VS OLD POLICY
    # --------------------------------------------------
    {
        "category": "policy_version",
        "subject": "mobile reimbursement",
        "positive": "The currently active mobile reimbursement limit is {a} rupees per month.",
        "negative": "The previous mobile reimbursement limit was {b} rupees per month and has been superseded.",
        "values": [(1800, 1200), (2200, 1500), (2500, 1600), (3000, 2000)],
        "queries": [
            "What is the current mobile reimbursement limit?",
            "Abhi mobile reimbursement kitna milta hai?",
            "Which mobile allowance is active now?",
            "Latest mobile reimbursement policy kya hai?",
            "How much can an employee currently claim for mobile expenses?",
            "What is the presently applicable mobile reimbursement amount?",
            "Current policy me mobile allowance kitna hai?",
            "Which reimbursement value replaced the earlier mobile allowance?",
            "What amount should finance use under the active mobile policy?",
            "Tell me the latest approved mobile reimbursement amount."
        ]
    },

    {
        "category": "policy_version",
        "subject": "internet allowance",
        "positive": "The active monthly internet allowance is {a} rupees.",
        "negative": "The old monthly internet allowance was {b} rupees and is no longer applicable.",
        "values": [(1500, 1000), (1800, 1200), (2000, 1400), (2400, 1600)],
        "queries": [
            "What internet allowance is active now?",
            "Latest internet allowance kitna hai?",
            "What is the current monthly internet reimbursement?",
            "Abhi employee internet ke liye kitna claim kar sakta hai?",
            "Which internet allowance should currently be used?",
            "What amount replaced the old internet reimbursement?",
            "Current internet expense rule kya kehta hai?",
            "How much is reimbursed under the latest internet policy?",
            "What is today's approved internet allowance?",
            "Which internet reimbursement limit is still valid?"
        ]
    },

    # --------------------------------------------------
    # PRIORITY / SEVERITY
    # --------------------------------------------------
    {
        "category": "priority",
        "subject": "service incident",
        "positive": "A critical service incident requires an initial response within {a} minutes.",
        "negative": "A high-priority service incident requires an initial response within {b} minutes.",
        "values": [(15, 90), (20, 120), (30, 180), (10, 60)],
        "queries": [
            "How quickly must a critical service incident be handled?",
            "Critical service issue ka response time kitna hai?",
            "What is the response target for the most severe service incident?",
            "How soon should the team respond to a critical outage?",
            "What SLA applies to a critical incident?",
            "Severe service failure ko kitni der me acknowledge karna hai?",
            "What is the required first-response time for a critical issue?",
            "How fast must operations react to the highest severity incident?",
            "Critical incident ke liye immediate response target kya hai?",
            "Which response time applies to a critical service problem?"
        ]
    },

    {
        "category": "priority",
        "subject": "payment incident",
        "positive": "A critical payment-processing failure must be investigated within {a} minutes.",
        "negative": "A high-priority payment issue must be investigated within {b} minutes.",
        "values": [(20, 120), (30, 180), (15, 90), (10, 60)],
        "queries": [
            "How fast must a critical payment failure be investigated?",
            "Critical payment issue ko kitni der me investigate karna hai?",
            "What investigation target applies to the highest severity payment failure?",
            "How soon should finance engineering investigate a critical payment outage?",
            "What is the critical payment incident SLA?",
            "Most severe payment failure ka investigation time kya hai?",
            "How quickly does a critical transaction-processing problem need attention?",
            "Which response target applies when payment processing is completely down?",
            "Critical payment processing failure ko kitni jaldi check karna hai?",
            "What is the required investigation time for a critical payment incident?"
        ]
    },

    # --------------------------------------------------
    # EMPLOYEE STATUS
    # --------------------------------------------------
    {
        "category": "employee_status",
        "subject": "training budget",
        "positive": "Confirmed employees may receive an annual professional training budget of {a} rupees.",
        "negative": "Employees under probation may receive a professional training budget of only {b} rupees.",
        "values": [(40000, 10000), (50000, 15000), (60000, 20000), (30000, 8000)],
        "queries": [
            "What training budget does a confirmed employee receive?",
            "Confirmed employee ko training ke liye kitna budget milta hai?",
            "How much professional development budget is available after confirmation?",
            "What annual training allowance applies to confirmed staff?",
            "Permanent confirmed staff ka training budget kya hai?",
            "How much can a confirmed employee spend on approved training?",
            "Which training amount applies after probation is completed?",
            "What is the professional training budget for confirmed employees?",
            "Confirmation ke baad employee ko training allowance kitna milta hai?",
            "What development budget should HR apply for confirmed staff?"
        ]
    },

    {
        "category": "employee_status",
        "subject": "equipment allowance",
        "positive": "Confirmed remote employees receive an equipment allowance of {a} rupees.",
        "negative": "Probationary remote employees receive an equipment allowance of {b} rupees.",
        "values": [(25000, 10000), (30000, 12000), (35000, 15000), (20000, 8000)],
        "queries": [
            "What equipment allowance is available to confirmed remote employees?",
            "Confirmed remote employee ko equipment ke liye kitna milta hai?",
            "How much home-office equipment allowance applies after confirmation?",
            "What amount can confirmed remote staff claim for equipment?",
            "Confirmed work-from-home staff ka equipment budget kya hai?",
            "Which equipment allowance applies to a confirmed employee?",
            "How much can a permanent remote worker claim for equipment?",
            "What remote-work equipment amount should be used after confirmation?",
            "Confirmation ke baad WFH equipment allowance kitna hai?",
            "What is the equipment benefit for confirmed remote staff?"
        ]
    },

    # --------------------------------------------------
    # CUSTOMER / VENDOR CONTEXT
    # --------------------------------------------------
    {
        "category": "payment_context",
        "subject": "customer payment",
        "positive": "Standard business customers must pay invoices within {a} days.",
        "negative": "Authorized resellers may pay invoices within {b} days.",
        "values": [(20, 45), (30, 60), (15, 40), (25, 50)],
        "queries": [
            "What is the invoice term for a normal business customer?",
            "Normal business customer ko payment ke liye kitne din milte hain?",
            "How soon must a standard customer pay an invoice?",
            "What payment period applies to ordinary business customers?",
            "Standard customer invoice due kab hota hai?",
            "Which payment term should be used for a regular customer?",
            "How many days does a non-reseller customer have to pay?",
            "What is the normal customer payment deadline?",
            "Regular business customer ka invoice term kya hai?",
            "When is payment due for a standard customer account?"
        ]
    },

    {
        "category": "payment_context",
        "subject": "supplier payment",
        "positive": "Approved domestic suppliers are normally paid within {a} days.",
        "negative": "Approved international suppliers are normally paid within {b} days.",
        "values": [(25, 55), (30, 60), (20, 50), (35, 70)],
        "queries": [
            "How long does payment take for a domestic supplier?",
            "Domestic vendor ko payment kitne din me hoti hai?",
            "What payment period applies to a local supplier?",
            "How soon is an approved domestic vendor normally paid?",
            "Local supplier ka standard payment term kya hai?",
            "Which payment duration applies to domestic suppliers?",
            "How many days does a local vendor normally wait for payment?",
            "What is the normal payment schedule for domestic suppliers?",
            "India-based supplier ko approved invoice ka payment kab milta hai?",
            "When should a domestic supplier expect payment?"
        ]
    },

    # --------------------------------------------------
    # EXIT VS REPAIR / REPLACEMENT
    # --------------------------------------------------
    {
        "category": "business_intent",
        "subject": "tablet",
        "positive": "Employees leaving the company must return the assigned company tablet before final clearance.",
        "negative": "Employees whose assigned tablet is damaged may request a replacement after IT inspection.",
        "values": [(1, 1)],
        "queries": [
            "What happens to the company tablet when an employee leaves?",
            "Employee company chhod raha hai to tablet ka kya karega?",
            "Must a departing employee return the assigned tablet?",
            "What is required for the tablet during employee exit?",
            "Exit clearance ke time company tablet ka kya hoga?",
            "What should an employee do with the tablet after resigning?",
            "Does company equipment have to be returned when employment ends?",
            "Departing staff ke assigned tablet ka process kya hai?",
            "What is the exit requirement for a company-issued tablet?",
            "When an employee leaves, where should the company tablet go?"
        ]
    },

    {
        "category": "business_intent",
        "subject": "access card",
        "positive": "Departing employees must return their company access card before exit clearance is completed.",
        "negative": "Employees with a damaged access card may request a replacement from facilities.",
        "values": [(1, 1)],
        "queries": [
            "What happens to an access card when an employee leaves?",
            "Employee resign kare to access card ka kya karega?",
            "Must departing staff return their access card?",
            "What is the exit process for a company access card?",
            "Exit clearance se pehle access card ka kya karna hai?",
            "Where should an employee return the card after leaving the company?",
            "What does HR require for the access card during employee exit?",
            "Departing employee ka access card return hona chahiye?",
            "What is required with an assigned access card when employment ends?",
            "Employee exit ke waqt access card ka process kya hai?"
        ]
    },

    # --------------------------------------------------
    # ACCESS LEVEL
    # --------------------------------------------------
    {
        "category": "access_control",
        "subject": "finance system",
        "positive": "Finance administrator access requires approval from both the finance director and security team.",
        "negative": "Finance read-only access requires approval only from the employee's manager.",
        "values": [(1, 1)],
        "queries": [
            "Who must approve finance administrator access?",
            "Finance admin access ke liye kiski approval chahiye?",
            "What approvals are required for privileged finance-system access?",
            "Does finance administrator access need security approval?",
            "Who authorizes admin-level access to the finance platform?",
            "Finance system me administrator rights kaun approve karega?",
            "What approvals are necessary for elevated finance access?",
            "Who must sign off on privileged finance access?",
            "Admin finance permissions ke liye kya approvals lagenge?",
            "Which teams approve finance administrator privileges?"
        ]
    },

    {
        "category": "access_control",
        "subject": "HR system",
        "positive": "HR administrator access requires approval from the HR director and information-security team.",
        "negative": "Standard HR employee access requires approval only from the reporting manager.",
        "values": [(1, 1)],
        "queries": [
            "Who approves HR administrator access?",
            "HR admin access ke liye security approval chahiye?",
            "What approvals are required for privileged HR-system access?",
            "Who must authorize administrative HR permissions?",
            "HR system administrator rights kiski approval se milte hain?",
            "What approval path applies to elevated HR access?",
            "Does admin-level HR access require information security?",
            "Who signs off on privileged access to HR records?",
            "HR administrator permissions ke liye kya approval chahiye?",
            "Which authorities approve HR administrator access?"
        ]
    }
]


rows = []
scenario_id = 1


for scenario in SCENARIOS:

    for a, b in scenario["values"]:

        positive = scenario["positive"].format(
            a=a,
            b=b
        )

        negative = scenario["negative"].format(
            a=a,
            b=b
        )

        for query in scenario["queries"]:

            rows.append({
                "scenario_id": scenario_id,
                "category": scenario["category"],
                "query": query,
                "document": positive,
                "label": 1
            })

            rows.append({
                "scenario_id": scenario_id,
                "category": scenario["category"],
                "query": query,
                "document": negative,
                "label": 0
            })

        scenario_id += 1


random.shuffle(rows)

scenario_ids = sorted(
    set(row["scenario_id"] for row in rows)
)

random.shuffle(scenario_ids)

validation_count = max(
    1,
    int(len(scenario_ids) * 0.15)
)

validation_ids = set(
    scenario_ids[:validation_count]
)


train_rows = [
    row for row in rows
    if row["scenario_id"] not in validation_ids
]

validation_rows = [
    row for row in rows
    if row["scenario_id"] in validation_ids
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
                ) + "\n"
            )


write_jsonl(
    OUTPUT_DIR / "train.jsonl",
    train_rows
)

write_jsonl(
    OUTPUT_DIR / "validation.jsonl",
    validation_rows
)


categories = sorted(
    set(row["category"] for row in rows)
)


print("Training V3 generated.")
print("Total rows:", len(rows))
print("Train rows:", len(train_rows))
print("Validation rows:", len(validation_rows))
print("Unique scenarios:", len(scenario_ids))
print("Categories:", len(categories))

for category in categories:

    count = sum(
        1 for row in rows
        if row["category"] == category
    )

    print(
        f"  {category}: {count}"
    )