import os
from app.rag import db, TOP_K, ask, rewrite

# (question, expected file stem). Aim for 15 of these.
IN_SCOPE = [
    ("How many casual leaves do I get per year?", "leave_policy"),
    ("Can I carry forward unused casual leave?", "leave_policy"),
    ("How many days per week can I work from home?", "wfh_policy"),
    ("Can interns work from home?", "wfh_policy"),
    ("Which days are mandatory in-office days?", "wfh_policy"),
    ("What is the hotel limit per night in Kochi?", "expense_policy"),
    ("Within how many days must I submit an expense claim?", "expense_policy"),
    ("What is the health insurance sum insured?", "benefits_policy"),
    ("How many weeks of maternity leave are given?", "benefits_policy"),
    ("What is the notice period for Grade L2?", "notice_period_policy"),
    ("What is the notice period during probation?", "notice_period_policy"),
    ("How long does the final settlement take?", "notice_period_policy"),
    ("What gift value must I report to HR?", "code_of_conduct"),
    ("How quickly must a data breach be reported?", "code_of_conduct"),
    ("Which form do I use to resign?", "notice_period_policy"),
]

OUT_OF_SCOPE = [
    "What is the CEO's salary?",
    "How do I plant tomatoes?",
    "What is the capital of France?",
    "Which stocks should I buy?",
    "Who won the last cricket world cup?",
    "What is the dress code at the office?",
    "How many stock options do new joiners get?",
    "What is the parental leave policy in the US office?",
    "Does the company reimburse home rent?",
    "What is the salary band for Grade L3?",
]
# (first question, canned answer, follow-up, keyword that must appear in the rewrite)
FOLLOWUPS = [
    ("How many casual leaves do I get?", "You get 12 casual leaves per year.",
     "Can I carry them forward?", "leave"),
    ("What is the WFH policy?", "Employees may work from home 2 days a week.",
     "Does that apply to interns?", "home"),
    
]

def raw_scores(q):
    res = db.similarity_search_with_score(q, k=TOP_K)
    # unit vectors: squared L2 distance d -> cosine = 1 - d/2
    return [(os.path.basename(d.metadata.get("source", "")), 1 - dist / 2) for d, dist in res]

inq = [(q, exp, raw_scores(q)) for q, exp in IN_SCOPE]
oos = [(q, raw_scores(q)) for q in OUT_OF_SCOPE]

header = f"hit@{TOP_K}"
print(f"{'thr':>5} {header:>7} {'false_refusal':>14} {'oos_declined':>13}")
for t in [0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    hit = sum(any(f.startswith(exp) and s >= t for f, s in sc) for _, exp, sc in inq)
    refused = sum(not any(s >= t for _, s in sc) for _, _, sc in inq)
    declined = sum(not any(s >= t for _, s in sc) for _, sc in oos)
    print(f"{t:>5} {hit/len(inq):>7.0%} {refused/len(inq):>14.0%} {declined/len(oos):>13.0%}")

print("\nEnd-to-end (calls the LLM, uses MIN_COSINE from rag.py):")
declined = sum(not ask(q, [])["grounded"] for q, _ in oos)
print(f"Out-of-scope declined: {declined}/{len(oos)}")
for q1, a1, q2, kw in FOLLOWUPS:
    rewritten = rewrite(q2, [("User", q1), ("Assistant", a1)])
    print(f"{'OK  ' if kw in rewritten.lower() else 'FAIL'} {q2!r} -> {rewritten!r}")