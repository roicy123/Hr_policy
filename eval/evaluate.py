import os
from app.rag import db, TOP_K, ask, rewrite

# (question, expected file stem, text the retrieved chunk must contain)
IN_SCOPE = [
    ("How many casual leaves do I get per year?", "leave_policy", "12 casual leaves"),
    ("Can I carry forward unused casual leave?", "leave_policy", "5 unused casual leaves"),
    ("How many days per week can I work from home?", "wfh_policy", "up to 2 days per week"),
    ("Can interns work from home?", "wfh_policy", "Interns and trainees must work from the office"),
    ("Which days are mandatory in-office days?", "wfh_policy", "Tuesday and Thursday"),
    ("What is the hotel limit per night in Kochi?", "expense_policy", "Tier 2 cities (Kochi"),
    ("Within how many days must I submit an expense claim?", "expense_policy", "within 30 days of the expense date"),
    ("What is the health insurance sum insured?", "benefits_policy", "5,00,000"),
    ("How many weeks of maternity leave are given?", "benefits_policy", "26 weeks"),
    ("What is the notice period for Grade L2?", "notice_period_policy", "Grades L1 to L3"),
    ("What is the notice period during probation?", "notice_period_policy", "15 days' notice"),
    ("How long does the final settlement take?", "notice_period_policy", "within 45 days"),
    ("What gift value must I report to HR?", "code_of_conduct", "Rs. 2,000 from clients or vendors"),
    ("How quickly must a data breach be reported?", "code_of_conduct", "24 hours"),
    ("Which form do I use to resign?", "notice_period_policy", "HR-12"),
    ("When do I have to notify security about a leaked file?", "code_of_conduct", "24 hours"),
    ("How much can I claim for a hotel in Pune?", "expense_policy", "Tier 2 cities (Kochi"),
    ("What happens to my sick leave at the end of the year?", "leave_policy", "lapses on 31 December"),
    ("How many paid learning days do employees get each year?", "benefits_policy", "2 paid learning days"),
    ("Can I take back a resignation I already submitted?", "notice_period_policy", "withdrawn within 7 days"),
    ("What is the mileage rate if I use my own car for work?", "expense_policy", "Rs. 12 per km for cars"),
    ("How much is the referral bonus?", "benefits_policy", "Rs. 25,000 for each successful hire"),
    ("Who handles sexual harassment complaints?", "code_of_conduct", "Internal Committee (IC)"),
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
    ("What is the notice period for Grade L2?", "It is 60 days.",
     "What about during probation?", "probation"),
    ("What is the hotel limit per night?", "It depends on the city tier.",
     "And in Kochi?", "kochi"),
    ("How many sick leaves do I get?", "You get 8 sick leaves per year.",
     "Do they carry forward?", "sick"),
]


def raw_scores(q):
    res = db.similarity_search_with_score(q, k=TOP_K)
    # unit vectors: squared L2 distance d -> cosine = 1 - d/2
    return [(os.path.basename(d.metadata.get("source", "")), 1 - dist / 2, d.page_content)
            for d, dist in res]


inq = [(q, exp, must, raw_scores(q)) for q, exp, must in IN_SCOPE]
oos = [(q, raw_scores(q)) for q in OUT_OF_SCOPE]

header = f"hit@{TOP_K}"
print(f"{'thr':>5} {header:>7} {'false_refusal':>14} {'oos_declined':>13}")
for t in [0.0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    hit = sum(any(exp in f and must in text and s >= t for f, s, text in sc)
              for _, exp, must, sc in inq)
    refused = sum(not any(s >= t for _, s, _ in sc) for _, _, _, sc in inq)
    declined = sum(not any(s >= t for _, s, _ in sc) for _, sc in oos)
    print(f"{t:>5} {hit/len(inq):>7.0%} {refused/len(inq):>14.0%} {declined/len(oos):>13.0%}")

print("\nChunk-level misses at 0.38:")
for q, exp, must, sc in inq:
    if not any(exp in f and must in text and s >= 0.38 for f, s, text in sc):
        print("  MISS:", q)

print("\nEnd-to-end (calls the LLM, uses MIN_COSINE from rag.py):")
declined = sum(not ask(q, [])["grounded"] for q, _ in oos)
print(f"Out-of-scope declined: {declined}/{len(oos)}")
for q1, a1, q2, kw in FOLLOWUPS:
    rewritten = rewrite(q2, [("User", q1), ("Assistant", a1)])
    print(f"{'OK  ' if kw in rewritten.lower() else 'FAIL'} {q2!r} -> {rewritten!r}")