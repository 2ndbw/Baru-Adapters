"""
BARU DEMO — Accounting Period Close
====================================

Scenario: A small company has five types of journal entries it makes
each month. The accounting period is "perfect" when the books balance
to zero — every dollar that came in is accounted for.

At the end of Q3, the accountant pulls the entry log. The books are
off by $300. Which entry was wrong? What's the fix?

Baru answers that in one call.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from baru import Baru

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Define the domain (the adapter — four functions)
# ─────────────────────────────────────────────────────────────────────────────

# These are the only entry types in this company's chart of accounts.
# Each maps to a dollar delta on the running balance.
ENTRY = {
    'sale':     +500,   # customer payment received
    'refund':   -500,   # customer refund issued
    'expense':  -200,   # operating expense paid
    'deposit':  +200,   # cash deposited to account
    'fee':      -100,   # bank or processing fee
    'credit':   +100,   # small credit received
}

def accounting_inverse(balance):
    """
    Given a non-zero balance, return the fewest entries that bring it to zero.
    Uses greedy fill: largest absolute entry first.
    """
    need   = -balance
    result = []
    pool   = sorted(ENTRY.items(), key=lambda x: abs(x[1]), reverse=True)
    for _ in range(abs(need) + 10):
        if need == 0:
            break
        for name, delta in pool:
            if (need > 0 and delta > 0 and delta <= need) or \
               (need < 0 and delta < 0 and delta >= need):
                result.append(name)
                need -= delta
                break
        else:
            # No single entry fits cleanly — use smallest available
            name, delta = pool[-1]
            result.append(name)
            need -= delta
    return result

ledger = Baru(
    segments = list(ENTRY.keys()),
    compose  = lambda balance, entry: balance + ENTRY[entry],
    perfect  = lambda balance: balance == 0,
    inverse  = accounting_inverse,
    start    = 0,
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: A clean month — books close perfectly
# ─────────────────────────────────────────────────────────────────────────────

clean_month = ['sale', 'sale', 'expense', 'expense', 'expense', 'expense', 'fee', 'fee']
# 500+500-200-200-200-200-100-100 = 0
result = ledger.run(clean_month)

print("─" * 60)
print("CLEAN MONTH")
print("─" * 60)
print(f"  Entries:  {clean_month}")
print(f"  Balance:  ${result.state}")
print(f"  Closed:   {result.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Q3 — books are off
# ─────────────────────────────────────────────────────────────────────────────

# What the accountant entered:
q3_entries = ['sale', 'sale', 'sale', 'expense', 'expense', 'fee', 'deposit']

result = ledger.run(q3_entries)

print()
print("─" * 60)
print("Q3 — BOOKS ARE OFF")
print("─" * 60)
print(f"  Entries:  {q3_entries}")
print(f"  Balance:  ${result.state}  ← should be $0")
print(f"  Closed:   {result.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Baru finds the correction
# ─────────────────────────────────────────────────────────────────────────────

fix = ledger.correct(q3_entries)

print()
print("─" * 60)
print("BARU CORRECTION")
print("─" * 60)
print(f"  Before:   {fix.before}")
print(f"  After:    {fix.after}")
print(f"  Swaps:    {fix.swaps}")
print(f"  Method:   {fix.method}")

# Show exactly what changed
changes = [(i, fix.before[i], fix.after[i])
           for i in range(min(len(fix.before), len(fix.after)))
           if fix.before[i] != fix.after[i]]
if len(fix.after) > len(fix.before):
    for i in range(len(fix.before), len(fix.after)):
        changes.append((i, '—', fix.after[i]))

print()
for pos, old, new in changes:
    old_val = ENTRY.get(old, '—')
    new_val = ENTRY.get(new, '—')
    print(f"  Position {pos}:  '{old}' (${old_val})  →  '{new}' (${new_val})")

# Verify
verify = ledger.run(fix.after)
print()
print(f"  Final balance: ${verify.state}")
print(f"  Books closed:  {verify.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Generate a perfect month from scratch
# ─────────────────────────────────────────────────────────────────────────────

generated = ledger.generate(8)
print()
print("─" * 60)
print("GENERATED PERFECT MONTH (Baru builds it)")
print("─" * 60)
print(f"  Entries:  {generated.segments}")
total = sum(ENTRY[e] for e in generated.segments)
print(f"  Balance:  ${total}")
print(f"  Closed:   {generated.perfect}")
print("─" * 60)
