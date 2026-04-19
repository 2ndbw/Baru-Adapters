"""
BARU DEMO — Staff Rotation Cycle
==================================

Scenario: A hospital has 12 nurses. They rotate through 12 weekly
shift slots on a fixed cycle. The cycle is "perfect" when every nurse
is back to their home slot after a full rotation.

The scheduling coordinator makes a series of swap adjustments
throughout the quarter. After 9 adjustments the rotation is off —
nobody is where they should be.

Baru finds the minimum swap sequence to restore the cycle.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from baru import Baru

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Define the domain
# ─────────────────────────────────────────────────────────────────────────────

# 12-position rotation ring (mod 12).
# State = how many positions the rotation is currently offset from home.
# Perfect = everyone is back in their home slot (offset == 0).

SHIFT_MOVE = {
    'hold':   0,    # no change — keep current rotation
    'next':  +1,    # advance everyone one slot forward
    'prev':  -1,    # move everyone one slot back
    'skip':  +2,    # skip two slots forward (cover for absence)
    'back':  -2,    # pull two slots back
    'jump':  +3,    # emergency 3-slot jump
    'pull':  -3,    # emergency 3-slot pull
    'half':  +6,    # flip to opposite shift group
}

def rotation_inverse(offset):
    """Return the fewest moves to bring the rotation offset back to 0."""
    need   = (-offset) % 12
    result = []
    moves  = sorted(SHIFT_MOVE.items(), key=lambda x: abs(x[1]), reverse=True)
    while need != 0:
        for name, delta in moves:
            d = delta % 12
            if d == 0:
                continue
            if d <= need:
                result.append(name)
                need = (need - d) % 12
                break
        else:
            break
    return result

roster = Baru(
    segments = list(SHIFT_MOVE.keys()),
    compose  = lambda offset, move: (offset + SHIFT_MOVE[move]) % 12,
    perfect  = lambda offset: offset == 0,
    inverse  = rotation_inverse,
    start    = 0,
)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: A well-managed quarter — rotation closes cleanly
# ─────────────────────────────────────────────────────────────────────────────

good_quarter = ['next', 'skip', 'jump', 'half', 'half', 'pull', 'back', 'prev']
# 1+2+3+6+6-3-2-1 = 12 mod 12 = 0
result = roster.run(good_quarter)

print("─" * 60)
print("GOOD QUARTER")
print("─" * 60)
print(f"  Moves:   {good_quarter}")
print(f"  Offset:  {result.state} slots")
print(f"  Closed:  {result.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: A rough quarter — ad-hoc swaps left the rotation broken
# ─────────────────────────────────────────────────────────────────────────────

bad_quarter = ['next', 'skip', 'skip', 'jump', 'prev', 'pull', 'half',
               'jump', 'back', 'next', 'skip']

result = roster.run(bad_quarter)

print()
print("─" * 60)
print("ROUGH QUARTER — ROTATION IS OFF")
print("─" * 60)
print(f"  Moves:   {bad_quarter}")
print(f"  Offset:  {result.state} slots  ← everyone is {result.state} slots from home")
print(f"  Closed:  {result.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Baru finds the fix
# ─────────────────────────────────────────────────────────────────────────────

fix = roster.correct(bad_quarter)

print()
print("─" * 60)
print("BARU CORRECTION")
print("─" * 60)
print(f"  Before:  {fix.before}")
print(f"  After:   {fix.after}")
print(f"  Swaps:   {fix.swaps}")
print(f"  Method:  {fix.method}")

# Show what changed
if len(fix.after) > len(fix.before):
    appended = fix.after[len(fix.before):]
    print(f"\n  Appended moves to close: {appended}")
    total_delta = sum(SHIFT_MOVE[m] for m in appended) % 12
    print(f"  Those moves shift by: {total_delta} slots total → closes offset")
else:
    changes = [(i, fix.before[i], fix.after[i])
               for i in range(len(fix.before))
               if fix.before[i] != fix.after[i]]
    print()
    for pos, old, new in changes:
        print(f"  Week {pos+1}:  '{old}' (+{SHIFT_MOVE[old]})  →  '{new}' (+{SHIFT_MOVE[new]})")

verify = roster.run(fix.after)
print()
print(f"  Final offset: {verify.state} slots")
print(f"  Rotation closed: {verify.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Generate a perfectly balanced rotation schedule
# ─────────────────────────────────────────────────────────────────────────────

schedule = roster.generate(12)
print()
print("─" * 60)
print("GENERATED PERFECT SCHEDULE (Baru builds it)")
print("─" * 60)
print(f"  Moves:  {schedule.segments}")
net = sum(SHIFT_MOVE[m] for m in schedule.segments) % 12
print(f"  Net offset: {net} (mod 12)")
print(f"  Closed: {schedule.perfect}")
print("─" * 60)
