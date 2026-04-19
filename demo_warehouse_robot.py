"""
BARU DEMO — Warehouse Robot Path
===================================

Scenario: A warehouse robot navigates a grid to pick items and must
return to its charging dock (0, 0) at the end of every run.

The robot has 8 named moves. Each run is a sequence of moves.
The run is "perfect" when the robot ends up exactly back at the dock.

A new operator programs a pick route. The robot ends up 3 meters east,
2 meters north — not at the dock. The robot can't recharge. Which move
was wrong? What's the minimal fix?

Baru answers it.
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(__file__))
from baru import Baru

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Define the domain
# ─────────────────────────────────────────────────────────────────────────────

# State: (x, y) grid position. Start and home = (0, 0).
# Moves: named cardinal steps, each 1 meter.

MOVE = {
    'N':  ( 0, +1),
    'S':  ( 0, -1),
    'E':  (+1,  0),
    'W':  (-1,  0),
    'NE': (+1, +1),
    'NW': (-1, +1),
    'SE': (+1, -1),
    'SW': (-1, -1),
}

def robot_inverse(pos):
    """Return the shortest move sequence to get from pos back to (0, 0)."""
    x, y = pos
    result = []
    # Diagonal moves first to handle both axes simultaneously
    while x != 0 and y != 0:
        dx = -1 if x > 0 else +1
        dy = -1 if y > 0 else +1
        # find the diagonal name
        for name, (mx, my) in MOVE.items():
            if mx == dx and my == dy:
                result.append(name)
                x += mx
                y += my
                break
    # Handle remaining single-axis movement
    while x > 0: result.append('W'); x -= 1
    while x < 0: result.append('E'); x += 1
    while y > 0: result.append('S'); y -= 1
    while y < 0: result.append('N'); y += 1
    return result

robot = Baru(
    segments = list(MOVE.keys()),
    compose  = lambda pos, move: (pos[0] + MOVE[move][0], pos[1] + MOVE[move][1]),
    perfect  = lambda pos: pos == (0, 0),
    inverse  = robot_inverse,
    start    = (0, 0),
)

def fmt_pos(pos):
    x, y = pos
    xdir = f"{abs(x)}m {'E' if x >= 0 else 'W'}"
    ydir = f"{abs(y)}m {'N' if y >= 0 else 'S'}"
    if pos == (0, 0): return "dock (0, 0)"
    return f"({x}, {y}) — {xdir}, {ydir}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: A clean run — robot returns to dock
# ─────────────────────────────────────────────────────────────────────────────

clean_run = ['E', 'E', 'E', 'N', 'N', 'N', 'W', 'W', 'W', 'S', 'S', 'S']
# 3E+3N+3W+3S = (0,0) ✓
result = robot.run(clean_run)

print("─" * 60)
print("CLEAN RUN")
print("─" * 60)
print(f"  Route:    {clean_run}")
print(f"  End pos:  {fmt_pos(result.state)}")
print(f"  At dock:  {result.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: New operator's route — robot doesn't make it back
# ─────────────────────────────────────────────────────────────────────────────

operator_route = ['NE', 'NE', 'E', 'N', 'SW', 'SW', 'NE', 'SW', 'NE', 'E']

result = robot.run(operator_route)

print()
print("─" * 60)
print("OPERATOR ROUTE — ROBOT STRANDED")
print("─" * 60)
print(f"  Route:    {operator_route}")
print(f"  End pos:  {fmt_pos(result.state)}  ← not at dock")
dist = math.hypot(result.state[0], result.state[1])
print(f"  Distance from dock: {dist:.2f} meters")
print(f"  At dock:  {result.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Baru finds the fix
# ─────────────────────────────────────────────────────────────────────────────

fix = robot.correct(operator_route)

print()
print("─" * 60)
print("BARU CORRECTION")
print("─" * 60)
print(f"  Before:  {fix.before}")
print(f"  After:   {fix.after}")
print(f"  Swaps:   {fix.swaps}")
print(f"  Method:  {fix.method}")

if len(fix.after) > len(fix.before):
    appended = fix.after[len(fix.before):]
    print(f"\n  Appended moves: {appended}")
    print(f"  Net of appended: ({sum(MOVE[m][0] for m in appended)}, {sum(MOVE[m][1] for m in appended)})")
else:
    changes = [(i, fix.before[i], fix.after[i])
               for i in range(len(fix.before))
               if fix.before[i] != fix.after[i]]
    print()
    for pos, old, new in changes:
        print(f"  Step {pos+1}:  '{old}' {MOVE[old]}  →  '{new}' {MOVE[new]}")

verify = robot.run(fix.after)
print()
print(f"  Final position: {fmt_pos(verify.state)}")
print(f"  At dock: {verify.perfect}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Generate a perfect pick route
# ─────────────────────────────────────────────────────────────────────────────

route = robot.generate(14)
print()
print("─" * 60)
print("GENERATED PERFECT ROUTE (Baru builds it)")
print("─" * 60)
print(f"  Route:  {route.segments}")
print(f"  Steps:  {len(route.segments)}")
final = robot.run(route.segments)
print(f"  End:    {fmt_pos(final.state)}")
print(f"  At dock: {route.perfect}")
print("─" * 60)
