"""
BaruAdapter.py

The easy way to build a Baru adapter.
Answer four questions. Baru does the rest.

─────────────────────────────────────────────────────────────
QUICK START

    from BaruAdapter import BaruAdapter

    a = BaruAdapter("My Domain")
    a.segments = [+1, -1, +2]          # the valid moves
    a.start    = 0                      # state before anything happens
    a.compose  = lambda state, move: state + move   # how a move changes state
    a.perfect  = lambda state: state == 0           # what "closed" looks like
    a.inverse  = lambda state: [-1]*state if state > 0 else [+1]*(-state)

    a.check()                           # verify it works before you use it
    baru = a.build()                    # get a Baru instance

─────────────────────────────────────────────────────────────
THE FOUR QUESTIONS

    segments   What moves can be made?
               Example: ['left', 'right', 'straight']
               Example: [+1, -1, +3]
               Example: ['T', 'E', 'D']

    compose    How does one move change the state?
               compose(current_state, move) → new_state
               Example: lambda state, move: state + move

    perfect    What does a closed loop look like?
               perfect(state) → True or False
               Example: lambda state: state == 0
               Example: lambda state: state % 12 == 0

    inverse    Given an open state, what moves close it?
               inverse(state) → list of moves
               Example: lambda state: [-1] * state  (if state is positive drift)
               This is the hard one. If you can't define it,
               you don't yet understand your domain.

─────────────────────────────────────────────────────────────
"""

import random
from baru import Baru


class BaruAdapter:
    """
    Build a Baru adapter step by step.

    Set the five properties, then call .check() to verify
    and .build() to get a working Baru instance.
    """

    def __init__(self, name: str = "My Adapter"):
        self.name     = name      # just a label — used in describe() and check()
        self.segments = None      # list: the valid moves in your domain
        self.start    = 0         # the state before any moves are made
        self.compose  = None      # function(state, move) → new_state
        self.perfect  = None      # function(state) → True / False
        self.inverse  = None      # function(state) → list of moves that close it

    # ── build ──────────────────────────────────────────────────────────

    def build(self) -> Baru:
        """
        Validate and return a working Baru instance.
        Raises a clear error if anything is missing or broken.
        """
        self._validate()
        return Baru(
            segments = self.segments,
            start    = self.start,
            compose  = self.compose,
            perfect  = self.perfect,
            inverse  = self.inverse,
        )

    # ── check ──────────────────────────────────────────────────────────

    def check(self) -> bool:
        """
        Run a quick sanity check on your adapter.
        Prints a pass/fail report and returns True if everything looks good.

        Tests:
          1. All four required properties are defined
          2. compose() doesn't crash on every segment
          3. perfect() works on the start state
          4. inverse() closes a loop from each segment's resulting state
          5. correct() repairs 50 random broken loops
        """
        print(f'\n  Checking adapter: {self.name}')
        print(f'  {"─"*44}')
        passed = True

        # ── 1. completeness ──────────────────────────────
        missing = [k for k in ('segments', 'compose', 'perfect', 'inverse')
                   if getattr(self, k) is None]
        if missing:
            for k in missing:
                print(f'  ✗  {k} is not set')
                print(f'     → set adapter.{k} = ...')
            return False
        print(f'  ✓  all four properties are defined')

        try:
            baru = self.build()
        except Exception as e:
            print(f'  ✗  build() failed: {e}')
            return False

        # ── 2. compose: run every segment from start ──────
        compose_ok = True
        for s in self.segments:
            try:
                self.compose(self.start, s)
            except Exception as e:
                print(f'  ✗  compose(start, {s!r}) crashed: {e}')
                compose_ok = False
                passed = False
        if compose_ok:
            print(f'  ✓  compose() works on all {len(self.segments)} segments')

        # ── 3. perfect: start state ───────────────────────
        try:
            result = self.perfect(self.start)
            start_label = '(already perfect)' if result else '(open — needs moves to close)'
            print(f'  ✓  perfect(start) = {result}  {start_label}')
        except Exception as e:
            print(f'  ✗  perfect(start) crashed: {e}')
            passed = False

        # ── 4. inverse: can it close any reachable state? ─
        inv_fails = 0
        inv_tested = 0
        for _ in range(200):
            segs  = [random.choice(self.segments) for _ in range(random.randint(1, 15))]
            loop  = baru.run(segs)
            if loop.perfect:
                continue
            inv_tested += 1
            closure = self.inverse(loop.state)
            check   = baru.run(segs + list(closure))
            if not check.perfect:
                inv_fails += 1
                if inv_fails <= 2:
                    print(f'  ✗  inverse({loop.state!r}) returned {closure}')
                    print(f'     but the loop still isn\'t closed after appending it')

        if inv_fails == 0:
            print(f'  ✓  inverse() correctly closes all {inv_tested} tested states')
        else:
            print(f'  ✗  inverse() failed on {inv_fails} of {inv_tested} states')
            passed = False

        # ── 5. correct: 50 random loops ───────────────────
        corr_fails = 0
        for _ in range(50):
            segs = [random.choice(self.segments) for _ in range(random.randint(3, 20))]
            try:
                ram   = baru.correct(segs)
                check = baru.run(ram.after)
                if not check.perfect:
                    corr_fails += 1
            except Exception as e:
                corr_fails += 1

        if corr_fails == 0:
            print(f'  ✓  correct() repaired 50 random broken loops without errors')
        else:
            print(f'  ✗  correct() failed on {corr_fails} of 50 loops')
            passed = False

        # ── summary ───────────────────────────────────────
        print(f'  {"─"*44}')
        if passed:
            print(f'  ✓  {self.name} is ready  →  call .build()\n')
        else:
            print(f'  ✗  fix the errors above before calling .build()\n')

        return passed

    # ── describe ───────────────────────────────────────────────────────

    def describe(self):
        """Print a summary of this adapter."""
        baru = self.build() if self._is_complete() else None
        print(f'\n  Adapter: {self.name}')
        print(f'  {"─"*44}')
        print(f'  segments  {self.segments}')
        print(f'  start     {self.start!r}')
        print(f'  compose   {"set" if self.compose else "NOT SET"}')
        print(f'  perfect   {"set" if self.perfect else "NOT SET"}')
        print(f'  inverse   {"set" if self.inverse else "NOT SET"}')
        if baru:
            loop = baru.run([])
            print(f'  start is perfect: {loop.perfect}')
            print()
            print('  Example loops:')
            for _ in range(3):
                segs = [random.choice(self.segments) for _ in range(8)]
                loop = baru.run(segs)
                print(f'    {segs}  →  {loop.state!r}  perfect={loop.perfect}')
        print()

    # ── private ────────────────────────────────────────────────────────

    def _is_complete(self) -> bool:
        return all(getattr(self, k) is not None
                   for k in ('segments', 'compose', 'perfect', 'inverse'))

    def _validate(self):
        missing = [k for k in ('segments', 'compose', 'perfect', 'inverse')
                   if getattr(self, k) is None]
        if missing:
            lines = '\n'.join(f'  adapter.{k} = ...' for k in missing)
            raise ValueError(
                f'\n\n  Adapter "{self.name}" is missing: {", ".join(missing)}\n\n'
                f'  Set the missing properties before calling build():\n{lines}\n'
            )
        if not isinstance(self.segments, list) or len(self.segments) == 0:
            raise ValueError(
                f'  adapter.segments must be a non-empty list.\n'
                f'  Got: {self.segments!r}'
            )


# ── BUILT-IN EXAMPLES ─────────────────────────────────────────────────
#
#  Copy any of these as a starting point for your own adapter.
#  Each one answers the four questions for a different domain.
#
# ──────────────────────────────────────────────────────────────────────


def counter_adapter() -> BaruAdapter:
    """
    COUNTER — the simplest possible adapter.

    Moves add or subtract from a running total.
    A loop is closed when the total returns to zero.
    """
    a = BaruAdapter("Counter")
    a.segments = [+1, -1, +2, -2, +3, -3]
    a.start    = 0
    a.compose  = lambda state, move: state + move
    a.perfect  = lambda state: state == 0
    a.inverse  = lambda state: [-1] * state if state > 0 else [+1] * (-state)
    return a


def clock_adapter(hours: int = 12) -> BaruAdapter:
    """
    CLOCK mod N — moves advance a clock hand.

    A loop is closed when the hand returns to 12 (position 0).
    Works for any modulus (default: 12-hour clock).
    """
    a = BaruAdapter(f"Clock mod {hours}")
    a.segments = list(range(1, hours))        # +1 through +(hours-1)
    a.start    = 0
    a.compose  = lambda state, move: (state + move) % hours
    a.perfect  = lambda state: state == 0
    a.inverse  = lambda state: [] if state == 0 else [(hours - state) % hours]
    return a


def toggle_adapter() -> BaruAdapter:
    """
    TOGGLE — two moves that cancel each other.

    A and B each flip a switch. A closed loop has equal A's and B's.
    """
    a = BaruAdapter("Toggle")
    a.segments = ['A', 'B']
    a.start    = 0
    a.compose  = lambda state, move: state + (1 if move == 'A' else -1)
    a.perfect  = lambda state: state == 0
    a.inverse  = lambda state: ['B'] * state if state > 0 else ['A'] * (-state)
    return a


def steps_adapter() -> BaruAdapter:
    """
    STEPS — forward and backward steps on a number line.

    A closed loop returns to the start position.
    """
    a = BaruAdapter("Steps")
    a.segments = ['F', 'B', 'FF', 'BB']   # Forward, Backward, double step
    STEP = {'F': 1, 'B': -1, 'FF': 2, 'BB': -2}
    a.start   = 0
    a.compose = lambda state, move: state + STEP[move]
    a.perfect = lambda state: state == 0
    a.inverse = lambda state: (['B'] * state if state > 0
                               else ['F'] * (-state))
    return a


# ── DEMO ──────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print('\n' + '─'*50)
    print('  BaruAdapter — built-in examples')
    print('─'*50)

    for factory in [counter_adapter, toggle_adapter, steps_adapter, clock_adapter]:
        adapter = factory()
        adapter.check()

    # ── build one from scratch, step by step ──────────────────────────
    print('─'*50)
    print('  Custom adapter built from scratch')
    print('─'*50)

    custom = BaruAdapter("Temperature")
    custom.segments = ['hot', 'cold', 'warm']
    custom.start    = 0
    # hot = +2, cold = -2, warm = +1
    HEAT = {'hot': +2, 'cold': -2, 'warm': +1}
    custom.compose  = lambda state, move: state + HEAT[move]
    custom.perfect  = lambda state: state == 0
    # hot=+2, cold=-2, warm=+1
    # to close state > 0: use colds (−2 each), add a warm (+1) if odd
    # to close state < 0: use hots (+2 each), add a warm (+1) if odd
    def temp_inverse(state):
        if state == 0: return []
        if state > 0:
            n_cold = (state + 1) // 2          # enough colds to overshoot by at most 1
            n_warm = n_cold * 2 - state        # warms cancel the overshoot
            return ['cold'] * n_cold + ['warm'] * n_warm
        else:
            s = -state
            return ['hot'] * (s // 2) + ['warm'] * (s % 2)
    custom.inverse = temp_inverse

    if custom.check():
        baru   = custom.build()
        broken = ['hot', 'hot', 'warm', 'cold']
        ram    = baru.correct(broken)
        print(f'  broken:    {broken}')
        print(f'  corrected: {ram.after}   ({ram.swaps} added, {ram.method})')
        print(f'  perfect:   {baru.run(ram.after).perfect}\n')
