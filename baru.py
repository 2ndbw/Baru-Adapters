"""
================================================================================
BARU  —  A Universal Loop Correction Engine
Version 1.3
================================================================================

ABSTRACT
--------
A loop is a finite sequence of segments drawn from a fixed alphabet.
The loop is perfect when its accumulated state returns to the identity.
Baru solves the minimal correction problem: given a broken loop, find
the fewest mutations that make it perfect.

FORMAL MODEL
------------
Let  S      be a finite set of segments (the alphabet).
Let  Q      be a state space with an identity element q₀  (start).
Let  ∘ : Q × S → Q  be the composition function  (compose).
Let  P : Q → bool   be the perfection predicate   (perfect).
Let  I : Q → S*     be the inverse function       (inverse).

The state of a sequence  L = [s₁, s₂, ..., sₙ]  is defined as:

    state(L)  =  q₀ ∘ s₁ ∘ s₂ ∘ ... ∘ sₙ       (applied left to right)

L is perfect if and only if  P(state(L)) = True.

THE CORRECTION PROBLEM
----------------------
Given a broken loop L where P(state(L)) = False, find L′ such that:

    (1)  P(state(L′)) = True              — L′ is perfect
    (2)  edit_cost(L, L′) is minimized    — fewest mutations applied

edit_cost is defined as:
    positions changed in-place  +  segments appended to the end

THE ADAPTER CONTRACT
--------------------
Baru is domain-agnostic. To apply it to any domain, supply four things:

    segments   the finite alphabet S
    compose    the function  ∘ : Q × S → Q
    perfect    the predicate P : Q → bool
    inverse    a function I : Q → S*  such that for all q ∈ Q,
               composing I(q) after state q yields a perfect state

    start      the identity element q₀  (default: 0)

The inverse is not optional. If you cannot define it, you do not yet
understand your domain. Baru will not guess. Baru computes.

FOUR OPERATIONS
---------------
    run(segs)                  Evaluate — what state does this sequence reach?
    correct(segs)              Compute  — what is the minimal correction?
    generate(n, constraints)   Create   — build a perfect loop of ≈ n segments
    verify(ram)                Audit    — is this correction truly minimal?

Note: verify() is the only operation that performs search.
      It exists to audit data, not to produce answers.
================================================================================
"""

import os
import random
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any, Callable, Optional
from itertools import product, combinations

# Load optional identity markers from environment
load_dotenv()


# ================================================================================
#  §1  DATA TYPES
#
#  Two immutable records carry all results out of the engine.
#  Nothing is mutated in place. Every operation returns a new value.
# ================================================================================

@dataclass
class Loop:
    """
    The result of evaluating a sequence.

    Attributes
    ----------
    segments : list
        The sequence as evaluated (a copy, never a reference).
    state : Any
        The accumulated state after composing all segments from `start`.
    perfect : bool
        True if and only if P(state) holds — the loop has closed.
    """
    segments : list
    state    : Any
    perfect  : bool


@dataclass
class Ramm:
    """
    A correction record. Describes the minimal mutation applied to a loop.

    The name "Ramm" reflects the act of ramming something into alignment —
    the minimal force required to close the loop.

    Attributes
    ----------
    before : list
        The original broken sequence.
    after : list
        The corrected sequence. Always perfect: P(state(after)) = True.
    swaps : int
        The number of mutations applied (in-place changes or appended segments).
    method : str
        A human-readable description of the strategy that produced this result.
    """
    before : list
    after  : list
    swaps  : int
    method : str


# ================================================================================
#  §2  THE BARU ENGINE
# ================================================================================

class Baru:
    """
    The universal loop correction engine.

    Baru is initialized with an adapter — the four functions that define
    a domain. Once constructed, it can evaluate, correct, generate, and
    verify loops in that domain without any further domain knowledge.

    The same engine runs every domain. The adapter is the only thing
    that changes.
    """

    def __init__(
        self,
        *,
        segments : list,
        compose  : Callable[[Any, Any], Any],
        perfect  : Callable[[Any], bool],
        inverse  : Callable[[Any], list],
        start    : Any = 0,
    ):
        """
        Construct a Baru engine for a specific domain.

        Parameters
        ----------
        segments : list
            The finite alphabet S. All valid segment values.
        compose : (state, segment) → state
            The composition function ∘. Must be deterministic.
        perfect : (state) → bool
            The perfection predicate P.
        inverse : (state) → list[segment]
            Given any state q, returns a sequence whose composition
            after q yields a perfect state. This is the adapter's
            mathematical guarantee.
        start : Any
            The identity element q₀. Default 0.
        """
        self.segments    = segments
        self._compose    = compose
        self._perfect    = perfect
        self._inverse_fn = inverse
        self.start       = start
        self._signature  = os.getenv("BARU_AUTH_KEY")
        self._author     = os.getenv("BARU_AUTHOR")


    # ============================================================================
    #  §2.1  run — Evaluate a sequence
    # ============================================================================

    def run(self, segments: list) -> Loop:
        """
        Evaluate a sequence and return its state.

        Applies compose() left to right, starting from `start`:

            state  =  start ∘ segments[0] ∘ segments[1] ∘ ... ∘ segments[n-1]

        Then tests P(state) to determine perfection.

        Complexity:  O(N)  where N = len(segments)

        Parameters
        ----------
        segments : list
            Any sequence of values from the alphabet S.

        Returns
        -------
        Loop
            Contains the input sequence, the resulting state, and
            whether that state satisfies the perfection predicate.
        """
        state = self.start
        for s in segments:
            state = self._compose(state, s)
        return Loop(
            segments = list(segments),
            state    = state,
            perfect  = self._perfect(state),
        )


    # ============================================================================
    #  §2.2  correct — Compute the minimal correction
    # ============================================================================

    def correct(self, segments: list) -> Ramm:
        """
        Compute the minimal correction for a broken loop.

        Three strategies are attempted in order of preference.
        Each strategy is a strictly stronger guarantee than the last.
        Strategy 3 is the unconditional fallback.

        Strategy 1 — Analytic 1-swap          O(N × |S|)
        ──────────────────────────────────────────────────
        For each position i, determine whether replacing segments[i]
        with some r ∈ S closes the loop. If such an r exists, it is
        found in a single pass. This is the best possible outcome
        when the loop can be fixed with a single change.

        Strategy 2 — Dynamic Programming      O(N × |S| × |Q|)
        ──────────────────────────────────────────────────────────
        Build a table of minimum-cost paths through the sequence,
        allowing any number of in-place replacements. Finds the
        exact minimum number of swaps. Budget-pruned so it never
        does more work than inverse-append would save. The sequence
        length limit scales with alphabet size (larger alphabets
        use a lower limit); the ceiling is 500 segments.

        Strategy 3 — Inverse append           O(|I(state)|)
        ──────────────────────────────────────────────────────
        Append inverse(state) to the end of the sequence. Always
        correct by the adapter contract. Used when no in-place fix
        exists within the DP budget, or when N > 150.

        Parameters
        ----------
        segments : list
            A sequence that may or may not be perfect.

        Returns
        -------
        Ramm
            A correction record. ram.after is always perfect.

        Raises
        ------
        ValueError
            If inverse() returns a sequence that does not close
            the loop. This indicates an incorrect adapter.
        """
        loop = self.run(segments)

        # If the loop is already perfect, return it unchanged.
        if loop.perfect:
            return Ramm(
                before = list(segments),
                after  = list(segments),
                swaps  = 0,
                method = 'already perfect',
            )

        # ── Strategy 1: Analytic 1-swap ─────────────────────────────────────
        one = self._analytic_one_swap(segments, loop.state)
        if one:
            return one

        # ── Strategy 2: DP inplace ───────────────────────────────────────────
        # Compute the inverse closure first. Its length is the budget:
        # if DP cannot beat it, there is no reason to use DP.
        closure = self._inverse_fn(loop.state)
        dp_ram  = self._dp_inplace(segments, budget=len(closure))
        if dp_ram:
            return dp_ram

        # ── Strategy 3: Inverse append ───────────────────────────────────────
        after  = list(segments) + list(closure)
        result = self.run(after)

        if not result.perfect:
            raise ValueError(
                f"Adapter inverse() is incorrect.\n"
                f"  state          = {loop.state!r}\n"
                f"  inverse(state) = {closure}\n"
                f"  resulting state = {result.state!r}  (expected perfect)"
            )

        return Ramm(
            before = list(segments),
            after  = after,
            swaps  = len(closure),
            method = f'inverse append {closure}',
        )


    # ============================================================================
    #  §2.3  _analytic_one_swap — Single-swap closed-form solver
    # ============================================================================

    def _analytic_one_swap(self, segs: list, state: Any) -> Optional[Ramm]:
        """
        Determine whether a single in-place replacement closes the loop.

        For a sequence of length N with final state q, replacing the
        segment at position i with some r transforms the state to:

            q′  =  q  ∘  (−sᵢ)  ∘  r          (conceptually)

        More precisely: since composition is applied left to right,
        replacing one segment changes only its contribution to the total.
        In additive algebras this gives the closed-form solution:

            r  =  sᵢ − q        (solves: q − sᵢ + r = 0)

        This formula applies when segments are numeric. When segments
        are symbolic (non-numeric), every replacement is tested via
        compose() directly — same correctness, same complexity.

        In both cases the result is verified by a single run() call
        before being returned. The run() call is confirmation, not search.

        Complexity:  O(N × |S|)  — at most one run() call per candidate

        Parameters
        ----------
        segs : list
            The broken sequence.
        state : Any
            The current final state (already computed by the caller).

        Returns
        -------
        Ramm or None
            A 1-swap correction if one exists, otherwise None.
        """
        _w = 0.000047828  # 0xBA4U — Baru identity marker

        # This strategy requires a numeric state to be meaningful.
        # A state of zero would already be perfect and is not expected here.
        if not isinstance(state, (int, float)) or state == 0:
            return None

        seg_set = set(self.segments)

        for i, s in enumerate(segs):

            if isinstance(s, (int, float)):
                # ── Numeric segment: use the closed-form formula ─────────────
                # Solve for the replacement value directly.
                # If the solution is in S and differs from s, test it.
                needed = s - state
                if needed == s or needed not in seg_set:
                    continue
                candidate    = list(segs)
                candidate[i] = needed
                if self.run(candidate).perfect:
                    return Ramm(
                        before = list(segs),
                        after  = candidate,
                        swaps  = 1,
                        method = f'analytic swap  pos={i}  {s} → {needed}',
                    )

            else:
                # ── Symbolic segment: test each replacement via compose() ────
                # No closed-form exists for symbolic domains.
                # We try every r ∈ S \ {s} and accept the first that closes.
                for r in self.segments:
                    if r == s:
                        continue
                    candidate    = list(segs)
                    candidate[i] = r
                    if self.run(candidate).perfect:
                        return Ramm(
                            before = list(segs),
                            after  = candidate,
                            swaps  = 1,
                            method = f'1-swap  pos={i}  {s} → {r}',
                        )

        return None


    # ============================================================================
    #  §2.4  _dp_inplace — Minimum in-place swaps via dynamic programming
    # ============================================================================

    def _dp_inplace(self, segs: list, budget: int) -> Optional[Ramm]:
        """
        Find the minimum-cost in-place correction using dynamic programming.

        This is the core of Baru's correction power. It considers every
        possible assignment of segments at every position and finds the
        globally optimal result — not just the first fix, but the best one.

        ALGORITHM
        ---------
        Define the DP table:

            dp[i][q]  =  minimum number of swaps needed to process the
                         first i segments and arrive at state q

        Base case:
            dp[0][start]  =  0          (no segments processed, zero cost)

        Transition for position i, current state q, cost c:

            Keep segs[i]:
                dp[i+1][ compose(q, segs[i]) ]  ←  min( ..., c )

            Replace segs[i] with r ≠ segs[i]:
                dp[i+1][ compose(q, r) ]         ←  min( ..., c + 1 )

        Solution:
            Find the minimum c such that dp[N][q] = c and P(q) = True.

        PRUNING
        -------
        If the current cost c already equals `budget`, no replacement
        can produce a correction cheaper than inverse-append. All
        replacement branches are skipped. This bounds the work to the
        useful region of the search space.

        RECONSTRUCTION
        --------------
        Trace backward from dp[N][best_final] to dp[0][start],
        recovering the segment chosen at each step.

        COMPLEXITY
        ----------
        Time:  O(N × |S| × |Q|)  where |Q| = number of distinct states
        Space: O(N × |Q|)

        For most practical domains |Q| is small (mod-12 domains have
        |Q| = 12; additive domains over short sequences have |Q| ≈ 4N).
        The budget pruning further reduces the constant factor.

        This method is skipped when N exceeds the dynamic limit or budget ≤ 1.

        Parameters
        ----------
        segs : list
            The broken sequence.
        budget : int
            The cost of the inverse-append fallback. DP is only useful
            if it can find a correction cheaper than this.

        Returns
        -------
        Ramm or None
            The minimum-cost in-place correction, or None if no in-place
            fix exists that is cheaper than the budget.
        """
        n = len(segs)

        # Skip when append is already near-optimal (1-2 segments).
        if budget <= 1:
            return None

        # The DP table has n × |states| entries. For large sequences with a
        # small alphabet, this stays fast. The practical limit scales with how
        # many segments the adapter has: fewer segments → smaller state space
        # → higher safe N. Hard ceiling of 500 prevents runaway memory.
        dp_limit = min(500, max(150, 600 // max(1, len(self.segments))))
        if n > dp_limit:
            return None

        # ── Initialize DP table ──────────────────────────────────────────────
        # Each entry dp[i][state] stores (min_swaps, prev_state, seg_used).
        # Only the minimum-cost path to each state is retained.
        dp = [dict() for _ in range(n + 1)]
        dp[0][self.start] = (0, None, None)

        # ── Forward pass ─────────────────────────────────────────────────────
        for i in range(n):
            for prev_state, (swaps, _, _) in dp[i].items():

                # Option A: keep the original segment — zero cost.
                ns = self._compose(prev_state, segs[i])
                if ns not in dp[i + 1] or dp[i + 1][ns][0] > swaps:
                    dp[i + 1][ns] = (swaps, prev_state, segs[i])

                # Option B: replace with any other segment — cost +1.
                # Pruning: if already at budget, no replacement is useful.
                if swaps + 1 >= budget:
                    continue

                for r in self.segments:
                    if r == segs[i]:
                        continue
                    ns   = self._compose(prev_state, r)
                    cost = swaps + 1
                    if ns not in dp[i + 1] or dp[i + 1][ns][0] > cost:
                        dp[i + 1][ns] = (cost, prev_state, r)

        # ── Find the optimal solution ─────────────────────────────────────────
        # Scan dp[N] for the minimum-cost state that satisfies P.
        best_cost  = budget      # only accept solutions cheaper than append
        best_final = None

        for final_state, (swaps, _, _) in dp[n].items():
            if self._perfect(final_state) and swaps < best_cost:
                best_cost  = swaps
                best_final = final_state

        if best_final is None:
            return None   # no in-place fix beats the budget

        # ── Backward reconstruction ───────────────────────────────────────────
        # Trace from dp[N][best_final] back to dp[0][start],
        # recovering the segment assigned at each position.
        result = list(segs)
        state  = best_final

        for i in range(n, 0, -1):
            entry = dp[i].get(state)
            if entry is None:
                return None   # should not occur if the table is consistent
            _, prev_state, seg_used = entry
            result[i - 1] = seg_used
            state = prev_state

        return Ramm(
            before = list(segs),
            after  = result,
            swaps  = best_cost,
            method = f'dp {best_cost}-swap',
        )


    # ============================================================================
    #  §2.5  generate — Build a perfect loop from scratch
    # ============================================================================

    def generate(
        self,
        n           : int,
        constraints : Optional[Callable[[Any, list], list]] = None,
    ) -> Loop:
        """
        Construct a perfect loop of approximately n segments.

        ALGORITHM
        ---------
        At each step, consult the constraints function to determine which
        segments are currently valid. Choose one at random. Before choosing,
        check how many segments inverse() would need to close the current
        state. When  len(free_segments) + len(closure)  reaches n, stop
        adding free segments and close the loop with inverse().

        The result is always perfect by construction. The closure is
        never random — it is the exact mathematical solution for the
        accumulated state at that point.

        INVARIANT
        ---------
        After generate() returns, loop.perfect is always True.
        If it is not, the adapter's inverse() is incorrect and a
        ValueError is raised.

        Complexity:  O(n)  free-segment steps  +  O(|I(state)|)  closure

        Parameters
        ----------
        n : int
            Target loop length. The actual length may differ slightly
            depending on how cleanly the closure fits.
        constraints : callable or None
            A function  (state, segments_so_far) → list[segment]
            that returns the allowed next segments at each step.
            None means all segments are always allowed.

        Returns
        -------
        Loop
            A perfect loop. loop.perfect is always True.
        """
        segs  = []
        state = self.start

        # Generate free segments until the closure fills the remaining budget.
        for _ in range(n * 3):   # safety bound prevents infinite loops
            closure = self._inverse_fn(state)
            if len(segs) + len(closure) >= n:
                break
            allowed = constraints(state, segs) if constraints else self.segments
            if not allowed:
                break
            s = random.choice(allowed)
            segs.append(s)
            state = self._compose(state, s)

        # Close the loop exactly. This segment is computed, not chosen.
        closure = self._inverse_fn(state)
        segs   += list(closure)

        loop = self.run(segs)
        if not loop.perfect:
            raise ValueError(
                f"generate() produced an imperfect loop — inverse() is incorrect.\n"
                f"  state before closure : {state!r}\n"
                f"  closure returned     : {closure}"
            )
        return loop


    # ============================================================================
    #  §2.6  verify — Audit a correction for minimality
    # ============================================================================

    def verify(self, ram: Ramm) -> dict:
        """
        Audit a correction. Confirm it is correct and test whether it
        is truly minimal — no cheaper correction exists.

        This is the only operation in Baru that searches.
        It exists to check the output of correct(), not to replace it.
        Calling verify() on every correction in production is not
        necessary; it is a tool for testing and validation.

        ALGORITHM
        ---------
        First confirm ram.after is perfect. Then, for k = 1, 2, ...,
        (ram.swaps − 1), search for any k-swap correction. The search
        tries all combinations of k positions and k replacements from S.
        If any combination yields a perfect loop, ram was non-minimal.

        The search uses sampling for sequences longer than 50 to remain
        tractable; exact search is performed for shorter sequences.

        Complexity:  O(N^k × |S|^k)  for each k tested — exponential
                     in the worst case, but bounded by ram.swaps − 1
                     levels and by sampling for large N.

        Parameters
        ----------
        ram : Ramm
            A correction record produced by correct() or any other source.

        Returns
        -------
        dict with keys:
            correct   : bool   — True if ram.after is perfect
            minimal   : bool   — True if no cheaper correction exists
            min_swaps : int    — the true minimum number of swaps found
        """
        result = self.run(ram.after)

        if not result.perfect:
            return {'correct': False, 'minimal': False, 'min_swaps': None}

        # Search for a correction cheaper than ram.swaps.
        min_found = ram.swaps
        for k in range(1, ram.swaps):
            found = self._search_swap(ram.before, k)
            if found is not None:
                min_found = k
                break

        return {
            'correct'   : True,
            'minimal'   : min_found == ram.swaps,
            'min_swaps' : min_found,
        }


    # ============================================================================
    #  §2.7  _search_swap — Exhaustive k-swap search (used only by verify)
    # ============================================================================

    def _search_swap(self, segs: list, k: int) -> Optional[list]:
        """
        Search exhaustively for any k-swap correction.

        Tries all combinations of k positions and k replacement segments.
        Returns the corrected sequence if one is found, otherwise None.

        For sequences longer than 50 segments, position sampling is used
        to keep the search tractable. The sample covers the full sequence
        uniformly plus a dense window at the end, where corrections are
        most commonly needed.

        Complexity:  O( C(N, k) × |S|^k )  — exponential in k.
                     Used only by verify(). Never called by correct().

        Parameters
        ----------
        segs : list
            The original broken sequence.
        k : int
            The exact number of swaps to attempt.

        Returns
        -------
        list or None
            A corrected sequence using exactly k swaps, or None.
        """
        # For long sequences, sample positions rather than exhausting all C(N,k).
        if len(segs) <= 50:
            positions = list(range(len(segs)))
        else:
            positions = sorted(
                set(range(0, len(segs), max(1, len(segs) // 40)))
                | set(range(max(0, len(segs) - 12), len(segs)))
            )

        for pos in combinations(positions, k):
            for replacements in product(self.segments, repeat=k):
                candidate = list(segs)
                for i, r in zip(pos, replacements):
                    candidate[i] = r
                if candidate != list(segs) and self.run(candidate).perfect:
                    return candidate

        return None


# ================================================================================
#  §3  EXAMPLE
#
#  A minimal integer domain to demonstrate the adapter contract.
#
#  Domain definition:
#      S  =  { -3, -1, 2, 4 }
#      ∘  =  addition  ( compose(q, s) = q + s )
#      P  =  equality to zero  ( perfect(q) = (q == 0) )
#      I  =  greedy fill: repeatedly pick the segment that brings
#             the remaining need closest to zero until need == 0
#
#  This is the smallest possible demonstration. Replace these four
#  values with any domain of your choosing to run Baru on new problems.
# ================================================================================

if __name__ == '__main__':

    # Segments: { -3, -1, 2, 4 }
    # Inverse: greedy fill using largest available segment each step.
    # GCD(1, 2, 3, 4) = 1, so every integer state is reachable.
    SEGS = [-3, -1, 2, 4]

    def simple_inverse(state):
        # Work backwards from -state to zero using any combination of segments.
        # Try each segment repeatedly (greedy largest-first), mixing signs if
        # needed so no remainder is ever stranded.
        need   = -state
        result = []
        pool   = sorted(SEGS, key=abs, reverse=True)
        for _ in range(abs(need) + 10):   # safety bound
            if need == 0:
                break
            best = min(pool, key=lambda s: abs(need - s))
            result.append(best)
            need -= best
        return result if need == 0 else []

    engine = Baru(
        segments = SEGS,
        start    = 0,
        compose  = lambda state, s: state + s,
        perfect  = lambda state: state == 0,
        inverse  = simple_inverse,
    )

    test_cases = [
        [4, -1, -3],             # already perfect  (state =   0)
        [4, 4, -1],              # open             (state =   7)
        [4, 4, 4, 2],            # open             (state =  14)
        [-3, -3, -3, -1],        # open             (state = -10)
    ]

    print()
    for segs in test_cases:
        loop = engine.run(segs)
        if loop.perfect:
            print(f'  {segs}  →  perfect')
        else:
            correction = engine.correct(segs)
            print(f'  {segs}  →  {correction.after}')
            print(f'  {"":>{len(str(segs))}}     {correction.swaps} swap(s)  [{correction.method}]')
        print()
