"""Production Director: structured, inspectable editing decisions.

The Director turns a teaching script + recording analysis into an
`EditDecisionList` (EDL) that a human reviews; the deterministic engine
executes only accepted decisions. See docs/PRODUCTION_DIRECTOR.md for the
architecture and its invariants (meaning preservation, review gating,
determinism, reusable compositions).
"""

from __future__ import annotations
