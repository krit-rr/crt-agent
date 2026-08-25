"""Trace auditing: does the reasoning actually produce the answer?

A correct answer with an incoherent trace is not evidence of reasoning — it is
evidence that the answer arrived by some route the trace does not describe. In a
benchmark about *how* a system gets there, that run is worthless, and this module
marks it invalid regardless of correctness.

The audit is deliberately mechanical. It does not read the rationale text. It checks:

1. a PARSE step produced a `ProblemSpec`,
2. a SOLVE step consumed *that same spec* (compared by content hash, not by trust),
3. the solver verified its own back-substitution,
4. the answer the agent reported equals the value the solver returned.

Step 2 is the one that catches the interesting failure: an agent that formalises the
problem, quietly ignores the formalisation, and reports a remembered number. The
hash makes that undetectable-by-eye failure a hard error.
"""

from __future__ import annotations

import hashlib
import json

from crt_agent.schemas import AuditVerdict, ProblemSpec, StepKind, Trace

TOLERANCE = 1e-9


def spec_fingerprint(spec: ProblemSpec) -> str:
    """Content hash of a spec, stable under key ordering."""
    blob = json.dumps(
        {
            "variables": sorted(v.name for v in spec.variables),
            "equations": sorted(e.replace(" ", "") for e in spec.equations),
            "query": spec.query,
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def audit(trace: Trace, reported_value: float | None, *, grounded: bool = True) -> AuditVerdict:
    """Check that `reported_value` is derivable from `trace` alone.

    `grounded=False` marks the ablation control arm, where no solver exists by
    design. Those runs are reported as `applicable=False` rather than as failures,
    so the control's numbers stay interpretable.
    """
    if not grounded:
        return AuditVerdict(
            valid=False,
            applicable=False,
            reason="ungrounded control arm: no solver step exists to audit against",
        )

    if trace.last(StepKind.ABSTAIN) is not None and reported_value is None:
        return AuditVerdict(valid=True, reason="agent abstained, and reported no value")

    parse = trace.last(StepKind.PARSE)
    if parse is None:
        return AuditVerdict(valid=False, reason="no PARSE step: nothing was formalised")
    if "spec_fingerprint" not in parse.payload:
        return AuditVerdict(valid=False, reason="PARSE step did not record a spec fingerprint")

    solve = trace.last(StepKind.SOLVE)
    if solve is None:
        return AuditVerdict(valid=False, reason="no SOLVE step: no answer was computed")

    if solve.payload.get("spec_fingerprint") != parse.payload["spec_fingerprint"]:
        return AuditVerdict(
            valid=False,
            reason="SOLVE consumed a different spec than PARSE produced "
            f"({solve.payload.get('spec_fingerprint')} != {parse.payload['spec_fingerprint']})",
        )

    verify = trace.last(StepKind.VERIFY)
    if verify is None or not verify.payload.get("verified"):
        return AuditVerdict(valid=False, reason="solution failed back-substitution verification")

    solved = solve.payload.get("value")
    if solved is None or reported_value is None:
        return AuditVerdict(valid=False, reason="SOLVE step recorded no value")
    if abs(float(solved) - float(reported_value)) > TOLERANCE:
        return AuditVerdict(
            valid=False,
            reason=f"reported answer {reported_value} does not match solver output {solved}",
        )

    return AuditVerdict(valid=True, reason="answer is reconstructable from the trace")
