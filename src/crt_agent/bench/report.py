"""Plain-text rendering of a sweep."""

from __future__ import annotations

from crt_agent.bench.runner import SweepResult
from crt_agent.schemas import Answer, Item, StepKind


def _pct(x: float) -> str:
    return "  -  " if x != x else f"{x * 100:5.1f}%"


def summary_table(result: SweepResult) -> str:
    counts = {
        name: result.metrics(result.agents()[0], name).n if result.agents() else 0
        for name in ("canonical", "surface", "perturbed", "novel")
    }
    header = (
        f"{'agent':<10} "
        f"{'canonical':>10} {'surface':>9} {'perturbed':>10} {'novel':>8} "
        f"{'wording':>9} {'number':>8} {'abstain':>9}"
    )
    lines = [
        "",
        "ACCURACY BY ITEM SET" + "                              " + "GAP vs CANONICAL",
        "-" * 82,
        header,
        f"{'n =':<10} "
        f"{counts['canonical']:>10} {counts['surface']:>9} {counts['perturbed']:>10} "
        f"{counts['novel']:>8}",
        "-" * 82,
    ]
    for agent in result.agents():
        overall = result.metrics(agent)
        lines.append(
            f"{agent:<10} "
            f"{_pct(result.metrics(agent, 'canonical').accuracy):>10} "
            f"{_pct(result.metrics(agent, 'surface').accuracy):>9} "
            f"{_pct(result.metrics(agent, 'perturbed').accuracy):>10} "
            f"{_pct(result.metrics(agent, 'novel').accuracy):>8} "
            f"{_pct(result.wording_gap(agent)):>9} "
            f"{_pct(result.number_gap(agent)):>8} "
            f"{_pct(overall.abstention_rate):>9}"
        )

    lines += [
        "",
        "FAILURE STRUCTURE AND TRACE INTEGRITY  (all items)",
        "-" * 82,
        f"{'agent':<10} {'accuracy':>9} {'lure rate':>10} {'audit ok':>9} "
        f"{'S1 lure':>9} {'override':>9} {'ms/item':>9}",
        "-" * 82,
    ]
    for agent in result.agents():
        m = result.metrics(agent)
        lines.append(
            f"{agent:<10} "
            f"{_pct(m.accuracy):>9} "
            f"{_pct(m.lure_rate):>10} "
            f"{_pct(m.audit_validity):>9} "
            f"{_pct(m.system1_lure_rate):>9} "
            f"{_pct(m.override_rate):>9} "
            f"{m.mean_latency_ms:>9.0f}"
        )

    lines += [
        "",
        "  surface   = canonical numbers, different nouns  -> same correct answer.",
        "  perturbed = canonical structure, different numbers.",
        "  wording   = canonical - surface.   High: it was keyed to the famous phrasing.",
        "  number    = canonical - perturbed. High: it was reciting the famous answer.",
        "              Both near zero is the only result that supports 'it reasons'.",
        "  lure rate = answered with the item's designed wrong answer, not a random one.",
        "  audit ok  = share of answers reconstructable from their own trace",
        "              ('-' means the arm has no solver to audit, i.e. the control).",
        "  S1 lure   = how often the snap judgement took the bait (dual-process only).",
        "  override  = how often the derived answer overruled the snap one.",
        "",
    ]
    if counts["canonical"] < 10:
        lines += [
            f"  NOTE: only {counts['canonical']} canonical items exist (there are only three",
            "        famous CRT wordings), so both gap columns carry wide error bars.",
            "        Treat them as directional, not as point estimates.",
            "",
        ]
    return "\n".join(lines)


def failure_digest(result: SweepResult, limit: int = 6) -> str:
    lines = ["MISSES", "-" * 78]
    shown = 0
    for item, answer in result.rows:
        if item.is_correct(answer.value) or shown >= limit:
            continue
        shown += 1
        if item.is_lure(answer.value):
            kind = "lure"
        elif answer.abstained:
            kind = "abstain"
        else:
            kind = "other"
        lines.append(f"[{answer.agent}] {item.item_id}  ({kind})")
        lines.append(f"  {item.text}")
        lines.append(f"  expected {item.answer}   got {answer.value}   lure {item.lure}")
        parse = answer.trace.last(StepKind.PARSE)
        if parse:
            lines.append(f"  equations: {parse.payload.get('spec', {}).get('equations')}")
        if answer.error:
            lines.append(f"  error: {answer.error}")
        lines.append("")
    if shown == 0:
        lines.append("(none)")
    return "\n".join(lines)


def render_answer(item: Item, answer: Answer) -> str:
    """Full trace for a single attempt — what `crt ask` prints."""
    lines = [
        "",
        f"QUESTION  {item.text}",
        f"AGENT     {answer.agent}   trace {answer.trace.trace_id}",
        "",
        "REASONING",
        answer.trace.render(),
        "",
    ]

    # An ad-hoc question typed at the CLI has no known answer. Scoring it against a
    # NaN would print "[wrong]" for a perfectly good answer, so don't score it at all.
    scored = item.answer == item.answer  # False for NaN

    if answer.abstained:
        verdict = "abstained"
    elif not scored:
        verdict = "unscored — no reference answer for an ad-hoc question"
    elif item.is_lure(answer.value):
        verdict = "wrong (fell for the lure)"
    elif item.is_correct(answer.value):
        verdict = "correct"
    else:
        verdict = "wrong"

    unit = answer.unit or ""
    lines.append(f"ANSWER    {answer.value} {unit}".rstrip() + f"   [{verdict}]")
    if scored:
        lines.append(f"EXPECTED  {item.answer} {item.unit}   (lure: {item.lure})")

    if answer.audit:
        if not answer.audit.applicable:
            lines.append(f"AUDIT     n/a — {answer.audit.reason}")
        else:
            state = "VALID" if answer.audit.valid else "INVALID"
            lines.append(f"AUDIT     {state} — {answer.audit.reason}")
    lines.append(f"COST      {answer.latency_ms:.0f} ms, {answer.tokens_out} output tokens")
    lines.append("")
    return "\n".join(lines)
