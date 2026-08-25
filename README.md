# crt-agent

An agent that answers Cognitive Reflection Test questions so you can watch it reason,
and a benchmark that tries to prove it isn't just remembering.

The CRT is three famous word problems (Frederick, 2005). Each one has a wrong answer
that arrives fast and a right answer that requires overriding it:

| item | fast answer | right answer |
|---|---|---|
| bat + ball = $1.10, bat is $1.00 more | $0.10 | $0.05 |
| 5 machines, 5 widgets, 5 minutes → 100 machines, 100 widgets | 100 min | 5 min |
| lily patch doubles daily, covers lake in 48 days | 24 days | 47 days |

Ask a language model these and you get the right answer instantly — which tells you
nothing, because those three items and their answers appear thousands of times in any
pretraining corpus. Getting `$0.05` is not evidence of reasoning. It's evidence of
having read the internet.

This repo is built around that problem. It has three parts:

1. **An architecture where the model cannot state an answer.** It emits equations; a
   symbolic solver computes the number. Recall has no channel to leak through.
2. **An item generator** that produces unbounded new CRT items with the same trap
   structure, splitting *wording* changes from *number* changes so you can tell which
   kind of memorisation you're looking at.
3. **A benchmark** comparing four architectures, including an unaided control, with
   metrics designed so a table of results actually answers the question.

---

## Quick start

```bash
pip install -e ".[dev]"

crt bench                   # full sweep — works offline, no API key needed
crt ask "A racket and a shuttlecock cost \$4.20 in total. The racket costs \$4.00 more than the shuttlecock. How much does the shuttlecock cost, in dollars?"
```

With no `ANTHROPIC_API_KEY` set, everything runs against a deterministic mock provider,
so the graphs, tracing, auditing, metrics and storage are all exercised without a
network call. Set the key in `.env` to run the real thing.

---

## What one answer looks like

```
QUESTION  A racket and a shuttlecock cost $4.20 in total. The racket costs $4.00
          more than the shuttlecock. How much does the shuttlecock cost, in dollars?
AGENT     tool   trace c954200503c4

REASONING
  [0] PARSE     (parse, 812ms)
      big + small = 4.2; big - small = 4.0  ->  solve for small
  [1] SOLVE     (solve, 64ms)
      small = 1/10 dollars
  [2] VERIFY    (verify, 0ms)
      all residuals zero

ANSWER    0.1 dollars   [unscored — no reference answer for an ad-hoc question]
AUDIT     VALID — answer is reconstructable from the trace
COST      71 ms, 60 output tokens
```

The model produced line 0 and nothing else. Lines 1 and 2 are sympy. The `AUDIT` line
is a mechanical check that the number on the `ANSWER` line came from the equations on
line 0 — see [Trace auditing](#trace-auditing). Questions typed at the CLI have no
reference answer, so they are deliberately not scored; scored comparisons come from
`crt bench`.

The same question through the control arm, for contrast:

```
AGENT     ablation
  [0] DIRECT    (direct, 0ms)
      (mock) went with the obvious read
ANSWER    0.2
AUDIT     n/a — ungrounded control arm: no solver step exists to audit against
```

Same model, no equations, no derivation, and the answer is the lure: $4.20 − $4.00.

---

## The four arms

| arm | pipeline | what it's for |
|---|---|---|
| `symbolic` | `match → solve → verify` | Rule-based, no LLM. The floor. Abstains on any phrasing its regexes don't cover. |
| `tool` | `parse → solve → verify` | LLM formalises under a forced tool schema, sympy computes. The main proposal. |
| `dual` | `intuit → parse → solve → verify → reconcile` | Samples System 1 *before* deliberating, then reports the override. |
| `ablation` | `direct` | **Control.** Same model, no solver, no schema. Makes every other number mean something. |

The three grounded arms share the identical `solve` and `verify` nodes, so any
difference between them is attributable to the architecture rather than to the
arithmetic.

### Why the control arm is not optional

Without `ablation`, "the tool agent scored 94%" is unfalsifiable as a claim about the
scaffolding — maybe the model scores 94% unaided. The control is the only thing that
converts an accuracy number into a statement about what you built. It is also where
the memorisation effect is directly visible, since it's the only arm that *can*
recall.

Its traces are reported as `audit: n/a` rather than `audit: failed`, because there is
no derivation to audit. That's honest reporting, not a gap.

---

## The core idea: the model is a parser, not an oracle

```python
class ProblemSpec(BaseModel):
    variables: list[Variable]
    equations: list[str]     # "big + small = 1.10"
    query: str               # "small"
    assumptions: list[str]
```

This is the only channel from the model to the answer. There is no field in the tool
schema in which the model *can* write a number as its result. It says what the
problem is; sympy says what the answer is.

Two properties make that a real guarantee rather than a stylistic preference:

**The spec is charset-validated before it reaches the solver.** Equations may contain
lowercase identifiers, digits, and `+ - * / ( ) ^ .` — nothing else. Anything that
looks like a function call, an import, or an attribute access is rejected by Pydantic.

**The evaluation namespace is sealed.** `parse_expr` runs with an empty global dict
plus four numeric constructors. Only the variables the model declared resolve as
symbols; `pi`, `exp`, and `__import__` all fail to parse rather than evaluating.
Arithmetic is exact — `(1.10 - 1.00) / 2` is `1/20`, not `0.050000000000000044`.

```python
solve_spec(spec).exact   # '1/20'
```

---

## Anti-memorisation: separating wording from numbers

The benchmark runs four item sets:

| set | wording | numbers | correct answer |
|---|---|---|---|
| `canonical` | famous | famous | famous |
| `surface` | **new** | famous | **same as canonical** |
| `perturbed` | famous structure | **new** | new |
| `novel` | new family entirely | new | new |

`surface` and `perturbed` are separate on purpose. Perturbing wording and numbers at
once is the obvious thing to do and it confounds two different failure modes:

```
wording gap = accuracy(canonical) - accuracy(surface)      # keyed to the phrasing
number gap  = accuracy(canonical) - accuracy(perturbed)    # reciting the answer
```

An agent that drops on `surface` was pattern-matching the sentence. An agent that
drops on `perturbed` was reproducing `$0.05`. Only *both gaps near zero* supports the
claim that something transfers. This is the first thing anyone will ask you about the
result, so the benchmark answers it by construction.

Items are generated from parametric templates, each of which knows its own **lure** —
the specific wrong answer System 1 produces:

```python
BAT_BALL.answer(p)  # (total - diff) / 2
BAT_BALL.lure(p)    # total - diff
```

Tracking the lure separately from "wrong" is the whole psychological point. An agent
that is wrong *in the lure direction* is failing differently from one that is wrong at
random, and a benchmark that reports only accuracy throws that distinction away. The
generator rejects any draw where the answer coincides with the lure — if the trap
isn't a trap, it isn't a CRT item.

Three novel families (`discount`, `rank`, `drift`) were written for this repo. They
have the same trap structure and no canonical wording to recall at all.

---

## Trace auditing

A correct answer with an incoherent trace is worthless in a benchmark about *how* the
system got there. The auditor is mechanical — it never reads the rationale text:

1. a `PARSE` step produced a spec,
2. a `SOLVE` step consumed **that same spec**, compared by content hash,
3. the solver verified its own back-substitution,
4. the reported answer equals the solver's output.

Check 2 is the load-bearing one. It catches an agent that formalises the problem,
quietly ignores the formalisation, and reports a remembered number — a failure that is
invisible to the eye and a hard error to the fingerprint comparison.

An abstention with no reported value passes the audit. Declining to answer is an
honest outcome and the benchmark scores it as one.

---

## What it doesn't prove

Worth being straight about, because these are the holes an ML engineer will poke:

- **`canonical` has n = 3.** There are only three famous CRT wordings. Both gap
  columns carry wide error bars and the report says so on every run. They're
  directional, not point estimates.
- **A well-formed spec of the wrong problem still passes the audit.** Grounding moves
  the failure from arithmetic into formalisation; it doesn't eliminate it. The mock
  provider deliberately does this on ~8% of items so the failure mode is visible in
  the output rather than theoretical. Scoring the *spec* against a reference spec is
  the obvious next metric and isn't built yet.
- **Solver-friendly problems only.** Every family here reduces to algebra. CRT-2 items
  ("if you pass the person in second place, what place are you in?") don't, and the
  `ProblemSpec` contract has nothing to say about them.
- **The templates are the ceiling.** Perturbation explores inside a structure a human
  wrote. It is not the same as unseen problems.

---

## Stack

Python 3.11+ · Anthropic SDK (forced tool-use for structured output) · LangGraph ·
sympy · Pydantic v2 · SQLAlchemy 2 / Postgres · LangFuse · pytest · GitHub Actions

```
src/crt_agent/
  schemas.py            ProblemSpec, Trace, Item, Answer — the typed contracts
  solver/symbolic.py    sealed sympy evaluation + back-substitution
  items/
    templates.py        six parametric families, each with answer + lure + spec
    bank.py             canonical / surface / perturbed / novel set construction
  agents/
    base.py             shared graph scaffolding
    nodes.py            parse, solve, verify, intuit — shared across arms
    {symbolic,tool,dual_process,ablation}.py
  llm/{client,mock}.py  Anthropic provider + deterministic offline provider
  tracing/              LangFuse adapter (no-op fallback) + the auditor
  store/                three-table schema: runs / attempts / trace_steps
  bench/                sweep, metrics, text report
```

### Observability

```bash
docker compose up -d          # LangFuse + Postgres + ClickHouse + MinIO
# create a project at localhost:3000, put the keys in .env
crt bench --store
```

Every graph node becomes a LangFuse span. If the keys aren't set, `get_tracer()`
returns a null tracer and nothing else changes — observability is never load-bearing
for correctness, and the local `Trace` object is the source of truth for scoring.

### Storage

Trace steps are stored as **rows**, not as a JSON blob on the attempt. Once they're
rows, the questions worth asking are SQL rather than a re-run:

```sql
-- what does the parser emit when it gets bat/ball wrong?
select t.payload->'spec'->>'equations'
from trace_steps t join attempts a on a.id = t.attempt_id
where t.kind = 'parse' and a.family = 'bat_ball' and not a.correct;
```

### CI

Three jobs: lint + tests on 3.11/3.12 against the mock provider, the store tests
against a real `postgres:16` service container, and a full offline benchmark sweep as
a smoke test. No API key, no network, no tokens spent on any push.

---

## Commands

```bash
crt agents                              # list the arms
crt items --set surface -n 5            # inspect generated items and their traps
crt ask "<question>" --agent dual       # one question, full trace
crt bench --perturbed 20 --store -v     # sweep, persist, stream progress
```

---

## Reading

- Frederick, S. (2005). *Cognitive Reflection and Decision Making.* Journal of
  Economic Perspectives 19(4). The original three items.
- Kahneman, D. (2011). *Thinking, Fast and Slow.* The dual-process framing the
  `dual` arm implements.
- Toplak, West & Stanovich (2014). On the CRT as a predictor of rational thinking —
  and on how badly the original items suffer from prior exposure. Which is, in
  miniature, the same problem this repo is about.
