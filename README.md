# crt-agent

An agent that answers Cognitive Reflection Test (CRT) questions so you can watch it reason,
and a benchmark that tries to prove it isn't just pulling from data.

The CRT is three famous word problems (Frederick, 2005), designed so that people's
instinctual, immediate answer is incorrect. The test measures response inhibition:
whether you can override your instinct to get to the right answer.

| item | instinctual answer | right answer |
|---|---|---|
| bat + ball = $1.10, bat is $1.00 more | $0.10 | $0.05 |
| 5 machines, 5 widgets, 5 minutes → 100 machines, 100 widgets | 100 min | 5 min |
| lily patch doubles daily, covers lake in 48 days | 24 days | 47 days |

However, if you ask an LLM these questions and get the right answer instantly...that
means absolutely nothing. The CRT questions and their answers appear thousands of
times in the data it was trained on. 

This repo is built around that problem. It has three parts:

1. **An architecture where the model can't state an answer:** it emits equations; a
   symbolic solver computes the number. Recall has no channel to leak through.
2. **An item generator:** produces unbounded new CRT items with the same trap
   structure, splitting *wording* changes from *number* changes so you can tell which
   kind of memorisation you're looking at.
3. **A benchmark comparing four architectures:** includes an unaided control, with
   metrics designed so a table of results actually answers the question.

---

## Core Premise: The model is simply a parser

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

## Anti-Memorization: Separating wording from numbers

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

## The Four Arms

| arm | pipeline | what it's for |
|---|---|---|
| `symbolic` | `match → solve → verify` | Rule-based, no LLM. The floor. Abstains on any phrasing its regexes don't cover. |
| `tool` | `parse → solve ⇄ repair → verify` | LLM formalises under a forced tool schema, sympy computes. The main proposal. |
| `dual` | `intuit → parse → solve ⇄ repair → verify → reconcile` | Samples System 1 *before* deliberating, then reports the override. |
| `ablation` | `direct` | **Control.** Same model, no solver, no schema. Makes every other number mean something. |

The `⇄ repair` cycle is the only feedback loop in the system — see
[Trust machinery](#trust-machinery) for why it is blindfolded.

The three grounded arms share the identical `solve` and `verify` nodes, so any
difference between them is attributable to the architecture rather than to the
arithmetic.

### Why the Control Arm is **mandatory**

Without `ablation`, "the tool agent scored 94%" is unfalsifiable as a claim about the
scaffolding — maybe the model scores 94% unaided. The control is the only thing that
converts an accuracy number into a statement about what you built. It is also where
the memorisation effect is directly visible, since it's the only arm that *can*
recall. Its traces are reported as `audit: n/a` rather than `audit: failed`, because there is
no derivation to audit.

---

## Trust Machinery

Three mechanisms are in place to help ensure the reported numbers are honest.

### (1) Trace Auditing
#### Determining whether the reported answer came from the equations

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

### (2) Repair Loop
#### Seeing whether broken algebra can be fixed without seeing the label
 
When the solver rejects a spec, the model gets one chance to see its own equations and
the solver's complaint and try again. It never sees the expected answer — so it can
converge on *solvable algebra* but not on a target number, which is the difference
between a repair loop and fitting to the label. There is a test asserting exactly that
(`test_repair_never_sees_the_expected_answer`). The audit is passive and inspects
output; the repair loop is active and feeds the model input — which is why only the
latter needs a blindfold.

### (3) Capability Gating
#### Refusing to print a number that wasn't measured

A provider declares what it cannot do. `supports_unreflective_sampling = False` on the
`claude -p` backend means the `S1 lure` and `override` columns print `n/a` rather than a
number that looks like a System 1 measurement but isn't. Withholding a number you did
not measure is the whole point.

---

## Example Answer

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
line 0 — see [trace auditing](#1-trace-auditing). Questions typed at the CLI have no
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

## Results

### Benchmark Discriminates
A 60-item sweep on `qwen2.5:7b` via Ollama (`LLM_PROVIDER=openai-compat`), September 2026. 
Only the model changed, everything else remained the same (items, arms, solver as Haiku run below)
```
agent       canonical   surface  perturbed    novel   wording   number   abstain
n =                 3        15         30       12
symbolic       100.0%    100.0%     100.0%   100.0%      0.0%     0.0%      0.0%
tool            66.7%     60.0%      56.7%    91.7%      6.7%    10.0%     28.3%
dual           100.0%     86.7%      60.0%    91.7%     13.3%    40.0%     10.0%
ablation       100.0%    100.0%      76.7%    41.7%      0.0%    23.3%      0.0%

agent       accuracy  lure rate  audit ok   S1 lure  override
tool           65.0%       5.0%    100.0%       -         -
dual           75.0%       5.0%    100.0%     10.2%     39.0%
ablation       76.7%       8.3%       -         -         -
​```

Three Findings.

**1. The unaided model is reciting, and the item design catches it.** Bare qwen scores
100% on `canonical` and `surface`, falls to 76.7% on `perturbed`, and collapses to
41.7% on `novel`. `Surface` changes the nouns but keeps the famous numbers, so the
famous answer is still correct there. A model that has memorised `$0.05` passes it
without reading. `Novel` has no famous anything, so there is nothing to retrieve.
Wording gap 0, number gap 23.3: the model is not keyed to the phrasing, it is keyed
to the answers.

**2. Grounding pays exactly where recall can't.** On `novel`, the tool arm scores
91.7% against the control's 41.7%, a 50-point difference on the only items where
memory is useless. In aggregate, the tool arm scores *lower* (65.0% vs 76.7%), and
that comparison is the one this benchmark exists to distrust: the control's total is
propped up by items it already has the answers to. The grounded arm cannot recall, so
it pays a formalisation cost on every item and only collects where recall fails.

The failures matter too (or so I tell myself). The tool arm abstains on 28.3% of items 
with a 5% lure rate and every answer passing the audit. When the 7B wrote broken equations, 
the solver refused to guess and the pipeline said so. The control never abstains; when
it is wrong, it is wrong in the trap's direction. Obviously, the model can't say
"I don't know", hence why my pipeline is there.

**3. The CRT's trap works on a machine, and so does the override.** For the first
time in this project, models gave the designed lure answer: 8.3% for the control,
10.2% for the `dual` arm's unreflective first sample. Given room to deliberate, the
derived answer overruled that snap judgement 39% of the time. This is the
fast-intuition, slow-correction pattern the CRT was built to measure in people,
measured in a model. It is a functional analogy, not an identity claim: the 7B does
not have a System 1. It produces lure-consistent errors under shallow processing,
the same error signature by a different mechanism. Haiku never produced this row,
because it cannot answer without deliberating.

The failure digest shows the same disease as the 83.3% run below, in a weaker patient:
unbound symbols and under-determined systems (`days_to_cover_half = total_days - 1`,
with `total_days` never tied to the stated 48), clustered in the rate and doubling
families.

**Caveats specific to this run.** One sweep, one model, temperature 0. `Novel` has
n = 12, so 91.7% vs 41.7% is 11 items vs 5. `Canonical` is still n = 3. Treat the
50-point gap as a large directional effect, not a precise estimate. Rerunning with
more `novel` items and a second small model is the obvious next check.

### But ofc bugs  

The first live sweep (on Haiku) scored `tool` at **83.3%**, *below* the 100% control. The failure
digest showed why:

```
[tool] drift:novel:0  (abstain)
  equations: ['net_daily_reduction = seen_per_day - join_per_day',
              'initial_waitlist = net_daily_reduction * days_until_empty']
  error: solver: 'days_until_empty' is under-determined
```

The model had written the general formula and never bound `seen_per_day` to the 12
stated in the prose. That is a bug in the *prompt*, not the model: the formalisation
instructions never said every stated number must appear as a literal. Two changes
followed: (1) an explicit grounding rule in the prompt, and (2) the `repair` cycle that feeds
the solver's complaint back for one retry, and the arm went to 100%.

Worth noticing what made that debuggable at all: the trace recorded the exact equations
the model committed to, and the solver refused to guess rather than returning a
plausible number. An ungrounded arm that gets the same item wrong tells you nothing
about why.

### And then it saturated (yay)

After the grounding fix, a 48-call sweep on Haiku 4.5 via the subscription route, August 2026:

```
agent       canonical   surface  perturbed    novel   wording   number   abstain
symbolic       100.0%    100.0%     100.0%   100.0%      0.0%     0.0%      0.0%
tool           100.0%    100.0%     100.0%   100.0%      0.0%     0.0%      0.0%
dual           100.0%    100.0%     100.0%   100.0%      0.0%     0.0%      0.0%
ablation       100.0%    100.0%     100.0%   100.0%      0.0%     0.0%      0.0%
```

**The benchmark is saturated at this model tier and the grounding delta is zero.**
Haiku solves every CRT item unaided, so the control arm matches the grounded arms and
the scaffolding demonstrably buys nothing here. In simpler words, on a 2026 frontier-family model, 
this benchmark has stopped, measuring what it was built to measure.

However, it still means something; the memorisation worry that started the project
does not bite for these models. They transfer perfectly to perturbed wordings and
perturbed numbers, with both gaps at zero, which is exactly the signature of reasoning
rather than recall. The `symbolic` row is a sanity check: a rule-based system with no
model in it also scores 100%, confirming the items are correctly generated.

This is what led to the run with a local 7-8B model through an OpenAI-compatible endpoint [Benchmark Discriminates](#benchmark-discriminates).

---

## Limitations 

- **`canonical` has n = 3.** There are only three famous CRT wordings. Both gap
  columns carry wide error bars and the report says so on every run. They're
  directional, not point estimates.
- **Frontier models saturate it.** Measured above: every arm at 100% on Haiku, grounding
  delta zero. The benchmark discriminates architectures only on models weak enough to
  fall for the lures. However, the 7B run shows the benchmark discriminates below that tier. 
- **The `dual` arm is unmeasurable through the subscription backend.** `claude -p`
  always deliberates, so its System 1 columns are withheld rather than reported. Only
  the raw API path can sample unreflectively.
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

## Quick Start

```bash
pip install -e ".[dev]"

crt bench                   # Full sweep — Works offline, no API key needed
crt ask "A racket and a shuttlecock cost \$4.20 in total. The racket costs \$4.00 more than the shuttlecock. How much does the shuttlecock cost, in dollars?"
```

With no `ANTHROPIC_API_KEY` set, everything runs against a deterministic mock provider,
so the graphs, tracing, auditing, metrics and storage are all exercised without a
network call.

---

## Stack

Python 3.11+ · Anthropic SDK (forced tool-use for structured output) · Claude CLI ·
OpenAI-compatible endpoints (stdlib only) · LangGraph · sympy · Pydantic v2 · SQLAlchemy 2 / Postgres · LangFuse · pytest · GitHub Actions

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
  llm/
    registry.py         LLM_PROVIDER -> factory; each backend registers itself
    client.py           LLMProvider protocol, tool schemas, Anthropic provider
    claude_cli.py       `claude -p` subscription backend
    openai_compat.py    Ollama / LM Studio / vLLM — the local-model backend
    json_contract.py    shared JSON-contract fallback for backends without tool_choice
    mock.py             deterministic offline provider
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
crt agents                                    # List the arms
crt items --set surface -n 5                  # Inspect generated items and their traps
crt ask "<question>" --agent dual             # One question, full trace
crt bench --perturbed 20 --store -v           # Sweep, persist, stream progress

# Against a real model with a Pro/Max subscription instead of an API key:
LLM_PROVIDER=claude-cli crt bench --model haiku --concurrency 8 --max-cost 5.00
LLM_PROVIDER=claude-cli crt bench --model haiku --model sonnet --concurrency 8

# Against a local model (Ollama):
LLM_PROVIDER=openai-compat crt bench --model qwen2.5:7b --concurrency 2 -v
```

---

## Sources / Further Reading

- Frederick, S. (2005). *Cognitive Reflection and Decision Making.* Journal of
  Economic Perspectives 19(4). The original three items.
- Kahneman, D. (2011). *Thinking, Fast and Slow.* The dual-process framing the
  `dual` arm implements.
- Toplak, West & Stanovich (2014). On the CRT as a predictor of rational thinking —
  and on how badly the original items suffer from prior exposure. Which is, in
  miniature, the same problem this repo is about.
