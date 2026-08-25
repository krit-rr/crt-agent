"""Command line entry point.

crt agents                       list the arms
crt items --set perturbed        show the generated item set
crt ask "..." --agent tool       one question, full reasoning trace
crt bench --perturbed 10         sweep every arm, print the table, store it
"""

from __future__ import annotations

import argparse
import sys

from crt_agent.agents import AGENTS
from crt_agent.bench import failure_digest, render_answer, run_sweep, summary_table
from crt_agent.config import settings
from crt_agent.items import (
    canonical_items,
    default_suite,
    novel_items,
    perturbed_items,
    surface_items,
)
from crt_agent.items.templates import TEMPLATES
from crt_agent.llm.client import build_provider
from crt_agent.schemas import Item
from crt_agent.store.repository import Repository
from crt_agent.tracing.langfuse_client import get_tracer


def _ad_hoc_item(text: str) -> Item:
    """Wrap free text as an item. Answer/lure are unknown, so scoring is meaningless."""
    return Item(
        item_id="ad-hoc",
        family="bat_ball",
        item_set="novel",
        text=text,
        answer=float("nan"),
        lure=float("nan"),
    )


def cmd_agents(_: argparse.Namespace) -> int:
    print()
    for name, cls in AGENTS.items():
        print(f"  {name:<10} {cls.blurb}")
        print(f"  {'':<10} grounded={cls.grounded}")
    print()
    return 0


def cmd_items(args: argparse.Namespace) -> int:
    items = {
        "canonical": canonical_items,
        "surface": lambda: surface_items(args.n, seed=settings.seed),
        "perturbed": lambda: perturbed_items(args.n, seed=settings.seed),
        "novel": lambda: novel_items(args.n, seed=settings.seed),
        "all": lambda: default_suite(args.n, args.n, seed=settings.seed),
    }[args.set]()
    for item in items:
        print(f"\n{item.item_id}   [answer {item.answer}, lure {item.lure}]")
        print(f"  {item.text}")
        print(f"  trap: {TEMPLATES[item.family].trap}")
    print()
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    provider, tracer = build_provider(), get_tracer()
    item = _ad_hoc_item(args.question)
    for name in args.agent:
        agent = AGENTS[name](provider=provider, tracer=tracer)
        print(render_answer(item, agent.answer(item)))
    tracer.flush()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    provider, tracer = build_provider(), get_tracer()
    agents = [AGENTS[name](provider=provider, tracer=tracer) for name in args.agent]
    items = default_suite(args.perturbed, args.novel, seed=settings.seed)

    print(f"provider={provider.name}  items={len(items)}  agents={[a.name for a in agents]}")
    result = run_sweep(
        agents, items, progress=(lambda line: print(f"  {line}")) if args.verbose else None
    )
    tracer.flush()

    print(summary_table(result))
    if args.failures:
        print(failure_digest(result, limit=args.failures))

    if args.store:
        repo = Repository()
        repo.create_schema()
        run_id = repo.save_run(
            provider=provider.name,
            model=settings.model,
            seed=settings.seed,
            agents=[a.name for a in agents],
            results=result.rows,
            notes=args.notes,
        )
        print(f"stored run {run_id} in {repo.url}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crt", description="CRT reasoning-agent benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("agents", help="list the agent arms").set_defaults(func=cmd_agents)

    items = sub.add_parser("items", help="print generated items")
    items.add_argument(
        "--set",
        choices=["canonical", "surface", "perturbed", "novel", "all"],
        default="all",
    )
    items.add_argument("-n", type=int, default=3, help="items per family")
    items.set_defaults(func=cmd_items)

    ask = sub.add_parser("ask", help="answer one question and print the trace")
    ask.add_argument("question")
    ask.add_argument("--agent", action="append", choices=list(AGENTS), default=None)
    ask.set_defaults(func=cmd_ask)

    bench = sub.add_parser("bench", help="run the full sweep")
    bench.add_argument("--agent", action="append", choices=list(AGENTS), default=None)
    bench.add_argument(
        "--perturbed", type=int, default=10, help="perturbed and surface items per family"
    )
    bench.add_argument("--novel", type=int, default=4, help="novel items per family")
    bench.add_argument("--failures", type=int, default=6, help="misses to print (0 to skip)")
    bench.add_argument("--store", action="store_true", help="persist the run to the database")
    bench.add_argument("--notes", default="")
    bench.add_argument("-v", "--verbose", action="store_true")
    bench.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "agent", None) is None and args.command in {"ask", "bench"}:
        args.agent = ["tool"] if args.command == "ask" else list(AGENTS)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
