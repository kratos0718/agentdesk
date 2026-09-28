"""Command-line demo.

    python -m agentdesk.cli --customer C001 "Where is my order 1042?"
    python -m agentdesk.cli --approvals
    python -m agentdesk.cli --approve 1
"""

from __future__ import annotations

import argparse
import logging

from . import db
from .agents import handle


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentDesk command-line demo")
    parser.add_argument("message", nargs="?", help="customer message")
    parser.add_argument("--customer", default="C001", help="customer id (C001, C002, C003)")
    parser.add_argument("--session", default="cli")
    parser.add_argument("--trace", action="store_true", help="show the plan and tool results")
    parser.add_argument("--approvals", action="store_true", help="list pending refund approvals")
    parser.add_argument("--approve", type=int, metavar="ID", help="approve a pending refund (supervisor)")
    parser.add_argument("--reset", action="store_true", help="recreate the demo database")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR)
    db.init_db(reset=args.reset)
    if args.approvals:
        for a in db.list_approvals():
            print(f"#{a['id']}  order {a['order_id']}  Rs. {a['amount']:,.0f}  reason: {a['reason']}")
        return
    if args.approve:
        print(db.decide_approval(args.approve, True, decided_by="supervisor") or "No pending approval with that id")
        return
    if not args.message:
        parser.error("give a message, or use --approvals / --approve")

    result = handle(args.message, args.customer, args.session)
    if args.trace or result.blocked:
        print(f"[mode: {result.mode}]  blocked: {result.blocked}")
        for s in result.steps:
            print(f"  {s.agent:<9} -> {s.tool}({s.args})  {'ok' if s.ok else 'error'}")
        print()
    print(result.answer)
    if result.sources:
        print(f"\nSources: {', '.join(result.sources)}")
    print(f"({result.latency_ms:.0f} ms, {result.usage.input_tokens + result.usage.output_tokens} tokens)")


if __name__ == "__main__":
    main()
