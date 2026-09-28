"""Tools the agents can call.

Every tool declares:
- which agent owns it (an agent can only call its own tools),
- a Pydantic model for its arguments (the same schema is shown to the LLM),
- a handler that receives validated arguments and the request context.

Business rules that must never depend on the model (ownership, refund limits,
final-sale items) are enforced here, in code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field, ValidationError

from . import db
from .config import settings
from .knowledge import get_index


class ToolError(Exception):
    """Raised when a tool call is not allowed or cannot be completed."""


@dataclass
class ToolContext:
    customer_id: str
    role: str = "agent"          # who is operating the assistant
    session_id: str = "default"


@dataclass
class Tool:
    name: str
    agent: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[Any, ToolContext], dict]
    side_effect: bool = False

    def schema(self) -> dict:
        return {"name": self.name, "agent": self.agent, "description": self.description,
                "parameters": self.args_model.model_json_schema()}


# ---- argument models ---------------------------------------------------------

class SearchArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300, description="What to look up in the policy documents")


class OrderArgs(BaseModel):
    order_id: int = Field(gt=0, description="Order number, for example 1042")


class NoArgs(BaseModel):
    pass


class RefundArgs(BaseModel):
    order_id: int = Field(gt=0)
    reason: str = Field(min_length=3, max_length=300)
    amount: Optional[float] = Field(default=None, gt=0, description="Leave empty to refund the remaining order value")


class EmailArgs(BaseModel):
    subject: str = Field(min_length=3, max_length=120)
    body: str = Field(min_length=3, max_length=2000)


# ---- handlers ----------------------------------------------------------------

def _owned_order(order_id: int, ctx: ToolContext) -> dict:
    order = db.get_order(order_id)
    # Same message for "missing" and "not yours", so the tool can't be used to
    # probe which order numbers exist.
    if order is None or order["customer_id"] != ctx.customer_id:
        raise ToolError(f"Order {order_id} was not found on this account.")
    return order


def search_policies(args: SearchArgs, ctx: ToolContext) -> dict:
    hits = get_index().search(args.query, k=3)
    return {"results": [{"source": h.source, "text": h.text, "score": h.score} for h in hits]}


def get_order(args: OrderArgs, ctx: ToolContext) -> dict:
    order = _owned_order(args.order_id, ctx)
    return {k: order[k] for k in ("id", "item", "amount", "status", "tracking", "days_in_transit")}


def list_my_orders(args: NoArgs, ctx: ToolContext) -> dict:
    return {"orders": db.list_orders(ctx.customer_id)}


def request_refund(args: RefundArgs, ctx: ToolContext) -> dict:
    order = _owned_order(args.order_id, ctx)
    if order["final_sale"]:
        return {"status": "refused", "order_id": order["id"],
                "reason": "Final-sale items, gift cards and digital products are not refundable."}
    delayed = order["status"] == "in_transit" and order["days_in_transit"] > 7
    if order["status"] != "delivered" and not delayed:
        return {"status": "refused", "order_id": order["id"],
                "reason": f"The order is {order['status'].replace('_', ' ')} and not eligible for a refund yet."}

    remaining = round(order["amount"] - db.refunded_amount(order["id"]), 2)
    if remaining <= 0:
        return {"status": "refused", "order_id": order["id"], "reason": "This order has already been refunded."}
    amount = min(args.amount or remaining, remaining)

    if amount > settings.refund_auto_limit:
        refund_id = db.create_refund(order["id"], amount, args.reason, "pending_approval")
        approval_id = db.create_approval(refund_id, requested_by=ctx.session_id)
        return {"status": "pending_approval", "order_id": order["id"], "amount": amount,
                "approval_id": approval_id,
                "reason": f"Refunds above Rs. {settings.refund_auto_limit:,.0f} need a supervisor's approval."}
    db.create_refund(order["id"], amount, args.reason, "issued")
    return {"status": "issued", "order_id": order["id"], "amount": amount}


def send_email(args: EmailArgs, ctx: ToolContext) -> dict:
    # Emails can only go to the customer who owns the session.
    email_id = db.queue_email(ctx.customer_id, args.subject, args.body)
    return {"status": "queued", "email_id": email_id}


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool("search_policies", "knowledge", "Search the company policy documents (refunds, shipping, returns, "
         "warranty, account security).", SearchArgs, search_policies),
    Tool("get_order", "data", "Get the status, item, amount and tracking of one of the customer's orders.",
         OrderArgs, get_order),
    Tool("list_my_orders", "data", "List the customer's orders.", NoArgs, list_my_orders),
    Tool("request_refund", "action", "Request a refund for one of the customer's orders.", RefundArgs,
         request_refund, side_effect=True),
    Tool("send_email", "action", "Email the customer a summary or confirmation.", EmailArgs, send_email,
         side_effect=True),
]}

AGENT_TOOLS: dict[str, list[str]] = {}
for _tool in TOOLS.values():
    AGENT_TOOLS.setdefault(_tool.agent, []).append(_tool.name)


def run_tool(agent: str, name: str, raw_args: dict, ctx: ToolContext) -> dict:
    tool = TOOLS.get(name)
    if tool is None:
        raise ToolError(f"Unknown tool '{name}'.")
    if tool.agent != agent:
        raise ToolError(f"The {agent} agent is not allowed to call '{name}'.")
    try:
        args = tool.args_model.model_validate(raw_args or {})
    except ValidationError as exc:
        raise ToolError(f"Invalid arguments for '{name}': {exc.errors()[0]['msg']}") from exc
    return tool.handler(args, ctx)
