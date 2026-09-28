import pytest

from agentdesk import db
from agentdesk.tools import ToolContext, ToolError, run_tool

PRIYA = ToolContext(customer_id="C001")
ANANYA = ToolContext(customer_id="C003")


def test_customer_can_read_own_order():
    out = run_tool("data", "get_order", {"order_id": 1042}, PRIYA)
    assert out["item"] == "Wireless Earbuds"


def test_customer_cannot_read_someone_elses_order():
    with pytest.raises(ToolError, match="not found"):
        run_tool("data", "get_order", {"order_id": 1044}, PRIYA)


def test_agent_cannot_call_another_agents_tool():
    with pytest.raises(ToolError, match="not allowed"):
        run_tool("knowledge", "request_refund", {"order_id": 1043, "reason": "test"}, PRIYA)


def test_arguments_are_validated():
    with pytest.raises(ToolError, match="Invalid arguments"):
        run_tool("data", "get_order", {"order_id": "abc"}, PRIYA)


def test_small_refund_is_issued():
    out = run_tool("action", "request_refund", {"order_id": 1043, "reason": "not needed"}, PRIYA)
    assert out["status"] == "issued" and out["amount"] == 1299.0


def test_large_refund_waits_for_approval():
    out = run_tool("action", "request_refund", {"order_id": 1047, "reason": "dead pixels"}, ANANYA)
    assert out["status"] == "pending_approval"
    assert db.list_approvals()[0]["order_id"] == 1047
    assert db.decide_approval(out["approval_id"], True, "supervisor")["status"] == "approved"
    assert db.list_approvals() == []


def test_final_sale_item_is_refused():
    out = run_tool("action", "request_refund", {"order_id": 1045, "reason": "changed mind"},
                   ToolContext(customer_id="C002"))
    assert out["status"] == "refused"


def test_order_cannot_be_refunded_twice():
    run_tool("action", "request_refund", {"order_id": 1043, "reason": "first"}, PRIYA)
    out = run_tool("action", "request_refund", {"order_id": 1043, "reason": "second"}, PRIYA)
    assert out["status"] == "refused"


def test_delayed_order_in_transit_can_be_refunded():
    out = run_tool("action", "request_refund", {"order_id": 1042, "reason": "late"}, PRIYA)
    assert out["status"] == "issued"
