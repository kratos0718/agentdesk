"""REST API.

Every request needs an X-API-Key header. Keys map to roles:
- agent:      can chat on behalf of a customer
- supervisor: can also list and decide refund approvals
"""

from __future__ import annotations

import json
import logging
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from . import db
from .agents import METRICS, handle
from .config import settings
from .guardrails import redact_pii


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {"level": record.levelname, "msg": redact_pii(record.getMessage())}
        for key in ("request_id", "session", "tools", "blocked", "latency_ms", "path", "status"):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry)


_handler = logging.StreamHandler()
_handler.setFormatter(JsonFormatter())
logging.getLogger("agentdesk").handlers = [_handler]
logging.getLogger("agentdesk").setLevel(logging.INFO)
log = logging.getLogger("agentdesk.api")

app = FastAPI(title="AgentDesk", version="0.1.0")
db.init_db()


def require_role(*roles: str):
    def check(x_api_key: str = Header(default="")) -> str:
        role = settings.api_keys.get(x_api_key)
        if role is None:
            raise HTTPException(status_code=401, detail="Missing or invalid API key")
        if role not in roles:
            raise HTTPException(status_code=403, detail="Not allowed for this role")
        return role
    return check


@app.middleware("http")
async def request_log(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    log.info("http", extra={"request_id": request_id, "path": request.url.path, "status": response.status_code})
    return response


class ChatIn(BaseModel):
    customer_id: str = Field(min_length=1, max_length=20)
    message: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=64)


class Decision(BaseModel):
    approve: bool


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": "online" if settings.online else "offline"}


@app.post("/chat")
def chat(body: ChatIn, role: str = Depends(require_role("agent", "supervisor"))) -> dict:
    if db.get_customer(body.customer_id) is None:
        raise HTTPException(status_code=404, detail="Unknown customer")
    return handle(body.message, body.customer_id, body.session_id, role).to_dict()


@app.get("/approvals")
def approvals(_: str = Depends(require_role("supervisor"))) -> list[dict]:
    return db.list_approvals("pending")


@app.post("/approvals/{approval_id}")
def decide(approval_id: int, body: Decision, role: str = Depends(require_role("supervisor"))) -> dict:
    outcome = db.decide_approval(approval_id, body.approve, decided_by=role)
    if outcome is None:
        raise HTTPException(status_code=404, detail="No pending approval with that id")
    return outcome


@app.get("/metrics")
def metrics(_: str = Depends(require_role("supervisor"))) -> dict:
    n = max(METRICS["requests"], 1)
    return {**METRICS, "avg_latency_ms": round(METRICS["total_latency_ms"] / n, 1),
            "cost_usd": round(METRICS["cost_usd"], 6)}
