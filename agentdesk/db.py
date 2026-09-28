"""SQLite storage: customers, orders, refunds, approvals, conversation memory and an email outbox.

Agents never write SQL. They call the small set of functions below, which all use
parameterised queries.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL, phone TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id),
    item TEXT NOT NULL, amount REAL NOT NULL, status TEXT NOT NULL,
    tracking TEXT, days_in_transit INTEGER DEFAULT 0, final_sale INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS refunds (
    id INTEGER PRIMARY KEY AUTOINCREMENT, order_id INTEGER NOT NULL, amount REAL NOT NULL,
    reason TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT, refund_id INTEGER NOT NULL, status TEXT NOT NULL,
    requested_by TEXT, decided_by TEXT, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT, to_customer TEXT NOT NULL, subject TEXT NOT NULL,
    body TEXT NOT NULL, created_at TEXT NOT NULL
);
"""

CUSTOMERS = [
    ("C001", "Priya Sharma", "priya.sharma@example.com", "9876543210"),
    ("C002", "Rahul Verma", "rahul.verma@example.com", "9123456780"),
    ("C003", "Ananya Rao", "ananya.rao@example.com", "9988776655"),
]

ORDERS = [
    (1042, "C001", "Wireless Earbuds", 2499.0, "in_transit", "BD1042IN", 9, 0),
    (1043, "C001", "Laptop Stand", 1299.0, "delivered", "BD1043IN", 0, 0),
    (1044, "C002", "Smartwatch", 8999.0, "delivered", "BD1044IN", 0, 0),
    (1045, "C002", "Gift Card", 2000.0, "delivered", None, 0, 1),
    (1046, "C003", "Mechanical Keyboard", 4599.0, "processing", None, 0, 0),
    (1047, "C003", "4K Monitor", 21999.0, "delivered", "BD1047IN", 0, 0),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path or settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None, reset: bool = False) -> None:
    path = path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset and path.exists():
        path.unlink()
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        if conn.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0:
            conn.executemany("INSERT INTO customers VALUES (?, ?, ?, ?)", CUSTOMERS)
            conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ORDERS)


# ---- orders ---------------------------------------------------------------

def get_order(order_id: int) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    return dict(row) if row else None


def list_orders(customer_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, item, amount, status FROM orders WHERE customer_id = ? ORDER BY id", (customer_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_customer(customer_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
    return dict(row) if row else None


# ---- refunds and approvals ---------------------------------------------------

def refunded_amount(order_id: int) -> float:
    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM refunds WHERE order_id = ? AND status != 'rejected'",
            (order_id,),
        ).fetchone()
    return float(row[0])


def create_refund(order_id: int, amount: float, reason: str, status: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO refunds (order_id, amount, reason, status, created_at) VALUES (?, ?, ?, ?, ?)",
            (order_id, amount, reason, status, _now()),
        )
        return int(cur.lastrowid)


def create_approval(refund_id: int, requested_by: str) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO approvals (refund_id, status, requested_by, created_at) VALUES (?, 'pending', ?, ?)",
            (refund_id, requested_by, _now()),
        )
        return int(cur.lastrowid)


def list_approvals(status: str = "pending") -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT a.id, a.status, a.requested_by, a.created_at, r.order_id, r.amount, r.reason "
            "FROM approvals a JOIN refunds r ON r.id = a.refund_id WHERE a.status = ? ORDER BY a.id",
            (status,),
        ).fetchall()
    return [dict(r) for r in rows]


def decide_approval(approval_id: int, approve: bool, decided_by: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)).fetchone()
        if row is None or row["status"] != "pending":
            return None
        new_status = "approved" if approve else "rejected"
        conn.execute("UPDATE approvals SET status = ?, decided_by = ? WHERE id = ?",
                     (new_status, decided_by, approval_id))
        conn.execute("UPDATE refunds SET status = ? WHERE id = ?",
                     ("issued" if approve else "rejected", row["refund_id"]))
    return {"approval_id": approval_id, "status": new_status}


# ---- memory and outbox -------------------------------------------------------

def add_message(session_id: str, role: str, content: str) -> None:
    with connect() as conn:
        conn.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                     (session_id, role, content, _now()))


def recent_messages(session_id: str, limit: int = 6) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def queue_email(customer_id: str, subject: str, body: str) -> int:
    with connect() as conn:
        cur = conn.execute("INSERT INTO outbox (to_customer, subject, body, created_at) VALUES (?, ?, ?, ?)",
                           (customer_id, subject, body, _now()))
        return int(cur.lastrowid)
