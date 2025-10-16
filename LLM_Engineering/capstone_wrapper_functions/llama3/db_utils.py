from __future__ import annotations
import datetime as dt
from decimal import Decimal
from typing import Any, Dict, List, Optional, Literal

from sqlalchemy import Table, MetaData, select, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

JSONLike = Dict[str, Any]

# ---- Configure your DSN ----
# postgresql+asyncpg://user:pass@host:5432/dbname
def make_engine(dsn: str) -> AsyncEngine:
    return create_async_engine(dsn, pool_pre_ping=True)

# ---- Hard allowlist to prevent SQL injection via identifiers ----
# Map *public API names* to actual table + allowed columns
ALLOWED: Dict[str, Dict[str, Any]] = {
    "orders": {
        "table": "orders",
        "columns": ["id", "region", "amount", "status", "created_at"]
    },
    "customers": {
        "table": "customers",
        "columns": ["id", "name", "segment", "created_at"]
    }
}

def _jsonify(val: Any) -> Any:
    if isinstance(val, (dt.datetime, dt.date)):
        return val.isoformat()
    if isinstance(val, Decimal):
        # choose float or str depending on your precision needs
        return float(val)
    return val

def _rows_to_json(rows) -> List[JSONLike]:
    out: List[JSONLike] = []
    for r in rows:
        m = dict(r._mapping)  # SQLAlchemy Row -> mapping
        out.append({k: _jsonify(v) for k, v in m.items()})
    return out

async def fetch_rows_as_json(
    engine: AsyncEngine,
    dataset: Literal["orders", "customers"],   # keys from ALLOWED
    columns: List[str],
    where: Optional[Dict[str, Any]] = None,    # simple equality filters
    limit: int = 500
) -> List[JSONLike]:
    spec = ALLOWED.get(dataset)
    if not spec:
        raise ValueError(f"Unknown dataset '{dataset}'")

    # Validate columns against allowlist
    allowed_cols = spec["columns"]
    for c in columns:
        if c not in allowed_cols:
            raise ValueError(f"Column '{c}' not allowed for dataset '{dataset}'")
    if not columns:
        raise ValueError("At least one column is required")

    meta = MetaData()
    async with engine.begin() as conn:
        # Reflect just the needed table (cheap)
        table = Table(spec["table"], meta, autoload_with=conn.sync_engine)

        sel_cols = [table.c[c] for c in columns]
        stmt = select(*sel_cols)

        # Basic equality filters (opt-in). You can extend this to support ranges/operators.
        if where:
            for k, v in where.items():
                if k not in allowed_cols:
                    raise ValueError(f"Filter on disallowed column '{k}'")
                stmt = stmt.where(table.c[k] == v)

        stmt = stmt.limit(limit)
        res = await conn.execute(stmt)
        rows = res.fetchall()
        return _rows_to_json(rows)
