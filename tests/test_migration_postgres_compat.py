"""Regression pin for the Postgres-invalid boolean `server_default` bug.

`sa.text('0')` / `sa.text('1')` render as a bare, unquoted SQL literal. On
SQLite that's harmless. On PostgreSQL, pairing it with `sa.Boolean()` renders
`BOOLEAN ... DEFAULT 0`, and Postgres has no implicit integer->boolean cast
in DDL, so `CREATE TABLE` fails outright (`column "..." is of type boolean
but default expression is of type integer`). That took down
`CREATE TABLE broker_positions` and, with it, the whole `docker compose up`
stack on Postgres — while the SQLite-only test suite stayed green, because
SQLite happily stores 0/1 in a column typed "boolean" regardless of how the
default was spelled.

Two independent checks, because either the ORM models or the frozen Alembic
migration could drift back into this bug independently of one another:

1. Model-DDL compile check — compiles every mapped table's CREATE TABLE
   under the PostgreSQL dialect (pure string rendering, no connection) and
   asserts no Boolean column renders a bare-integer DEFAULT.
2. Migration-source check — parses the frozen migration's AST and asserts no
   `sa.Boolean()` column pairs with `server_default=sa.text('0'/'1')`. This
   is the half that must fail (RED) against the pre-fix migration file.

Needs no live Postgres: CreateTable(...).compile(dialect=...) only renders a
SQL string, it never opens a connection.
"""
import ast
import re
from pathlib import Path

from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

import form4lab.models  # noqa: F401 — registers every table on Base.metadata
from form4lab.database import Base

MIGRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "alembic"
    / "versions"
    / "0001_initial_schema.py"
)

# Matches "BOOLEAN DEFAULT 0" / "BOOLEAN NOT NULL DEFAULT 1", but not
# "BOOLEAN DEFAULT false" or "BOOLEAN DEFAULT 'f'".
_BARE_INT_BOOL_DEFAULT = re.compile(r"BOOLEAN[^,)]*DEFAULT\s+[01]\b", re.IGNORECASE)


def test_model_ddl_has_no_bare_integer_boolean_defaults():
    """Every ORM-mapped table must compile to Postgres-valid DDL."""
    offenders = []
    for table in Base.metadata.tables.values():
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        if _BARE_INT_BOOL_DEFAULT.search(ddl):
            offenders.append(table.name)

    assert not offenders, (
        "Tables whose compiled Postgres DDL has a bare-integer BOOLEAN "
        f"DEFAULT (invalid on Postgres): {offenders}"
    )


def _boolean_columns_with_bare_int_text_default(source: str) -> list[str]:
    """Column names where an `sa.Column(..., sa.Boolean(), ...)` call pairs
    `server_default=sa.text('0')` or `sa.text('1')` — the bare-integer-text
    pattern Postgres rejects for a Boolean column. A plain string default
    (`server_default='0'`) is untouched: SQLAlchemy renders that as a quoted
    literal (`DEFAULT '0'`), which Postgres casts fine via the boolean input
    function, so it isn't part of this bug class.
    """
    tree = ast.parse(source)
    offenders: list[str] = []

    for node in ast.walk(tree):
        is_column_call = (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "Column"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "sa"
        )
        if not is_column_call:
            continue

        col_name = (
            node.args[0].value
            if node.args and isinstance(node.args[0], ast.Constant)
            else "<unknown>"
        )

        is_boolean_type = any(
            isinstance(arg, ast.Call)
            and isinstance(arg.func, ast.Attribute)
            and arg.func.attr == "Boolean"
            for arg in node.args
        )
        if not is_boolean_type:
            continue

        for kw in node.keywords:
            if kw.arg != "server_default":
                continue
            val = kw.value
            is_bare_int_text = (
                isinstance(val, ast.Call)
                and isinstance(val.func, ast.Attribute)
                and val.func.attr == "text"
                and val.args
                and isinstance(val.args[0], ast.Constant)
                and val.args[0].value in ("0", "1")
            )
            if is_bare_int_text:
                offenders.append(col_name)

    return offenders


def test_migration_source_has_no_bare_integer_boolean_defaults():
    """Regression pin: the frozen 0001 migration must not reintroduce a
    Boolean column with a bare-integer `sa.text('0'/'1')` server_default —
    the exact bug that broke `CREATE TABLE broker_positions` on Postgres.
    Use `sa.false()` / `sa.true()` instead (dialect-aware: SQLite gets 0/1,
    Postgres gets false/true).
    """
    source = MIGRATION_PATH.read_text()
    offenders = _boolean_columns_with_bare_int_text_default(source)

    assert not offenders, (
        f"Boolean column(s) in {MIGRATION_PATH.name} using the "
        f"Postgres-invalid server_default=sa.text('0'/'1'): {offenders}. "
        "Use sa.false()/sa.true() instead."
    )
