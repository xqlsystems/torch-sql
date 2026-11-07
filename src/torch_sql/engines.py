"""Running programs until the runtime exists (design §12.1).

An :class:`Engine` registers tables, runs queries, saves a query's rows as a
table, and runs ddx's gradient programs. Two engines implement it:
:class:`DataFusionEngine` and :class:`DuckDBEngine`. Both run ddx's programs
through ddxdb's public ``Backend`` implementations, so the same program runs
on either.

Every engine object owns its own session (a DataFusion ``SessionContext``
or a DuckDB connection), so two engines never see each other's tables.
"""

from __future__ import annotations

from dataclasses import dataclass

import ddxdb
import pyarrow as pa


@dataclass
class Gradients:
    """What a ddx gradient program left on the engine: the loss's value, and
    one table per ``wrt`` column, in order, each with the table's key
    columns and the gradient under ``v``. The caller drops them
    (:meth:`Engine.drop`) once it has read them."""

    value: str
    gradients: list[str]

    def tables(self) -> list[str]:
        return [self.value, *self.gradients]


class Engine:
    """The operations torch-sql needs from an engine."""

    name: str

    def register(self, name: str, table: pa.Table) -> None:
        """Register ``table`` as ``name``, replacing any table of that name."""
        raise NotImplementedError

    def drop(self, name: str) -> None:
        """Drop the table ``name``, if there is one."""
        raise NotImplementedError

    def query(self, sql: str) -> pa.Table:
        """Run ``sql`` and return its rows."""
        raise NotImplementedError

    def materialize(self, name: str, sql: str) -> None:
        """Run ``sql`` and save its rows as the table ``name``."""
        raise NotImplementedError

    def backend(self) -> ddxdb.Backend:
        """ddxdb's backend for this engine's session."""
        raise NotImplementedError

    def grad(self, loss: str, wrt: list[str]) -> Gradients:
        """Run ddx's gradient program for ``loss``, SQL that computes one row
        with one column, with respect to each ``"table.column"`` in ``wrt``:
        one program, with one backward pass for all of them (§11.1)."""
        backend = self.backend()
        program = ddxdb.grad_plan(backend.plan_sql(loss), wrt)
        ddxdb.run(backend, program)
        return Gradients(program.value.step, [g.step for g in program.gradients])


class DataFusionEngine(Engine):
    """Apache DataFusion, a query engine in Rust, through datafusion-python."""

    name = "datafusion"

    def __init__(self):
        from datafusion import SessionConfig, SessionContext
        from ddxdb.ad import DataFusionBackend, register_stop_gradient

        # ddx reads views through the information schema.
        self.ctx = SessionContext(SessionConfig().with_information_schema(True))
        register_stop_gradient(self.ctx)
        self._backend = DataFusionBackend(self.ctx)

    def register(self, name, table):
        self.ctx.deregister_table(name)
        batches = table.to_batches() or [pa.RecordBatch.from_pylist([], schema=table.schema)]
        self.ctx.register_record_batches(name, [batches])

    def drop(self, name):
        self.ctx.deregister_table(name)

    def query(self, sql):
        return self.ctx.sql(sql).to_arrow_table()

    def materialize(self, name, sql):
        self.register(name, self.query(sql))

    def backend(self):
        return self._backend


class DuckDBEngine(Engine):
    """DuckDB, an embedded analytical database. ddx plans on it through
    DuckDB's community Substrait extension, which exists for DuckDB 1.5
    (design §12.3)."""

    name = "duckdb"

    def __init__(self):
        import duckdb
        from ddxdb.duckdb import DuckDBBackend

        self.con = duckdb.connect()
        self._backend = DuckDBBackend(self.con)
        # Registered Arrow tables, kept alive as long as they're registered.
        self._arrow: dict[str, pa.Table] = {}

    def register(self, name, table):
        self.drop(name)
        self._arrow[name] = table
        self.con.register(name, table)

    def drop(self, name):
        if self._arrow.pop(name, None) is not None:
            self.con.unregister(name)
        self.con.execute(f"DROP TABLE IF EXISTS {name}")

    def query(self, sql):
        return self.con.sql(sql).to_arrow_table()

    def materialize(self, name, sql):
        self.drop(name)
        self.con.execute(f"CREATE TEMP TABLE {name} AS {sql}")

    def backend(self):
        return self._backend


ENGINES = {"datafusion": DataFusionEngine, "duckdb": DuckDBEngine}


def engine(name: str) -> Engine:
    """A new engine of the kind ``name``: ``"datafusion"`` or ``"duckdb"``."""
    try:
        return ENGINES[name]()
    except KeyError:
        raise ValueError(f"unknown engine {name!r}; torch-sql runs on {sorted(ENGINES)}") from None
