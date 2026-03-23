"""Postgres event store placeholder.

Implemented as a soft dependency hook to keep current install lightweight.
"""

from __future__ import annotations


class PostgresEventStore:
    def __init__(self, dsn: str) -> None:
        raise NotImplementedError(
            "PostgresEventStore is not yet implemented in this iteration. Use sqlite backend for now."
        )
