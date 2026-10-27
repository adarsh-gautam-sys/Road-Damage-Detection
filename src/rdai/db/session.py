"""SQLAlchemy 2.0 engine and session factory with SQLite concurrency tuning (`A09`)."""

from __future__ import annotations

from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from rdai.db.models import Base


def create_db_engine_and_session(database_url: str) -> tuple[Engine, sessionmaker[Session]]:
    """Create a SQLAlchemy 2.0 Engine and Session factory, initializing tables automatically."""
    connect_args: dict[str, Any] = {}
    if database_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": 30.0}

    engine = create_engine(database_url, connect_args=connect_args, future=True)

    if database_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=15000;")
            cursor.close()

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, expire_on_commit=False
    )
    return engine, session_factory
