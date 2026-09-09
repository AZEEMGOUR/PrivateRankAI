import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.storage.document_store import (
    DEFAULT_DATABASE_PATH,
)


class UserStore:

    def __init__(self, database_path=DEFAULT_DATABASE_PATH):
        self.database_path = Path(database_path)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(
            self.database_path,
            timeout=30,
        )
        connection.row_factory = sqlite3.Row

        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _utc_now():
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _normalize_email(email):
        return str(email or "").strip().casefold()

    @staticmethod
    def _row_to_user(
        row,
        *,
        include_password_hash=False,
    ):
        if row is None:
            return None

        user = dict(row)
        user["is_active"] = bool(
            user["is_active"]
        )

        if not include_password_hash:
            user.pop(
                "password_hash",
                None,
            )

        return user

    def initialize_database(self):
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                        CHECK (is_active IN (0, 1)),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )

    def create_user(
        self,
        *,
        email,
        full_name,
        password_hash,
        role,
        is_active=True,
    ):
        email = self._normalize_email(email)
        full_name = str(full_name or "").strip()
        password_hash = str(
            password_hash or ""
        ).strip()
        role = str(role or "").strip()

        if not email:
            raise ValueError(
                "Email is required."
            )

        if not full_name:
            raise ValueError(
                "Full name is required."
            )

        if not password_hash:
            raise ValueError(
                "Password hash is required."
            )

        if not role:
            raise ValueError(
                "Role is required."
            )

        user_id = str(uuid4())
        timestamp = self._utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    id,
                    email,
                    full_name,
                    password_hash,
                    role,
                    is_active,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    email,
                    full_name,
                    password_hash,
                    role,
                    int(bool(is_active)),
                    timestamp,
                    timestamp,
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

        return self._row_to_user(row)

    def get_user_by_id(self, user_id):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

        return self._row_to_user(row)

    def get_user_by_email(self, email):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM users
                WHERE email = ? COLLATE NOCASE
                """,
                (
                    self._normalize_email(email),
                ),
            ).fetchone()

        return self._row_to_user(row)

    def get_user_credentials_by_email(self, email):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM users
                WHERE email = ? COLLATE NOCASE
                """,
                (
                    self._normalize_email(email),
                ),
            ).fetchone()

        return self._row_to_user(
            row,
            include_password_hash=True,
        )

    def set_user_active(
        self,
        user_id,
        is_active,
    ):
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET is_active = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    int(bool(is_active)),
                    self._utc_now(),
                    user_id,
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

        return self._row_to_user(row)
