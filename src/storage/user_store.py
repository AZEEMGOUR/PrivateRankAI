import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.storage.document_store import (
    DEFAULT_DATABASE_PATH,
)


class DuplicateEmailError(ValueError):
    pass


class LastActiveAdminError(ValueError):
    pass


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

        try:
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
        except sqlite3.IntegrityError as error:
            raise DuplicateEmailError(
                "A user with this email already exists."
            ) from error

        return self._row_to_user(row)

    def list_users(self):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM users
                ORDER BY full_name COLLATE NOCASE,
                    email COLLATE NOCASE
                """
            ).fetchall()

        return [
            self._row_to_user(row)
            for row in rows
        ]

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

    def update_user(
        self,
        user_id,
        *,
        full_name=None,
        role=None,
        is_active=None,
        protected_role=None,
    ):
        updates = {}

        if full_name is not None:
            full_name = str(
                full_name
            ).strip()

            if not full_name:
                raise ValueError(
                    "Full name cannot be empty."
                )

            updates["full_name"] = full_name

        if role is not None:
            role = str(role).strip()

            if not role:
                raise ValueError(
                    "Role cannot be empty."
                )

            updates["role"] = role

        if is_active is not None:
            updates["is_active"] = int(
                bool(is_active)
            )

        with self._connect() as connection:
            connection.execute(
                "BEGIN IMMEDIATE"
            )

            current_row = connection.execute(
                """
                SELECT *
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

            if current_row is None:
                return None

            current_user = dict(current_row)

            if protected_role:
                current_is_protected = (
                    bool(current_user["is_active"])
                    and str(
                        current_user["role"]
                    ).casefold()
                    == str(
                        protected_role
                    ).casefold()
                )
                updated_role = updates.get(
                    "role",
                    current_user["role"],
                )
                updated_is_active = bool(
                    updates.get(
                        "is_active",
                        current_user["is_active"],
                    )
                )
                remains_protected = (
                    updated_is_active
                    and str(
                        updated_role
                    ).casefold()
                    == str(
                        protected_role
                    ).casefold()
                )

                if (
                    current_is_protected
                    and not remains_protected
                ):
                    active_count = (
                        connection.execute(
                            """
                            SELECT COUNT(*)
                            FROM users
                            WHERE is_active = 1
                                AND role = ? COLLATE NOCASE
                            """,
                            (protected_role,),
                        ).fetchone()[0]
                    )

                    if active_count <= 1:
                        raise LastActiveAdminError(
                            "The last active administrator "
                            "cannot be deactivated or demoted."
                        )

            if updates:
                updates["updated_at"] = (
                    self._utc_now()
                )
                assignments = ", ".join(
                    f"{column} = ?"
                    for column in updates
                )
                values = [
                    *updates.values(),
                    user_id,
                ]

                connection.execute(
                    f"""
                    UPDATE users
                    SET {assignments}
                    WHERE id = ?
                    """,
                    values,
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

    def update_password(
        self,
        user_id,
        password_hash,
    ):
        password_hash = str(
            password_hash or ""
        ).strip()

        if not password_hash:
            raise ValueError(
                "Password hash is required."
            )

        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE users
                SET password_hash = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    password_hash,
                    self._utc_now(),
                    user_id,
                ),
            )

            if cursor.rowcount == 0:
                return None

            row = connection.execute(
                """
                SELECT *
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()

        return self._row_to_user(row)

    def count_active_users_by_role(self, role):
        with self._connect() as connection:
            count = connection.execute(
                """
                SELECT COUNT(*)
                FROM users
                WHERE is_active = 1
                    AND role = ? COLLATE NOCASE
                """,
                (str(role or "").strip(),),
            ).fetchone()[0]

        return count

    def set_user_active(
        self,
        user_id,
        is_active,
    ):
        return self.update_user(
            user_id,
            is_active=is_active,
        )
