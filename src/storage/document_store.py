import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "private_rank.db"


class DocumentStore:

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
    def _serialize_tags(tags):
        normalized = []
        seen = set()

        for value in tags or []:
            tag = str(value).strip()
            key = tag.casefold()

            if tag and key not in seen:
                normalized.append(tag)
                seen.add(key)

        return json.dumps(
            normalized,
            ensure_ascii=False,
        )

    @staticmethod
    def _row_to_document(row):
        if row is None:
            return None

        document = dict(row)

        try:
            document["tags"] = json.loads(
                document["tags"]
            )
        except (TypeError, json.JSONDecodeError):
            document["tags"] = []

        return document

    def initialize_database(self):
        self.database_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    department TEXT NOT NULL,
                    category TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    tags TEXT NOT NULL DEFAULT '[]',
                    uploaded_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )

    def create_document(
        self,
        *,
        filename,
        display_name,
        department="General",
        category="Other",
        description="",
        tags=None,
        status="indexed",
    ):
        document_id = str(uuid4())
        timestamp = self._utc_now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    id,
                    filename,
                    display_name,
                    department,
                    category,
                    description,
                    tags,
                    uploaded_at,
                    updated_at,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    str(filename).strip(),
                    str(display_name).strip(),
                    str(department).strip(),
                    str(category).strip(),
                    str(description).strip(),
                    self._serialize_tags(tags),
                    timestamp,
                    timestamp,
                    str(status).strip(),
                ),
            )

            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()

        return self._row_to_document(row)

    def get_document_by_id(self, document_id):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()

        return self._row_to_document(row)

    def get_document_by_filename(self, filename):
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE filename = ? COLLATE NOCASE
                """,
                (filename,),
            ).fetchone()

        return self._row_to_document(row)

    def list_documents(self):
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM documents
                ORDER BY uploaded_at DESC, filename COLLATE NOCASE
                """
            ).fetchall()

        return [
            self._row_to_document(row)
            for row in rows
        ]

    def list_filenames_by_metadata(
        self,
        *,
        department=None,
        category=None,
    ):
        conditions = []
        values = []

        department = str(
            department or ""
        ).strip()
        category = str(
            category or ""
        ).strip()

        if department:
            conditions.append(
                "department = ? COLLATE NOCASE"
            )
            values.append(department)

        if category:
            conditions.append(
                "category = ? COLLATE NOCASE"
            )
            values.append(category)

        query = "SELECT filename FROM documents"

        if conditions:
            query += " WHERE " + " AND ".join(
                conditions
            )

        query += " ORDER BY filename COLLATE NOCASE"

        with self._connect() as connection:
            rows = connection.execute(
                query,
                values,
            ).fetchall()

        return [
            row["filename"]
            for row in rows
        ]

    def update_document_metadata(
        self,
        document_id,
        *,
        display_name=None,
        department=None,
        category=None,
        description=None,
        tags=None,
        status=None,
    ):
        updates = {}

        if display_name is not None:
            updates["display_name"] = str(
                display_name
            ).strip()

        if department is not None:
            updates["department"] = str(
                department
            ).strip()

        if category is not None:
            updates["category"] = str(
                category
            ).strip()

        if description is not None:
            updates["description"] = str(
                description
            ).strip()

        if tags is not None:
            updates["tags"] = self._serialize_tags(
                tags
            )

        if status is not None:
            updates["status"] = str(
                status
            ).strip()

        if not updates:
            return self.get_document_by_id(
                document_id
            )

        updates["updated_at"] = self._utc_now()

        assignments = ", ".join(
            f"{column} = ?"
            for column in updates
        )

        values = [
            *updates.values(),
            document_id,
        ]

        with self._connect() as connection:
            connection.execute(
                f"""
                UPDATE documents
                SET {assignments}
                WHERE id = ?
                """,
                values,
            )

            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()

        return self._row_to_document(row)

    def delete_document_metadata(self, document_id):
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM documents
                WHERE id = ?
                """,
                (document_id,),
            )

        return cursor.rowcount > 0

    def bootstrap_existing_pdfs(self, documents_directory):
        self.initialize_database()

        documents_directory = Path(
            documents_directory
        )

        if not documents_directory.exists():
            return {
                "scanned": 0,
                "created": 0,
                "existing": 0,
            }

        pdf_files = sorted(
            path
            for path in documents_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".pdf"
        )

        created = 0
        timestamp = self._utc_now()

        with self._connect() as connection:
            for pdf_file in pdf_files:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO documents (
                        id,
                        filename,
                        display_name,
                        department,
                        category,
                        description,
                        tags,
                        uploaded_at,
                        updated_at,
                        status
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid4()),
                        pdf_file.name,
                        pdf_file.stem,
                        "General",
                        "Other",
                        "",
                        self._serialize_tags([]),
                        timestamp,
                        timestamp,
                        "indexed",
                    ),
                )

                created += cursor.rowcount

        return {
            "scanned": len(pdf_files),
            "created": created,
            "existing": len(pdf_files) - created,
        }
