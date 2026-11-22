from pathlib import Path
from uuid import UUID

import duckdb

from nonslop.capsules.schemas import CapsuleEntry, CapsuleRecord, EntryEnrichment
from nonslop.data.jobs import JobStore


class CapsuleStore:
    def __init__(self, jobs: JobStore) -> None:
        self.database = jobs.database
        self.lock = jobs.lock
        self.commit = jobs.commit
        self.root = jobs.root / "capsules"
        with self.lock:
            self.root.mkdir(parents=True, exist_ok=True)
            with duckdb.connect(str(self.database)) as connection:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS capsules "
                    "(capsule_id UUID PRIMARY KEY, owner_id VARCHAR, "
                    "created_at TIMESTAMPTZ, payload JSON)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS capsule_entries "
                    "(entry_id UUID PRIMARY KEY, capsule_id UUID, owner_id VARCHAR, "
                    "created_at TIMESTAMPTZ, occurred_on DATE, payload JSON)"
                )
            self.commit()

    def media_path(self, capsule_id: UUID, filename: str) -> Path:
        media_name = Path(filename)
        if media_name.name != filename or filename in {".", ".."}:
            raise ValueError("Capsule media must use a UUID filename")
        UUID(media_name.stem)
        return self.root / str(capsule_id) / filename

    def create(self, capsule: CapsuleRecord, anchor_bytes: bytes | None = None) -> None:
        if anchor_bytes is not None and capsule.anchor_filename is None:
            raise ValueError("Anchor media requires a filename")
        with self.lock:
            with duckdb.connect(str(self.database)) as connection:
                connection.execute("BEGIN TRANSACTION")
                connection.execute(
                    "INSERT INTO capsules (capsule_id, owner_id, created_at, payload) "
                    "VALUES (?, ?, ?, ?)",
                    [
                        capsule.capsule_id,
                        capsule.owner_id,
                        capsule.created_at,
                        capsule.model_dump_json(),
                    ],
                )
                if anchor_bytes is not None and capsule.anchor_filename is not None:
                    destination = self.media_path(
                        capsule.capsule_id, capsule.anchor_filename
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as media:
                        media.write(anchor_bytes)
                connection.execute("COMMIT")
            self.commit()

    def get(self, capsule_id: UUID, owner_id: str) -> CapsuleRecord | None:
        with self.lock, duckdb.connect(str(self.database)) as connection:
            row = connection.execute(
                "SELECT payload FROM capsules WHERE capsule_id = ? AND owner_id = ?",
                [capsule_id, owner_id],
            ).fetchone()
        return None if row is None else CapsuleRecord.model_validate_json(row[0])

    def recent(self, owner_id: str) -> list[CapsuleRecord]:
        with self.lock, duckdb.connect(str(self.database)) as connection:
            rows = connection.execute(
                "SELECT payload FROM capsules WHERE owner_id = ? "
                "ORDER BY created_at DESC LIMIT 30",
                [owner_id],
            ).fetchall()
        return [CapsuleRecord.model_validate_json(row[0]) for row in rows]

    def append(self, entry: CapsuleEntry, media_bytes: bytes | None = None) -> None:
        if media_bytes is not None and entry.media_filename is None:
            raise ValueError("Entry media requires a filename")
        with self.lock:
            if self.get(entry.capsule_id, entry.owner_id) is None:
                raise ValueError("Capsule not found in this workspace")
            with duckdb.connect(str(self.database)) as connection:
                connection.execute("BEGIN TRANSACTION")
                connection.execute(
                    "INSERT INTO capsule_entries "
                    "(entry_id, capsule_id, owner_id, created_at, occurred_on, payload) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    [
                        entry.entry_id,
                        entry.capsule_id,
                        entry.owner_id,
                        entry.created_at,
                        entry.occurred_on,
                        entry.model_dump_json(),
                    ],
                )
                if media_bytes is not None and entry.media_filename is not None:
                    destination = self.media_path(
                        entry.capsule_id, entry.media_filename
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open("xb") as media:
                        media.write(media_bytes)
                connection.execute("COMMIT")
            self.commit()

    def get_entry(self, entry_id: UUID, owner_id: str) -> CapsuleEntry | None:
        with self.lock, duckdb.connect(str(self.database)) as connection:
            row = connection.execute(
                "SELECT payload FROM capsule_entries WHERE entry_id = ? AND owner_id = ?",
                [entry_id, owner_id],
            ).fetchone()
        return None if row is None else CapsuleEntry.model_validate_json(row[0])

    def entries(self, capsule_id: UUID, owner_id: str) -> list[CapsuleEntry]:
        with self.lock:
            if self.get(capsule_id, owner_id) is None:
                raise ValueError("Capsule not found in this workspace")
            with duckdb.connect(str(self.database)) as connection:
                rows = connection.execute(
                    "SELECT payload FROM capsule_entries "
                    "WHERE capsule_id = ? AND owner_id = ? "
                    "ORDER BY occurred_on, created_at",
                    [capsule_id, owner_id],
                ).fetchall()
        return [CapsuleEntry.model_validate_json(row[0]) for row in rows]

    def update_enrichment(
        self, entry_id: UUID, owner_id: str, enrichment: EntryEnrichment
    ) -> CapsuleEntry:
        with self.lock:
            entry = self.get_entry(entry_id, owner_id)
            if entry is None:
                raise ValueError("Capsule entry not found in this workspace")
            updated = entry.model_copy(update={"enrichment": enrichment})
            with duckdb.connect(str(self.database)) as connection:
                connection.execute(
                    "UPDATE capsule_entries SET payload = ? "
                    "WHERE entry_id = ? AND owner_id = ?",
                    [updated.model_dump_json(), entry_id, owner_id],
                )
            self.commit()
        return updated

    def linked_entries(self, job_id: UUID, owner_id: str) -> list[CapsuleEntry]:
        with self.lock, duckdb.connect(str(self.database)) as connection:
            rows = connection.execute(
                "SELECT payload FROM capsule_entries WHERE owner_id = ? "
                "AND json_extract_string(payload, '$.linked_job_id') = ? ORDER BY occurred_on, created_at",
                [owner_id, str(job_id)],
            ).fetchall()
        return [CapsuleEntry.model_validate_json(row[0]) for row in rows]
