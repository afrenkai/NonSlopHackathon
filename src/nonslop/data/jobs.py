from collections.abc import Callable
from pathlib import Path
from threading import RLock

import duckdb

from nonslop.generation.schemas import JobRecord


class JobStore:
    def __init__(self, root: Path, commit: Callable[[], None]) -> None:
        self.root = root
        self.database = root / "scenes.duckdb"
        self.commit = commit
        self.lock = RLock()
        (root / "uploads").mkdir(parents=True, exist_ok=True)
        with self.lock, duckdb.connect(str(self.database)) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS jobs "
                "(job_id UUID PRIMARY KEY, created_at TIMESTAMPTZ, status VARCHAR, payload JSON)"
            )
            connection.execute(
                "ALTER TABLE jobs ADD COLUMN IF NOT EXISTS owner_id VARCHAR DEFAULT ''"
            )
        self.commit()

    def save(self, job: JobRecord, reference_images: list[bytes] | None = None) -> None:
        with self.lock:
            if reference_images:
                (self.root / "uploads" / f"{job.job_id}.png").write_bytes(
                    reference_images[0]
                )
                folder = self.root / "uploads" / job.job_id
                folder.mkdir(exist_ok=True)
                for index, content in enumerate(reference_images):
                    (folder / f"{index}.png").write_bytes(content)
            with duckdb.connect(str(self.database)) as connection:
                connection.execute(
                    "INSERT INTO jobs (job_id, created_at, status, payload, owner_id) VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT (job_id) DO UPDATE SET status=excluded.status, payload=excluded.payload, owner_id=excluded.owner_id",
                    [
                        job.job_id,
                        job.created_at,
                        job.status,
                        job.model_dump_json(),
                        job.owner_id,
                    ],
                )
            self.commit()

    def reserve(self, job: JobRecord, reference_images: list[bytes]) -> bool:
        with self.lock:
            if self.active_count() >= 4:
                return False
            self.save(job, reference_images)
            return True

    def get(self, job_id: str) -> JobRecord | None:
        with self.lock, duckdb.connect(str(self.database)) as connection:
            row = connection.execute(
                "SELECT payload FROM jobs WHERE job_id = ?", [job_id]
            ).fetchone()
        return None if row is None else JobRecord.model_validate_json(row[0])

    def recent(self, owner_id: str | None = None) -> list[JobRecord]:
        with self.lock, duckdb.connect(str(self.database)) as connection:
            if owner_id is None:
                rows = connection.execute(
                    "SELECT payload FROM jobs ORDER BY created_at DESC LIMIT 12"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT payload FROM jobs WHERE owner_id = ? ORDER BY created_at DESC LIMIT 12",
                    [owner_id],
                ).fetchall()
        return [JobRecord.model_validate_json(row[0]) for row in rows]

    def active_count(self) -> int:
        with self.lock, duckdb.connect(str(self.database)) as connection:
            row = connection.execute(
                "SELECT count(*) FROM jobs WHERE status IN ('queued', 'running')"
            ).fetchone()
        assert row is not None
        return int(row[0])

    def active_world_count(self) -> int:
        with self.lock, duckdb.connect(str(self.database)) as connection:
            row = connection.execute(
                "SELECT count(*) FROM jobs WHERE json_extract_string(payload, '$.world.status') = 'running'"
            ).fetchone()
        assert row is not None
        return int(row[0])
