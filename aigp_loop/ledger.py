"""SQLite trial ledger with idempotence, leases, and tier checkpoints."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Sequence

from ._util import canonical_json, json_hash, sha256_file, sha256_text


SCHEMA_VERSION = 2
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
ALL_STATUSES = frozenset({"pending", "running", *TERMINAL_STATUSES})


def _utc_now(epoch_s: Optional[float] = None) -> str:
    stamp = time.time() if epoch_s is None else float(epoch_s)
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat(
        timespec="microseconds"
    )


@dataclass(frozen=True)
class TrialKey:
    code_hash: str
    config_hash: str
    dataset_hash: str
    seed: int
    evaluator_version: str

    def validate(self) -> None:
        for name in ("code_hash", "config_hash", "dataset_hash", "evaluator_version"):
            value = getattr(self, name)
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be an exact non-empty string")
        if type(self.seed) is not int:
            raise ValueError("seed must be an exact integer, not bool/string/float")


class TrialLedger:
    """Durable source of truth for autonomous evaluation work.

    Connections are short lived and every claim/lease transition uses
    ``BEGIN IMMEDIATE``.  WAL mode permits read-only status tools while one
    scheduler is recording a checkpoint.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path), timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    parent_trial_id TEXT REFERENCES trials(trial_id),
                    candidate_name TEXT,
                    status TEXT NOT NULL CHECK (
                        status IN ('pending','running','completed','failed','cancelled')
                    ),
                    lease_owner TEXT,
                    lease_expires_at REAL,
                    heartbeat TEXT,
                    started_at TEXT,
                    finished_at TEXT,
                    commit_hash TEXT NOT NULL,
                    dirty_diff_hash TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    resolved_config TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    dataset_hash TEXT NOT NULL,
                    artifact_hashes TEXT NOT NULL DEFAULT '{}',
                    simulator_build TEXT,
                    evaluator_version TEXT NOT NULL,
                    environment_fingerprint TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    phase_timings TEXT NOT NULL DEFAULT '{}',
                    safety_and_completion_metrics TEXT NOT NULL DEFAULT '{}',
                    failure_reason TEXT,
                    stdout_stderr_tail TEXT,
                    worktree_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(code_hash, config_hash, dataset_hash, seed, evaluator_version)
                );
                CREATE INDEX IF NOT EXISTS idx_trials_claim
                    ON trials(status, lease_expires_at, created_at);
                CREATE TABLE IF NOT EXISTS checkpoints (
                    trial_id TEXT NOT NULL REFERENCES trials(trial_id) ON DELETE CASCADE,
                    tier INTEGER NOT NULL CHECK(tier BETWEEN 0 AND 5),
                    status TEXT NOT NULL CHECK(status IN ('running','completed','failed')),
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    metrics TEXT NOT NULL DEFAULT '{}',
                    artifact_hashes TEXT NOT NULL DEFAULT '{}',
                    stdout_stderr_tail TEXT,
                    elapsed_s REAL,
                    PRIMARY KEY(trial_id, tier)
                );
                CREATE TABLE IF NOT EXISTS leases (
                    name TEXT PRIMARY KEY,
                    owner TEXT NOT NULL,
                    heartbeat TEXT NOT NULL,
                    expires_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS imports (
                    source_hash TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    row_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS promotion_rounds (
                    round_id TEXT PRIMARY KEY,
                    tier INTEGER NOT NULL CHECK(tier BETWEEN 0 AND 4),
                    status TEXT NOT NULL CHECK(status IN ('planned','decided','applied')),
                    member_trial_ids TEXT NOT NULL,
                    decision TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_promotion_round_open
                    ON promotion_rounds(tier,status,created_at);
                """
            )
            db.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )
            actual_version = db.execute(
                "SELECT value FROM metadata WHERE key='schema_version'"
            ).fetchone()[0]
            if actual_version == "1":
                columns = {
                    str(row[1]) for row in db.execute("PRAGMA table_info(checkpoints)")
                }
                if "elapsed_s" not in columns:
                    db.execute("ALTER TABLE checkpoints ADD COLUMN elapsed_s REAL")
                db.execute(
                    "UPDATE metadata SET value=? WHERE key='schema_version'",
                    (str(SCHEMA_VERSION),),
                )
                actual_version = str(SCHEMA_VERSION)
            if actual_version != str(SCHEMA_VERSION):
                raise RuntimeError(
                    f"unsupported trial ledger schema {actual_version}; "
                    f"this code requires {SCHEMA_VERSION}"
                )

    def create_or_get_trial(
        self,
        *,
        key: TrialKey,
        commit_hash: str,
        dirty_diff_hash: str,
        resolved_config: Mapping[str, Any],
        environment_fingerprint: str,
        parent_trial_id: Optional[str] = None,
        candidate_name: Optional[str] = None,
        simulator_build: Optional[str] = None,
        artifact_hashes: Optional[Mapping[str, Any]] = None,
        trial_id: Optional[str] = None,
    ) -> tuple[str, bool]:
        """Return ``(trial_id, created)`` for the exact deduplication key."""

        key.validate()
        if not isinstance(resolved_config, Mapping):
            raise TypeError("resolved_config must be an object")
        if json_hash(resolved_config) != key.config_hash:
            raise ValueError("config_hash does not match resolved_config")
        for name, value in (
            ("commit_hash", commit_hash),
            ("dirty_diff_hash", dirty_diff_hash),
            ("environment_fingerprint", environment_fingerprint),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError(f"{name} must be an exact non-empty string")
        for name, value in (
            ("parent_trial_id", parent_trial_id),
            ("candidate_name", candidate_name),
            ("simulator_build", simulator_build),
            ("trial_id", trial_id),
        ):
            if value is not None and (type(value) is not str or not value.strip()):
                raise ValueError(f"{name} must be null or an exact non-empty string")
        if artifact_hashes is not None and not isinstance(artifact_hashes, Mapping):
            raise TypeError("artifact_hashes must be an object")
        resolved_config_json = canonical_json(resolved_config)
        now = _utc_now()
        identifier = trial_id or uuid.uuid4().hex
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                """SELECT trial_id,parent_trial_id,candidate_name,commit_hash,
                          dirty_diff_hash,resolved_config,simulator_build,
                          environment_fingerprint
                   FROM trials WHERE code_hash=? AND config_hash=?
                   AND dataset_hash=? AND seed=? AND evaluator_version=?""",
                (
                    key.code_hash,
                    key.config_hash,
                    key.dataset_hash,
                    int(key.seed),
                    key.evaluator_version,
                ),
            ).fetchone()
            if existing is not None:
                expected = {
                    "parent_trial_id": parent_trial_id,
                    "commit_hash": commit_hash,
                    "dirty_diff_hash": dirty_diff_hash,
                    "resolved_config": resolved_config_json,
                    "simulator_build": simulator_build,
                    "environment_fingerprint": environment_fingerprint,
                }
                conflicts = [
                    name for name, value in expected.items() if existing[name] != value
                ]
                if (
                    candidate_name is not None
                    and existing["candidate_name"] != candidate_name
                ):
                    conflicts.append("candidate_name")
                if trial_id is not None and existing["trial_id"] != trial_id:
                    conflicts.append("trial_id")
                if conflicts:
                    db.rollback()
                    raise ValueError(
                        "existing trial metadata conflicts with the five-part key: "
                        + ", ".join(sorted(conflicts))
                    )
                db.commit()
                return str(existing["trial_id"]), False
            db.execute(
                """INSERT INTO trials(
                    trial_id,parent_trial_id,candidate_name,status,
                    commit_hash,dirty_diff_hash,code_hash,resolved_config,
                    config_hash,dataset_hash,artifact_hashes,simulator_build,
                    evaluator_version,environment_fingerprint,seed,created_at,updated_at
                ) VALUES(?,?,?,'pending',?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    identifier,
                    parent_trial_id,
                    candidate_name,
                    commit_hash,
                    dirty_diff_hash,
                    key.code_hash,
                    resolved_config_json,
                    key.config_hash,
                    key.dataset_hash,
                    canonical_json(artifact_hashes or {}),
                    simulator_build,
                    key.evaluator_version,
                    environment_fingerprint,
                    int(key.seed),
                    now,
                    now,
                ),
            )
            db.commit()
        return identifier, True

    def get_trial(self, trial_id: str) -> Dict[str, Any]:
        with self._connect() as db:
            row = db.execute("SELECT * FROM trials WHERE trial_id=?", (trial_id,)).fetchone()
        if row is None:
            raise KeyError(trial_id)
        result = dict(row)
        for field in (
            "resolved_config",
            "artifact_hashes",
            "phase_timings",
            "safety_and_completion_metrics",
        ):
            result[field] = json.loads(result[field])
        return result

    def list_trials(self, *, status: Optional[str] = None) -> list[Dict[str, Any]]:
        if status is not None and status not in ALL_STATUSES:
            raise ValueError(f"unknown status: {status}")
        query = "SELECT trial_id FROM trials"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY created_at, trial_id"
        with self._connect() as db:
            identifiers = [str(row[0]) for row in db.execute(query, params)]
        return [self.get_trial(identifier) for identifier in identifiers]

    def lease_trial(
        self,
        trial_id: str,
        owner: str,
        *,
        ttl_s: float = 60.0,
        now_s: Optional[float] = None,
    ) -> bool:
        if (
            type(owner) is not str
            or not owner.strip()
            or type(ttl_s) not in {int, float}
            or not math.isfinite(ttl_s)
            or ttl_s <= 0
        ):
            raise ValueError("owner must be non-empty and ttl_s finite and positive")
        if now_s is not None and (
            type(now_s) not in {int, float} or not math.isfinite(now_s)
        ):
            raise ValueError("now_s must be finite numeric evidence")
        epoch = time.time() if now_s is None else float(now_s)
        heartbeat = _utc_now(epoch)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT status,lease_owner,lease_expires_at FROM trials WHERE trial_id=?",
                (trial_id,),
            ).fetchone()
            if row is None:
                raise KeyError(trial_id)
            available = row["status"] == "pending" or (
                row["status"] == "running"
                and (
                    row["lease_owner"] == owner
                    or row["lease_expires_at"] is None
                    or float(row["lease_expires_at"]) <= epoch
                )
            )
            if not available:
                db.rollback()
                return False
            db.execute(
                """UPDATE trials SET status='running', lease_owner=?,
                   lease_expires_at=?, heartbeat=?, started_at=COALESCE(started_at,?),
                   updated_at=? WHERE trial_id=?""",
                (owner, epoch + ttl_s, heartbeat, heartbeat, heartbeat, trial_id),
            )
            db.commit()
        return True

    def claim_next(
        self,
        owner: str,
        *,
        ttl_s: float = 60.0,
        now_s: Optional[float] = None,
        through: int = 5,
    ) -> Optional[str]:
        if type(owner) is not str or not owner.strip():
            raise ValueError("owner must be a non-empty string")
        if type(ttl_s) not in {int, float} or not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("ttl_s must be finite and positive")
        if now_s is not None and (
            type(now_s) not in {int, float} or not math.isfinite(now_s)
        ):
            raise ValueError("now_s must be finite numeric evidence")
        if type(through) is not int or through not in range(6):
            raise ValueError("through must be 0..5")
        epoch = time.time() if now_s is None else float(now_s)
        with self._connect() as db:
            row = db.execute(
                """SELECT trial_id FROM trials
                   WHERE (status='pending' OR
                     (status='running' AND (lease_expires_at IS NULL OR lease_expires_at<=?)))
                     AND (
                       (SELECT COUNT(*) FROM checkpoints
                        WHERE checkpoints.trial_id=trials.trial_id
                          AND checkpoints.status='completed'
                          AND checkpoints.tier BETWEEN 0 AND ?) < ?
                       OR (
                         status='running'
                         AND (lease_expires_at IS NULL OR lease_expires_at<=?)
                         AND (SELECT COUNT(*) FROM checkpoints
                              WHERE checkpoints.trial_id=trials.trial_id
                                AND checkpoints.status='completed'
                                AND checkpoints.tier BETWEEN 0 AND ?) = ?
                       )
                     )
                   ORDER BY CASE WHEN
                     (SELECT COUNT(*) FROM checkpoints
                      WHERE checkpoints.trial_id=trials.trial_id
                        AND checkpoints.status='completed'
                        AND checkpoints.tier BETWEEN 0 AND ?) < ?
                     THEN 0 ELSE 1 END,
                     created_at, trial_id LIMIT 1""",
                (
                    epoch,
                    through,
                    through + 1,
                    epoch,
                    through,
                    through + 1,
                    through,
                    through + 1,
                ),
            ).fetchone()
        if row is None:
            return None
        identifier = str(row[0])
        return identifier if self.lease_trial(identifier, owner, ttl_s=ttl_s, now_s=epoch) else None

    def heartbeat(
        self,
        trial_id: str,
        owner: str,
        *,
        ttl_s: float = 60.0,
        now_s: Optional[float] = None,
    ) -> None:
        if type(owner) is not str or not owner.strip():
            raise ValueError("owner must be a non-empty string")
        if type(ttl_s) not in {int, float} or not math.isfinite(ttl_s) or ttl_s <= 0:
            raise ValueError("ttl_s must be finite and positive")
        if now_s is not None and (
            type(now_s) not in {int, float} or not math.isfinite(now_s)
        ):
            raise ValueError("now_s must be finite numeric evidence")
        epoch = time.time() if now_s is None else float(now_s)
        stamp = _utc_now(epoch)
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE trials SET heartbeat=?,lease_expires_at=?,updated_at=?
                   WHERE trial_id=? AND status='running' AND lease_owner=?
                     AND lease_expires_at>?""",
                (stamp, epoch + ttl_s, stamp, trial_id, owner, epoch),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("trial lease is not owned by this worker")

    def yield_trial(self, trial_id: str, owner: str) -> None:
        """Return a leased, non-terminal trial to the resumable pending queue."""

        if type(owner) is not str or not owner.strip():
            raise ValueError("owner must be a non-empty string")
        now = _utc_now()
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE trials SET status='pending',lease_owner=NULL,
                   lease_expires_at=NULL,heartbeat=?,updated_at=?
                   WHERE trial_id=? AND status='running' AND lease_owner=?""",
                (now, now, trial_id, owner),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("trial lease is not owned by this worker")

    def checkpoint(
        self,
        trial_id: str,
        tier: int,
        *,
        owner: str,
        status: str,
        metrics: Optional[Mapping[str, Any]] = None,
        artifact_hashes: Optional[Mapping[str, Any]] = None,
        stdout_stderr_tail: Optional[str] = None,
        elapsed_s: Optional[float] = None,
    ) -> None:
        if (
            type(tier) is not int
            or tier not in range(6)
            or type(status) is not str
            or status not in {"running", "completed", "failed"}
        ):
            raise ValueError("tier must be 0..5 and status must be running/completed/failed")
        if type(owner) is not str or not owner.strip():
            raise ValueError("owner must be a non-empty string")
        for name, value in (("metrics", metrics), ("artifact_hashes", artifact_hashes)):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be an object")
        if elapsed_s is not None and (
            type(elapsed_s) not in {int, float}
            or not math.isfinite(elapsed_s)
            or elapsed_s < 0.0
        ):
            raise ValueError("elapsed_s must be finite and non-negative")
        now = _utc_now()
        finished = now if status in {"completed", "failed"} else None
        encoded_metrics = canonical_json(metrics or {})
        encoded_artifacts = (
            canonical_json(artifact_hashes) if artifact_hashes is not None else None
        )
        with self._connect() as db:
            # Lease validation and checkpoint publication are one serialized
            # transition.  A reclaimer cannot acquire between the SELECT and
            # UPSERT and then receive stale evidence from the old owner.
            db.execute("BEGIN IMMEDIATE")
            lease = db.execute(
                """SELECT status,lease_owner,lease_expires_at FROM trials
                   WHERE trial_id=?""",
                (trial_id,),
            ).fetchone()
            if lease is None:
                raise KeyError(trial_id)
            if (
                lease["status"] != "running"
                or lease["lease_owner"] != owner
                or lease["lease_expires_at"] is None
                or float(lease["lease_expires_at"]) <= time.time()
            ):
                raise RuntimeError("checkpoint requires the current unexpired trial lease")
            existing = db.execute(
                "SELECT * FROM checkpoints WHERE trial_id=? AND tier=?",
                (trial_id, tier),
            ).fetchone()
            if existing is not None and existing["status"] in {"completed", "failed"}:
                effective_artifacts = (
                    existing["artifact_hashes"]
                    if encoded_artifacts is None
                    else encoded_artifacts
                )
                elapsed = float(elapsed_s) if elapsed_s is not None else None
                if (
                    existing["status"] == status
                    and existing["metrics"] == encoded_metrics
                    and existing["artifact_hashes"] == effective_artifacts
                    and existing["stdout_stderr_tail"] == stdout_stderr_tail
                    and existing["elapsed_s"] == elapsed
                ):
                    return
                raise RuntimeError("terminal checkpoint is immutable")
            db.execute(
                """INSERT INTO checkpoints(
                       trial_id,tier,status,started_at,finished_at,metrics,
                       artifact_hashes,stdout_stderr_tail,elapsed_s)
                   VALUES(?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(trial_id,tier) DO UPDATE SET
                       status=excluded.status,
                       finished_at=excluded.finished_at,
                       metrics=excluded.metrics,
                       stdout_stderr_tail=excluded.stdout_stderr_tail,
                       artifact_hashes=CASE
                           WHEN ? IS NULL THEN checkpoints.artifact_hashes
                           ELSE excluded.artifact_hashes END,
                       elapsed_s=excluded.elapsed_s""",
                (
                    trial_id,
                    tier,
                    status,
                    now,
                    finished,
                    encoded_metrics,
                    encoded_artifacts or "{}",
                    stdout_stderr_tail,
                    float(elapsed_s) if elapsed_s is not None else None,
                    encoded_artifacts,
                ),
            )

    def completed_tiers(self, trial_id: str) -> tuple[int, ...]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT tier FROM checkpoints WHERE trial_id=? AND status='completed' ORDER BY tier",
                (trial_id,),
            )
            return tuple(int(row[0]) for row in rows)

    def get_checkpoint(self, trial_id: str, tier: int) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM checkpoints WHERE trial_id=? AND tier=?",
                (trial_id, int(tier)),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["metrics"] = json.loads(result["metrics"])
        result["artifact_hashes"] = json.loads(result["artifact_hashes"])
        return result

    def create_or_get_promotion_round(
        self,
        tier: int,
        member_trial_ids: Sequence[str],
        *,
        cohort_identity_sha256: Optional[str] = None,
    ) -> Dict[str, Any]:
        if type(tier) is not int or tier not in range(5):
            raise ValueError("promotion round tier must be 0..4")
        raw_members = tuple(member_trial_ids)
        if any(type(value) is not str or not value.strip() for value in raw_members):
            raise ValueError("promotion round members must be non-empty strings")
        if len(raw_members) != len(set(raw_members)):
            raise ValueError("promotion round members must be unique")
        members = tuple(sorted(raw_members))
        if not members:
            raise ValueError("promotion round requires members")
        if cohort_identity_sha256 is not None and (
            type(cohort_identity_sha256) is not str
            or len(cohort_identity_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in cohort_identity_sha256
            )
        ):
            raise ValueError("promotion round cohort identity must be SHA-256")
        round_id = json_hash(
            {
                "tier": tier,
                "members": members,
                **(
                    {"cohort_identity_sha256": cohort_identity_sha256}
                    if cohort_identity_sha256 is not None
                    else {}
                ),
            }
        )
        now = _utc_now()
        with self._connect() as db:
            db.execute(
                """INSERT OR IGNORE INTO promotion_rounds(
                       round_id,tier,status,member_trial_ids,created_at,updated_at)
                   VALUES(?,?,'planned',?,?,?)""",
                (round_id, tier, canonical_json(members), now, now),
            )
        return self.get_promotion_round(round_id)

    def get_promotion_round(self, round_id: str) -> Dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM promotion_rounds WHERE round_id=?", (round_id,)
            ).fetchone()
        if row is None:
            raise KeyError(round_id)
        result = dict(row)
        result["member_trial_ids"] = tuple(json.loads(result["member_trial_ids"]))
        result["decision"] = (
            json.loads(result["decision"]) if result["decision"] is not None else None
        )
        return result

    def open_promotion_round(self, tier: int) -> Optional[Dict[str, Any]]:
        with self._connect() as db:
            row = db.execute(
                """SELECT round_id FROM promotion_rounds
                   WHERE tier=? AND status!='applied' ORDER BY created_at LIMIT 1""",
                (int(tier),),
            ).fetchone()
        return self.get_promotion_round(str(row[0])) if row is not None else None

    def decide_promotion_round(
        self, round_id: str, decision: Mapping[str, Any]
    ) -> None:
        now = _utc_now()
        encoded = canonical_json(decision)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                "SELECT status,decision FROM promotion_rounds WHERE round_id=?",
                (round_id,),
            ).fetchone()
            if current is None:
                raise KeyError(round_id)
            if current["status"] == "decided":
                if current["decision"] == encoded:
                    return
                raise RuntimeError("promotion round decision is immutable")
            if current["status"] != "planned":
                raise RuntimeError("promotion round cannot be decided")
            cursor = db.execute(
                """UPDATE promotion_rounds SET status='decided',decision=?,updated_at=?
                   WHERE round_id=? AND status='planned'""",
                (encoded, now, round_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("promotion round cannot be decided")

    def mark_promotion_round_applied(self, round_id: str) -> None:
        with self._connect() as db:
            cursor = db.execute(
                """UPDATE promotion_rounds SET status='applied',updated_at=?
                   WHERE round_id=? AND status='decided'""",
                (_utc_now(), round_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("promotion round is not decided")

    def next_tier(self, trial_id: str, *, through: int = 4) -> Optional[int]:
        if type(through) is not int or through not in range(6):
            raise ValueError("through must be 0..5")
        completed = set(self.completed_tiers(trial_id))
        return next((tier for tier in range(through + 1) if tier not in completed), None)

    def finish_trial(
        self,
        trial_id: str,
        owner: str,
        *,
        success: bool,
        phase_timings: Optional[Mapping[str, Any]] = None,
        safety_and_completion_metrics: Optional[Mapping[str, Any]] = None,
        artifact_hashes: Optional[Mapping[str, Any]] = None,
        failure_reason: Optional[str] = None,
        stdout_stderr_tail: Optional[str] = None,
        worktree_path: Optional[str] = None,
    ) -> None:
        if type(success) is not bool:
            raise TypeError("success must be an exact bool")
        if type(owner) is not str or not owner.strip():
            raise ValueError("owner must be a non-empty string")
        for name, value in (
            ("phase_timings", phase_timings),
            ("safety_and_completion_metrics", safety_and_completion_metrics),
            ("artifact_hashes", artifact_hashes),
        ):
            if value is not None and not isinstance(value, Mapping):
                raise TypeError(f"{name} must be an object")
        now = _utc_now()
        status = "completed" if success else "failed"
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute(
                """SELECT artifact_hashes,phase_timings,
                          safety_and_completion_metrics
                   FROM trials WHERE trial_id=?""",
                (trial_id,),
            ).fetchone()
            if current is None:
                raise KeyError(trial_id)
            merged_artifacts = json.loads(current["artifact_hashes"])
            if artifact_hashes is not None:
                merged_artifacts.update(artifact_hashes)
            merged_timings = json.loads(current["phase_timings"])
            if phase_timings is not None:
                merged_timings.update(phase_timings)
            merged_safety = json.loads(current["safety_and_completion_metrics"])
            if safety_and_completion_metrics is not None:
                merged_safety.update(safety_and_completion_metrics)
            cursor = db.execute(
                """UPDATE trials SET status=?, finished_at=?, updated_at=?,
                   lease_owner=NULL,lease_expires_at=NULL,heartbeat=?,
                   phase_timings=?,safety_and_completion_metrics=?,artifact_hashes=?,
                   failure_reason=?,stdout_stderr_tail=?,worktree_path=?
                   WHERE trial_id=? AND status='running' AND lease_owner=?
                     AND lease_expires_at>?""",
                (
                    status,
                    now,
                    now,
                    now,
                    canonical_json(merged_timings),
                    canonical_json(merged_safety),
                    canonical_json(merged_artifacts),
                    failure_reason,
                    stdout_stderr_tail,
                    worktree_path,
                    trial_id,
                    owner,
                    time.time(),
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("trial lease is not owned by this worker")

    def acquire_global_lease(
        self,
        name: str,
        owner: str,
        *,
        ttl_s: float = 60.0,
        now_s: Optional[float] = None,
    ) -> bool:
        if (
            type(name) is not str
            or not name.strip()
            or type(owner) is not str
            or not owner.strip()
            or type(ttl_s) not in {int, float}
            or ttl_s <= 0
            or not math.isfinite(ttl_s)
        ):
            raise ValueError("lease name/owner and finite positive ttl_s are required")
        if now_s is not None and (
            type(now_s) not in {int, float} or not math.isfinite(now_s)
        ):
            raise ValueError("now_s must be finite numeric evidence")
        epoch = time.time() if now_s is None else float(now_s)
        stamp = _utc_now(epoch)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT owner,expires_at FROM leases WHERE name=?", (name,)).fetchone()
            if row is not None and row["owner"] != owner and float(row["expires_at"]) > epoch:
                db.rollback()
                return False
            db.execute(
                """INSERT INTO leases(name,owner,heartbeat,expires_at) VALUES(?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET owner=excluded.owner,
                       heartbeat=excluded.heartbeat,expires_at=excluded.expires_at""",
                (name, owner, stamp, epoch + ttl_s),
            )
            db.commit()
        return True

    def release_global_lease(self, name: str, owner: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM leases WHERE name=? AND owner=?", (name, owner))
            return cursor.rowcount == 1

    def import_legacy_benchmark_history(self, source: Path | str) -> int:
        """Import the historical concatenated pretty-JSON stream exactly once."""

        path = Path(source).resolve()
        source_hash = sha256_file(path)
        with self._connect() as db:
            existing = db.execute(
                "SELECT row_count FROM imports WHERE source_hash=?", (source_hash,)
            ).fetchone()
        if existing is not None:
            return int(existing["row_count"])

        text = path.read_text(encoding="utf-8")
        def reject_constant(value: str) -> None:
            raise ValueError(f"legacy stream contains non-standard JSON constant: {value}")

        def unique_object(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"legacy stream contains duplicate key: {key}")
                result[key] = value
            return result

        decoder = json.JSONDecoder(
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
        objects: list[Mapping[str, Any]] = []
        offset = 0
        while offset < len(text):
            while offset < len(text) and text[offset].isspace():
                offset += 1
            if offset >= len(text):
                break
            value, offset = decoder.raw_decode(text, offset)
            if not isinstance(value, Mapping):
                raise ValueError("legacy history contains a non-object value")
            objects.append(value)

        imported = 0
        for index, payload in enumerate(objects):
            config = {
                "legacy_source_hash": source_hash,
                "legacy_object_index": index,
                "historical_only": True,
                "promotion_eligible": False,
                "evidence_trust": "untrusted-historical-report",
                "comparison_series": "legacy-non-comparable",
            }
            config_hash = json_hash(config)
            raw_commit = payload.get("commit_hash") or payload.get("git_commit") or "unknown"
            dirty = payload.get("dirty_diff_hash") or "unknown"
            code_hash = sha256_text(f"legacy\0{raw_commit}\0{dirty}\0{index}")
            trial_id, created = self.create_or_get_trial(
                key=TrialKey(
                    code_hash=code_hash,
                    config_hash=config_hash,
                    dataset_hash=f"legacy:{source_hash}",
                    seed=index,
                    evaluator_version="legacy-benchmark-history-v1",
                ),
                commit_hash=str(raw_commit),
                dirty_diff_hash=str(dirty),
                resolved_config=config,
                environment_fingerprint="historical-unknown",
                simulator_build=None,
                artifact_hashes={"source": source_hash, "object_index": index},
                candidate_name=f"legacy-history-{index:04d}",
            )
            row = self.get_trial(trial_id)
            if row["status"] not in TERMINAL_STATUSES:
                owner = f"legacy-importer:{source_hash}"
                if not self.lease_trial(trial_id, owner):
                    raise RuntimeError(
                        f"historical import row {trial_id} has an active foreign lease"
                    )
                overall = payload.get("overall_passed")
                overall_is_exact_boolean = type(overall) is bool
                self.finish_trial(
                    trial_id,
                    owner,
                    # Completion means the evidence row was imported, not that
                    # the historical candidate passed a modern promotion gate.
                    success=True,
                    safety_and_completion_metrics={
                        "promotion_eligible": False,
                        "trusted": False,
                        "comparable_to_current_evaluator": False,
                        "historical_reported_overall_passed": overall,
                        "historical_report_is_exact_boolean": overall_is_exact_boolean,
                        "historical_result": payload,
                    },
                    failure_reason=None,
                    stdout_stderr_tail=(
                        "UNTRUSTED HISTORICAL IMPORT: not rerun, not comparable, "
                        "and never valid promotion evidence."
                    ),
                )
                imported += 1
        nonterminal = [
            row["trial_id"]
            for row in self.list_trials()
            if row["evaluator_version"] == "legacy-benchmark-history-v1"
            and row["dataset_hash"] == f"legacy:{source_hash}"
            and row["status"] not in TERMINAL_STATUSES
        ]
        if nonterminal:
            raise RuntimeError(
                "legacy import remains incomplete; marker not written: "
                + ", ".join(nonterminal)
            )
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO imports(source_hash,source_path,imported_at,row_count) VALUES(?,?,?,?)",
                (source_hash, str(path), _utc_now(), len(objects)),
            )
        return len(objects)
