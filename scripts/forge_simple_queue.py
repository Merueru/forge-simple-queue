from __future__ import annotations

import copy
import pickle
import hashlib
import html
import json
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI
import gradio as gr

from modules import call_queue, img2img, progress, script_callbacks, scripts, shared, txt2img


HISTORY_LIMIT = 30

FORGE_STATE_KEYS = (
    "forge_preset",
    "sd_model_checkpoint",
    "forge_additional_modules",
    "forge_unet_storage_dtype",
)


def _copy_value(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def capture_forge_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    opts = getattr(shared, "opts", None)
    if opts is None:
        return state

    for key in FORGE_STATE_KEYS:
        if hasattr(opts, key):
            state[key] = _copy_value(getattr(opts, key))
    return state


def summarize_forge_state(state: dict[str, Any]) -> dict[str, Any]:
    modules = state.get("forge_additional_modules") or []
    return {
        "preset": state.get("forge_preset") or "",
        "checkpoint": state.get("sd_model_checkpoint") or "",
        "dtype": state.get("forge_unet_storage_dtype") or "",
        "modules": [str(item).split("\\")[-1].split("/")[-1] for item in modules],
    }


def forge_states_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    for key in set(left) | set(right):
        if left.get(key) != right.get(key):
            return False
    return True


def apply_forge_state(state: dict[str, Any]):
    if not state:
        return

    opts = getattr(shared, "opts", None)
    if opts is None:
        return

    changed = False
    for key, value in state.items():
        if not hasattr(opts, key):
            continue
        if getattr(opts, key) == value:
            continue
        opts.set(key, _copy_value(value), run_callbacks=False)
        changed = True

    if changed:
        try:
            from modules_forge import main_entry

            main_entry.refresh_model_loading_parameters(refresh=True)
        except Exception:
            print("[Forge Simple Queue] Failed to refresh Forge model state:")
            traceback.print_exc()


def stable_fingerprint_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if len(value) > 512:
            digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()
            return {"type": "str", "length": len(value), "sha256": digest}
        return value
    if isinstance(value, bytes):
        return {"type": "bytes", "length": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (list, tuple)):
        return [stable_fingerprint_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): stable_fingerprint_value(value[key]) for key in sorted(value, key=str)}

    name = getattr(value, "name", None)
    if name:
        return {"type": type(value).__name__, "name": str(name)}

    mode = getattr(value, "mode", None)
    size = getattr(value, "size", None)
    if mode is not None and size is not None:
        return {"type": type(value).__name__, "mode": str(mode), "size": stable_fingerprint_value(size)}

    return {"type": type(value).__name__, "repr": repr(value)}


class QueueJob:
    def __init__(
        self,
        job_id: str,
        task_id: str,
        tab: str,
        args: list[Any],
        input_keys: list[str | None],
        forge_state: dict[str, Any],
        username: str | None,
        repeat_id: str | None = None,
        repeat_placeholder: bool = False,
    ):
        self.id = job_id
        self.task_id = task_id
        self.tab = tab
        self.args = args
        self.input_keys = input_keys
        self.forge_state = forge_state
        self.username = username
        self.repeat_id = repeat_id or job_id
        self.repeat_placeholder = repeat_placeholder
        self.created_at = time.time()
        self.status = "queued"
        self.paused = repeat_placeholder
        self.cancel_requested = False
        self.error: str | None = None
        self._fingerprint_cache: str | None = None
        self._prompt_snapshot: str | None = None
        self._negative_prompt_snapshot: str | None = None
        self._summary_snapshot: dict[str, Any] | None = None

    @property
    def prompt(self) -> str:
        if self._prompt_snapshot is not None:
            return self._prompt_snapshot
        return self._string_value(f"{self.tab}_prompt", fallback_index=self.prompt_index())

    @property
    def negative_prompt(self) -> str:
        if self._negative_prompt_snapshot is not None:
            return self._negative_prompt_snapshot
        return self._string_value(f"{self.tab}_neg_prompt", fallback_index=self.negative_index())

    def prompt_index(self) -> int:
        return 1 if self.tab == "txt2img" else 2

    def negative_index(self) -> int:
        return 2 if self.tab == "txt2img" else 3

    def _string_arg(self, index: int) -> str:
        if len(self.args) <= index:
            return ""
        return str(self.args[index] or "")

    def _arg_by_key(self, *keys: str, default: Any = None) -> Any:
        for key in keys:
            try:
                index = self.input_keys.index(key)
            except ValueError:
                continue
            if len(self.args) > index:
                return self.args[index]
        return default

    def _string_value(self, *keys: str, fallback_index: int | None = None) -> str:
        value = self._arg_by_key(*keys, default=None)
        if value is None and fallback_index is not None:
            return self._string_arg(fallback_index)
        return str(value or "")

    def update_text(self, prompt: str | None, negative_prompt: str | None):
        if prompt is not None and len(self.args) > self.prompt_index():
            self.args[self.prompt_index()] = prompt
        if negative_prompt is not None and len(self.args) > self.negative_index():
            self.args[self.negative_index()] = negative_prompt
        self._fingerprint_cache = None
        self._prompt_snapshot = None
        self._negative_prompt_snapshot = None
        self._summary_snapshot = None

    def to_dict(
        self,
        index: int | None = None,
        repeat_selected: bool = False,
        runs: int = 1,
        failures: int = 0,
        deleted: int = 0,
    ) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "repeat_id": self.repeat_id,
            "repeat_selected": repeat_selected,
            "repeat_placeholder": self.repeat_placeholder,
            "tab": self.tab,
            "index": index,
            "status": "paused" if self.paused and self.status == "queued" else self.status,
            "paused": self.paused,
            "progress_active": self.task_id == getattr(progress, "current_task", None),
            "progress_queued": self.task_id in getattr(progress, "pending_tasks", {}),
            "age": max(0, int(time.time() - self.created_at)),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "summary": self.summary(),
            "forge": summarize_forge_state(self.forge_state),
            "error": self.error,
            "editable": self.status in ("queued", "waiting") and self.task_id != getattr(progress, "current_task", None),
            "runs": runs,
            "failures": failures,
            "deleted": deleted,
        }

    def summary(self) -> dict[str, Any]:
        if self._summary_snapshot is not None:
            return dict(self._summary_snapshot)
        try:
            width = self._arg_by_key(f"{self.tab}_width")
            height = self._arg_by_key(f"{self.tab}_height")
            batch_count = self._arg_by_key(f"{self.tab}_batch_count")
            batch_size = self._arg_by_key(f"{self.tab}_batch_size")
            return {
                "steps": self._arg_by_key(f"{self.tab}_steps"),
                "sampler": self._arg_by_key(f"{self.tab}_sampling"),
                "schedule": self._arg_by_key(f"{self.tab}_scheduler"),
                "size": f"{width}x{height}" if width and height else "",
                "batch": f"{batch_count}x{batch_size}" if batch_count and batch_size else "",
            }
        except Exception:
            return {}

    def clone_for_history(self) -> "QueueJob":
        fingerprint = self.fingerprint()
        history_job = QueueJob(
            self.id,
            self.task_id,
            self.tab,
            [self.task_id],
            [self.input_keys[0] if self.input_keys else None],
            _copy_value(self.forge_state),
            self.username,
            repeat_id=self.repeat_id,
            repeat_placeholder=self.repeat_placeholder,
        )
        history_job.created_at = self.created_at
        history_job.status = self.status
        history_job.paused = self.paused
        history_job.cancel_requested = self.cancel_requested
        history_job.error = self.error
        history_job._fingerprint_cache = fingerprint
        history_job._prompt_snapshot = self.prompt
        history_job._negative_prompt_snapshot = self.negative_prompt
        history_job._summary_snapshot = self.summary()
        return history_job

    def clone_for_repeat(self, repeat_placeholder: bool = False) -> "QueueJob":
        job_id = uuid4().hex[:12]
        task_id = f"task(simple-queue-{job_id})"
        args = [_copy_value(value) for value in self.args]
        if args:
            args[0] = task_id
        return QueueJob(
            job_id,
            task_id,
            self.tab,
            args,
            list(self.input_keys),
            _copy_value(self.forge_state),
            self.username,
            repeat_id=self.repeat_id,
            repeat_placeholder=repeat_placeholder,
        )

    def fingerprint(self) -> str:
        if self._fingerprint_cache is not None:
            return self._fingerprint_cache

        payload = {
            "tab": self.tab,
            "inputs": [
                [key, stable_fingerprint_value(value)]
                for key, value in zip(self.input_keys[1:], self.args[1:])
            ],
            "forge": stable_fingerprint_value(self.forge_state),
        }
        self._fingerprint_cache = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return self._fingerprint_cache


class SimpleRequest:
    def __init__(self, username: str | None):
        self.username = username


class SimpleQueue:
    def __init__(self):
        self._condition = threading.Condition()
        self._pending: deque[QueueJob] = deque()
        self._history: deque[QueueJob] = deque(maxlen=HISTORY_LIMIT)
        self._active: QueueJob | None = None
        self._waiting: QueueJob | None = None
        self._queue_paused = False
        self._queue_mode = "play"
        self._queue_restore_state: dict[str, Any] | None = None
        self._snapshot_path = Path(__file__).resolve().parent.parent / "data" / "saved_queue.pkl"
        self._saved_snapshot: dict[str, Any] | None = self._load_saved_snapshot()
        self._repeat_enabled = False
        self._repeat_order: list[str] = []
        self._repeat_templates: dict[str, QueueJob] = {}
        self._worker_started = False
        self._api_registered = False

    def start(self):
        with self._condition:
            if self._worker_started:
                return
            thread = threading.Thread(target=self._worker_loop, name="ForgeSimpleQueueWorker", daemon=True)
            self._worker_started = True
            thread.start()

    def register_api(self, app: FastAPI):
        if self._api_registered:
            return
        self._api_registered = True

        @app.get("/forge-simple-queue/status")
        def status():
            return self.snapshot()

        @app.get("/forge-simple-queue/status-lite")
        def status_lite():
            return self.snapshot(compact=True)

        @app.post("/forge-simple-queue/delete")
        def delete(payload: dict[str, Any] = Body(default={})):
            return self.delete(str(payload.get("id", "")))

        @app.post("/forge-simple-queue/delete-many")
        def delete_many(payload: dict[str, Any] = Body(default={})):
            ids = payload.get("ids", [])
            if not isinstance(ids, list):
                ids = []
            return self.delete_many([str(item) for item in ids])

        @app.post("/forge-simple-queue/pause")
        def pause(payload: dict[str, Any] = Body(default={})):
            return self.set_paused(str(payload.get("id", "")), True)

        @app.post("/forge-simple-queue/resume")
        def resume(payload: dict[str, Any] = Body(default={})):
            return self.set_paused(str(payload.get("id", "")), False)

        @app.post("/forge-simple-queue/move")
        def move(payload: dict[str, Any] = Body(default={})):
            return self.move(str(payload.get("id", "")), int(payload.get("index", 0)))

        @app.post("/forge-simple-queue/update")
        def update(payload: dict[str, Any] = Body(default={})):
            return self.update_job(
                str(payload.get("id", "")),
                payload.get("prompt"),
                payload.get("negative_prompt"),
            )

        @app.post("/forge-simple-queue/repeat")
        def repeat(payload: dict[str, Any] = Body(default={})):
            return self.set_repeat(
                payload.get("enabled"),
                payload.get("id"),
                payload.get("selected"),
            )

        @app.post("/forge-simple-queue/repeat-all")
        def repeat_all(payload: dict[str, Any] = Body(default={})):
            return self.set_repeat_all(bool(payload.get("selected")))

        @app.get("/forge-simple-queue/saved-queue/status")
        def saved_queue_status():
            return self.saved_queue_status()

        @app.post("/forge-simple-queue/saved-queue/save")
        def saved_queue_save():
            return self.save_queue_snapshot()

        @app.post("/forge-simple-queue/saved-queue/restore")
        def saved_queue_restore():
            return self.restore_queue_snapshot()

        @app.post("/forge-simple-queue/saved-queue/clear")
        def saved_queue_clear():
            return self.clear_queue_snapshot()

        @app.post("/forge-simple-queue/control")
        def control(payload: dict[str, Any] = Body(default={})):
            return self.control(str(payload.get("action", "")))

        @app.post("/forge-simple-queue/interrupt")
        def interrupt():
            return self.control("stop")

        @app.post("/forge-simple-queue/skip")
        def skip():
            shared.state.skip()
            return self.snapshot()

    def enqueue(
        self,
        tab: str,
        request: gr.Request | None,
        raw_args: tuple[Any, ...],
        input_keys: list[str | None],
    ) -> str:
        if len(raw_args) != len(input_keys):
            print(f"[Forge Simple Queue] Trimmed captured args from {len(raw_args)} to {len(input_keys)} inputs.")
        args = [_copy_value(value) for value in raw_args[: len(input_keys)]]
        if not args:
            raise RuntimeError("Could not read generation inputs for this tab.")

        job_id = uuid4().hex[:12]
        task_id = f"task(simple-queue-{job_id})"
        args[0] = task_id
        username = getattr(request, "username", None)
        job = QueueJob(job_id, task_id, tab, args, input_keys, capture_forge_state(), username)

        with self._condition:
            self._pending.append(job)
            self._condition.notify_all()

        print(f"[Forge Simple Queue] Queued {tab} job {job_id}.")
        return (
            f"<div class='forge-simple-queue-status-line' data-tab='{html.escape(tab)}' data-task-id='{html.escape(task_id)}'>"
            f"Queued {html.escape(tab)} job {html.escape(job_id)}."
            "</div>"
        )

    def snapshot(self, compact: bool = False) -> dict[str, Any]:
        with self._condition:
            active_job = self._active or self._waiting
            active = self._compact_job_dict(active_job) if compact else (self._job_dict(active_job) if active_job is not None else None)
            pending_jobs = [job for job in self._pending if job is not self._waiting]
            pending = [] if compact else [self._job_dict(job, index=i) for i, job in enumerate(pending_jobs)]
            queue_count = len(pending_jobs) + (1 if self._job_counts_as_queued_locked(active_job) else 0)
            history = [] if compact else self._history_snapshot_locked()
            recent_tasks = [job.task_id for job in list(self._history)[:HISTORY_LIMIT]]
            repeat = {
                "enabled": self._repeat_enabled,
                "count": len(self._repeat_order),
                "order": list(self._repeat_order),
            }
            control = {
                "mode": self._queue_mode,
                "paused": self._queue_paused,
            }
            saved_queue = self._saved_queue_summary_locked()
        return {
            "active": active,
            "pending": pending,
            "history": history,
            "pending_count": len(pending_jobs),
            "queue_count": queue_count,
            "compact": compact,
            "recent_tasks": recent_tasks,
            "repeat": repeat,
            "control": control,
            "saved_queue": saved_queue,
        }

    def saved_queue_status(self) -> dict[str, Any]:
        with self._condition:
            return self._saved_queue_summary_locked()

    def save_queue_snapshot(self) -> dict[str, Any]:
        with self._condition:
            jobs = self._snapshot_queue_jobs_locked()
            if not jobs:
                data = self.snapshot()
                data["ok"] = False
                data["message"] = "No queued jobs to save."
                return data

            payload = {
                "version": 1,
                "saved_at": time.time(),
                "repeat_enabled": self._repeat_enabled,
                "repeat_order": list(self._repeat_order),
                "queue_paused": self._queue_paused,
                "queue_mode": self._queue_mode,
                "jobs": [self._job_snapshot(job) for job in jobs],
            }

        try:
            self._write_saved_snapshot(payload)
        except Exception:
            print("[Forge Simple Queue] Failed to save queue snapshot:")
            traceback.print_exc()
            data = self.snapshot()
            data["ok"] = False
            data["message"] = "Failed to save queue snapshot."
            return data

        with self._condition:
            self._saved_snapshot = payload
        data = self.snapshot()
        data["ok"] = True
        data["message"] = f"Saved {len(payload['jobs'])} queued job(s)."
        return data

    def restore_queue_snapshot(self) -> dict[str, Any]:
        with self._condition:
            snapshot = self._saved_snapshot
            count = self._saved_queue_count_locked()
            if snapshot is None or count == 0:
                data = self.snapshot()
                data["ok"] = False
                data["message"] = "No saved queue to restore."
                return data
            if self._active is not None or self._waiting is not None:
                data = self.snapshot()
                data["ok"] = False
                data["message"] = "Pause or stop the active queue job before restoring."
                return data

            jobs = [self._job_from_snapshot(item) for item in snapshot.get("jobs", [])]
            jobs = [job for job in jobs if job is not None]
            self._pending = deque(jobs)
            self._queue_paused = bool(snapshot.get("queue_paused", False))
            self._queue_mode = str(snapshot.get("queue_mode") or ("pause" if self._queue_paused else "play"))
            if self._queue_mode not in ("play", "pause", "stop"):
                self._queue_mode = "pause" if self._queue_paused else "play"

            self._repeat_enabled = bool(snapshot.get("repeat_enabled", False))
            self._repeat_order = [str(item) for item in snapshot.get("repeat_order", [])]
            self._repeat_templates = {}
            for job in jobs:
                if job.repeat_id in self._repeat_order:
                    self._repeat_templates[job.repeat_id] = job

            self._sync_repeat_order_locked()
            if self._repeat_enabled:
                self._unpark_repeat_placeholders_locked()
                self._ensure_repeat_jobs_locked()
            else:
                self._ensure_selected_placeholders_locked()
            self._condition.notify_all()

        data = self.snapshot()
        data["ok"] = True
        data["message"] = f"Restored {len(jobs)} saved job(s)."
        return data

    def clear_queue_snapshot(self) -> dict[str, Any]:
        try:
            self._snapshot_path.unlink(missing_ok=True)
        except Exception:
            print("[Forge Simple Queue] Failed to clear saved queue snapshot:")
            traceback.print_exc()
            data = self.snapshot()
            data["ok"] = False
            data["message"] = "Failed to clear saved queue."
            return data

        with self._condition:
            count = self._saved_queue_count_locked()
            self._saved_snapshot = None
        data = self.snapshot()
        data["ok"] = True
        data["message"] = f"Cleared {count} saved job(s)."
        return data

    def _load_saved_snapshot(self) -> dict[str, Any] | None:
        try:
            if not self._snapshot_path.exists():
                return None
            with self._snapshot_path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
                return None
            return payload
        except Exception:
            print("[Forge Simple Queue] Failed to load saved queue snapshot:")
            traceback.print_exc()
            return None

    def _write_saved_snapshot(self, payload: dict[str, Any]):
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._snapshot_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self._snapshot_path)

    def _saved_queue_count_locked(self) -> int:
        if not self._saved_snapshot:
            return 0
        jobs = self._saved_snapshot.get("jobs", [])
        return len(jobs) if isinstance(jobs, list) else 0

    def _saved_queue_summary_locked(self) -> dict[str, Any]:
        count = self._saved_queue_count_locked()
        return {
            "exists": count > 0,
            "count": count,
            "saved_at": self._saved_snapshot.get("saved_at") if self._saved_snapshot else None,
        }

    def _job_counts_as_queued_locked(self, job: QueueJob | None) -> bool:
        if job is None or job.cancel_requested:
            return False
        return job.task_id != getattr(progress, "current_task", None)

    def _snapshot_queue_jobs_locked(self) -> list[QueueJob]:
        jobs: list[QueueJob] = []
        active_job = self._active or self._waiting
        if self._job_counts_as_queued_locked(active_job):
            jobs.append(active_job)
        for job in self._pending:
            if job is self._waiting or job.cancel_requested:
                continue
            jobs.append(job)
        return jobs

    def _job_snapshot(self, job: QueueJob) -> dict[str, Any]:
        return {
            "tab": job.tab,
            "args": [_copy_value(value) for value in job.args],
            "input_keys": list(job.input_keys),
            "forge_state": _copy_value(job.forge_state),
            "username": job.username,
            "repeat_id": job.repeat_id,
            "repeat_selected": job.repeat_id in self._repeat_order,
            "paused": job.paused,
            "repeat_placeholder": job.repeat_placeholder,
        }

    def _job_from_snapshot(self, item: dict[str, Any]) -> QueueJob | None:
        if not isinstance(item, dict):
            return None
        args = [_copy_value(value) for value in item.get("args", [])]
        if not args:
            return None
        job_id = uuid4().hex[:12]
        task_id = f"task(simple-queue-{job_id})"
        args[0] = task_id
        job = QueueJob(
            job_id,
            task_id,
            str(item.get("tab") or "txt2img"),
            args,
            list(item.get("input_keys") or []),
            _copy_value(item.get("forge_state") or {}),
            item.get("username"),
            repeat_id=str(item.get("repeat_id") or job_id),
            repeat_placeholder=bool(item.get("repeat_placeholder", False)),
        )
        job.paused = bool(item.get("paused", False)) or job.repeat_placeholder
        job.status = "queued"
        return job

    def _job_dict(self, job: QueueJob, index: int | None = None, **kwargs) -> dict[str, Any]:
        return job.to_dict(index=index, repeat_selected=job.repeat_id in self._repeat_order, **kwargs)

    def _compact_job_dict(self, job: QueueJob | None) -> dict[str, Any] | None:
        if job is None:
            return None
        return {
            "id": job.id,
            "task_id": job.task_id,
            "tab": job.tab,
            "status": job.status,
            "progress_active": job.task_id == getattr(progress, "current_task", None),
            "progress_queued": job.task_id in getattr(progress, "pending_tasks", {}),
        }

    def _history_snapshot_locked(self) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        by_fingerprint: dict[str, dict[str, Any]] = {}
        for job in list(self._history):
            fingerprint = job.fingerprint()
            if fingerprint in by_fingerprint:
                group = by_fingerprint[fingerprint]
                group["runs"] += 1
                if job.status == "failed":
                    group["failures"] += 1
                if job.status == "deleted":
                    group["deleted"] += 1
                continue

            group = self._job_dict(
                job,
                runs=1,
                failures=1 if job.status == "failed" else 0,
                deleted=1 if job.status == "deleted" else 0,
            )
            by_fingerprint[fingerprint] = group
            groups.append(group)
            if len(groups) >= 30:
                break
        return groups

    def _append_history_locked(self, job: QueueJob):
        if any(item.id == job.id for item in self._history):
            return
        self._history.appendleft(job.clone_for_history())

    def delete(self, job_id: str) -> dict[str, Any]:
        deleted = False
        with self._condition:
            for job in list(self._pending):
                if job.id == job_id:
                    self._delete_pending_job_locked(job)
                    deleted = True
                    break
            if deleted:
                self._condition.notify_all()
                self._restore_forge_state_if_idle_locked()
        return self.snapshot()

    def delete_many(self, job_ids: list[str]) -> dict[str, Any]:
        ids = {job_id for job_id in job_ids if job_id}
        deleted = 0
        if ids:
            with self._condition:
                for job in list(self._pending):
                    if job.id in ids:
                        self._delete_pending_job_locked(job, add_history=deleted < HISTORY_LIMIT)
                        deleted += 1
                if deleted:
                    self._condition.notify_all()
                    self._restore_forge_state_if_idle_locked()
        data = self.snapshot()
        data["ok"] = True
        data["deleted_count"] = deleted
        data["skipped_count"] = max(0, len(ids) - deleted)
        return data

    def _delete_pending_job_locked(self, job: QueueJob, add_history: bool = True):
        job.cancel_requested = True
        if job in self._pending:
            self._pending.remove(job)
        job.status = "deleted"
        if self._waiting is job:
            self._waiting = None
        if add_history:
            self._append_history_locked(job)
        self._clear_progress_task(job, completed=True)
        self._remove_repeat_locked(job.repeat_id)

    def set_paused(self, job_id: str, paused: bool) -> dict[str, Any]:
        with self._condition:
            for job in self._pending:
                if job.id == job_id:
                    job.paused = paused
                    if not paused:
                        job.repeat_placeholder = False
                    if paused and job is self._waiting:
                        self._clear_progress_task(job)
                        self._waiting = None
                        job.status = "queued"
                    if not paused and job.status == "waiting":
                        job.status = "queued"
                    self._condition.notify_all()
                    break
        return self.snapshot()

    def move(self, job_id: str, index: int) -> dict[str, Any]:
        with self._condition:
            jobs = list(self._pending)
            job = next((item for item in jobs if item.id == job_id), None)
            if job is not None:
                jobs.remove(job)
                index = max(0, min(index, len(jobs)))
                jobs.insert(index, job)
                self._pending = deque(jobs)
                self._sync_repeat_order_locked()
                self._condition.notify_all()
        return self.snapshot()

    def update_job(self, job_id: str, prompt: str | None, negative_prompt: str | None) -> dict[str, Any]:
        with self._condition:
            for job in self._pending:
                if job.id == job_id:
                    job.update_text(prompt, negative_prompt)
                    if job.repeat_id in self._repeat_order:
                        self._repeat_templates[job.repeat_id] = job
                    break
        return self.snapshot()

    def set_repeat(self, enabled: Any = None, job_id: Any = None, selected: Any = None) -> dict[str, Any]:
        with self._condition:
            if enabled is not None:
                self._repeat_enabled = bool(enabled)
                if self._repeat_enabled and self._queue_mode == "stop":
                    self._queue_mode = "pause" if self._queue_paused else "play"

            if job_id is not None and selected is not None:
                job = self._find_job_locked(str(job_id))
                if job is not None:
                    if bool(selected):
                        self._add_repeat_locked(job)
                    else:
                        self._remove_repeat_locked(job.repeat_id)

            self._sync_repeat_order_locked()
            if self._repeat_enabled:
                self._ensure_repeat_jobs_locked()
            else:
                self._ensure_selected_placeholders_locked()
            self._condition.notify_all()
            self._restore_forge_state_if_idle_locked()
        return self.snapshot()

    def set_repeat_all(self, selected: bool) -> dict[str, Any]:
        with self._condition:
            for job in self._iter_jobs_locked(include_history=False):
                if selected:
                    self._add_repeat_locked(job)
                else:
                    self._remove_repeat_locked(job.repeat_id)
            self._sync_repeat_order_locked()
            if self._repeat_enabled:
                self._ensure_repeat_jobs_locked()
            else:
                self._ensure_selected_placeholders_locked()
            self._condition.notify_all()
            self._restore_forge_state_if_idle_locked()
        return self.snapshot()

    def disable_repeat(self):
        with self._condition:
            self._repeat_enabled = False
            self._ensure_selected_placeholders_locked()
            self._condition.notify_all()

    def control(self, action: str) -> dict[str, Any]:
        action = (action or "").lower()
        interrupt = False
        with self._condition:
            if action == "play":
                self._queue_paused = False
                self._queue_mode = "play"
                self._unpark_repeat_placeholders_locked()
                self._ensure_repeat_jobs_locked()
                self._condition.notify_all()
            elif action == "pause":
                self._queue_paused = True
                self._queue_mode = "pause"
                self._pause_waiting_locked()
                self._ensure_selected_placeholders_locked()
                self._condition.notify_all()
            elif action == "stop":
                self._queue_paused = True
                self._queue_mode = "pause"
                self._repeat_enabled = False
                interrupt = self._active is not None
                self._pause_waiting_locked()
                self._ensure_selected_placeholders_locked()
                self._condition.notify_all()
                self._restore_forge_state_if_idle_locked()

        if interrupt:
            shared.state.interrupt()
        return self.snapshot()

    def _find_job_locked(self, job_id: str) -> QueueJob | None:
        for job in self._iter_jobs_locked(include_history=True):
            if job.id == job_id:
                return job
        return None

    def _iter_jobs_locked(self, include_history: bool = False):
        seen: set[int] = set()
        for job in [self._active, self._waiting, *list(self._pending)]:
            if job is None or id(job) in seen:
                continue
            seen.add(id(job))
            yield job
        if include_history:
            for job in list(self._history):
                if id(job) in seen:
                    continue
                seen.add(id(job))
                yield job

    def _add_repeat_locked(self, job: QueueJob):
        if job.repeat_id not in self._repeat_order:
            self._repeat_order.append(job.repeat_id)
        self._repeat_templates[job.repeat_id] = job

    def _remove_repeat_locked(self, repeat_id: str):
        self._repeat_order = [item for item in self._repeat_order if item != repeat_id]
        self._repeat_templates.pop(repeat_id, None)
        for job in list(self._pending):
            if job.repeat_id == repeat_id and job.repeat_placeholder:
                self._pending.remove(job)

    def _sync_repeat_order_locked(self):
        if not self._repeat_order:
            return

        visible_order: list[str] = []
        for job in self._iter_jobs_locked(include_history=False):
            if job.repeat_id in self._repeat_order and job.repeat_id not in visible_order:
                visible_order.append(job.repeat_id)

        for repeat_id in self._repeat_order:
            if repeat_id not in visible_order:
                visible_order.append(repeat_id)

        self._repeat_order = visible_order

    def _clear_progress_task(self, job: QueueJob, completed: bool = False):
        progress.pending_tasks.pop(job.task_id, None)
        if completed:
            progress.finish_task(job.task_id)

    def _live_repeat_ids_locked(self) -> set[str]:
        return {
            job.repeat_id
            for job in self._iter_jobs_locked(include_history=False)
            if job.repeat_id in self._repeat_order and not job.cancel_requested
        }

    def _ordered_repeat_ids_after_locked(self, after_repeat_id: str | None = None) -> list[str]:
        order = list(self._repeat_order)
        if after_repeat_id in order:
            index = order.index(after_repeat_id) + 1
            return order[index:] + order[:index]
        return order

    def _ensure_repeat_jobs_locked(self, after_repeat_id: str | None = None):
        if not self._repeat_enabled or self._queue_paused or not self._repeat_order:
            return

        live_repeat_ids = self._live_repeat_ids_locked()
        for repeat_id in self._ordered_repeat_ids_after_locked(after_repeat_id):
            template = self._repeat_templates.get(repeat_id)
            if template is None:
                self._remove_repeat_locked(repeat_id)
                continue
            if repeat_id in live_repeat_ids:
                for job in self._pending:
                    if job.repeat_id == repeat_id and job.repeat_placeholder:
                        job.repeat_placeholder = False
                        job.paused = False
                        job.status = "queued"
                continue

            next_job = template.clone_for_repeat()
            self._pending.append(next_job)
            live_repeat_ids.add(repeat_id)
            print(f"[Forge Simple Queue] Auto repeat queued {next_job.tab} job {next_job.id}.")

    def _ensure_selected_placeholders_locked(self):
        if not self._repeat_order:
            return

        live_repeat_ids = self._live_repeat_ids_locked()
        for repeat_id in list(self._repeat_order):
            if repeat_id in live_repeat_ids:
                continue
            template = self._repeat_templates.get(repeat_id)
            if template is None:
                self._remove_repeat_locked(repeat_id)
                continue
            self._pending.append(template.clone_for_repeat(repeat_placeholder=True))

    def _unpark_repeat_placeholders_locked(self):
        for job in self._pending:
            if job.repeat_id in self._repeat_order and job.repeat_placeholder:
                job.repeat_placeholder = False
                job.paused = False
                job.status = "queued"

    def _pause_waiting_locked(self):
        job = self._waiting
        if job is None:
            return
        self._clear_progress_task(job)
        self._waiting = None
        if job in self._pending and not job.cancel_requested:
            job.status = "queued"

    def _restore_forge_state_if_idle_locked(self):
        if self._queue_restore_state is None:
            return
        if self._active is not None or self._waiting is not None or self._pending:
            return
        restore_state = self._queue_restore_state
        self._queue_restore_state = None
        apply_forge_state(restore_state)

    def _worker_loop(self):
        while True:
            with self._condition:
                job = self._next_ready_job_locked()
                while job is None:
                    self._condition.wait()
                    job = self._next_ready_job_locked()
                job.status = "waiting"
                self._waiting = job
                progress.add_task_to_queue(job.task_id)

            call_queue.queue_lock.acquire()

            try:
                with self._condition:
                    if job.cancel_requested or job.paused or job.repeat_placeholder or self._queue_paused or job not in self._pending:
                        if self._waiting is job:
                            self._waiting = None
                        if not job.cancel_requested and job in self._pending and job.status == "waiting":
                            job.status = "queued"
                        self._clear_progress_task(job, completed=job.cancel_requested)
                        self._condition.notify_all()
                        continue

                    self._pending.remove(job)
                    if self._waiting is job:
                        self._waiting = None
                    job.status = "running"
                    self._active = job

                self._run_job_with_lock(job)
                if job.status != "deleted":
                    job.status = "done"
            except Exception as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                print("[Forge Simple Queue] Job failed:")
                traceback.print_exc()
            finally:
                call_queue.queue_lock.release()
                with self._condition:
                    if self._active is job:
                        self._active = None
                    if self._waiting is job:
                        self._waiting = None
                    if job not in self._pending:
                        self._append_history_locked(job)
                    if job.status == "done" and job.repeat_id in self._repeat_order:
                        if self._repeat_enabled and not self._queue_paused:
                            self._ensure_repeat_jobs_locked(after_repeat_id=job.repeat_id)
                        else:
                            self._ensure_selected_placeholders_locked()
                    self._condition.notify_all()
                    self._restore_forge_state_if_idle_locked()

    def _next_ready_job_locked(self) -> QueueJob | None:
        if self._queue_paused:
            return None
        return next((job for job in self._pending if not job.paused and not job.repeat_placeholder), None)

    def _run_job_with_lock(self, job: QueueJob):
        runner = txt2img.txt2img if job.tab == "txt2img" else img2img.img2img
        args = [job.task_id, SimpleRequest(job.username)] + job.args[1:]
        if self._queue_restore_state is None:
            self._queue_restore_state = capture_forge_state()
        apply_forge_state(job.forge_state)

        def run_without_lock(*runner_args):
            shared.state.begin(job=job.task_id)
            progress.start_task(job.task_id)
            try:
                return runner(*runner_args)
            finally:
                progress.finish_task(job.task_id)
                shared.state.end()

        wrapped = call_queue.wrap_gradio_call(run_without_lock, extra_outputs=[None, None, "", ""], add_stats=True)
        try:
            result = wrapped(*args)
            self._record_restore_result(job, result)
        finally:
            pass

    def _record_restore_result(self, job: QueueJob, result: Any):
        if isinstance(result, tuple):
            result = list(result)
        if isinstance(result, list) and len(result) >= 5:
            restore_result = [result[0], result[2], result[3], result[4]]
        else:
            restore_result = result

        progress.recorded_results.clear()
        progress.record_results(job.task_id, restore_result)


queue = SimpleQueue()


SIMPLE_QUEUE_ASSETS = r"""
<style>
#txt2img_simple_queue_box, #img2img_simple_queue_box {
  gap: 8px;
}
.forge-simple-queue-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 44px;
  gap: 8px;
  align-items: stretch;
  width: 100%;
}
.forge-simple-queue-row button {
  min-width: 0 !important;
}
.forge-simple-queue-view-button {
  width: 44px !important;
  min-width: 44px !important;
  padding: 0 !important;
  font-size: 0 !important;
}
.forge-simple-queue-view-button::before {
  content: "";
  width: 20px;
  height: 20px;
  margin: auto;
  display: block;
  background: currentColor;
  -webkit-mask: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M5 7h14M5 12h14M5 17h14M3 7h.01M3 12h.01M3 17h.01' stroke='black' stroke-width='2.2' stroke-linecap='round'/%3E%3C/svg%3E") center / contain no-repeat;
  mask: url("data:image/svg+xml,%3Csvg viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M5 7h14M5 12h14M5 17h14M3 7h.01M3 12h.01M3 17h.01' stroke='black' stroke-width='2.2' stroke-linecap='round'/%3E%3C/svg%3E") center / contain no-repeat;
}
.forge-simple-queue-hidden-status {
  min-height: 20px;
}
.forge-simple-queue-status-line {
  margin-top: 6px;
  color: #93c5fd;
  font-size: 12px;
  font-weight: 700;
}
.fsq-backdrop {
  position: fixed;
  inset: 0;
  z-index: 100000;
  display: none;
  align-items: center;
  justify-content: center;
  background: rgba(5, 8, 14, 0.62);
  backdrop-filter: blur(5px);
}
.fsq-backdrop.fsq-open {
  display: flex;
}
.fsq-panel {
  width: min(940px, calc(100vw - 36px));
  max-height: min(760px, calc(100vh - 44px));
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.22);
  border-radius: 8px;
  background: #111827;
  box-shadow: 0 24px 70px rgba(0, 0, 0, 0.45);
  color: #e5e7eb;
}
.fsq-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 14px 16px;
  border-bottom: 1px solid rgba(148, 163, 184, 0.18);
}
.fsq-head-main {
  display: flex;
  align-items: center;
  min-width: 0;
  gap: 10px;
  flex-wrap: wrap;
}
.fsq-head-actions {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 0 0 auto;
}
.fsq-select-all {
  width: 34px;
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 8px;
  background: #172033;
  color: #9ca3af;
  cursor: pointer;
  transition: background 0.14s ease, color 0.14s ease, box-shadow 0.14s ease;
}
.fsq-select-all svg {
  width: 16px;
  height: 16px;
}
.fsq-select-all:hover {
  background: #263244;
  color: #f9fafb;
}
.fsq-select-all.fsq-active {
  background: rgba(34, 197, 94, 0.17);
  color: #86efac;
  box-shadow: inset 0 -2px 0 rgba(34, 197, 94, 0.42);
}
.fsq-bulk-delete {
  height: 32px;
  padding: 0 10px;
  border: 1px solid rgba(239, 68, 68, 0.34);
  border-radius: 8px;
  background: rgba(127, 29, 29, 0.22);
  color: #fecaca;
  cursor: pointer;
  font-size: 12px;
  font-weight: 700;
  transition: background 0.14s ease, color 0.14s ease, opacity 0.14s ease;
}
.fsq-bulk-delete:hover:not(:disabled) {
  background: rgba(185, 28, 28, 0.34);
  color: #fee2e2;
}
.fsq-bulk-delete:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
.fsq-bulk-delete[hidden] {
  display: none;
}
.fsq-saved-controls {
  display: grid;
  grid-template-columns: repeat(3, 34px);
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 8px;
  background: #172033;
}
.fsq-saved-control {
  width: 34px;
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-right: 1px solid rgba(148, 163, 184, 0.18);
  background: transparent;
  color: #9ca3af;
  cursor: pointer;
  transition: background 0.14s ease, color 0.14s ease, opacity 0.14s ease;
}
.fsq-saved-control:last-child {
  border-right: 0;
}
.fsq-saved-control svg {
  width: 16px;
  height: 16px;
  fill: none;
  stroke: currentColor;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.fsq-saved-control:hover:not(:disabled) {
  background: #263244;
  color: #f9fafb;
}
.fsq-saved-control:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}
.fsq-saved-control.fsq-ready[data-saved-queue-action="save"] {
  color: #bfdbfe;
}
.fsq-saved-control.fsq-ready[data-saved-queue-action="restore"] {
  color: #93c5fd;
}
.fsq-saved-control.fsq-ready[data-saved-queue-action="clear"] {
  color: #fca5a5;
}
.fsq-saved-control.fsq-control-working {
  background: rgba(96, 165, 250, 0.14);
}
.fsq-queue-controls {
  display: grid;
  grid-template-columns: repeat(3, 36px);
  overflow: hidden;
  border: 1px solid rgba(148, 163, 184, 0.24);
  border-radius: 8px;
  background: #172033;
}
.fsq-control {
  width: 36px;
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-right: 1px solid rgba(148, 163, 184, 0.18);
  background: transparent;
  color: #9ca3af;
  cursor: pointer;
  transition: background 0.14s ease, color 0.14s ease, box-shadow 0.14s ease;
}
.fsq-control:last-child {
  border-right: 0;
}
.fsq-control svg {
  width: 16px;
  height: 16px;
  fill: currentColor;
}
.fsq-control:hover {
  background: #263244;
  color: #f9fafb;
}
.fsq-control-active[data-queue-control="play"] {
  background: rgba(34, 197, 94, 0.17);
  color: #86efac;
  box-shadow: inset 0 -2px 0 rgba(34, 197, 94, 0.42);
}
.fsq-control-active[data-queue-control="pause"] {
  background: rgba(245, 158, 11, 0.17);
  color: #fbbf24;
  box-shadow: inset 0 -2px 0 rgba(245, 158, 11, 0.42);
}
.fsq-control-active[data-queue-control="stop"], .fsq-control-working[data-queue-control="stop"] {
  background: rgba(239, 68, 68, 0.17);
  color: #fca5a5;
  box-shadow: inset 0 -2px 0 rgba(239, 68, 68, 0.42);
}
.fsq-title {
  font-size: 15px;
  font-weight: 700;
}
.fsq-tabs {
  display: flex;
  gap: 6px;
  margin-left: 14px;
}
.fsq-repeat-toggle {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: #cbd5e1;
  font-size: 12px;
  white-space: nowrap;
  user-select: none;
}
.fsq-repeat-toggle input {
  position: absolute;
  opacity: 0;
  pointer-events: none;
}
.fsq-switch {
  width: 36px;
  height: 20px;
  border: 1px solid rgba(148, 163, 184, 0.32);
  border-radius: 999px;
  background: #0f172a;
  position: relative;
  box-sizing: border-box;
}
.fsq-switch::after {
  content: "";
  position: absolute;
  top: 2px;
  left: 2px;
  width: 14px;
  height: 14px;
  border-radius: 999px;
  background: #94a3b8;
  transition: transform 0.14s ease, background 0.14s ease;
}
.fsq-repeat-toggle input:checked + .fsq-switch {
  background: rgba(37, 99, 235, 0.34);
  border-color: rgba(96, 165, 250, 0.45);
}
.fsq-repeat-toggle input:checked + .fsq-switch::after {
  transform: translateX(16px);
  background: #bfdbfe;
}
#forge-simple-queue-repeat-count {
  color: #93c5fd;
}
.fsq-tab {
  height: 28px;
  padding: 0 10px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 7px;
  background: transparent;
  color: #9ca3af;
  cursor: pointer;
}
.fsq-tab.fsq-active {
  background: #263244;
  color: #f9fafb;
}
.fsq-close {
  width: 34px;
  height: 34px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 7px;
  background: #1f2937;
  color: #f9fafb;
  cursor: pointer;
  transition: background 0.14s ease, border-color 0.14s ease, color 0.14s ease, transform 0.14s ease;
}
.fsq-close svg {
  width: 16px;
  height: 16px;
  display: block;
}
.fsq-close:hover {
  background: rgba(127, 29, 29, 0.36);
  border-color: rgba(239, 68, 68, 0.52);
  color: #fecaca;
}
.fsq-close:active {
  transform: translateY(1px);
}
.fsq-body {
  max-height: calc(min(760px, calc(100vh - 44px)) - 64px);
  overflow: auto;
  padding: 12px;
  scrollbar-width: thin;
  scrollbar-color: rgba(148, 163, 184, 0.35) transparent;
}
.fsq-body::-webkit-scrollbar {
  width: 8px;
}
.fsq-body::-webkit-scrollbar-track {
  background: transparent;
}
.fsq-body::-webkit-scrollbar-thumb {
  background: rgba(148, 163, 184, 0.28);
  border-radius: 999px;
}
.fsq-section-title {
  margin: 10px 4px 8px;
  color: #9ca3af;
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.fsq-empty {
  padding: 18px;
  border: 1px dashed rgba(148, 163, 184, 0.22);
  border-radius: 8px;
  color: #9ca3af;
  text-align: center;
}
.fsq-job {
  display: grid;
  grid-template-columns: 30px 26px minmax(0, 1fr) auto;
  gap: 10px;
  align-items: start;
  padding: 10px;
  margin-bottom: 8px;
  border: 1px solid rgba(148, 163, 184, 0.18);
  border-radius: 8px;
  background: #172033;
}
.fsq-job[draggable="true"] {
  cursor: grab;
}
.fsq-job.fsq-dragging {
  opacity: 0.55;
}
.fsq-repeat-check, .fsq-repeat-spacer {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 32px;
}
.fsq-repeat-check input {
  width: 22px;
  height: 22px;
  accent-color: #60a5fa;
  cursor: pointer;
}
.fsq-handle {
  width: 24px;
  height: 28px;
  border: 0;
  background: transparent;
  color: #9ca3af;
  cursor: grab;
  font-size: 17px;
  line-height: 1;
}
.fsq-prompt {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  font-weight: 700;
  color: #f9fafb;
}
.fsq-meta {
  margin-top: 4px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  color: #9ca3af;
  font-size: 12px;
}
.fsq-status {
  display: inline-flex;
  align-items: center;
  height: 22px;
  padding: 0 8px;
  border-radius: 999px;
  background: rgba(59, 130, 246, 0.14);
  color: #93c5fd;
  font-size: 12px;
  font-weight: 700;
}
.fsq-status.paused {
  background: rgba(245, 158, 11, 0.14);
  color: #fbbf24;
}
.fsq-status.done {
  background: rgba(34, 197, 94, 0.14);
  color: #86efac;
}
.fsq-status.failed, .fsq-status.deleted {
  background: rgba(239, 68, 68, 0.14);
  color: #fca5a5;
}
.fsq-actions {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.fsq-actions button, .fsq-save {
  height: 30px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 7px;
  background: #263244;
  color: #f9fafb;
  cursor: pointer;
}
.fsq-actions button {
  padding: 0 9px;
}
.fsq-editor {
  display: none;
  grid-column: 3 / 5;
  gap: 8px;
  margin-top: 8px;
}
.fsq-details {
  display: none;
  grid-column: 3 / 5;
  margin-top: 8px;
  color: #cbd5e1;
  font-size: 12px;
  line-height: 1.45;
}
.fsq-editor.fsq-open, .fsq-details.fsq-open {
  display: grid;
}
.fsq-editor-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}
.fsq-editor textarea {
  min-height: 74px;
  resize: vertical;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 7px;
  background: #0f172a;
  color: #f9fafb;
  padding: 8px;
}
.fsq-cancel {
  background: transparent !important;
  color: #cbd5e1 !important;
}
</style>
"""


class Script(scripts.Script):
    def __init__(self):
        super().__init__()
        self.generate_button = None
        self.queue_column = None
        self.queue_button = None
        self.queue_status = None
        self.queue_bound = False
        script_callbacks.on_app_started(lambda _block, app: self.on_app_started(app))

    def title(self):
        return "Forge Simple Queue"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def after_component(self, component, **_kwargs):
        generate_id = "img2img_generate" if self.is_img2img else "txt2img_generate"
        actions_column_id = "img2img_actions_column" if self.is_img2img else "txt2img_actions_column"

        if component.elem_id != generate_id:
            return

        self.generate_button = component
        self._hook_generate_click(component)
        parent = component.parent
        while parent is not None:
            if parent.elem_id == actions_column_id:
                self._add_queue_button()
                component.parent.children.pop()
                parent.add(self.queue_column)
                break
            parent = parent.parent

    def on_app_started(self, app: FastAPI):
        queue.start()
        queue.register_api(app)

    def _hook_generate_click(self, component):
        if getattr(component, "_forge_simple_queue_hooked", False):
            return

        original_click = component.click
        tab = "img2img" if self.is_img2img else "txt2img"

        def click_with_queue(*args, **kwargs):
            dependency = original_click(*args, **kwargs)
            try:
                inputs = kwargs.get("inputs")
                outputs = kwargs.get("outputs")
                if self._looks_like_generation_click(inputs, outputs):
                    self._bind_queue_button(tab, inputs)
            except Exception:
                print("[Forge Simple Queue] Failed to bind Queue button:")
                traceback.print_exc()
            return dependency

        component.click = click_with_queue
        component._forge_simple_queue_hooked = True

    def _looks_like_generation_click(self, inputs, outputs) -> bool:
        return (
            self.queue_button is not None
            and not self.queue_bound
            and isinstance(inputs, list)
            and isinstance(outputs, list)
            and len(outputs) in (4, 5)
        )

    def _add_queue_button(self):
        tab = "img2img" if self.is_img2img else "txt2img"
        with gr.Column(elem_id=f"{tab}_simple_queue_box") as column:
            self.queue_column = column
            gr.HTML(SIMPLE_QUEUE_ASSETS if not self.is_img2img else "")
            with gr.Row(elem_classes=["forge-simple-queue-row"]):
                self.queue_button = gr.Button("Queue", elem_id=f"{tab}_simple_queue_button", variant="secondary")
                gr.Button("", elem_id=f"{tab}_simple_queue_view", elem_classes=["forge-simple-queue-view-button"])
            self.queue_status = gr.HTML("", elem_id=f"{tab}_simple_queue_status", elem_classes=["forge-simple-queue-hidden-status"])

    def _bind_queue_button(self, tab: str, inputs: list[Any]):
        input_keys = [getattr(component, "elem_id", None) for component in inputs]

        def enqueue_current(request: gr.Request, *args):
            return queue.enqueue(tab, request, args, input_keys)

        self.queue_button.click(
            fn=enqueue_current,
            inputs=inputs,
            outputs=self.queue_status,
            show_progress=False,
            queue=False,
        )
        self.queue_bound = True
        print(f"[Forge Simple Queue] Queue button bound for {tab}.")
