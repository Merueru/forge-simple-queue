from __future__ import annotations

import copy
import pickle
import re
import hashlib
import html
from io import BytesIO
import json
import math
import statistics
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import Body, FastAPI
import gradio as gr
from gradio.context import get_blocks_context

from modules import call_queue, img2img, progress, script_callbacks, scripts, shared, txt2img


HISTORY_LIMIT = 30
THUMBNAIL_SIZE = 96
ETA_FILE_LIMIT = 1_000_000
ETA_SAMPLES_PER_PROFILE = 10


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


def loaded_model_signature() -> str | None:
    try:
        from modules.sd_models import model_data
    except ImportError:
        return None
    return model_data.forge_hash


def timing_digest(value: Any) -> str:
    serialized = json.dumps(stable_fingerprint_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("ascii")).hexdigest()[:32]


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


def forge_generation_active() -> bool:
    state = getattr(shared, "state", None)
    job_count = getattr(state, "job_count", 0)
    return (
        getattr(progress, "current_task", None) is not None
        or bool(getattr(state, "job", ""))
        or (isinstance(job_count, int) and job_count != 0)
    )


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
        source: str = "queue",
        name: str = "",
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
        self.source = source
        self.name = name
        self.thumbnail: bytes | None = None
        self.created_at = time.time()
        self.started_at: float | None = None
        self.generation_started_at: float | None = None
        self.tracks_generation_stage = False
        self.eta_load_state: str | None = None
        self.live_eta_enabled = False
        self.live_eta_steps: int | None = None
        self.live_eta_observations: deque[tuple[int, float]] = deque(maxlen=5)
        self.completed_at: float | None = None
        self.status = "queued"
        self.paused = repeat_placeholder
        self.cancel_requested = False
        self.error: str | None = None
        self.termination_reason: str | None = None
        self.editing = False
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

    def duration_seconds(self) -> float | None:
        if self.started_at is None or self.completed_at is None:
            return None
        return max(0.0, self.completed_at - self.started_at)

    def supports_live_eta(self, processing: Any) -> bool:
        if self.tab == "img2img":
            # Forge sends the tab mode after the task ID and the resize mode
            # selector immediately before height. Older saved jobs use this layout too.
            if "img2img_height" not in self.input_keys:
                return False
            height_index = self.input_keys.index("img2img_height")
            if height_index <= 1 or len(self.args) <= height_index or self.args[1] != 0 or self.args[height_index - 1] != 0:
                return False
        elif self.tab != "txt2img":
            return False
        return (
            getattr(processing, "n_iter", None) == 1
            and getattr(processing, "batch_size", None) == 1
            and not getattr(processing, "enable_hr", False)
            and getattr(processing, "refiner_checkpoint", None) in (None, "", "None", "none")
            and not any(value for key, value in zip(self.input_keys, self.args)
                        if key and key.endswith("ad_main_accordion-checkbox"))
        )

    def eta_duration_seconds(self) -> float | None:
        if self.generation_started_at is not None and self.completed_at is not None:
            return max(0.0, self.completed_at - self.generation_started_at)
        return self.duration_seconds()

    def eta_sample_profile(self) -> str:
        profile = self.timing_profile()
        if self.generation_started_at is None:
            return profile
        state = self.eta_load_state
        return f"{profile}:generation:{state}" if state else f"{profile}:generation"

    def timing_profile(self) -> str:
        loras = re.findall(r"<lora:([^>]+)>", self.prompt, flags=re.IGNORECASE)
        if self._arg_by_key("txt2img_hr-checkbox", "txt2img_hr", "txt2img_enable_hr"):
            hires_prompt = self._string_value("hires_prompt", "txt2img_hr_prompt")
            hires_loras = re.findall(r"<lora:([^>]+)>", hires_prompt, flags=re.IGNORECASE)
        else:
            hires_loras = []
        return timing_digest({
            "tab": self.tab,
            "summary": self.summary(),
            "forge": self.forge_state,
            "workload": self.timing_workload(),
            "loras": sorted(loras),
            "hires_loras": sorted(hires_loras),
        })

    def timing_workload(self) -> dict[str, Any]:
        def is_prompt(key: str) -> bool:
            return key in ("prompt", "negative_prompt") or key.endswith(("_prompt", "_prompts"))

        def timing_value(value: Any) -> Any:
            if isinstance(value, dict):
                return {str(key): timing_value(item) for key, item in value.items()
                        if not is_prompt(str(key))}
            if isinstance(value, (list, tuple)):
                return [timing_value(item) for item in value]
            return stable_fingerprint_value(value)

        def normalize(key: str | None, value: Any) -> Any:
            if key and key.endswith("ad_tab_order") and isinstance(value, str):
                try:
                    value = json.loads(value)
                except (TypeError, ValueError):
                    pass
            return timing_value(value)

        hires = self._arg_by_key("txt2img_hr-checkbox", "txt2img_hr", "txt2img_enable_hr")
        refiner = self._arg_by_key(f"{self.tab}_checkpoint")

        def is_hires(key: str) -> bool:
            return key.startswith("hr_") or "_hr_" in key or "hires" in key or key.endswith(("_hr", "_hr-checkbox", "_enable_hr"))

        # Forge's first unnamed input is the per-run task ID, not a workload setting.
        workload = {
            str(key) if key else f"unnamed_{index}": normalize(key, value)
            for index, (key, value) in enumerate(zip(self.input_keys, self.args))
            if not (index == 0 and not key) and not (key and is_prompt(key))
            and (not key or key.startswith("script_") or "controlnet" in key.lower()
            or is_hires(key) or key.endswith("_denoising_strength"))
            and not (hires is False and key and (is_hires(key) or key == "txt2img_denoising_strength"))
        }
        if self._arg_by_key(f"{self.tab}_enable-checkbox") and refiner not in (None, "", "None", "none"):
            workload["refiner"] = [refiner, self._arg_by_key(f"{self.tab}_switch_at"), self._arg_by_key(f"{self.tab}_cfg")]
        return workload

    def to_dict(
        self,
        index: int | None = None,
        repeat_selected: bool = False,
        runs: int = 1,
        failures: int = 0,
        deleted: int = 0,
        skipped: int = 0,
        interrupted: int = 0,
    ) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "repeat_id": self.repeat_id,
            "repeat_selected": repeat_selected,
            "repeat_placeholder": self.repeat_placeholder,
            "source": self.source,
            "name": self.name,
            "tab": self.tab,
            "index": index,
            "status": "paused" if self.paused and self.status == "queued" else self.status,
            "editing": self.editing,
            "paused": self.paused,
            "progress_active": self.task_id == getattr(progress, "current_task", None),
            "progress_queued": self.task_id in getattr(progress, "pending_tasks", {}),
            "age": max(0, int(time.time() - self.created_at)),
            "duration_seconds": self.duration_seconds(),
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "summary": self.summary(),
            "forge": summarize_forge_state(self.forge_state),
            "error": self.error,
            "editable": self.status in ("queued", "waiting") and self.task_id != getattr(progress, "current_task", None),
            "runs": runs,
            "failures": failures,
            "deleted": deleted,
            "skipped": skipped,
            "interrupted": interrupted,
        }

    def summary(self) -> dict[str, Any]:
        if self._summary_snapshot is not None:
            return dict(self._summary_snapshot)
        try:
            width = self._arg_by_key(f"{self.tab}_width")
            height = self._arg_by_key(f"{self.tab}_height")
            batch_count = self._arg_by_key(f"{self.tab}_batch_count")
            batch_size = self._arg_by_key(f"{self.tab}_batch_size")
            refiner = self._arg_by_key(f"{self.tab}_checkpoint")
            if not self._arg_by_key(f"{self.tab}_enable-checkbox") or refiner in (None, "", "None", "none"):
                refiner = None
            return {
                "width": width,
                "height": height,
                "batch_count": batch_count,
                "batch_size": batch_size,
                "has_refiner": all(f"{self.tab}_{suffix}" in self.input_keys for suffix in ("enable-checkbox", "checkpoint")),
                "cfg": self._arg_by_key(f"{self.tab}_cfg_scale"),
                "distilled_cfg": self._arg_by_key(f"{self.tab}_distilled_cfg_scale"),
                "denoising": self._arg_by_key(f"{self.tab}_denoising_strength") if self.tab == "img2img" else None,
                "refiner": refiner,
                "refiner_switch": self._arg_by_key(f"{self.tab}_switch_at") if refiner else None,
                "refiner_cfg": self._arg_by_key(f"{self.tab}_cfg") if refiner else None,
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
            [_copy_value(value) for value in self.args],
            list(self.input_keys),
            _copy_value(self.forge_state),
            self.username,
            repeat_id=self.repeat_id,
            repeat_placeholder=self.repeat_placeholder,
            source=self.source,
            name=self.name,
        )
        history_job.thumbnail = self.thumbnail if self.status == "done" else None
        history_job.created_at = self.created_at
        history_job.started_at = self.started_at
        history_job.generation_started_at = self.generation_started_at
        history_job.eta_load_state = self.eta_load_state
        history_job.completed_at = self.completed_at
        history_job.status = self.status
        history_job.paused = self.paused
        history_job.cancel_requested = self.cancel_requested
        history_job.error = self.error
        history_job.termination_reason = self.termination_reason
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
            name=self.name,
        )

    def clone_as_copy(self) -> "QueueJob":
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
        self._eta_path = Path(__file__).resolve().parent.parent / "data" / "eta_timings.json"
        self._eta_timings = self._load_eta_timings()
        self._eta_fallback_cache: dict[tuple[str, str | None], tuple[float | None, int, bool]] = {}
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
        self._recovery_path = Path(__file__).resolve().parent.parent / "data" / "recovery_queue.pkl"
        self._recovery_snapshot: dict[str, Any] | None = self._load_recovery_snapshot()
        self._recovery_enabled = bool(self._recovery_snapshot.get("enabled", True)) if self._recovery_snapshot else True
        self._recovered_count = 0
        self._recovery_checkpoint_timer: threading.Timer | None = None
        self._direct_termination_reason: str | None = None
        with self._condition:
            self._restore_recovery_payload_locked(self._recovery_snapshot)
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
            return self.move(
                str(payload.get("id", "")),
                int(payload.get("index", 0)),
                target_id=payload.get("target_id"),
                after=payload.get("after") is True,
            )

        @app.get("/forge-simple-queue/thumbnail/{job_id}")
        def thumbnail(job_id: str):
            from fastapi import HTTPException, Response

            with self._condition:
                data = next((job.thumbnail for job in self._history if job.id == job_id), None)
            if not data:
                raise HTTPException(status_code=404)
            return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

        @app.get("/forge-simple-queue/checkpoints")
        def checkpoints():
            return shared.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)

        @app.post("/forge-simple-queue/settings")
        def settings(payload: dict[str, Any] = Body(default={})):
            return self.update_settings(str(payload.get("id", "")), payload)

        @app.post("/forge-simple-queue/rename")
        def rename(payload: dict[str, Any] = Body(default={})):
            return self.rename_job(str(payload.get("id", "")), payload.get("name"))

        @app.post("/forge-simple-queue/update")
        def update(payload: dict[str, Any] = Body(default={})):
            return self.update_job(
                str(payload.get("id", "")),
                payload.get("prompt"),
                payload.get("negative_prompt"),
            )

        @app.post("/forge-simple-queue/full-edit/cancel")
        def cancel_full_edit(payload: dict[str, Any] = Body(default={})):
            return self.cancel_full_edit(str(payload.get("id", "")))

        @app.post("/forge-simple-queue/copy")
        def copy_job(payload: dict[str, Any] = Body(default={})): 
            return self.copy_job(str(payload.get("id", "")))

        @app.post("/forge-simple-queue/reuse")
        def reuse_job(payload: dict[str, Any] = Body(default={})): 
            result = self.reuse_job(str(payload.get("id", "")))
            return {key: value for key, value in result.items() if key != "args"}

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

        @app.post("/forge-simple-queue/recovery")
        def recovery(payload: dict[str, Any] = Body(default={})):
            return self.set_recovery_enabled(bool(payload.get("enabled", True)))

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
            return self.skip()

    def enqueue(
        self,
        tab: str,
        request: gr.Request | None,
        raw_args: tuple[Any, ...],
        input_keys: list[str | None],
        *,
        checkpoint: str | None = None,
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
        forge_state = capture_forge_state()
        if checkpoint:
            forge_state["sd_model_checkpoint"] = checkpoint
        job = QueueJob(job_id, task_id, tab, args, input_keys, forge_state, username)

        with self._condition:
            job.name = self._default_job_name_locked(job)
            self._pending.append(job)
            self._condition.notify_all()
        self._schedule_recovery_checkpoint()

        print(f"[Forge Simple Queue] Queued {tab} job {job_id}.")
        return (
            f"<div class='forge-simple-queue-status-line' data-tab='{html.escape(tab)}' data-task-id='{html.escape(task_id)}'>"
            f"Queued {html.escape(tab)} job {html.escape(job_id)}."
            "</div>"
        )

    def _is_queue_generation(self, task_id: str | None = None) -> bool:
        task_id = str(task_id or getattr(getattr(shared, "state", None), "job", "") or getattr(progress, "current_task", None) or "")
        if task_id.startswith("task(simple-queue-"):
            return True
        with self._condition:
            active = self._active or self._waiting
            return active is not None and (not task_id or active.task_id == task_id)

    def record_generation_thumbnail(self, processed: Any) -> bytes | None:
        try:
            from PIL import Image

            images = processed.images
            index = processed.index_of_first_image
            if not 0 <= index < len(images) or not isinstance(images[index], Image.Image):
                return None
            thumbnail = images[index]
            scale = THUMBNAIL_SIZE / max(thumbnail.size)
            if scale < 1:
                size = tuple(max(1, round(side * scale)) for side in thumbnail.size)
                thumbnail = thumbnail.resize(size, Image.Resampling.BILINEAR, reducing_gap=2.0)
            with BytesIO() as buffer:
                thumbnail.convert("RGB").save(buffer, "JPEG", quality=72)
                return buffer.getvalue()
        except Exception:
            return None

    def record_direct_generation(self, tab: str, processing: Any, processed: Any = None) -> bool:
        if getattr(processing, "is_api", False) or getattr(processing, "txt2img_upscale", False):
            return False
        if getattr(processing, "_forge_simple_queue_history_recorded", False):
            return False
        with self._condition:
            requested_termination = self._direct_termination_reason
            self._direct_termination_reason = None
        if requested_termination:
            return False
        state = getattr(shared, "state", None)
        if state is not None and any(
            getattr(state, flag, False)
            for flag in ("interrupted", "skipped", "stopping_generation")
        ):
            return False

        task_id = str(
            getattr(getattr(shared, "state", None), "job", "")
            or getattr(progress, "current_task", None)
            or f"task(generate-{uuid4().hex[:12]})"
        )
        if self._is_queue_generation(task_id):
            return False

        job_id = uuid4().hex[:12]
        input_keys = [
            "id_task",
            f"{tab}_prompt",
            f"{tab}_neg_prompt",
            f"{tab}_steps",
            f"{tab}_sampling",
            f"{tab}_scheduler",
            f"{tab}_width",
            f"{tab}_height",
            f"{tab}_batch_count",
            f"{tab}_batch_size",
            f"{tab}_cfg_scale",
            f"{tab}_denoising_strength",
        ]
        args = [
            task_id,
            _copy_value(getattr(processing, "prompt", "")),
            _copy_value(getattr(processing, "negative_prompt", "")),
            _copy_value(getattr(processing, "steps", None)),
            _copy_value(getattr(processing, "sampler_name", None)),
            _copy_value(getattr(processing, "scheduler", None)),
            _copy_value(getattr(processing, "width", None)),
            _copy_value(getattr(processing, "height", None)),
            _copy_value(getattr(processing, "n_iter", None)),
            _copy_value(getattr(processing, "batch_size", None)),
            _copy_value(getattr(processing, "cfg_scale", None)),
            _copy_value(getattr(processing, "denoising_strength", None)),
        ]
        refiner = getattr(processing, "refiner_checkpoint", None)
        input_keys.extend([
            f"{tab}_distilled_cfg_scale", f"{tab}_enable-checkbox", f"{tab}_checkpoint",
            f"{tab}_switch_at", f"{tab}_cfg",
        ])
        args.extend([
            getattr(processing, "distilled_cfg_scale", None), bool(refiner), refiner,
            getattr(processing, "refiner_switch_at", None), getattr(processing, "refiner_cfg", None) or 0,
        ])
        if tab == "txt2img":
            input_keys.append("txt2img_hr")
            args.append(bool(getattr(processing, "enable_hr", False)))
            if getattr(processing, "enable_hr", False):
                for attribute in ("hr_scale", "hr_resize_x", "hr_resize_y", "hr_second_pass_steps",
                                  "hr_upscaler", "hr_checkpoint_name", "hr_sampler_name", "hr_scheduler"):
                    input_keys.append(f"{tab}_{attribute}")
                    args.append(_copy_value(getattr(processing, attribute, None)))
        if getattr(processing, "script_args", None):
            input_keys.append("script_direct_args")
            args.append(_copy_value(processing.script_args))
        job = QueueJob(
            job_id,
            task_id,
            tab,
            args,
            input_keys,
            capture_forge_state(),
            getattr(processing, "user", None),
            source="generate",
        )
        started_at = getattr(processing, "_forge_simple_queue_started_at", None) or time.time()
        job.created_at = started_at
        job.started_at = started_at
        job.generation_started_at = getattr(processing, "_forge_simple_queue_generation_started_at", None)
        job.eta_load_state = getattr(processing, "_forge_simple_queue_eta_load_state", None)
        job.completed_at = time.time()
        job.status = "done"
        if processed is not None:
            job.thumbnail = self.record_generation_thumbnail(processed)
        with self._condition:
            self._append_history_locked(job)
        processing._forge_simple_queue_history_recorded = True
        return True

    def snapshot(self, compact: bool = False) -> dict[str, Any]:
        with self._condition:
            active_job = self._active or self._waiting
            active = self._compact_job_dict(active_job) if compact else (self._job_dict(active_job) if active_job is not None else None)
            pending_jobs = [job for job in self._pending if job is not self._waiting]
            pending = (
                [self._compact_job_dict(job) for job in pending_jobs]
                if compact
                else [self._job_dict(job, index=i) for i, job in enumerate(pending_jobs)]
            )
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
            eta = self._estimate_remaining_locked()
        return {
            "active": active,
            "pending": pending,
            "history": history,
            "pending_count": len(pending_jobs),
            "queue_count": queue_count,
            "generation_active": forge_generation_active(),
            "compact": compact,
            "recent_tasks": recent_tasks,
            "repeat": repeat,
            "control": control,
            "saved_queue": saved_queue,
            "eta": eta,
            "recovery": {
                "enabled": self._recovery_enabled,
                "recovered_count": self._recovered_count,
            },
        }

    def _recovery_payload_locked(self) -> dict[str, Any]:
        return {
            "version": 1,
            "saved_at": time.time(),
            "enabled": self._recovery_enabled,
            "jobs": [self._job_snapshot(job) for job in self._recovery_queue_jobs_locked()],
        }

    def _recovery_queue_jobs_locked(self) -> list[QueueJob]:
        jobs: list[QueueJob] = []
        active_job = self._active or self._waiting
        if active_job is not None and not active_job.cancel_requested:
            jobs.append(active_job)
        for job in self._pending:
            if job is active_job or job is self._waiting or job.cancel_requested:
                continue
            jobs.append(job)
        return jobs

    def _checkpoint_recovery(self):
        with self._condition:
            self._recovery_checkpoint_timer = None
            if not self._recovery_enabled:
                return
            payload = self._recovery_payload_locked()
        try:
            self._write_recovery_snapshot(payload)
        except Exception:
            print("[Forge Simple Queue] Failed to checkpoint recovery queue:")
            traceback.print_exc()

    def _schedule_recovery_checkpoint(self):
        with self._condition:
            if not self._recovery_enabled:
                return
            if self._recovery_checkpoint_timer is not None:
                self._recovery_checkpoint_timer.cancel()
            timer = threading.Timer(0.5, self._checkpoint_recovery)
            timer.daemon = True
            self._recovery_checkpoint_timer = timer
            timer.start()

    def _restore_recovery_payload_locked(self, payload: dict[str, Any] | None):
        if not self._recovery_enabled or not isinstance(payload, dict):
            return
        jobs = [self._job_from_snapshot(item) for item in payload.get("jobs", [])]
        jobs = [job for job in jobs if job is not None]
        if not jobs:
            return
        self._pending = deque(jobs)
        self._queue_paused = True
        self._queue_mode = "pause"
        self._repeat_enabled = False
        self._repeat_order = []
        self._repeat_templates = {}
        self._recovered_count = len(jobs)

    def set_recovery_enabled(self, enabled: bool) -> dict[str, Any]:
        with self._condition:
            self._recovery_enabled = enabled
            if self._recovery_checkpoint_timer is not None:
                self._recovery_checkpoint_timer.cancel()
                self._recovery_checkpoint_timer = None
            payload = self._recovery_payload_locked()
            if not enabled:
                payload["jobs"] = []
        try:
            self._write_recovery_snapshot(payload)
        except Exception:
            print("[Forge Simple Queue] Failed to update recovery setting:")
            traceback.print_exc()
            return {"ok": False, "message": "Failed to update recovery setting."}
        data = self.snapshot()
        data.update({"ok": True, "message": "Recovery enabled." if enabled else "Recovery disabled."})
        return data

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

        self._checkpoint_recovery()
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

    def _load_eta_timings(self) -> dict[str, Any]:
        try:
            with self._eta_path.open("rb") as handle:
                data = handle.read(ETA_FILE_LIMIT + 1)
            if len(data) > ETA_FILE_LIMIT:
                return {}
            payload = json.loads(data)
            if not isinstance(payload, dict) or payload.get("version") != 1:
                return {}
            timings = payload.get("profiles")
            if not isinstance(timings, dict):
                return {}
            valid = {}
            for profile, entry in timings.items():
                if not isinstance(entry, dict):
                    continue
                features, durations = entry.get("features"), entry.get("durations")
                if not isinstance(features, dict) or not isinstance(durations, list):
                    continue
                if not all(key in features for key in (
                    "tab", "forge", "checkpoint", "steps", "dimensions", "batch", "batch_size",
                    "sampler", "schedule", "workload", "extra_processing",
                )):
                    continue
                dimensions = features["dimensions"]
                numbers = [features[key] for key in ("steps", "batch", "batch_size")]
                if dimensions is not None:
                    if not isinstance(dimensions, list) or len(dimensions) != 2:
                        continue
                    numbers.extend(dimensions)
                if any(value is not None and (type(value) not in (int, float) or not 0 < value < 1e9) for value in numbers):
                    continue
                if dimensions is not None and any(value is None for value in dimensions):
                    continue
                samples = [duration for duration in durations if type(duration) in (int, float) and 0 < duration < 1e9]
                if samples:
                    valid[profile] = {"features": features, "durations": samples[-ETA_SAMPLES_PER_PROFILE:]}
            return valid
        except (OSError, ValueError, RecursionError):
            return {}

    def _record_eta_timing_locked(self, job: QueueJob, duration: float):
        self._eta_fallback_cache.clear()
        profile = job.eta_sample_profile()
        existing = self._eta_timings.pop(profile, None)
        durations = existing["durations"] if existing else []
        self._eta_timings[profile] = {
            "features": self._timing_features(job),
            "durations": [*durations, duration][-ETA_SAMPLES_PER_PROFILE:],
        }

        def encode_timings():
            return json.dumps({"version": 1, "profiles": self._eta_timings}, separators=(",", ":")).encode("ascii")

        data = encode_timings()
        while len(data) > ETA_FILE_LIMIT and self._eta_timings:
            self._eta_timings.pop(next(iter(self._eta_timings)))
            data = encode_timings()

        temporary = self._eta_path.with_suffix(f".{uuid4().hex}.tmp")
        try:
            self._eta_path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(data)
            temporary.replace(self._eta_path)
        except OSError as exc:
            print(f"[Forge Simple Queue] Could not save ETA timings: {exc}")
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    def _load_recovery_snapshot(self) -> dict[str, Any] | None:
        try:
            if not self._recovery_path.exists():
                return None
            with self._recovery_path.open("rb") as handle:
                payload = pickle.load(handle)
            if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), list):
                return None
            return payload
        except Exception:
            print("[Forge Simple Queue] Failed to load recovery queue:")
            traceback.print_exc()
            return None

    def _write_saved_snapshot(self, payload: dict[str, Any]):
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._snapshot_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self._snapshot_path)

    def _write_recovery_snapshot(self, payload: dict[str, Any]):
        self._recovery_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._recovery_path.with_suffix(".tmp")
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(self._recovery_path)

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
            "name": job.name,
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
            name=str(item.get("name") or "")[:60],
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
            "editing": job.editing,
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
                if job.thumbnail and not group.get("thumbnail"):
                    group["thumbnail"] = f"/forge-simple-queue/thumbnail/{job.id}"
                if job.status == "done":
                    group["runs"] += 1
                if job.status == "failed":
                    group["failures"] += 1
                if job.status == "deleted":
                    group["deleted"] += 1
                if job.status == "skipped":
                    group["skipped"] += 1
                if job.status == "interrupted":
                    group["interrupted"] += 1
                self._add_history_duration(group, job)
                continue

            group = self._job_dict(
                job,
                runs=1 if job.status == "done" else 0,
                failures=1 if job.status == "failed" else 0,
                deleted=1 if job.status == "deleted" else 0,
                skipped=1 if job.status == "skipped" else 0,
                interrupted=1 if job.status == "interrupted" else 0,
            )
            if job.thumbnail:
                group["thumbnail"] = f"/forge-simple-queue/thumbnail/{job.id}"
            group["timed_runs"] = 0
            group["_duration_total"] = 0.0
            self._add_history_duration(group, job)
            by_fingerprint[fingerprint] = group
            groups.append(group)
            if len(groups) >= 30:
                break
        for group in groups:
            timed_runs = group.pop("timed_runs")
            duration_total = group.pop("_duration_total")
            group["duration_seconds"] = duration_total / timed_runs if timed_runs else None
            group["timed_runs"] = timed_runs
        return groups

    @staticmethod
    def _add_history_duration(group: dict[str, Any], job: QueueJob):
        if job.status != "done":
            return
        duration = job.duration_seconds()
        if duration is not None:
            group["timed_runs"] += 1
            group["_duration_total"] += duration

    @staticmethod
    def _timing_features(job: QueueJob) -> dict[str, Any]:
        def number(key: str) -> float | None:
            value = job._arg_by_key(key, default=None)
            try:
                result = float(value) if value is not None else None
                return result if result is not None and math.isfinite(result) and result > 0 else None
            except (TypeError, ValueError):
                return None

        width = number(f"{job.tab}_width")
        height = number(f"{job.tab}_height")
        dimensions = sorted((width, height)) if width and height else None
        batch_count = number(f"{job.tab}_batch_count")
        batch_size = number(f"{job.tab}_batch_size")
        batch = batch_count * batch_size if batch_count and batch_size else None
        workload = job.timing_workload()
        checkpoint = job.forge_state.get("sd_model_checkpoint")
        sampler = job._arg_by_key(f"{job.tab}_sampling")
        schedule = job._arg_by_key(f"{job.tab}_scheduler")
        evaluations = None
        if isinstance(sampler, str) and sampler:
            try:
                from modules.sd_samplers import all_samplers_map
            except ImportError:
                all_samplers_map = {}
            config = all_samplers_map.get(sampler)
            if config is not None:
                evaluations = config.total_steps(1)
        return {
            "tab": job.tab,
            "forge": timing_digest({key: value for key, value in job.forge_state.items() if value is not None}),
            "checkpoint": timing_digest(checkpoint) if checkpoint else None,
            "steps": number(f"{job.tab}_steps"),
            "sampler": timing_digest(sampler) if sampler is not None else None,
            "schedule": timing_digest(schedule) if schedule is not None else None,
            "evaluations": evaluations,
            "dimensions": dimensions,
            "batch": batch,
            "batch_size": batch_size,
            "workload": timing_digest(workload),
            "extra_processing": any(value for key, value in workload.items() if not key.endswith("_denoising_strength")),
            "load_state": job.eta_load_state,
        }

    @classmethod
    def _timing_similarity_score(cls, target: dict[str, Any], candidate: dict[str, Any]) -> float | None:
        if target["tab"] != candidate["tab"] or target["forge"] != candidate["forge"]:
            return None
        if target["workload"] != candidate["workload"]:
            return None
        if any(target[key] is None or candidate[key] is None
               for key in ("dimensions", "steps", "batch", "batch_size", "sampler", "schedule")):
            return None
        if any(target[key] != candidate[key] for key in ("sampler", "schedule", "batch_size")):
            return None

        score = 0.0
        comparable = 0

        target_dimensions = target["dimensions"]
        candidate_dimensions = candidate["dimensions"]
        if target_dimensions and candidate_dimensions:
            comparable += 1
            dimension_delta = max(
                target_size / candidate_size if target_size >= candidate_size else candidate_size / target_size
                for target_size, candidate_size in zip(target_dimensions, candidate_dimensions)
            ) - 1.0
            if dimension_delta > 0.4:
                return None
            score += 4.0 if dimension_delta <= 0.05 else 3.0 if dimension_delta <= 0.15 else 1.0

        target_steps = target["steps"]
        candidate_steps = candidate["steps"]
        if target_steps and candidate_steps:
            comparable += 1
            steps_delta = abs(target_steps - candidate_steps) / max(target_steps, candidate_steps, 1.0)
            if steps_delta > 0.5:
                return None
            score += 3.0 if steps_delta <= 0.1 else 2.0 if steps_delta <= 0.25 else 1.0

        target_batch = target["batch"]
        candidate_batch = candidate["batch"]
        if target_batch and candidate_batch:
            comparable += 1
            batch_delta = max(target_batch, candidate_batch) / min(target_batch, candidate_batch) - 1.0
            if batch_delta > 1.0:
                return None
            score += 2.0 if batch_delta <= 0.05 else 1.0

        if target["sampler"] is not None and candidate["sampler"] is not None:
            comparable += 1
            if target["sampler"] == candidate["sampler"]:
                score += 1.0
        if target["schedule"] is not None and candidate["schedule"] is not None:
            comparable += 1
            if target["schedule"] == candidate["schedule"]:
                score += 1.0

        return score if comparable and score >= 3.0 else None

    @classmethod
    def _estimate_similar_duration_locked(
        cls,
        job: QueueJob,
        history_samples: list[tuple[dict[str, Any], float]],
    ) -> tuple[float | None, int]:
        target = cls._timing_features(job)
        ranked: list[tuple[float, float]] = []
        for candidate, duration in history_samples:
            score = cls._timing_similarity_score(target, candidate)
            if score is not None:
                ratio = (target["steps"] / candidate["steps"]
                         * math.prod(target["dimensions"]) / math.prod(candidate["dimensions"])
                         * target["batch"] / candidate["batch"])
                if ratio != 1.0 and target["extra_processing"]:
                    # Keep a fixed allowance for setup and extra passes when only
                    # the base step count changes in an otherwise identical warm run.
                    if (candidate.get("load_state") != "warm"
                            or target["dimensions"] != candidate["dimensions"]
                            or target["batch"] != candidate["batch"]
                            or not 0.8 <= ratio <= 1.25):
                        continue
                    ratio = 1.0 + (ratio - 1.0) * 0.85
                # Avoid extrapolating far beyond measured workloads.
                if 0.5 <= ratio <= 2.0:
                    ranked.append((score, duration * ratio))
        if not ranked:
            return None, 0

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score = ranked[0][0]
        durations = [duration for score, duration in ranked if score >= best_score - 1.0][:5]
        return float(statistics.median(durations)), len(durations)

    @classmethod
    def _estimate_rough_duration(cls, job: QueueJob, history_samples: list[tuple[dict[str, Any], float]]) -> tuple[float | None, int]:
        target = cls._timing_features(job)
        checkpoint = target["checkpoint"]
        if not checkpoint or any(target[key] is None for key in ("steps", "dimensions", "batch")):
            return None, 0
        ranked = []
        for candidate, duration in history_samples:
            if any(candidate[key] is None for key in ("steps", "dimensions", "batch")):
                continue
            ratio = (target["steps"] / candidate["steps"]
                     * math.prod(target["dimensions"]) / math.prod(candidate["dimensions"])
                     * target["batch"] / candidate["batch"])
            if not math.isfinite(ratio) or ratio <= 0:
                continue
            # Prefer timings from the same model and similar pipelines.
            score = (
                4 * (checkpoint == candidate["checkpoint"])
                + 2 * (target["tab"] == candidate["tab"])
                + 2 * (target["workload"] == candidate["workload"])
                + (target["sampler"] == candidate["sampler"])
                + (target["batch_size"] == candidate["batch_size"])
                - abs(math.log2(ratio))
            )
            # Use Forge's per-step evaluation count only for an otherwise similar
            # workload. Keep some of the measured time for work outside the sampler.
            target_evaluations = target.get("evaluations")
            candidate_evaluations = candidate.get("evaluations")
            if (target_evaluations in (1, 2) and candidate_evaluations in (1, 2)
                    and target_evaluations != candidate_evaluations
                    and checkpoint == candidate["checkpoint"] and target["tab"] == candidate["tab"]
                    and target["workload"] == candidate["workload"]
                    and target["dimensions"] == candidate["dimensions"] and target["batch"] == candidate["batch"]):
                ratio *= target_evaluations / candidate_evaluations
                scaled = duration * (1.0 + 0.85 * (max(0.25, min(4.0, ratio)) - 1.0))
            else:
                scaled = duration * math.sqrt(max(0.25, min(4.0, ratio)))
            ranked.append((score, scaled))
        if not ranked:
            return None, 0
        ranked.sort(key=lambda sample: sample[0], reverse=True)
        best = ranked[0][0]
        durations = [duration for score, duration in ranked if score >= best - 1.5][:5]
        return float(statistics.median(durations)), len(durations)

    @staticmethod
    def _predict_eta_load_state(job: QueueJob) -> str | None:
        if job.eta_load_state is not None:
            return job.eta_load_state
        try:
            from modules.sd_models import checkpoint_aliases, model_data
            from modules_forge.main_entry import forge_unet_storage_dtype_options
        except ImportError:
            return None

        if job._arg_by_key(f"{job.tab}_override_settings"):
            return None
        checkpoint = job.forge_state.get("sd_model_checkpoint")
        if not isinstance(checkpoint, str) or not checkpoint:
            return None
        expected = checkpoint_aliases.get(checkpoint)
        if expected is None:
            return None
        modules = job.forge_state.get("forge_additional_modules", getattr(shared.opts, "forge_additional_modules", []))
        dtype_name = job.forge_state.get("forge_unet_storage_dtype", getattr(shared.opts, "forge_unet_storage_dtype", None))
        dtype, _ = forge_unet_storage_dtype_options.get(dtype_name, (None, False))
        requested = dict(checkpoint_info=expected, additional_modules=modules, unet_storage_dtype=dtype)
        return "warm" if model_data.forge_hash == str(requested) else "reload"

    @staticmethod
    def _live_remaining_seconds(job: QueueJob, now: float) -> float | None:
        if not job.live_eta_enabled or getattr(progress, "current_task", None) != job.task_id:
            return None
        state = getattr(shared, "state", None)
        if state is None or getattr(state, "job_no", None) != 0:
            return None
        if getattr(state, "interrupted", False) or getattr(state, "skipped", False):
            return None
        total, completed = getattr(state, "sampling_steps", 0), getattr(state, "preview_step", 0)
        if total < 8:
            return None
        observations = job.live_eta_observations
        if observations and (completed < observations[-1][0] or total != job.live_eta_steps):
            job.live_eta_enabled = False
            return None
        if not 0 < completed < total:
            return None
        job.live_eta_steps = total
        if not observations or (completed > observations[-1][0] and now - observations[-1][1] >= 0.3):
            observations.append((completed, now))
        first_step, first_at = observations[0]
        last_step, last_at = observations[-1]
        span = last_step - first_step
        if completed != last_step or span < 4 or last_at - first_at < 0.75:
            return None
        seconds_per_step = (last_at - first_at) / span
        stale_for = now - last_at
        if stale_for > max(2.0, seconds_per_step * 3):
            return None
        # Leave about one step's worth of time for decoding and finalization.
        return max(seconds_per_step, seconds_per_step * (total - last_step + 1) - stale_for)

    def _estimate_remaining_locked(self, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        active_job = self._active or self._waiting
        stage = None
        if self._active is not None and self._active.status == "running" and self._active.tracks_generation_stage:
            stage = "generating" if self._active.generation_started_at is not None else "preparing"
        queue_jobs: list[QueueJob] = []
        seen: set[int] = set()
        for job in [active_job, *self._pending]:
            if job is None or id(job) in seen or job.cancel_requested:
                continue
            seen.add(id(job))
            queue_jobs.append(job)
        if not queue_jobs:
            return {"seconds": None, "sample_count": 0, "unknown_jobs": 0}

        samples: dict[str, list[float]] = {}
        recent_samples = []
        for job in self._history:
            duration = job.eta_duration_seconds()
            if job.status != "done" or duration is None or not math.isfinite(duration) or duration <= 0:
                continue
            profile = job.eta_sample_profile()
            if profile in self._eta_timings:
                continue
            profile_samples = samples.setdefault(profile, [])
            if len(profile_samples) < ETA_SAMPLES_PER_PROFILE:
                profile_samples.append(duration)
            recent_samples.append((self._timing_features(job), duration))

        history_samples = None
        fallback_pools = {}
        exact_samples: dict[str, int] = {}
        unknown_jobs = 0
        total_seconds = 0.0
        fallback_sample_count = 0
        fallback_estimates: dict[str, tuple[float | None, int]] = {}
        approximate = False
        overdue = False
        live = False
        rough_profiles: set[str] = set()
        for job in queue_jobs:
            live_remaining = (
                self._live_remaining_seconds(job, now)
                if job is self._active and stage == "generating" else None
            )
            profile = job.timing_profile()
            load_state = self._predict_eta_load_state(job)
            generation_profile = f"{profile}:generation"
            warm_profile = f"{generation_profile}:warm"
            preferred_profile = f"{generation_profile}:reload" if load_state == "reload" else warm_profile
            durations = None
            for key in dict.fromkeys((preferred_profile, warm_profile, generation_profile, profile)):
                stored = self._eta_timings.get(key)
                durations = stored["durations"] if stored else samples.get(key)
                if durations:
                    break
            estimate_key = (profile, load_state)
            if durations:
                estimate = float(statistics.median(durations))
                exact_samples[key] = len(durations)
                if load_state is not None and key != preferred_profile:
                    rough_profiles.add(estimate_key)
                    approximate = True
            else:
                if estimate_key not in fallback_estimates:
                    cached = self._eta_fallback_cache.get(estimate_key) if not recent_samples else None
                    if cached is None:
                        if history_samples is None:
                            history_samples = recent_samples + [
                                (entry["features"], float(statistics.median(entry["durations"])))
                                for key, entry in reversed(self._eta_timings.items())
                                if f"{key}:generation" not in self._eta_timings and f"{key}:warm" not in self._eta_timings
                            ]
                        if load_state not in fallback_pools:
                            warm, legacy, reloads = [], [], []
                            for sample in history_samples:
                                state = sample[0].get("load_state")
                                (warm if state == "warm" else reloads if state == "reload" else legacy).append(sample)
                            fallback_pools[load_state] = (reloads, warm, legacy) if load_state == "reload" else (warm, legacy)
                        estimate, count, rough = None, 0, False
                        for index, pool in enumerate(fallback_pools[load_state]):
                            if not pool:
                                continue
                            estimate, count = self._estimate_similar_duration_locked(job, pool)
                            if estimate is None:
                                estimate, count = self._estimate_rough_duration(job, pool)
                                rough = estimate is not None
                            if estimate is not None:
                                rough = rough or (load_state is not None and index > 0)
                                break
                        if not recent_samples and len(self._eta_fallback_cache) < 128:
                            self._eta_fallback_cache[estimate_key] = estimate, count, rough
                    else:
                        estimate, count, rough = cached
                    if rough:
                        rough_profiles.add(estimate_key)
                    fallback_estimates[estimate_key] = estimate, count
                    fallback_sample_count += count
                estimate, _ = fallback_estimates[estimate_key]
                if estimate is not None:
                    approximate = True
            if job is self._active and job.status == "running" and stage != "preparing":
                started_at = job.generation_started_at if job.generation_started_at is not None else job.started_at
                elapsed = max(0.0, now - started_at) if started_at is not None else 0.0
                historical_remaining = estimate - elapsed if estimate is not None else None
                if live_remaining is not None and (
                    historical_remaining is None
                    or historical_remaining <= 0
                    or abs(live_remaining - historical_remaining) >= max(0.3, estimate * 0.15)
                ):
                    total_seconds += live_remaining
                    live = approximate = True
                    continue
                if historical_remaining is not None and historical_remaining <= 0:
                    overdue = True
                    unknown_jobs += 1
                    continue
                estimate = historical_remaining
            if estimate is None:
                unknown_jobs += 1
            else:
                total_seconds += estimate

        if unknown_jobs:
            return {
                "seconds": None,
                "known_seconds": round(total_seconds, 3) if total_seconds > 0 else None,
                "stage": stage,
                "live": live,
                "live_enabled": stage == "generating" and active_job.live_eta_enabled,
                "overdue": overdue,
                "rough": bool(rough_profiles),
                "sample_count": sum(exact_samples.values()) + fallback_sample_count,
                "unknown_jobs": unknown_jobs,
                "approximate": approximate,
            }
        return {
            "seconds": round(total_seconds, 3),
            "stage": stage,
            "live": live,
            "live_enabled": stage == "generating" and active_job.live_eta_enabled,
            "overdue": overdue,
            "rough": bool(rough_profiles),
            "sample_count": sum(exact_samples.values()) + fallback_sample_count,
            "unknown_jobs": 0,
            "approximate": approximate,
        }

    def _append_history_locked(self, job: QueueJob):
        if any(item.id == job.id for item in self._history):
            return
        self._history.appendleft(job.clone_for_history())
        duration = job.eta_duration_seconds()
        if job.status == "done" and duration is not None and 0 < duration < 1e9:
            self._record_eta_timing_locked(job, duration)

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
        if deleted:
            self._schedule_recovery_checkpoint()
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
        if deleted:
            self._schedule_recovery_checkpoint()
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
        changed = False
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
                    changed = True
                    break
        if changed:
            self._schedule_recovery_checkpoint()
        return self.snapshot()

    def move(self, job_id: str, index: int, *, target_id: str | None = None, after: bool = False) -> dict[str, Any]:
        moved = False
        with self._condition:
            jobs = list(self._pending)
            job = next((item for item in jobs if item.id == job_id), None)
            target = None
            if target_id is not None:
                target = next((item for item in jobs if item.id == target_id and item is not job), None)
            if job is not None and (target_id is None or target is not None):
                jobs.remove(job)
                if target is not None:
                    index = jobs.index(target) + int(after)
                index = max(0, min(index, len(jobs)))
                jobs.insert(index, job)
                self._pending = deque(jobs)
                self._sync_repeat_order_locked()
                self._condition.notify_all()
                moved = True
        if moved:
            self._schedule_recovery_checkpoint()
        return self.snapshot()

    def update_job(self, job_id: str, prompt: str | None, negative_prompt: str | None) -> dict[str, Any]:
        updated = False
        with self._condition:
            for job in self._pending:
                if job.id == job_id:
                    job.update_text(prompt, negative_prompt)
                    if job.repeat_id in self._repeat_order:
                        self._repeat_templates[job.repeat_id] = job
                    updated = True
                    break
        if updated:
            self._schedule_recovery_checkpoint()
        return self.snapshot()

    @staticmethod
    def _job_has_refiner_inputs(job: QueueJob) -> bool:
        return all(f"{job.tab}_{suffix}" in job.input_keys for suffix in ("enable-checkbox", "checkpoint"))

    def update_settings(self, job_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        with self._condition:
            job = next((item for item in self._pending if item.id == job_id), None)
            if job is None or job.editing or job.cancel_requested:
                return {"ok": False, "message": "This job is no longer available for editing."}
            checkpoint = settings.get("checkpoint")
            refiner = settings.get("refiner")
            choices = ()
            if checkpoint is not None or refiner is not None:
                choices = shared.list_checkpoint_tiles(shared.opts.sd_checkpoint_dropdown_use_short)
            if checkpoint is not None and checkpoint != job.forge_state.get("sd_model_checkpoint"):
                if checkpoint not in choices:
                    return {"ok": False, "message": "Choose an available Checkpoint."}
                overrides = job._arg_by_key(f"{job.tab}_override_settings") or []
                if overrides:
                    from modules.infotext_utils import create_override_settings_dict

                    if "sd_model_checkpoint" in create_override_settings_dict(overrides):
                        return {"ok": False, "message": "Clear the model override in Full Edit before changing Checkpoint."}

            tab = job.tab
            if refiner is not None:
                if refiner != "" and refiner not in choices:
                    return {"ok": False, "message": "Choose an available Refiner."}
                if not self._job_has_refiner_inputs(job):
                    return {"ok": False, "message": "Refiner is unavailable for this job."}

            img2img = tab == "img2img"
            numeric_fields = {
                "cfg": ("cfg_scale", 1, 24, 0.5),
                "distilled_cfg": ("distilled_cfg_scale", 0 if img2img else 1, 24, 0.5),
                "steps": ("steps", 1, 150, 1),
                "width": ("width", 64, 2048 if img2img else 4096, 1),
                "height": ("height", 64, 2048 if img2img else 4096, 1),
                "batch_count": ("batch_count", 1, 128, 1),
                "batch_size": ("batch_size", 1, 8, 1),
                "denoising": ("denoising_strength", 0, 1, 0.01 if img2img else 0.05),
                "refiner_switch": ("switch_at", 0, 1, 0.025),
                "refiner_cfg": ("cfg", 0, 24, 0.5),
            }
            updates = {}
            for field, (suffix, minimum, maximum, step) in numeric_fields.items():
                value = settings.get(field)
                if value is None:
                    continue
                key = f"{tab}_{suffix}"
                if job._arg_by_key(key) is None:
                    return {"ok": False, "message": f"{field.replace('_', ' ').capitalize()} is unavailable for this job."}
                if type(value) not in (int, float) or not minimum <= value <= maximum or not math.isclose(
                    value / step, round(value / step), abs_tol=1e-8
                ):
                    return {"ok": False, "message": f"Enter a value from {minimum} to {maximum} in steps of {step}."}
                updates[job.input_keys.index(key)] = int(value) if step == 1 else value

            if checkpoint is not None:
                job.forge_state["sd_model_checkpoint"] = checkpoint
            if refiner is not None:
                updates[job.input_keys.index(f"{tab}_enable-checkbox")] = bool(refiner)
                updates[job.input_keys.index(f"{tab}_checkpoint")] = refiner or "None"
            for index, value in updates.items():
                job.args[index] = value
            job._fingerprint_cache = None
            job._summary_snapshot = None
            if job.repeat_id in self._repeat_order:
                self._repeat_templates[job.repeat_id] = job
            self._condition.notify_all()
        self._schedule_recovery_checkpoint()
        return {"ok": True, "message": "Queued settings updated."}

    def begin_full_edit(self, job_id: str, input_keys: list[str | None] | None = None) -> dict[str, Any]:
        with self._condition:
            job = next((item for item in self._pending if item.id == job_id), None)
            if job is None or job.status not in ("queued", "waiting") or job.cancel_requested:
                return {"ok": False, "message": "This queued job can no longer be edited."}
            if input_keys is not None and (job.input_keys != input_keys or len(job.args) != len(input_keys)):
                return {"ok": False, "message": "The current page inputs do not match this queued job."}
            job.editing = True
            self._condition.notify_all()
            result = {"ok": True, "args": [_copy_value(value) for value in job.args]}
        self._schedule_recovery_checkpoint()
        return result

    def finish_full_edit(self, job_id: str, raw_args: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
        with self._condition:
            job = next((item for item in self._pending if item.id == job_id), None)
            if job is None or not job.editing:
                return {"ok": False, "message": "This queued job is not being edited."}
            if len(raw_args) != len(job.input_keys):
                return {"ok": False, "message": "The current page inputs do not match this queued job."}
            job.args = [_copy_value(value) for value in raw_args]
            job.args[0] = job.task_id
            job.editing = False
            job._fingerprint_cache = None
            job._prompt_snapshot = None
            job._negative_prompt_snapshot = None
            job._summary_snapshot = None
            if job.repeat_id in self._repeat_order:
                self._repeat_templates[job.repeat_id] = job
            self._condition.notify_all()
            result = {"ok": True, "message": "Queued job updated."}
        self._schedule_recovery_checkpoint()
        return result

    def cancel_full_edit(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            job = next((item for item in self._pending if item.id == job_id), None)
            if job is None or not job.editing:
                return {"ok": False, "message": "This queued job is not being edited."}
            job.editing = False
            self._condition.notify_all()
            result = {"ok": True, "message": "Full edit cancelled."}
        self._schedule_recovery_checkpoint()
        return result

    def _default_job_name_locked(self, job: QueueJob, base: str = "") -> str:
        jobs = list(self._iter_jobs_locked(include_history=True))
        if not base:
            previous = max(
                (item for item in jobs if item.tab == job.tab and item.source == "queue"),
                key=lambda item: item.created_at,
                default=None,
            )
            if previous is not None:
                previous_parts = {part.strip().casefold() for part in previous.prompt.split(",")}
                parts = [part.strip() for part in job.prompt.split(",") if part.strip()]
                if any(part.casefold() in previous_parts for part in parts):
                    base = next((part for part in parts if part.casefold() not in previous_parts), "")

        names = {item.name.casefold() for item in jobs}
        base = base[:55].strip()
        if base and base.casefold() not in names:
            return base
        base = base or "Job"
        number = 2 if base != "Job" else 1
        while f"{base} {number}".casefold() in names:
            number += 1
        return f"{base} {number}"

    def rename_job(self, job_id: str, name: Any) -> dict[str, Any]:
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 60:
            return {"ok": False, "message": "Enter a name from 1 to 60 characters."}
        with self._condition:
            job = self._find_job_locked(job_id)
            if job is None:
                return {"ok": False, "message": "The job is no longer available."}
            job.name = name.strip()
        self._schedule_recovery_checkpoint()
        return {"ok": True, "name": job.name}

    def copy_job(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            source = self._find_job_locked(job_id)
            if source is None:
                return {"ok": False, "message": "The source job is no longer available."}
            job = source.clone_as_copy()
            job.name = self._default_job_name_locked(job, source.name)
            self._pending.append(job)
            self._condition.notify_all()
        self._schedule_recovery_checkpoint()
        result = self.snapshot()
        result.update({"ok": True, "message": "Copied job to the end of the queue."})
        return result

    def reuse_job(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            source = self._find_job_locked(job_id)
            if source is None or not source.args or len(source.args) != len(source.input_keys):
                return {"ok": False, "message": "The source job settings are no longer available."}
            return {
                "ok": True,
                "tab": source.tab,
                "args": [_copy_value(value) for value in source.args],
            }

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
        self._schedule_recovery_checkpoint()
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
        self._schedule_recovery_checkpoint()
        return self.snapshot()

    def disable_repeat(self):
        with self._condition:
            self._repeat_enabled = False
            self._ensure_selected_placeholders_locked()
            self._condition.notify_all()
        self._schedule_recovery_checkpoint()

    def skip(self) -> dict[str, Any]:
        with self._condition:
            if self._active is not None:
                self._active.termination_reason = "skipped"
            elif getattr(shared.state, "job", "") or getattr(progress, "current_task", None):
                self._direct_termination_reason = "skipped"
        shared.state.skip()
        return self.snapshot()

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
                if self._active is not None:
                    self._active.termination_reason = "interrupted"
                elif getattr(shared.state, "job", "") or getattr(progress, "current_task", None):
                    self._direct_termination_reason = "interrupted"
                self._pause_waiting_locked()
                self._ensure_selected_placeholders_locked()
                self._condition.notify_all()
                self._restore_forge_state_if_idle_locked()

        if interrupt:
            shared.state.interrupt()
        self._schedule_recovery_checkpoint()
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
                    if job.cancel_requested or job.paused or job.repeat_placeholder or job.editing or self._queue_paused or job not in self._pending:
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
                    job.started_at = time.time()
                    job.tracks_generation_stage = True
                    self._active = job

                self._checkpoint_recovery()
                self._run_job_with_lock(job)
                if job.status != "deleted":
                    job.status = job.termination_reason or "done"
            except Exception as exc:
                job.status = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                print("[Forge Simple Queue] Job failed:")
                traceback.print_exc()
            finally:
                call_queue.queue_lock.release()
                with self._condition:
                    job.completed_at = time.time()
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
                self._checkpoint_recovery()

    def _next_ready_job_locked(self) -> QueueJob | None:
        if self._queue_paused:
            return None
        for job in self._pending:
            if job.editing:
                return None
            if not job.paused and not job.repeat_placeholder:
                return job
        return None

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
                result = runner(*runner_args)
                if job.termination_reason is None:
                    if shared.state.interrupted or shared.state.stopping_generation:
                        job.termination_reason = "interrupted"
                    elif shared.state.skipped:
                        job.termination_reason = "skipped"
                return result
            except Exception as exc:
                job.termination_reason = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                raise
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
#txt2img_simple_queue_button, #img2img_simple_queue_button,
#txt2img_simple_queue_button button, #img2img_simple_queue_button button,
.forge-simple-queue-view-button,
#txt2img_simple_queue_update, #img2img_simple_queue_update,
#txt2img_simple_queue_update button, #img2img_simple_queue_update button,
#txt2img_simple_queue_cancel, #img2img_simple_queue_cancel,
#txt2img_simple_queue_cancel button, #img2img_simple_queue_cancel button {
  border-radius: 8px !important;
}
.forge-simple-queue-full-edit-actions {
  width: 100%;
}
.forge-simple-queue-full-edit-actions > .form {
  gap: 8px;
}
.forge-simple-queue-full-edit-actions .form,
.forge-simple-queue-full-edit-actions .gradio-row {
  gap: 8px !important;
}
.forge-simple-queue-full-edit-actions button,
#txt2img_simple_queue_update, #img2img_simple_queue_update,
#txt2img_simple_queue_cancel, #img2img_simple_queue_cancel,
#txt2img_simple_queue_update button, #img2img_simple_queue_update button,
#txt2img_simple_queue_cancel button, #img2img_simple_queue_cancel button {
  flex: 1 1 0;
}
#txt2img_simple_queue_update button, #img2img_simple_queue_update button,
#txt2img_simple_queue_cancel button, #img2img_simple_queue_cancel button {
  transition: background 0.14s ease, border-color 0.14s ease, box-shadow 0.14s ease, transform 0.14s ease;
}
#txt2img_simple_queue_update button:hover, #img2img_simple_queue_update button:hover {
  filter: brightness(1.12);
  box-shadow: 0 3px 10px rgba(15, 23, 42, 0.3);
  transform: translateY(-1px);
}
#txt2img_simple_queue_cancel button:hover, #img2img_simple_queue_cancel button:hover {
  background: #334155;
  border-color: rgba(147, 197, 253, 0.56);
  box-shadow: 0 3px 10px rgba(15, 23, 42, 0.3);
  transform: translateY(-1px);
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
.fsq-queue-status-icon {
  display: inline-flex;
  width: 1em;
  height: 1em;
  margin-left: 0.2em;
  vertical-align: middle;
  color: currentColor;
}
.fsq-queue-status-icon svg {
  width: 1em;
  height: 1em;
  fill: currentColor;
}
button.fsq-queue-paused {
  background: rgba(245, 158, 11, 0.18) !important;
  color: #f59e0b !important;
}
button.fsq-queue-paused:hover,
button.fsq-queue-paused:focus {
  background: rgba(245, 158, 11, 0.26) !important;
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
.fsq-cross-tab-status {
  color: #93c5fd;
  font-size: 12px;
  font-weight: 600;
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
  box-sizing: border-box;
  width: 34px;
  height: 34px;
  padding: 0;
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
  pointer-events: none;
}
.fsq-close:hover {
  background: rgba(127, 29, 29, 0.36);
  border-color: rgba(239, 68, 68, 0.52);
  color: #fecaca;
}
.fsq-close:active {
  transform: translateY(1px);
}
.fsq-panel button {
  transition: background 0.14s ease, border-color 0.14s ease, color 0.14s ease, box-shadow 0.14s ease, transform 0.14s ease;
}
.fsq-panel button:not(:disabled):not(.fsq-close):not(.fsq-job-name):hover {
  background: #334155;
  border-color: rgba(147, 197, 253, 0.56);
  color: #f8fafc;
  box-shadow: 0 3px 10px rgba(15, 23, 42, 0.32);
  transform: translateY(-1px);
}
.fsq-panel button:not(:disabled):not(.fsq-close):not(.fsq-job-name):active {
  transform: translateY(0);
  box-shadow: none;
}
.fsq-panel .fsq-control-active[data-queue-control="play"]:hover {
  background: rgba(34, 197, 94, 0.28);
  color: #bbf7d0;
}
.fsq-panel .fsq-control-active[data-queue-control="pause"]:hover {
  background: rgba(245, 158, 11, 0.28);
  color: #fde68a;
}
.fsq-panel .fsq-control-active[data-queue-control="stop"]:hover {
  background: rgba(239, 68, 68, 0.28);
  color: #fecaca;
}
.fsq-body {
  max-height: calc(min(760px, calc(100vh - 44px)) - 64px);
  overflow: auto;
  padding: 12px;
  scrollbar-width: thin;
  scrollbar-color: rgba(148, 163, 184, 0.35) transparent;
}
.fsq-body::-webkit-scrollbar, .fsq-model-choices::-webkit-scrollbar {
  width: 5px;
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
.fsq-eta {
  margin-left: 8px;
  color: rgba(156, 163, 175, 0.72);
  font-size: 11px;
  font-weight: 500;
  letter-spacing: 0;
  text-transform: none;
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
.fsq-job.fsq-drop-before {
  box-shadow: 0 -3px 0 #60a5fa;
}
.fsq-job.fsq-drop-after {
  box-shadow: 0 3px 0 #60a5fa;
}
.fsq-job.fsq-reordered {
  border-color: rgba(96, 165, 250, 0.82);
  box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.18);
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
  display: grid;
  place-items: center;
  width: 24px;
  height: 28px;
  border: 0;
  background: transparent;
  color: #9ca3af;
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
.fsq-job-name, .fsq-name-input {
  min-width: 0;
  max-width: 100%;
  font: inherit;
  color: inherit;
}
.fsq-job-name {
  appearance: none;
  overflow: hidden;
  padding: 0;
  border: 0;
  box-shadow: none;
  text-align: left;
  text-overflow: ellipsis;
  white-space: nowrap;
  background: transparent;
  cursor: text;
}
.fsq-job-name:hover {
  text-decoration: underline;
}
.fsq-name-input {
  width: min(100%, 240px);
  padding: 0 3px;
  border-radius: 4px;
  background: #111827;
}
.fsq-identity {
  display: flex;
  gap: 6px;
  margin-bottom: 4px;
  color: #93c5fd;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  font-size: 11px;
  font-weight: 700;
}
.fsq-history-latest {
  display: inline-flex;
  align-items: center;
  padding: 1px 6px;
  border: 1px solid rgba(74, 222, 128, 0.34);
  border-radius: 999px;
  color: #86efac;
  font-family: inherit;
  font-size: 10px;
  line-height: 1.2;
  letter-spacing: 0.02em;
  text-transform: uppercase;
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
.fsq-status.editing {
  background: rgba(168, 85, 247, 0.16);
  color: #d8b4fe;
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
  flex-wrap: nowrap;
  align-items: center;
  justify-content: flex-end;
  white-space: nowrap;
}
.fsq-actions button, .fsq-save {
  height: 30px;
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 7px;
  background: #263244;
  color: #f9fafb;
  cursor: pointer;
}
.fsq-actions .fsq-icon-action {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  box-sizing: border-box !important;
  width: 30px !important;
  min-width: 30px !important;
  max-width: 30px !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  padding: 0;
}
.fsq-icon-action svg {
  display: block;
  flex: 0 0 auto;
  width: 16px;
  height: 16px;
}
.fsq-actions .fsq-delete:hover {
  background: rgba(185, 28, 28, 0.42) !important;
  border-color: rgba(252, 165, 165, 0.68) !important;
  color: #fee2e2 !important;
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
.fsq-thumbnail {
  grid-column: 1 / 3;
  width: 56px;
  height: 56px;
  object-fit: contain;
  border-radius: 6px;
  background: #111827;
}
.fsq-settings-editor {
  position: relative;
  display: inline-block;
  vertical-align: middle;
  z-index: 2;
}
.fsq-settings-editor input, .fsq-model-choices {
  position: absolute;
  left: 0;
  width: min(16rem, 55vw);
  box-sizing: border-box;
  min-width: 0;
  padding: 2px 6px;
  border: 1px solid #475569;
  border-radius: 5px;
  background: #111827;
  color: inherit;
  font: inherit;
}
.fsq-settings-editor input {
  top: 50%;
  height: 22px;
  transform: translateY(-50%);
}
.fsq-settings-editor input[type="number"] {
  width: 5em;
}
.fsq-model-choices {
  top: calc(50% + 13px);
  display: block;
  max-height: 8rem;
  overflow-y: auto;
  overflow-x: hidden;
  scrollbar-width: thin;
  scrollbar-color: #475569 transparent;
}
.fsq-model-choices-above {
  top: auto;
  bottom: calc(50% + 13px);
}
.fsq-model-choice {
  display: block;
  min-height: 22px;
  width: 100%;
  padding: 2px 4px;
}
.fsq-model-choice:hover, .fsq-model-choice:focus-visible {
  color: #e2e8f0;
  background: #253246;
}
.fsq-toast {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 100000;
  max-width: min(360px, calc(100vw - 48px));
  padding: 10px 14px;
  border: 1px solid #488465;
  border-radius: 8px;
  background: #132a22;
  color: #dcfce7;
}
.fsq-toast-error {
  border-color: #995454;
  background: #351f24;
  color: #fee2e2;
}
.fsq-details-panel {
  display: none;
  grid-column: 3 / 5;
  margin-top: 8px;
  color: #cbd5e1;
  font-size: 12px;
  line-height: 1.45;
}
.fsq-editor.fsq-open, .fsq-details-panel.fsq-open {
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
        self.queue_actions = None
        self.full_edit_actions = None
        self.full_edit_job_id = None
        self.full_edit_load_button = None
        self.full_edit_update_button = None
        self.full_edit_cancel_button = None
        self.reuse_job_id = None
        self.reuse_load_button = None
        self.queue_bound = False
        script_callbacks.on_app_started(lambda _block, app: self.on_app_started(app))

    def title(self):
        return "Forge Simple Queue"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def before_process(self, processing, *args):
        if not hasattr(processing, "_forge_simple_queue_model_before"):
            processing._forge_simple_queue_model_before = loaded_model_signature()
        if not queue._is_queue_generation() and not getattr(processing, "_forge_simple_queue_started_at", None):
            processing._forge_simple_queue_started_at = time.time()

    def process(self, processing, *args):
        if getattr(processing, "is_api", False):
            return
        now = time.time()
        if not getattr(processing, "_forge_simple_queue_generation_started_at", None):
            processing._forge_simple_queue_generation_started_at = now
            before = getattr(processing, "_forge_simple_queue_model_before", None)
            after = loaded_model_signature()
            if before is not None and after:
                processing._forge_simple_queue_eta_load_state = "warm" if before == after else "reload"
        load_state = getattr(processing, "_forge_simple_queue_eta_load_state", None)
        task_id = getattr(progress, "current_task", None) or getattr(getattr(shared, "state", None), "job", None)
        with queue._condition:
            job = queue._active
            if job is not None and job.task_id == task_id and job.generation_started_at is None:
                job.generation_started_at = now
                job.eta_load_state = load_state
                job.live_eta_enabled = job.supports_live_eta(processing)

    def postprocess(self, processing, processed, *args):
        tab = "img2img" if self.is_img2img else "txt2img"
        if getattr(processing, "is_api", False):
            return
        task_id = getattr(progress, "current_task", None) or getattr(getattr(shared, "state", None), "job", None)
        with queue._condition:
            active_job = queue._active
        if active_job is not None and task_id == active_job.task_id:
            active_job.thumbnail = queue.record_generation_thumbnail(processed)
        elif queue.record_direct_generation(tab, processing, processed):
            print(f"[Forge Simple Queue] Recorded direct {tab} generation in history.")

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
                with gr.Group() as queue_actions:
                    self.queue_button = gr.Button("Queue", elem_id=f"{tab}_simple_queue_button", variant="secondary")
                self.queue_actions = queue_actions
                with gr.Group(visible=False, elem_id=f"{tab}_simple_queue_full_edit_actions", elem_classes=["forge-simple-queue-full-edit-actions"]) as full_edit_actions:
                    with gr.Row():
                        self.full_edit_update_button = gr.Button("Update", variant="secondary", elem_id=f"{tab}_simple_queue_update", elem_classes=["forge-simple-queue-full-edit-update"])
                        self.full_edit_cancel_button = gr.Button("Cancel", elem_id=f"{tab}_simple_queue_cancel", elem_classes=["forge-simple-queue-full-edit-cancel"])
                self.full_edit_actions = full_edit_actions
                gr.Button("", elem_id=f"{tab}_simple_queue_view", elem_classes=["forge-simple-queue-view-button"], variant="secondary")
            self.full_edit_job_id = gr.Textbox("", visible=False, elem_id=f"{tab}_simple_queue_full_edit_job_id")
            self.full_edit_load_button = gr.Button("", visible=False, elem_id=f"{tab}_simple_queue_full_edit_load")
            self.reuse_job_id = gr.Textbox("", visible=False, elem_id=f"{tab}_simple_queue_reuse_job_id")
            self.reuse_load_button = gr.Button("", visible=False, elem_id=f"{tab}_simple_queue_reuse_load")
            self.queue_status = gr.HTML("", elem_id=f"{tab}_simple_queue_status", elem_classes=["forge-simple-queue-hidden-status"])

    def _bind_queue_button(self, tab: str, inputs: list[Any]):
        input_keys = [getattr(component, "elem_id", None) for component in inputs]

        def enqueue_current(request: gr.Request, checkpoint: str | None, *args):
            return queue.enqueue(tab, request, args, input_keys, checkpoint=checkpoint)

        # Gradio also constructs dropdowns when processing updates outside UI setup.
        enqueue_bound = False

        def bind_enqueue(component, **_kwargs):
            nonlocal enqueue_bound
            if enqueue_bound or component.elem_id != "setting_sd_model_checkpoint" or get_blocks_context() is None:
                return
            label = "img2img" if tab == "img2img" else "Txt2img"
            if label in shared.opts.hidden_tabs:
                return
            self.queue_button.click(
                fn=enqueue_current,
                inputs=[component, *inputs],
                outputs=self.queue_status,
                show_progress=False,
                queue=False,
            )
            enqueue_bound = True

        script_callbacks.on_after_component(bind_enqueue)

        main_page_settings = gr.State(value=None)

        def load_full_edit(job_id: str, existing_settings, *current_args):
            result = queue.begin_full_edit(job_id, input_keys)
            if not result["ok"]:
                status = full_edit_status(job_id, result, close=True)
                return [gr.skip() for _ in inputs] + [gr.update(visible=True), gr.update(visible=False), status, gr.skip()]
            original_settings = existing_settings if existing_settings is not None else [_copy_value(value) for value in current_args]
            return result["args"] + [gr.update(visible=False), gr.update(visible=True), gr.skip(), original_settings]

        self.full_edit_load_button.click(
            fn=load_full_edit,
            inputs=[self.full_edit_job_id, main_page_settings, *inputs],
            outputs=[*inputs, self.queue_actions, self.full_edit_actions, self.queue_status, main_page_settings],
            show_progress=False,
            queue=False,
        )

        def load_reuse(job_id: str):
            result = queue.reuse_job(job_id)
            if not result["ok"]:
                return [gr.update() for _ in inputs]
            return result["args"]

        self.reuse_load_button.click(
            fn=load_reuse,
            inputs=[self.reuse_job_id],
            outputs=inputs,
            show_progress=False,
            queue=False,
        )

        def full_edit_status(job_id: str, result: dict[str, Any], close: bool):
            return (
                f'<span data-fsq-full-edit-job="{html.escape(job_id or "")}" '
                f'data-fsq-ok="{int(result["ok"])}" data-fsq-close="{int(close)}" '
                f'data-fsq-event="{uuid4().hex}">{html.escape(result["message"])}</span>'
            )

        def restore_main_page(status, original_settings):
            values = original_settings if original_settings is not None else [gr.skip() for _ in inputs]
            return status, gr.update(value=""), gr.update(visible=True), gr.update(visible=False), *values, None

        def finish_full_edit(job_id: str, original_settings, *args):
            result = queue.finish_full_edit(job_id, args)
            status = full_edit_status(job_id, result, close=result["ok"])
            if not result["ok"]:
                return status, gr.skip(), gr.skip(), gr.skip(), *[gr.skip() for _ in inputs], gr.skip()
            return restore_main_page(status, original_settings)

        def cancel_full_edit(job_id: str, original_settings):
            result = queue.cancel_full_edit(job_id)
            return restore_main_page(full_edit_status(job_id, result, close=True), original_settings)

        full_edit_outputs = [self.queue_status, self.full_edit_job_id, self.queue_actions, self.full_edit_actions, *inputs, main_page_settings]

        self.full_edit_update_button.click(
            fn=finish_full_edit,
            inputs=[self.full_edit_job_id, main_page_settings, *inputs],
            outputs=full_edit_outputs,
            show_progress=False,
            queue=False,
        )
        self.full_edit_cancel_button.click(
            fn=cancel_full_edit,
            inputs=[self.full_edit_job_id, main_page_settings],
            outputs=full_edit_outputs,
            show_progress=False,
            queue=False,
        )
        self.queue_bound = True
        print(f"[Forge Simple Queue] Queue button bound for {tab}.")
