import importlib.util
import sys
import types
import unittest
from pathlib import Path


def load_queue_module():
    fastapi = types.ModuleType("fastapi")
    fastapi.Body = lambda default=None: default
    fastapi.FastAPI = object
    sys.modules["fastapi"] = fastapi

    modules = types.ModuleType("modules")
    modules.call_queue = types.SimpleNamespace()
    modules.img2img = types.SimpleNamespace()
    modules.progress = types.SimpleNamespace(current_task=None, pending_tasks={})
    modules.script_callbacks = types.SimpleNamespace()
    modules.scripts = types.SimpleNamespace(Script=object, AlwaysVisible=True)
    modules.shared = types.SimpleNamespace()
    modules.txt2img = types.SimpleNamespace()
    sys.modules["modules"] = modules

    gradio = types.ModuleType("gradio")
    gradio.Request = object
    gradio.update = lambda **kwargs: kwargs
    sys.modules["gradio"] = gradio

    source = Path(__file__).parents[1] / "scripts" / "forge_simple_queue.py"
    spec = importlib.util.spec_from_file_location("forge_simple_queue_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FullEditContractTests(unittest.TestCase):
    def setUp(self):
        self.module = load_queue_module()
        self.queue = self.module.SimpleQueue()
        self.job = self.module.QueueJob(
            "job-1", "task-1", "txt2img", ["task-1", "before", "negative"],
            ["txt2img_task_id", "txt2img_prompt", "txt2img_neg_prompt"], {}, None,
        )
        self.queue._pending.append(self.job)

    def test_full_edit_blocks_the_job_at_the_front_of_the_queue(self):
        self.assertTrue(self.queue.begin_full_edit("job-1")["ok"])
        self.assertIsNone(self.queue._next_ready_job_locked())

    def test_update_full_edit_keeps_position_and_releases_the_job(self):
        self.queue.begin_full_edit("job-1")
        result = self.queue.finish_full_edit("job-1", ["task-1", "after", "negative"])

        self.assertTrue(result["ok"])
        self.assertEqual([job.id for job in self.queue._pending], ["job-1"])
        self.assertEqual(self.job.args[1], "after")
        self.assertIs(self.queue._next_ready_job_locked(), self.job)

    def test_cancel_full_edit_keeps_the_original_values_and_releases_the_job(self):
        self.queue.begin_full_edit("job-1")
        result = self.queue.cancel_full_edit("job-1")

        self.assertTrue(result["ok"])
        self.assertEqual(self.job.args[1], "before")
        self.assertIs(self.queue._next_ready_job_locked(), self.job)

    def test_copy_appends_an_independent_job_with_a_new_identity(self):
        result = self.queue.copy_job("job-1")

        self.assertTrue(result["ok"])
        self.assertEqual(len(self.queue._pending), 2)
        copied = self.queue._pending[-1]
        self.assertNotEqual(copied.id, self.job.id)
        self.assertEqual(copied.args[1:], self.job.args[1:])
        self.assertIsNot(copied.args, self.job.args)
        self.assertNotEqual(copied.repeat_id, self.job.repeat_id)

    def test_reuse_reads_full_settings_from_history_without_requeueing(self):
        self.queue._append_history_locked(self.job)
        self.queue._pending.clear()

        result = self.queue.reuse_job("job-1")

        self.assertTrue(result["ok"])
        self.assertEqual(result["args"], ["task-1", "before", "negative"])
        self.assertEqual(len(self.queue._pending), 0)


if __name__ == "__main__":
    unittest.main()
