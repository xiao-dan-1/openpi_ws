"""GPU monitoring fixtures; no training or GPU workloads are started."""

import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pipeline.ops.gpu_watch import (BackgroundSampler, MonitorState, collect_snapshot, main,
                                    compact_pages, parse_snapshot, read_process, render, run_terminal,
                                    task_summary, terminal_frame, terminal_width)


XML = """<nvidia_smi_log>
<driver_version>550.90.07</driver_version>
<gpu><product_name>NVIDIA GeForce RTX 3090</product_name><uuid>GPU-first</uuid>
<fb_memory_usage><used>9117 MiB</used><total>24576 MiB</total><free>15459 MiB</free></fb_memory_usage>
<utilization><gpu_util>8 %</gpu_util></utilization><temperature><gpu_temp>58 C</gpu_temp></temperature>
<power_readings><power_draw>173 W</power_draw><power_limit>350 W</power_limit></power_readings>
<processes>
<process_info><pid>2609</pid><type>G</type><process_name>/usr/lib/xorg/Xorg</process_name><used_memory>4 MiB</used_memory></process_info>
<process_info><pid>258415</pid><type>C</type><process_name>/env/bin/python</process_name><used_memory>5982 MiB</used_memory></process_info>
</processes></gpu>
<gpu><product_name>NVIDIA GeForce RTX 3090</product_name><uuid>GPU-second</uuid>
<processes><process_info><pid>2609</pid><type>G</type><process_name>/usr/lib/xorg/Xorg</process_name><used_memory>N/A</used_memory></process_info></processes>
</gpu></nvidia_smi_log>"""


def large_snapshot():
    snapshot = parse_snapshot(XML)
    template = snapshot["gpus"][0]
    snapshot["gpus"] = [copy.deepcopy(template) for _ in range(4)]
    for index, gpu in enumerate(snapshot["gpus"]):
        gpu["index"] = index
        gpu["processes"] = []
    snapshot["processes"] = []
    for index in range(7):
        pid = 100 + index
        snapshot["gpus"][index % 4]["processes"].append(
            {"pid": pid, "type": "C", "memory": "2048 MiB", "name": "/env/bin/python"})
        snapshot["processes"].append({"pid": pid, "user": "other_user", "ppid": 1,
                                      "elapsed_seconds": 3600,
                                      "command": "/env/bin/python task.py --name " + "测试" * 60 + f" END_COMMAND_{pid}",
                                      "cwd": None, "executable": None,
                                      "notes": ["cwd unavailable: PermissionError", "executable unavailable: PermissionError"]})
    return snapshot


class SnapshotTests(unittest.TestCase):
    def test_graphics_compute_and_unavailable_metrics(self):
        snapshot = parse_snapshot(XML)
        self.assertEqual(snapshot["driver"], "550.90.07")
        first, second = snapshot["gpus"]
        self.assertEqual(first["index"], 0)
        self.assertEqual(first["power"], "173 W")
        self.assertEqual(first["memory_free"], "15459 MiB")
        self.assertEqual([process["type"] for process in first["processes"]], ["G", "C"])
        self.assertEqual(second["index"], 1)
        self.assertEqual(second["utilization"], "N/A")
        self.assertEqual(second["processes"][0]["memory"], "N/A")

    def test_newer_driver_power_tags(self):
        xml = XML.replace("power_readings", "gpu_power_readings").replace("power_draw", "instant_power_draw").replace("power_limit", "current_power_limit")
        gpu = parse_snapshot(xml)["gpus"][0]
        self.assertEqual(gpu["power"], "173 W")
        self.assertEqual(gpu["power_limit"], "350 W")

    @patch("pipeline.ops.gpu_watch.read_process")
    @patch("pipeline.ops.gpu_watch.subprocess.run")
    def test_multi_gpu_pid_read_once_and_physical_filter(self, run, read):
        run.return_value.returncode = 0
        run.return_value.stdout = XML
        read.side_effect = lambda pid: {"pid": pid}
        snapshot = collect_snapshot()
        self.assertEqual([process["pid"] for process in snapshot["processes"]], [2609, 258415])
        self.assertEqual(read.call_count, 2)
        snapshot = collect_snapshot({1})
        self.assertEqual([gpu["index"] for gpu in snapshot["gpus"]], [1])
        self.assertEqual(snapshot["processes"], [{"pid": 2609}])
        with self.assertRaisesRegex(ValueError, "indices not found: 4"):
            collect_snapshot({4})

    def test_full_command_retained_and_control_characters_visible(self):
        snapshot = parse_snapshot(XML)
        command = "/long/environment/bin/python train.py --checkpoint " + "a" * 250 + "\x1b[2J"
        snapshot["processes"] = [{"pid": 2609, "user": "tester", "ppid": 1,
                                  "elapsed_seconds": 90061, "command": command,
                                  "cwd": "/project", "executable": "/env/bin/python", "notes": []}]
        output = render(snapshot, details=True)
        self.assertEqual(output.count("PID 2609 |"), 2)
        self.assertIn("GPU 0: G 0.0G", output)
        self.assertIn("GPU 1: G N/A", output)
        self.assertIn("8.9G / 24.0G", output)
        self.assertIn("173.0W / 350.0W", output)
        self.assertIn("58.0C", output)
        for column in ["UTIL", "FAN", "PERF", "FREE"]:
            self.assertNotIn(column, output.splitlines()[2].split())
        self.assertNotIn("Ctrl+C", output)
        self.assertNotIn("--gpus", output)
        self.assertNotIn("MiB", output)
        self.assertIn("a" * 250, output)
        self.assertNotIn("\x1b", output)
        self.assertIn("1d 01:01:01", output)

    def test_invalid_intervals_rejected_before_collecting(self):
        for interval in ["0", "-1", "nan", "inf"]:
            with self.subTest(interval=interval), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main(["--interval", interval])
                self.assertEqual(error.exception.code, 2)


class GroupingTests(unittest.TestCase):
    def test_gpu_order_memory_descending_and_multigpu_pid_in_each_group(self):
        snapshot = parse_snapshot(XML)
        snapshot["gpus"].reverse()
        snapshot["processes"] = []
        output = render(snapshot, details=True)
        self.assertLess(output.index("GPU 0 |"), output.index("GPU 1 |"))
        group0, group1 = output.split("GPU 0 |", 1)[1].split("GPU 1 |", 1)
        self.assertLess(group0.index("PID 258415 |"), group0.index("PID 2609 |"))
        self.assertIn("GPU 0: C 5.8G", group0)
        self.assertIn("GPU 1: G N/A", group1)
        self.assertEqual(output.count("PID 2609 |"), 2)

    def test_sort_uses_raw_memory_with_pid_ties_and_unknowns_last(self):
        snapshot = parse_snapshot(XML)
        gpu = snapshot["gpus"][0]
        snapshot["gpus"] = [gpu]
        snapshot["processes"] = []
        gpu["processes"] = [{"pid": pid, "memory": memory, "type": "C", "name": "python"}
                            for pid, memory in [(1, "N/A"), (50, "[N/A]"), (30, "1025 MiB"),
                                                (20, "1024 MiB"), (10, "1024 MiB"), (5, "0 MiB")]]
        output = render(snapshot, details=True)
        pids = [int(line.split()[1]) for line in output.splitlines() if line.startswith("PID ")]
        self.assertEqual(pids, [30, 10, 20, 5, 1, 50])
        # Rounded displays are equal, but sorting must preserve 1025 > 1024.
        self.assertEqual(output.count("C 1.0G"), 3)

    def test_empty_gpu_remains_visible_and_filter_does_not_renumber(self):
        snapshot = parse_snapshot(XML)
        snapshot["processes"] = []
        gpu = snapshot["gpus"][1]
        gpu["index"] = 3
        gpu["processes"] = []
        snapshot["gpus"] = [gpu]
        output = render(snapshot)
        self.assertIn("GPU 3 | 0 processes", output)
        self.assertIn("No process records returned", output)
        self.assertNotIn("GPU 0 |", output)


class CompactViewTests(unittest.TestCase):
    def test_python_scripts_preserve_roles_workers_and_experiment_arguments(self):
        self.assertEqual(task_summary("/env/bin/python -u /long/runtime_entry.py server", "python"),
                         "runtime_entry.py server")
        self.assertEqual(task_summary("/env/bin/python3.11 -B -u /long/runtime_entry.py client", "python"),
                         "runtime_entry.py client")
        self.assertEqual(task_summary("/env/bin/python /long/run.py --worker rice_kcenter --attempt 1 --gpu GPU-ab", "python"),
                         "run.py --worker rice_kcenter --attempt 1")
        self.assertEqual(task_summary("python -m pipeline.q5.train --gpus=2,3 --exp-name test --steps 120", "python"),
                         "-m pipeline.q5.train --exp-name test --steps 120")

    def test_quoted_arguments_and_python_options_are_not_misparsed(self):
        command = "python -W ignore -X utf8 '/project/my task.py' --exp-name 'two words' --config '/path with spaces/config.py'"
        self.assertEqual(task_summary(command, "python"),
                         "'my task.py' --exp-name 'two words' --config '/path with spaces/config.py'")
        self.assertEqual(task_summary("python run.py -- --gpu literal", "python"), "run.py -- --gpu literal")
        self.assertEqual(task_summary("python run.py --gpu --worker test", "python"), "run.py --gpu --worker test")

    def test_unknown_launchers_and_inline_python_fall_back_to_original(self):
        for command in ["bash -lc 'python train.py --exp-name test'", "python -c 'print(42)'",
                        "python -unknown /long/train.py", "python", "python -m", "python 'unterminated"]:
            with self.subTest(command=command):
                self.assertEqual(task_summary(command, "python"), command)
        self.assertEqual(task_summary(None, "/env/bin/python"), "N/A (driver: /env/bin/python)")

    def test_xorg_is_short_in_compact_view_and_complete_in_details(self):
        snapshot = parse_snapshot(XML)
        snapshot["processes"] = [{"pid": 2609, "user": "root", "ppid": 1,
                                  "elapsed_seconds": 100, "command": "/usr/lib/xorg/Xorg vt1 -nolisten tcp",
                                  "cwd": None, "executable": None,
                                  "notes": ["cwd unavailable: PermissionError", "executable unavailable: PermissionError"]}]
        compact, full = render(snapshot), render(snapshot, details=True)
        self.assertIn("Xorg", compact)
        self.assertNotIn("-nolisten", compact)
        self.assertNotIn("CWD  N/A", compact)
        self.assertNotIn("PermissionError", compact)
        self.assertEqual(compact.count("Metadata access denied for 1 PID(s)"), 1)
        self.assertIn("/usr/lib/xorg/Xorg vt1 -nolisten tcp", full)
        self.assertIn("cwd unavailable: PermissionError", full)

    def test_compact_keeps_gpu_and_memory_sort_order(self):
        snapshot = parse_snapshot(XML)
        snapshot["processes"] = []
        output = render(snapshot)
        group0, group1 = output.split("GPU 0 |", 1)[1].split("GPU 1 |", 1)
        rows = [line.split() for line in group0.splitlines() if len(line.split()) >= 5 and line.split()[0].isdigit()]
        self.assertEqual([row[0] for row in rows], ["258415", "2609"])
        self.assertEqual([row[3] for row in rows], ["5.8G", "0.0G"])
        self.assertIn("2609", group1)

    def test_interactive_task_elision_respects_wide_characters(self):
        state = MonitorState()
        state.accept(large_snapshot())
        frame, _, pages = terminal_frame(state, 24, 80)
        self.assertIn("COMPACT", "\n".join(frame))
        self.assertIn("...", "\n".join(frame))
        self.assertTrue(all(terminal_width(line) <= 79 for line in frame))
        _, _, detail_pages = terminal_frame(state, 24, 80, details=True)
        self.assertLess(pages, detail_pages)
        self.assertIn("END_COMMAND_100", render(state.snapshot, details=True))

    def test_gpu_heading_and_table_repeat_on_continued_pages(self):
        lines = ["GPU 0 | 4 processes", "PID USER TYPE VRAM TASK", "101 user C 4.0G server",
                 "102 user C 3.0G worker", "103 user G 1.0G client", "104 root G 0.0G Xorg"]
        pages = compact_pages(lines, 4)
        self.assertEqual(len(pages), 2)
        for page in pages:
            self.assertEqual(page[:2], lines[:2])
            self.assertLessEqual(len(page), 4)
        records = [line for page in pages for line in page if line.split()[0].isdigit()]
        self.assertEqual(records, lines[2:])

    @patch("pipeline.ops.gpu_watch.sys.platform", "linux")
    @patch("pipeline.ops.gpu_watch.collect_snapshot")
    def test_once_defaults_compact_and_details_can_be_requested(self, collect):
        collect.return_value = large_snapshot()
        compact, full = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(compact):
            self.assertEqual(main(["--once"]), 0)
        with contextlib.redirect_stdout(full):
            self.assertEqual(main(["--once", "--details"]), 0)
        self.assertIn("TASK", compact.getvalue())
        self.assertNotIn("  CMD  ", compact.getvalue())
        self.assertIn("  CMD  ", full.getvalue())
        self.assertIn("END_COMMAND_100", full.getvalue())


class MonitorTests(unittest.TestCase):
    @patch("pipeline.ops.gpu_watch.sys.platform", "linux")
    @patch("pipeline.ops.gpu_watch.time.sleep")
    @patch("pipeline.ops.gpu_watch.collect_snapshot")
    def test_initial_failures_retry_and_recover(self, collect, sleep):
        snapshot = parse_snapshot(XML)
        snapshot["processes"] = []
        collect.side_effect = [subprocess.TimeoutExpired("nvidia-smi", 15),
                               RuntimeError("driver temporarily unavailable"), snapshot, KeyboardInterrupt]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["--plain", "--interval", "0.1"]), 0)
        self.assertEqual(collect.call_count, 4)
        self.assertIn("NO DATA", stderr.getvalue())
        self.assertIn("retrying", stderr.getvalue())
        self.assertIn("GPU watch", stdout.getvalue())
        self.assertGreaterEqual(sleep.call_args_list[0].args[0], 1)
        self.assertGreaterEqual(sleep.call_args_list[1].args[0], 1)

    def test_failed_sample_preserves_snapshot_then_clears_stale_state(self):
        state = MonitorState()
        old = large_snapshot()
        state.accept(old)
        state.accept(error=subprocess.TimeoutExpired("nvidia-smi", 15))
        self.assertIs(state.snapshot, old)
        self.assertIn("STALE", state.status())
        self.assertIn(old["timestamp"], state.status())
        self.assertNotIn("LIVE", state.status())
        self.assertIsNotNone(state.failed_at)
        new = copy.deepcopy(old)
        new["timestamp"] = "2026-10-29T18:00:00+08:00"
        state.accept(new)
        self.assertIs(state.snapshot, new)
        self.assertEqual(state.status(), "LIVE")
        self.assertIsNone(state.error)

    @patch("pipeline.ops.gpu_watch.sys.platform", "linux")
    @patch("pipeline.ops.gpu_watch.time.sleep")
    @patch("pipeline.ops.gpu_watch.collect_snapshot")
    def test_plain_error_log_identifies_last_success_and_recovers(self, collect, sleep):
        snapshot = parse_snapshot(XML)
        snapshot["processes"] = []
        collect.side_effect = [snapshot, RuntimeError("temporary error"), snapshot, KeyboardInterrupt]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(["--plain"]), 0)
        self.assertIn("STALE", stderr.getvalue())
        self.assertIn("last success: " + snapshot["timestamp"], stderr.getvalue())
        self.assertEqual(stdout.getvalue().count("GPU watch"), 2)

    @patch("pipeline.ops.gpu_watch.sys.platform", "linux")
    @patch("pipeline.ops.gpu_watch.collect_snapshot")
    def test_once_and_json_fail_without_retry(self, collect):
        for option in ["--once", "--json"]:
            with self.subTest(option=option):
                collect.reset_mock()
                collect.side_effect = subprocess.TimeoutExpired("nvidia-smi", 15)
                with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main([option]), 1)
                self.assertEqual(collect.call_count, 1)

    @patch("pipeline.ops.gpu_watch.sys.platform", "linux")
    @patch("pipeline.ops.gpu_watch.collect_snapshot")
    def test_invalid_gpu_or_missing_executable_does_not_retry(self, collect):
        for error in [ValueError("GPU indices not found: 4"), FileNotFoundError("nvidia-smi")]:
            with self.subTest(error=error):
                collect.reset_mock()
                collect.side_effect = error
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(main(["--plain"]), 1)
                self.assertEqual(collect.call_count, 1)

    @patch("pipeline.ops.gpu_watch.sys.platform", "linux")
    @patch("pipeline.ops.gpu_watch.collect_snapshot")
    def test_json_remains_one_raw_snapshot(self, collect):
        snapshot = parse_snapshot(XML)
        snapshot["processes"] = []
        collect.return_value = snapshot
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(main(["--json"]), 0)
        self.assertEqual(json.loads(stdout.getvalue()), snapshot)
        self.assertEqual(collect.call_count, 1)

    @patch("pipeline.ops.gpu_watch.collect_snapshot")
    def test_slow_background_query_is_nonblocking_and_never_overlaps(self, collect):
        released = threading.Event()
        entered = threading.Event()
        snapshot = large_snapshot()

        def slow_query(_):
            entered.set()
            released.wait(timeout=2)
            return snapshot

        collect.side_effect = slow_query
        sampler = BackgroundSampler(None)
        try:
            sampler.start()
            self.assertTrue(entered.wait(timeout=1))
            sampler.start()
            self.assertIsNone(sampler.poll())
            self.assertTrue(sampler.pending)
            self.assertEqual(collect.call_count, 1)
        finally:
            released.set()
        deadline = time.monotonic() + 2
        result = None
        while result is None and time.monotonic() < deadline:
            result = sampler.poll()
            time.sleep(0.005)
        self.assertEqual(result, (snapshot, None))
        self.assertFalse(sampler.pending)


class PagingTests(unittest.TestCase):
    def test_four_gpus_stay_visible_on_every_page_and_commands_are_complete(self):
        state = MonitorState()
        state.accept(large_snapshot())
        first, _, pages = terminal_frame(state, 24, 80, details=True)
        self.assertGreater(pages, 1)
        section = next(index for index, line in enumerate(first) if line.startswith("PROCESSES |"))
        all_details = []
        for page in range(pages):
            frame, current, _ = terminal_frame(state, 24, 80, page, details=True)
            self.assertEqual(current, page)
            self.assertEqual(len(frame), 24)
            self.assertEqual(frame[:section], first[:section])
            for gpu in range(4):
                self.assertTrue(any(line.startswith(str(gpu) + " ") for line in frame[:section]))
            self.assertEqual(frame[-1], "n/p PgDn/PgUp:page | Space:pause | d:details | q:quit")
            all_details.extend(frame[section + 1:-1])
        joined = "".join(all_details)
        for process in state.snapshot["processes"]:
            self.assertIn(process["command"], joined)

    def test_resize_reflows_pages_and_small_terminal_has_explicit_message(self):
        state = MonitorState()
        state.accept(large_snapshot())
        _, _, small_pages = terminal_frame(state, 24, 80)
        large_frame, _, large_pages = terminal_frame(state, 40, 120)
        self.assertLess(large_pages, small_pages)
        self.assertEqual(len(large_frame), 40)
        small_frame, _, _ = terminal_frame(state, 8, 40)
        self.assertLessEqual(len(small_frame), 8)
        self.assertIn("Terminal too small", " ".join(small_frame))

    def test_failure_keeps_overview_and_marks_retained_data_stale(self):
        state = MonitorState()
        state.accept(large_snapshot())
        before, _, _ = terminal_frame(state, 24, 80)
        state.accept(error=RuntimeError("driver unavailable"))
        frame, _, _ = terminal_frame(state, 24, 80)
        self.assertEqual(frame[:7], before[:7])
        self.assertIn("STALE", "\n".join(frame))
        self.assertIn("QUERY FAILED", "\n".join(frame))
        self.assertIn("driver unavailable", "\n".join(frame))

    def test_initial_error_and_paused_state_are_explicit(self):
        state = MonitorState()
        state.accept(error=RuntimeError("driver unavailable"))
        frame, _, _ = terminal_frame(state, 24, 80, paused=True)
        output = "\n".join(frame)
        self.assertIn("PAUSED | NO DATA", output)
        self.assertIn("retry on resume", output)


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux terminal controls")
class TerminalControlTests(unittest.TestCase):
    def test_pause_freezes_sample_allows_paging_and_resume_accepts_new_data(self):
        import curses

        first = large_snapshot()
        second = copy.deepcopy(first)
        second["timestamp"] = "2026-10-29T18:00:00+08:00"

        class Sampler:
            pending = False

            def __init__(self, _):
                self.started = 0
                self.waiting = None

            def start(self):
                self.pending = True
                self.started += 1
                self.waiting = (first if self.started == 1 else second, None)

            def poll(self):
                result, self.waiting = self.waiting, None
                if result is not None:
                    self.pending = False
                return result

        class Screen:
            def __init__(self):
                self.frame = {}
                self.frames = []
                # First query starts; accept it, pause, page, remain frozen,
                # resume, then accept second query and quit.
                self.keys = iter([-1, ord(" "), ord("n"), ord("d"), ord("n"), ord("d"), ord(" "), -1, ord("q")])

            def keypad(self, _):
                pass

            def timeout(self, _):
                pass

            def getmaxyx(self):
                return 24, 80

            def erase(self):
                self.frame = {}

            def addstr(self, row, column, text):
                self.frame[row] = text

            def refresh(self):
                pass

            def getch(self):
                self.frames.append("\n".join(self.frame.values()))
                return next(self.keys)

        screen = Screen()
        with patch("pipeline.ops.gpu_watch.BackgroundSampler", Sampler), patch.object(curses, "curs_set"):
            self.assertEqual(run_terminal(screen, SimpleNamespace(gpus=None, interval=0), curses), 0)
        self.assertIn("PAUSED", screen.frames[2])
        self.assertIn("PROCESSES | page 2/", screen.frames[3])
        self.assertIn("DETAILS", screen.frames[4])
        self.assertIn("PROCESSES | page 1/", screen.frames[4])
        self.assertIn("PROCESSES | page 2/", screen.frames[5])
        self.assertIn("COMPACT", screen.frames[6])
        self.assertIn("PROCESSES | page 1/", screen.frames[6])
        for frame in screen.frames[2:7]:
            self.assertIn(first["timestamp"].replace("T", " "), frame)
            self.assertNotIn("18:00:00", frame)
        self.assertIn("18:00:00", screen.frames[-1])

    def test_terminal_quit_does_not_wait_for_blocked_query(self):
        import curses

        released = threading.Event()
        entered = threading.Event()

        def slow_query(_):
            entered.set()
            released.wait(timeout=2)
            return large_snapshot()

        screen = unittest.mock.Mock()
        screen.getmaxyx.return_value = (24, 80)
        screen.getch.side_effect = lambda: ord("q") if entered.wait(timeout=1) else -1
        try:
            with patch("pipeline.ops.gpu_watch.collect_snapshot", side_effect=slow_query), patch.object(curses, "curs_set"):
                started = time.monotonic()
                self.assertEqual(run_terminal(screen, SimpleNamespace(gpus=None, interval=2), curses), 0)
                self.assertLess(time.monotonic() - started, 1)
        finally:
            released.set()


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux /proc metadata")
class ProcessTests(unittest.TestCase):
    def make_process(self, root):
        process = root / "42"
        process.mkdir()
        # /proc/stat comm may itself contain spaces and closing parentheses.
        (process / "stat").write_text("42 (worker ) name) S 7 " + "0 " * 17 + "500\n")
        (process / "cmdline").write_bytes(b"/env/bin/python\0train.py\0--name\0two words\0")
        (root / "uptime").write_text("100.00 0.0\n")
        (process / "cwd").symlink_to("/project with spaces")
        (process / "exe").symlink_to("/env/bin/python")
        return process

    @patch("pipeline.ops.gpu_watch.os.sysconf", return_value=100)
    def test_full_argv_workdir_owner_and_elapsed(self, _):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_process(root)
            info = read_process(42, root)
        self.assertEqual(info["ppid"], 7)
        self.assertEqual(info["elapsed_seconds"], 95)
        self.assertEqual(info["command"], "/env/bin/python train.py --name 'two words'")
        self.assertEqual(info["cwd"], "/project with spaces")
        self.assertNotEqual(info["user"], "N/A")
        self.assertEqual(info["notes"], [])

    def test_missing_pid_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            info = read_process(42, Path(directory))
        self.assertIsNone(info["command"])
        self.assertIn("PID not visible", info["notes"][0])

    def test_permission_denied_does_not_discard_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_process(root)
            with patch("pipeline.ops.gpu_watch.os.readlink", side_effect=PermissionError):
                info = read_process(42, root)
        self.assertIn("train.py", info["command"])
        self.assertIsNone(info["cwd"])
        self.assertIn("cwd unavailable: PermissionError", info["notes"])


if __name__ == "__main__":
    unittest.main()
