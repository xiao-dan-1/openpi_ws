"""Read-only NVIDIA GPU and full Linux process monitor; Python standard library only."""

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import socket
import subprocess
import sys
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET


def xml_value(node, *paths):
    for path in paths:
        value = node.findtext(path)
        if value and value.strip() not in {"N/A", "[N/A]", "Not Supported"}:
            return value.strip()
    return "N/A"


def parse_snapshot(xml):
    root = ET.fromstring(xml)
    devices = []
    for index, node in enumerate(root.findall("gpu")):
        processes = []
        for process in node.findall("processes/process_info"):
            pid = xml_value(process, "pid")
            if pid.isdigit():
                processes.append({"pid": int(pid), "type": xml_value(process, "type"),
                                  "memory": xml_value(process, "used_memory"),
                                  "name": xml_value(process, "process_name")})
        devices.append({
            "index": index, "uuid": xml_value(node, "uuid"),
            "bus_id": xml_value(node, "pci/pci_bus_id"),
            "name": xml_value(node, "product_name"),
            "utilization": xml_value(node, "utilization/gpu_util"),
            "memory_used": xml_value(node, "fb_memory_usage/used"),
            "memory_total": xml_value(node, "fb_memory_usage/total"),
            "memory_free": xml_value(node, "fb_memory_usage/free"),
            "temperature": xml_value(node, "temperature/gpu_temp"),
            "power": xml_value(node, "gpu_power_readings/instant_power_draw",
                               "gpu_power_readings/power_draw", "power_readings/power_draw"),
            "power_limit": xml_value(node, "gpu_power_readings/current_power_limit",
                                     "gpu_power_readings/power_limit", "power_readings/power_limit"),
            "fan": xml_value(node, "fan_speed"),
            "performance_state": xml_value(node, "performance_state"),
            "processes": processes,
        })
    return {"timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "driver": xml_value(root, "kmd_version", "driver_version"),
            "gpus": devices}


def read_process(pid, proc_root=Path("/proc")):
    """Read each field independently: a denied cwd must not hide the command."""
    process = proc_root / str(pid)
    info = {"pid": pid, "user": "N/A", "ppid": None, "elapsed_seconds": None,
            "command": None, "cwd": None, "executable": None, "notes": []}
    try:
        stat = (process / "stat").read_text().rsplit(")", 1)[1].split()
        start_ticks = int(stat[19])
        info["ppid"] = int(stat[1])
        uptime = float((proc_root / "uptime").read_text().split()[0])
        info["elapsed_seconds"] = max(0, int(uptime - start_ticks / os.sysconf("SC_CLK_TCK")))
    except FileNotFoundError:
        info["notes"].append("PID not visible (exited or different PID namespace)")
        return info
    except (OSError, ValueError, IndexError) as exc:
        start_ticks = None
        info["notes"].append(f"process age unavailable: {type(exc).__name__}")
    try:
        import pwd

        uid = process.stat().st_uid
        try:
            info["user"] = pwd.getpwuid(uid).pw_name
        except KeyError:
            info["user"] = str(uid)
    except OSError as exc:
        info["notes"].append(f"user unavailable: {type(exc).__name__}")
    try:
        raw = (process / "cmdline").read_bytes()
        argv = raw.split(b"\0")
        if argv and argv[-1] == b"":
            argv.pop()
        if argv:
            info["command"] = shlex.join([os.fsdecode(arg) for arg in argv])
        else:
            info["notes"].append("empty command line (exited or kernel thread)")
    except OSError as exc:
        info["notes"].append(f"command unavailable: {type(exc).__name__}")
    for key, filename in [("cwd", "cwd"), ("executable", "exe")]:
        try:
            info[key] = os.readlink(process / filename)
        except OSError as exc:
            info["notes"].append(f"{key} unavailable: {type(exc).__name__}")
    # Avoid presenting mixed metadata if a PID is reused during this read.
    try:
        current = (process / "stat").read_text().rsplit(")", 1)[1].split()
        if start_ticks is not None and int(current[19]) != start_ticks:
            return {"pid": pid, "user": "N/A", "ppid": None, "elapsed_seconds": None,
                    "command": None, "cwd": None, "executable": None,
                    "notes": ["PID reused during sampling; wait for next refresh"]}
    except (OSError, ValueError, IndexError):
        info["notes"].append("process changed or exited during sampling")
    return info


def collect_snapshot(selected=None):
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=15)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "nvidia-smi failed")
    snapshot = parse_snapshot(result.stdout)
    if selected is not None:
        missing = selected - {gpu["index"] for gpu in snapshot["gpus"]}
        if missing:
            raise ValueError(f"GPU indices not found: {','.join(map(str, sorted(missing)))}")
        snapshot["gpus"] = [gpu for gpu in snapshot["gpus"] if gpu["index"] in selected]
    pids = {process["pid"] for gpu in snapshot["gpus"] for process in gpu["processes"]}
    snapshot["processes"] = [read_process(pid) for pid in sorted(pids)]
    return snapshot


def elapsed_text(seconds):
    if seconds is None:
        return "N/A"
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return (f"{days}d " if days else "") + f"{hours:02}:{minutes:02}:{seconds:02}"


def terminal_text(value):
    """Show control characters visibly without truncating process arguments."""
    return "".join(char if char.isprintable() else ascii(char)[1:-1] for char in str(value))


def format_metric(value, unit, divisor=1):
    """Format driver values for display; retain unknowns instead of inventing zeros."""
    try:
        number = float(str(value).split()[0]) / divisor
    except (ValueError, IndexError):
        return "N/A"
    return f"{number:.1f}{unit}" if math.isfinite(number) else "N/A"


def process_memory_order(entry):
    """Sort raw per-GPU MiB descending, unknowns last, with stable PID ties."""
    try:
        memory = float(entry["memory"].split()[0])
    except (ValueError, IndexError):
        memory = float("nan")
    if not math.isfinite(memory) or memory < 0:
        return True, 0, entry["pid"]
    return False, -memory, entry["pid"]


def task_summary(command, driver_name):
    """Shorten known launchers; preserve task arguments and conservative fallbacks."""
    if not command:
        return "N/A (driver: " + driver_name + ")"
    try:
        argv = shlex.split(command)
    except ValueError:
        return command
    if not argv:
        return command
    executable = os.path.basename(argv[0])
    if executable == "Xorg":
        return "Xorg"
    if not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", executable):
        return command
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument == "-m" and index + 1 < len(argv):
            target = "-m " + shlex.quote(argv[index + 1])
            index += 2
            break
        if argument in {"-W", "-X"} and index + 1 < len(argv):
            index += 2
        elif argument in {"-u", "-B", "-O", "-OO", "-b", "-bb", "-E", "-s", "-S", "-I", "-q"}:
            index += 1
        elif argument.endswith((".py", ".pyw")) and not argument.startswith("-"):
            target = shlex.quote(os.path.basename(argument))
            index += 1
            break
        else:
            return command
    else:
        return command
    arguments = []
    while index < len(argv):
        argument = argv[index]
        # Selection already appears in the group heading. Respect the end-of-options marker.
        if argument == "--":
            arguments.extend(argv[index:])
            break
        if argument in {"--gpu", "--gpus"} and index + 1 < len(argv) and not argv[index + 1].startswith("-"):
            index += 2
        elif argument.startswith(("--gpu=", "--gpus=")):
            index += 1
        else:
            arguments.append(argument)
            index += 1
    return target + (" " + shlex.join(arguments) if arguments else "")


def process_metadata(processes, pid):
    return processes.get(pid, {"user": "N/A", "ppid": None, "elapsed_seconds": None,
                               "command": None, "cwd": None, "executable": None,
                               "notes": ["process metadata unavailable"]})


def compact_process_rows(entries, processes, width=None):
    rows = [["PID", "USER", "TYPE", "VRAM", "TASK"]]
    for entry in entries:
        process = process_metadata(processes, entry["pid"])
        rows.append([str(entry["pid"]), process["user"], entry["type"],
                     format_metric(entry["memory"], "G", 1024),
                     task_summary(process["command"], entry["name"])])
    widths = [max(terminal_width(row[column]) for row in rows) for column in range(4)]
    lines = []
    for row in rows:
        prefix = "  ".join(" " * (size - terminal_width(cell)) + terminal_text(cell) if index == 3
                           else terminal_text(cell) + " " * (size - terminal_width(cell))
                           for index, (cell, size) in enumerate(zip(row[:4], widths))) + "  "
        task = terminal_text(row[4])
        if width is not None and terminal_width(prefix + task) > width:
            # Interactive compact mode deliberately elides the task; details remain lossless.
            available = width - terminal_width(prefix)
            if available >= 4:
                parts = wrap_terminal_lines([task], available - 3)
                task = parts[0] + "..."
        lines.append(prefix + task)
    return lines


def render(snapshot, details=False, width=None):
    timestamp = snapshot["timestamp"].replace("T", " ", 1)
    lines = [f"GPU watch | {snapshot['host']} | {timestamp} | driver {snapshot['driver']}", ""]
    rows = [["GPU", "NAME", "VRAM USED/TOTAL", "TEMP", "POWER/LIMIT"]]
    devices = sorted(snapshot["gpus"], key=lambda gpu: gpu["index"])
    for gpu in devices:
        rows.append([str(gpu["index"]), gpu["name"],
                     f"{format_metric(gpu['memory_used'], 'G', 1024)} / {format_metric(gpu['memory_total'], 'G', 1024)}",
                     format_metric(gpu["temperature"], "C"),
                     f"{format_metric(gpu['power'], 'W')} / {format_metric(gpu['power_limit'], 'W')}"])
    widths = [max(len(terminal_text(row[column])) for row in rows) for column in range(len(rows[0]))]
    lines.extend("  ".join(terminal_text(cell).ljust(width) if column < 2 else terminal_text(cell).rjust(width)
                           for column, (cell, width) in enumerate(zip(row, widths))).rstrip() for row in rows)
    lines.extend(["", "PROCESSES"])
    processes = {process["pid"]: process for process in snapshot["processes"]}
    for gpu in devices:
        entries = sorted(gpu["processes"], key=process_memory_order)
        count = len(entries)
        lines.extend(["", f"GPU {gpu['index']} | {count} process" + ("es" if count != 1 else "")])
        if not entries:
            lines.append("  No process records returned (GPU activity may still exist).")
        if not details and entries:
            lines.extend(compact_process_rows(entries, processes, width))
            continue
        for entry in entries:
            pid = entry["pid"]
            process = process_metadata(processes, pid)
            lines.extend(["", f"PID {pid} | USER {process['user']} | PPID {process['ppid']} | AGE {elapsed_text(process['elapsed_seconds'])}",
                          f"  GPU {gpu['index']}: {entry['type']} {format_metric(entry['memory'], 'G', 1024)}",
                          "  CMD  " + (process["command"] or "N/A (driver name: " + entry["name"] + ")"),
                          "  CWD  " + (process["cwd"] or "N/A"),
                          "  EXE  " + (process["executable"] or "N/A")])
            if process["notes"]:
                lines.append("  NOTE " + "; ".join(process["notes"]))
    if not details:
        denied = {process["pid"] for process in snapshot["processes"]
                  if any("PermissionError" in note for note in process["notes"])}
        other = {process["pid"] for process in snapshot["processes"]
                 if any("PermissionError" not in note for note in process["notes"])}
        if denied or other:
            lines.append("")
        if denied:
            lines.append(f"NOTE Metadata access denied for {len(denied)} PID(s); d / --details for reasons.")
        if other:
            lines.append(f"NOTE Other metadata unavailable for {len(other)} PID(s); d / --details for reasons.")
    return "\n".join(terminal_text(line) for line in lines)


class MonitorState:
    """Only successful samples replace the displayed snapshot."""

    def __init__(self):
        self.snapshot = None
        self.error = None
        self.failed_at = None

    def accept(self, snapshot=None, error=None):
        if error is not None:
            if isinstance(error, FileNotFoundError) or not isinstance(
                    error, (OSError, RuntimeError, ET.ParseError, subprocess.TimeoutExpired)):
                raise error
            self.error = terminal_text(error)
            self.failed_at = datetime.now().astimezone().isoformat(timespec="seconds")
        else:
            self.snapshot = snapshot
            self.error = None
            self.failed_at = None

    def status(self, paused=False, pending=False):
        mode = "PAUSED" if paused else "LIVE"
        if self.error:
            last = self.snapshot["timestamp"] if self.snapshot else "none"
            label = "STALE" if self.snapshot else "NO DATA"
            return ("PAUSED | " if paused else "") + f"{label} | last success: {last}"
        if self.snapshot is None:
            return f"{mode} | NO DATA | collecting first snapshot"
        return mode + (" | collecting next snapshot" if pending else "")


class BackgroundSampler:
    """Keep slow driver queries off the terminal thread; never overlap queries."""

    def __init__(self, selected):
        self.selected = selected
        self.pending = False
        self.results = queue.Queue()

    def start(self):
        if self.pending:
            return
        self.pending = True
        threading.Thread(target=self._sample, daemon=True).start()

    def _sample(self):
        try:
            self.results.put((collect_snapshot(self.selected), None))
        except Exception as exc:
            self.results.put((None, exc))

    def poll(self):
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.pending = False
        return result


def terminal_width(text):
    return sum(0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in "WF" else 1
               for char in terminal_text(text))


def wrap_terminal_lines(lines, width):
    """Hard-wrap all content without losing long commands or wide characters."""
    wrapped = []
    for line in lines:
        current, cells = "", 0
        for char in terminal_text(line):
            size = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in "WF" else 1
            if cells + size > width and current:
                wrapped.append(current)
                current, cells = "", 0
            current += char
            cells += size
        wrapped.append(current)
    return wrapped


def compact_pages(lines, capacity):
    """Repeat a GPU title and table header when its rows continue on another page."""
    pages, current, context = [], [], []
    for line in lines:
        group = line.startswith("GPU ") and " | " in line
        if group:
            if current and capacity - len(current) < 3:
                pages.append(current)
                current = []
            context = [line]
        elif line.split()[:2] == ["PID", "USER"]:
            context = context[:1] + [line]
        elif line.startswith("NOTE "):
            context = []
        if len(current) == capacity:
            pages.append(current)
            current = context[:] if not group and line and len(context) < capacity else []
        if line or current:
            current.append(line)
    if current:
        pages.append(current)
    return pages or [[]]


def terminal_frame(state, rows, columns, page=0, paused=False, pending=False, details=False):
    """Bound the complete frame to the viewport; only process details paginate."""
    width = max(1, columns - 1)  # Avoid curses' lower-right-cell wrap behavior.
    if state.snapshot is None:
        overview, body = ["GPU watch"], ["Waiting for the first successful sample."]
    else:
        overview_text, _, body_text = render(state.snapshot, details=details, width=width).partition("\nPROCESSES\n")
        overview, body = overview_text.splitlines(), body_text.strip("\n").splitlines()
    if state.error:
        action = "retry on resume" if paused else "retrying"
        body = [f"QUERY FAILED {state.failed_at} | {action}", state.error, "", *body]
    header = wrap_terminal_lines(overview + ["", state.status(paused, pending)], width)
    room = rows - len(header) - 4  # Blank, section title, blank, controls.
    if columns < 30 or room < (1 if details else 3):
        message = wrap_terminal_lines(["Terminal too small: enlarge it or use --once.", "q: quit"], width)
        return message[:max(0, rows)], 0, 1
    detail_lines = wrap_terminal_lines(body, width)
    chunks = ([detail_lines[index:index + room] for index in range(0, len(detail_lines), room)] or [[]]
              if details else compact_pages(detail_lines, room))
    pages = len(chunks)
    page %= pages
    view = "DETAILS" if details else "COMPACT"
    frame = header + ["", f"PROCESSES | page {page + 1}/{pages} | {view}"] + chunks[page]
    frame.extend([""] * (rows - len(frame) - 1))
    controls = "n/p PgDn/PgUp:page | Space:pause | d:details | q:quit"
    # A short footer for narrow terminals; full help remains available via --help.
    frame.append(controls if len(controls) <= width else "n/p:page d:view q:quit")
    return frame, page, pages


def run_terminal(screen, args, curses):
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    screen.keypad(True)
    screen.timeout(100)
    state = MonitorState()
    sampler = BackgroundSampler(args.gpus)
    paused, page, next_sample, previous = False, 0, 0, None
    details = getattr(args, "details", False)
    while True:
        if not paused:
            result = sampler.poll()
            if result is not None:
                snapshot, error = result
                state.accept(snapshot, error)
                delay = max(1.0, args.interval) if state.error else args.interval
                next_sample = time.monotonic() + delay
            if not sampler.pending and time.monotonic() >= next_sample:
                sampler.start()
        rows, columns = screen.getmaxyx()
        frame, page, pages = terminal_frame(state, rows, columns, page, paused, sampler.pending, details)
        if frame != previous:
            screen.erase()
            for row, line in enumerate(frame):
                try:
                    screen.addstr(row, 0, line)
                except curses.error:
                    # A terminal resize can occur between getmaxyx and drawing.
                    pass
            screen.refresh()
            previous = frame
        key = screen.getch()
        if key in (ord("q"), ord("Q")):
            return 0
        if key == ord(" "):
            paused = not paused
        elif key in (ord("d"), ord("D")):
            details, page = not details, 0
        elif key in (ord("n"), curses.KEY_NPAGE, curses.KEY_RIGHT, curses.KEY_DOWN):
            page = (page + 1) % pages
        elif key in (ord("p"), curses.KEY_PPAGE, curses.KEY_LEFT, curses.KEY_UP):
            page = (page - 1) % pages
        elif key == curses.KEY_RESIZE:
            previous = None


def run_plain(args):
    state = MonitorState()
    while True:
        started = time.monotonic()
        try:
            state.accept(collect_snapshot(args.gpus))
        except (OSError, RuntimeError, ET.ParseError, subprocess.TimeoutExpired) as exc:
            state.accept(error=exc)
        if state.error:
            print(f"gpu_watch: {state.status()} | failed: {state.failed_at} | {state.error} | retrying",
                  file=sys.stderr, flush=True)
        else:
            print(render(state.snapshot, details=args.details), flush=True)
        delay = max(1.0, args.interval) if state.error else max(0, args.interval - (time.monotonic() - started))
        time.sleep(delay)


def gpu_indices(value):
    parts = value.split(",")
    if not parts or any(not part.isascii() or not part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("Use GPU indices such as 0,2")
    return {int(part) for part in parts}


def positive_interval(value):
    try:
        interval = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Interval must be a finite positive number") from exc
    if not math.isfinite(interval) or interval <= 0:
        raise argparse.ArgumentTypeError("Interval must be a finite positive number")
    return interval


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=positive_interval, default=2, help="Refresh interval in seconds (default: 2)")
    parser.add_argument("--gpus", type=gpu_indices, help="Physical nvidia-smi GPU indices, e.g. 0,2")
    parser.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    parser.add_argument("--json", action="store_true", help="Print one JSON snapshot and exit")
    parser.add_argument("--plain", action="store_true", help="Continuous full text output without interactive paging")
    parser.add_argument("--details", action="store_true", help="Show full commands and metadata instead of the compact task table")
    parser.epilog = "Interactive controls: n/p or PgDn/PgUp to page; d for compact/details; Space to pause/resume; q or Ctrl+C to quit."
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("Run on the Linux GPU host; full process details require its /proc filesystem")
    try:
        if args.once or args.json:
            snapshot = collect_snapshot(args.gpus)
            if args.json:
                print(json.dumps(snapshot, ensure_ascii=True, indent=2))
            else:
                print(render(snapshot, details=args.details), flush=True)
            return 0
        interactive = (not args.plain and sys.stdin.isatty() and sys.stdout.isatty()
                       and os.environ.get("TERM", "dumb") != "dumb")
        if interactive:
            try:
                import curses
            except ImportError:
                print("gpu_watch: curses unavailable; using full text output (--once for a single snapshot)", file=sys.stderr)
            else:
                try:
                    return curses.wrapper(run_terminal, args, curses)
                except curses.error as exc:
                    print(f"gpu_watch: terminal unavailable ({exc}); using full text output", file=sys.stderr)
        run_plain(args)
    except KeyboardInterrupt:
        return 0
    except BrokenPipeError:
        return 0
    except (OSError, RuntimeError, ValueError, ET.ParseError, subprocess.TimeoutExpired) as exc:
        print(f"gpu_watch: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
