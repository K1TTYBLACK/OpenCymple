"""The log lines of this process, kept in memory for the window's Logs tab and Export Logs. The entry
points attach BUFFER to the root logger (attach); nothing here needs Qt. The file opencymple.log in
the state directory keeps earlier runs as well."""
import logging
import threading
from collections import deque

FORMAT = "%(asctime)s %(levelname).1s %(name)s: %(message)s"
CAPACITY = 5000


class LogBuffer(logging.Handler):
    def __init__(self, capacity=CAPACITY):
        super().__init__()
        self.setFormatter(logging.Formatter(FORMAT, "%H:%M:%S"))
        self._lines = deque(maxlen=capacity)
        self._total = 0
        self._lock = threading.Lock()                  # emit runs on any thread, the window reads in its own

    def emit(self, record):
        try:
            line = self.format(record)
        except Exception:                              # noqa: BLE001
            self.handleError(record)
            return
        with self._lock:
            self._lines.append(line)
            self._total += 1

    def since(self, seen):
        """(lines logged after the first `seen`, still kept, and the number logged so far)."""
        with self._lock:
            new = self._total - seen
            return (list(self._lines)[-new:] if new > 0 else []), self._total

    def text(self):
        with self._lock:
            return "\n".join(self._lines) + "\n"


BUFFER = LogBuffer()


def attach():
    """Collect this process's log lines (once, however often it is called)."""
    root = logging.getLogger()
    if BUFFER not in root.handlers:
        root.addHandler(BUFFER)
