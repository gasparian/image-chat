import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from agent.logging_config import get_logger

logger = get_logger(__name__)

_tracker_context: ContextVar[Optional["StepTracker"]] = ContextVar("step_tracker", default=None)


class StepTracker:
    def __init__(self):
        self.steps: list[dict] = []
        self.current_step: Optional[dict] = None
        self.total_tokens: int = 0
        self.start_time: Optional[float] = None

    @contextmanager
    def track_step(self, name: str):
        start = time.time()
        step_data = {"name": name, "start": start, "tokens": 0}
        self.current_step = step_data
        self.steps.append(step_data)
        try:
            yield
        finally:
            step_data["duration"] = time.time() - start
            self.current_step = None

    def add_tokens(self, count: int):
        self.total_tokens += count
        if self.current_step is not None:
            self.current_step["tokens"] += count

    def print_summary(self):
        if not self.steps:
            return

        total_time = sum(s["duration"] for s in self.steps)

        for step in self.steps:
            name = step["name"]
            duration = step["duration"]
            tokens = step["tokens"]
            pct = (duration / total_time * 100) if total_time > 0 else 0

            logger.info(
                "Step performance",
                step=name,
                duration_sec=round(duration, 2),
                percentage=round(pct, 1),
                tokens=tokens if tokens > 0 else None
            )

        logger.info(
            "Total performance",
            total_duration_sec=round(total_time, 2),
            total_tokens=self.total_tokens if self.total_tokens > 0 else None
        )


def get_tracker() -> StepTracker:
    tracker = _tracker_context.get()
    if tracker is None:
        raise RuntimeError(
            "No StepTracker initialized for this context. "
            "Call init_tracker() at the start of your request handler."
        )
    return tracker


def init_tracker() -> StepTracker:
    tracker = StepTracker()
    _tracker_context.set(tracker)
    return tracker


def reset_tracker() -> StepTracker:
    return init_tracker()


@contextmanager
def track_step(name: str):
    tracker = get_tracker()
    with tracker.track_step(name):
        yield


def log_tokens(count: int):
    tracker = get_tracker()
    tracker.add_tokens(count)


def print_summary():
    tracker = get_tracker()
    tracker.print_summary()
