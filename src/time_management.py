import time
from typing import Dict, List, Optional, Union


class Timer:
    """Context manager for timing a block of code."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: Optional[float] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args: object) -> None:
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start


class EpochTimer:
    """Tracks per-epoch timing statistics during model training."""

    def __init__(self) -> None:
        self._epoch_times: List[float] = []
        self._start: Optional[float] = None

    def start_epoch(self) -> None:
        """Start timing the current epoch."""
        self._start = time.perf_counter()

    def end_epoch(self) -> float:
        """End timing the current epoch, record its duration, and return it."""
        if self._start is None:
            raise RuntimeError("end_epoch() called before start_epoch().")
        elapsed = time.perf_counter() - self._start
        self._epoch_times.append(elapsed)
        self._start = None
        return elapsed

    @property
    def mean_epoch_time(self) -> float:
        """Mean epoch duration in seconds."""
        if not self._epoch_times:
            return 0.0
        return sum(self._epoch_times) / len(self._epoch_times)

    @property
    def total_time(self) -> float:
        """Total elapsed time across all completed epochs in seconds."""
        return sum(self._epoch_times)

    def estimate_remaining(self, epochs_remaining: int) -> float:
        """Estimate remaining training time in seconds based on mean epoch duration."""
        return self.mean_epoch_time * epochs_remaining

    def get_summary(self) -> Dict[str, Union[float, int]]:
        """Return a summary dict with timing statistics."""
        return {
            "mean_epoch_time_s": self.mean_epoch_time,
            "total_time_s": self.total_time,
            "num_epochs": len(self._epoch_times),
        }
