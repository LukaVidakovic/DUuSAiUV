"""
Lane-change detector based on a sequence of predicted steering angles.

Detection heuristic
-------------------
A lane change is triggered when the instantaneous absolute steering angle
exceeds ``threshold`` for at least ``min_hold_frames`` consecutive frames.
This sustained deviation from straight-ahead driving is characteristic of a
lane-change manoeuvre.

A ``cooldown_frames`` counter prevents repeated triggers during the same
manoeuvre: after an event fires, no new event is raised for the configured
number of frames.
"""

from collections import deque
from typing import Optional


class LaneChangeDetector:
    """Stateful detector that consumes one steering angle at a time.

    A lane-change event is fired when the instantaneous absolute steering
    angle exceeds *threshold* for at least *min_hold_frames* consecutive
    frames.  After an event fires, a *cooldown_frames* counter suppresses
    further events so a single manoeuvre does not trigger repeatedly.

    A rolling buffer of the last *window_size* angles is maintained and
    exposed via the :attr:`rolling_mean` property for external inspection
    (e.g. dashboards or logging), but is not used in the detection logic
    itself.

    Args:
        window_size:      Size of the rolling history buffer (for
                          :attr:`rolling_mean` only).
        threshold:        Absolute-angle level (0-1) that, when sustained,
                          indicates a lane change.  Typical value: 0.15-0.30.
        min_hold_frames:  Minimum number of consecutive above-threshold frames
                          before an event is raised.
        cooldown_frames:  Frames to wait after an event before the next one
                          can be raised (prevents repeated triggers).
    """

    def __init__(
        self,
        window_size: int = 15,
        threshold: float = 0.2,
        min_hold_frames: int = 5,
        cooldown_frames: int = 20,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.min_hold_frames = min_hold_frames
        self.cooldown_frames = cooldown_frames

        self._buffer: deque = deque(maxlen=window_size)
        self._above_count: int = 0       # consecutive frames above threshold
        self._cooldown_count: int = 0    # frames remaining in cooldown
        self._in_event: bool = False     # currently inside a lane-change event

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, steering_angle: float) -> bool:
        """Feed the next predicted steering angle and return whether a lane
        change has just been *detected* this frame.

        Returns ``True`` only on the **first** frame of a new lane-change
        event; subsequent frames of the same manoeuvre return ``False``.
        """
        self._buffer.append(steering_angle)

        # Count down cooldown
        if self._cooldown_count > 0:
            self._cooldown_count -= 1
            self._in_event = False
            return False

        if abs(steering_angle) > self.threshold:
            self._above_count += 1
        else:
            self._above_count = 0
            self._in_event = False

        if self._above_count >= self.min_hold_frames and not self._in_event:
            self._in_event = True
            self._cooldown_count = self.cooldown_frames
            return True

        return False

    def reset(self) -> None:
        """Clear all internal state."""
        self._buffer.clear()
        self._above_count = 0
        self._cooldown_count = 0
        self._in_event = False

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def rolling_mean(self) -> Optional[float]:
        """Mean steering angle over the current window (``None`` if empty)."""
        if not self._buffer:
            return None
        return float(sum(self._buffer) / len(self._buffer))

    @property
    def is_in_event(self) -> bool:
        """``True`` while a lane-change event is active."""
        return self._in_event
