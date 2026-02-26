"""
Lane-change detector based on a sequence of predicted steering angles.

Detection heuristic
-------------------
The detector uses a two-step heuristic:
1) Candidate start: absolute steering angle exceeds ``threshold`` for at
   least ``min_hold_frames`` consecutive frames with the same steering sign.
2) Candidate confirmation: the candidate is confirmed as a lane-change event
   only after a stabilization cue appears within ``max_settle_frames``:
   either steering returns near straight driving (``settle_threshold``) or a
   counter-steer with opposite sign occurs.

This reduces false positives on long constant curves, where steering can stay
large but does not show the transient "leave lane + settle" pattern.

After an event fires, ``cooldown_frames`` suppresses repeated triggers.
"""

from collections import deque
from typing import Optional


class LaneChangeDetector:
    """Stateful detector that consumes one steering angle at a time.

    A lane-change event is fired after a two-step pattern:
    sustained steering in one direction (candidate), then settling/counter-
    steering confirmation within a time limit.

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
                          before candidate creation.
        settle_threshold: Angle magnitude considered "settled" back near
                          straight driving (default: 0.08).
        max_settle_frames:
                          Maximum frames allowed for candidate confirmation.
        cooldown_frames:  Frames to wait after an event before the next one
                          can be raised (prevents repeated triggers).
    """

    def __init__(
        self,
        window_size: int = 15,
        threshold: float = 0.2,
        min_hold_frames: int = 5,
        settle_threshold: float = 0.08,
        max_settle_frames: int = 25,
        cooldown_frames: int = 20,
    ) -> None:
        self.window_size = window_size
        self.threshold = threshold
        self.min_hold_frames = min_hold_frames
        self.settle_threshold = settle_threshold
        self.max_settle_frames = max_settle_frames
        self.cooldown_frames = cooldown_frames

        self._buffer: deque = deque(maxlen=window_size)
        self._above_count: int = 0        # consecutive frames above threshold
        self._run_sign: int = 0           # current sign of sustained steering
        self._candidate_active: bool = False
        self._candidate_sign: int = 0
        self._candidate_frames: int = 0
        self._cooldown_count: int = 0     # frames remaining in cooldown
        self._in_event: bool = False      # currently inside a lane-change event

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

        sign = self._sign(steering_angle)

        # Candidate phase: wait for stabilization/counter-steer confirmation.
        if self._candidate_active:
            self._candidate_frames += 1
            settled = abs(steering_angle) <= self.settle_threshold
            opposite_sign = (
                sign != 0
                and sign != self._candidate_sign
                and abs(steering_angle) >= self.settle_threshold
            )

            if settled or opposite_sign:
                self._emit_event()
                return True

            if self._candidate_frames >= self.max_settle_frames:
                self._reset_candidate()
            return False

        # Candidate creation phase: sustained steering in one direction.
        if abs(steering_angle) > self.threshold:
            if sign != 0 and sign == self._run_sign:
                self._above_count += 1
            else:
                self._above_count = 1
                self._run_sign = sign
        else:
            self._above_count = 0
            self._run_sign = 0
            self._in_event = False

        if self._above_count >= self.min_hold_frames and self._run_sign != 0:
            self._candidate_active = True
            self._candidate_sign = self._run_sign
            self._candidate_frames = 0
            self._above_count = 0
            self._run_sign = 0

        return False

    def reset(self) -> None:
        """Clear all internal state."""
        self._buffer.clear()
        self._above_count = 0
        self._run_sign = 0
        self._reset_candidate()
        self._cooldown_count = 0
        self._in_event = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sign(value: float) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def _reset_candidate(self) -> None:
        self._candidate_active = False
        self._candidate_sign = 0
        self._candidate_frames = 0

    def _emit_event(self) -> None:
        self._in_event = True
        self._cooldown_count = self.cooldown_frames
        self._above_count = 0
        self._run_sign = 0
        self._reset_candidate()

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
