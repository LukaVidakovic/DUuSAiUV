import unittest

from lane_change_detector import LaneChangeDetector


class LaneChangeDetectorTests(unittest.TestCase):
    def _run(self, detector: LaneChangeDetector, seq):
        return [detector.update(x) for x in seq]

    def test_no_event_for_small_angles(self):
        detector = LaneChangeDetector(threshold=0.2, min_hold_frames=3)
        events = self._run(detector, [0.01, -0.05, 0.11, -0.12, 0.18, 0.0, 0.19])
        self.assertFalse(any(events))

    def test_event_after_sustained_angle_then_settle(self):
        detector = LaneChangeDetector(
            threshold=0.2,
            min_hold_frames=3,
            settle_threshold=0.08,
            max_settle_frames=12,
            cooldown_frames=3,
        )
        # 3x above-threshold (candidate), then settle below 0.08 (event).
        seq = [0.0, 0.21, 0.25, 0.23, 0.18, 0.06]
        events = self._run(detector, seq)
        self.assertEqual(sum(events), 1)
        self.assertTrue(events[-1])

    def test_cooldown_blocks_immediate_retrigger(self):
        detector = LaneChangeDetector(
            threshold=0.2,
            min_hold_frames=2,
            settle_threshold=0.08,
            max_settle_frames=8,
            cooldown_frames=4,
        )
        seq = [
            0.25, 0.24, 0.05,  # first event
            0.26, 0.27, 0.04,  # would be second event but inside cooldown
            0.0,
            0.28, 0.29, 0.03,  # second valid event after cooldown
        ]
        events = self._run(detector, seq)
        self.assertEqual(sum(events), 2)

    def test_long_constant_curve_is_not_lane_change(self):
        detector = LaneChangeDetector(
            threshold=0.2,
            min_hold_frames=3,
            settle_threshold=0.08,
            max_settle_frames=6,
            cooldown_frames=3,
        )
        # Sustained turn without settling/opposite sign within max_settle_frames.
        seq = [0.22, 0.24, 0.25, 0.26, 0.25, 0.24, 0.23, 0.24, 0.22, 0.21]
        events = self._run(detector, seq)
        self.assertFalse(any(events))

    def test_counter_steer_confirms_event(self):
        detector = LaneChangeDetector(
            threshold=0.2,
            min_hold_frames=3,
            settle_threshold=0.08,
            max_settle_frames=10,
            cooldown_frames=3,
        )
        # Positive sustained steering, then strong opposite sign.
        seq = [0.0, 0.22, 0.24, 0.25, -0.18, -0.22]
        events = self._run(detector, seq)
        self.assertEqual(sum(events), 1)
        self.assertTrue(events[4] or events[5])


if __name__ == "__main__":
    unittest.main()
