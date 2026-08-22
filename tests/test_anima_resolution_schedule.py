import unittest

from library.anima_resolution_schedule import ResolutionSchedule, ResolutionScheduleError


class ResolutionScheduleTest(unittest.TestCase):
    def test_final_stage_receives_remaining_percentage_and_exact_step_range(self):
        schedule = ResolutionSchedule.from_entries(
            [
                {"resolution": 512, "percent": 30, "batch_size": 4},
                {"resolution": 1024, "batch_size": 2},
            ],
            total_steps=10_001,
        )

        self.assertEqual([stage.percent for stage in schedule.stages], [30.0, 70.0])
        self.assertEqual((schedule.stages[0].start_step, schedule.stages[0].end_step), (0, 3_000))
        self.assertEqual((schedule.stages[1].start_step, schedule.stages[1].end_step), (3_000, 10_001))
        self.assertEqual(schedule.stage_for_step(2_999).resolution, 512)
        self.assertEqual(schedule.stage_for_step(3_000).resolution, 1024)

    def test_rejects_overallocated_or_non_final_automatic_percentage(self):
        with self.assertRaises(ResolutionScheduleError):
            ResolutionSchedule.from_entries(
                [
                    {"resolution": 512, "percent": 80, "batch_size": 2},
                    {"resolution": 768, "percent": 30, "batch_size": 2},
                ],
                total_steps=100,
            )

        with self.assertRaises(ResolutionScheduleError):
            ResolutionSchedule.from_entries(
                [
                    {"resolution": 512, "batch_size": 2},
                    {"resolution": 768, "percent": 100, "batch_size": 2},
                ],
                total_steps=100,
            )

    def test_fingerprint_is_stable_and_changes_with_batch_size(self):
        entries = [
            {"resolution": 512, "percent": 50, "batch_size": 4},
            {"resolution": 1024, "batch_size": 2},
        ]
        first = ResolutionSchedule.from_entries(entries, total_steps=100)
        second = ResolutionSchedule.from_entries(entries, total_steps=100)
        changed = ResolutionSchedule.from_entries(
            [
                {"resolution": 512, "percent": 50, "batch_size": 3},
                {"resolution": 1024, "batch_size": 2},
            ],
            total_steps=100,
        )

        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)


if __name__ == "__main__":
    unittest.main()
