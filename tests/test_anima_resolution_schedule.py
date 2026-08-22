import unittest

from library.anima_resolution_schedule import (
    ResolutionSchedule,
    ResolutionScheduleError,
    apply_stage_to_dataset_group,
)


class ResolutionScheduleTest(unittest.TestCase):
    def test_accepts_ui_toml_list_or_cli_json(self):
        entries = [{"resolution": 512, "percent": 40, "batch_size": 4}, {"resolution": 1024, "batch_size": 2}]

        from_toml = ResolutionSchedule.from_value(entries, total_steps=100)
        from_cli = ResolutionSchedule.from_value(str(entries).replace("'", '"'), total_steps=100)

        self.assertEqual(from_toml.to_dict(), from_cli.to_dict())

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

    def test_stage_rebuild_keeps_small_images_native_and_clears_old_latents(self):
        class Image:
            latents = object()
            latents_flipped = object()
            latents_npz = "old.npz"
            latents_aug_variants = [object()]

        class Dataset:
            width = 1024
            height = 1024
            size = 1024
            batch_size = 1
            enable_bucket = False
            bucket_no_upscale = False
            bucket_manager = object()
            image_data = {"image": Image()}

            def __init__(self):
                self.make_buckets_calls = 0

            def make_buckets(self):
                self.make_buckets_calls += 1

        dataset = Dataset()
        group = type("Group", (), {"datasets": [dataset]})()
        stage = ResolutionSchedule.from_entries(
            [{"resolution": 512, "batch_size": 4}], total_steps=10
        ).stages[0]

        apply_stage_to_dataset_group(group, stage)

        self.assertEqual((dataset.width, dataset.height, dataset.size), (512, 512, 512))
        self.assertEqual(dataset.batch_size, 4)
        self.assertTrue(dataset.enable_bucket)
        self.assertTrue(dataset.bucket_no_upscale)
        self.assertIsNone(dataset.bucket_manager)
        self.assertEqual(dataset.make_buckets_calls, 1)
        image = dataset.image_data["image"]
        self.assertIsNone(image.latents)
        self.assertIsNone(image.latents_flipped)
        self.assertIsNone(image.latents_npz)
        self.assertIsNone(image.latents_aug_variants)


if __name__ == "__main__":
    unittest.main()
