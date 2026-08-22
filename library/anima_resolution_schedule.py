"""Resolution-stage planning for Anima training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Callable, Mapping, Sequence


class ResolutionScheduleError(ValueError):
    """Raised when an Anima resolution schedule is invalid."""


@dataclass(frozen=True)
class ResolutionStage:
    resolution: int
    percent: float
    batch_size: int
    start_step: int
    end_step: int


class ResolutionSchedule:
    def __init__(self, stages: Sequence[ResolutionStage], total_steps: int) -> None:
        self.stages = tuple(stages)
        self.total_steps = total_steps

    @classmethod
    def from_value(cls, value: Any, total_steps: int) -> "ResolutionSchedule":
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as error:
                raise ResolutionScheduleError("resolution_schedule must be a JSON array") from error
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise ResolutionScheduleError("resolution_schedule must be a list of stages")
        return cls.from_entries(value, total_steps)

    @classmethod
    def from_entries(cls, entries: Sequence[Mapping[str, Any]], total_steps: int) -> "ResolutionSchedule":
        if total_steps <= 0:
            raise ResolutionScheduleError("total_steps must be greater than zero")
        if not entries:
            raise ResolutionScheduleError("resolution_schedule requires at least one stage")

        normalized: list[dict[str, Any]] = []
        specified_percent = 0.0
        for index, entry in enumerate(entries):
            try:
                resolution = int(entry["resolution"])
                batch_size = int(entry["batch_size"])
            except (KeyError, TypeError, ValueError) as error:
                raise ResolutionScheduleError(f"stage {index + 1} requires integer resolution and batch_size") from error
            if resolution <= 0 or batch_size <= 0:
                raise ResolutionScheduleError(f"stage {index + 1} resolution and batch_size must be greater than zero")

            percent = entry.get("percent")
            if percent is None:
                if index != len(entries) - 1:
                    raise ResolutionScheduleError("only the final stage may use the automatic percentage")
            else:
                try:
                    percent = float(percent)
                except (TypeError, ValueError) as error:
                    raise ResolutionScheduleError(f"stage {index + 1} percent must be numeric") from error
                if percent <= 0:
                    raise ResolutionScheduleError(f"stage {index + 1} percent must be greater than zero")
                specified_percent += percent
            normalized.append({"resolution": resolution, "batch_size": batch_size, "percent": percent})

        if specified_percent > 100.0:
            raise ResolutionScheduleError("resolution schedule percentages cannot exceed 100")

        final_percent = normalized[-1]["percent"]
        if final_percent is None:
            final_percent = 100.0 - specified_percent
            if final_percent <= 0:
                raise ResolutionScheduleError("automatic final percentage must be greater than zero")
            normalized[-1]["percent"] = final_percent
        elif specified_percent != 100.0:
            raise ResolutionScheduleError("explicit stage percentages must total exactly 100")

        stages: list[ResolutionStage] = []
        start_step = 0
        cumulative_percent = 0.0
        for index, entry in enumerate(normalized):
            cumulative_percent += entry["percent"]
            end_step = total_steps if index == len(normalized) - 1 else round(total_steps * cumulative_percent / 100.0)
            if end_step <= start_step:
                raise ResolutionScheduleError("each resolution stage must receive at least one optimizer step")
            stages.append(
                ResolutionStage(
                    resolution=entry["resolution"],
                    percent=entry["percent"],
                    batch_size=entry["batch_size"],
                    start_step=start_step,
                    end_step=end_step,
                )
            )
            start_step = end_step
        return cls(stages, total_steps)

    def stage_for_step(self, global_step: int) -> ResolutionStage:
        if global_step < 0 or global_step >= self.total_steps:
            raise ResolutionScheduleError(f"step {global_step} is outside the resolution schedule")
        for stage in self.stages:
            if stage.start_step <= global_step < stage.end_step:
                return stage
        raise ResolutionScheduleError(f"no resolution stage contains step {global_step}")

    def to_dict(self) -> dict[str, Any]:
        return {"total_steps": self.total_steps, "stages": [asdict(stage) for stage in self.stages]}

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return sha256(encoded).hexdigest()


def has_remaining_training_steps(global_step: int, total_steps: int) -> bool:
    """Return whether a resolution stage can be selected for this step."""

    return global_step < total_steps


def resolution_disk_cache_key(stage: ResolutionStage) -> str:
    """Stable disk-cache namespace for a target resolution, independent of its schedule position."""

    return f"resolution-{stage.resolution}"


class ResolutionStageCache:
    """In-memory snapshots of dataset/bucket state prepared before training.

    The snapshots intentionally retain tensor references. Latent tensors are
    already CPU-resident after caching; copying them would duplicate the RAM
    required for each scheduled resolution.
    """

    _DATASET_ATTRIBUTES = (
        "width",
        "height",
        "size",
        "batch_size",
        "enable_bucket",
        "bucket_no_upscale",
        "bucket_manager",
        "buckets_indices",
        "_length",
    )
    _IMAGE_ATTRIBUTES = (
        "bucket_reso",
        "resized_size",
        "latents",
        "latents_flipped",
        "latents_npz",
        "latents_original_size",
        "latents_crop_ltrb",
        "latents_aug_variants",
    )

    def __init__(self) -> None:
        self._snapshots: dict[ResolutionStage, list[tuple[dict[str, Any], dict[str, dict[str, Any]]]]] = {}

    def capture(self, dataset_group: Any, stage: ResolutionStage) -> None:
        datasets = []
        for dataset in dataset_group.datasets:
            dataset_state = {
                attribute: getattr(dataset, attribute)
                for attribute in self._DATASET_ATTRIBUTES
                if hasattr(dataset, attribute)
            }
            image_state = {
                image_key: {
                    attribute: getattr(image_info, attribute)
                    for attribute in self._IMAGE_ATTRIBUTES
                    if hasattr(image_info, attribute)
                }
                for image_key, image_info in dataset.image_data.items()
            }
            datasets.append((dataset_state, image_state))
        self._snapshots[stage] = datasets

    def restore(self, dataset_group: Any, stage: ResolutionStage) -> None:
        try:
            snapshots = self._snapshots[stage]
        except KeyError as error:
            raise ResolutionScheduleError(f"latents were not pre-cached for stage starting at step {stage.start_step}") from error
        if len(snapshots) != len(dataset_group.datasets):
            raise ResolutionScheduleError("dataset count changed after resolution-stage caching")

        for dataset, (dataset_state, image_state) in zip(dataset_group.datasets, snapshots):
            for attribute, value in dataset_state.items():
                setattr(dataset, attribute, value)
            for image_key, values in image_state.items():
                image_info = dataset.image_data[image_key]
                for attribute, value in values.items():
                    setattr(image_info, attribute, value)


def prepare_rebuilt_dataloader(accelerator: Any, factory: Callable[[], Any], previous: Any = None) -> Any:
    """Build a stage DataLoader and replace its predecessor in Accelerate's checkpoint registry."""

    prepared = accelerator.prepare_data_loader(factory())
    dataloaders = getattr(accelerator, "_dataloaders", None)
    if previous is not None and isinstance(dataloaders, list) and previous in dataloaders:
        dataloaders.remove(previous)
    return prepared


def apply_stage_to_dataset_group(dataset_group: Any, stage: ResolutionStage) -> None:
    """Rebuild dataset buckets for one stage without replacing the dataset object.

    ``bucket_no_upscale`` preserves native resolution for small images while the
    stage square limits the maximum image area for larger images.
    """

    for dataset in dataset_group.datasets:
        dataset.width = stage.resolution
        dataset.height = stage.resolution
        dataset.size = stage.resolution
        dataset.batch_size = stage.batch_size
        dataset.enable_bucket = True
        dataset.bucket_no_upscale = True
        dataset.bucket_manager = None

        for image_info in dataset.image_data.values():
            image_info.latents = None
            image_info.latents_flipped = None
            image_info.latents_npz = None
            image_info.latents_aug_variants = None

        dataset.make_buckets()
