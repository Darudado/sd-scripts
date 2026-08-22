"""Resolution-stage planning for Anima training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


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
