"""Project model: everything needed to edit one lesson, persisted as JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from openclip.engine.captions import CaptionStyle
from openclip.engine.cutplan import CutOptions, CutPlan
from openclip.engine.script import LessonScript

SCHEMA_VERSION = 1


@dataclass
class GazeOptions:
    enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return {"enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> GazeOptions:
        return cls(enabled=bool(data.get("enabled", False)))


@dataclass
class Project:
    source: Path
    script: LessonScript = field(default_factory=LessonScript)
    captions: CaptionStyle = field(default_factory=CaptionStyle)
    cut_options: CutOptions = field(default_factory=CutOptions)
    audio_preset: str = "teaching"
    align_language: str | None = None
    gaze: GazeOptions = field(default_factory=GazeOptions)
    alignment: list[dict[str, object]] | None = None
    cut_plan: CutPlan | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA_VERSION,
            "source": str(self.source),
            "script": self.script.to_dict(),
            "captions": self.captions.to_dict(),
            "cut_options": self.cut_options.to_dict(),
            "audio_preset": self.audio_preset,
            "align_language": self.align_language,
            "gaze": self.gaze.to_dict(),
            "alignment": self.alignment,
            "cut_plan": self.cut_plan.to_dict() if self.cut_plan else None,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Project:
        from openclip.engine.jsonio import (
            as_dict,
            as_int,
            as_list,
            as_optional_list,
            as_optional_str,
            as_str,
        )

        if as_int(data, "schema", 0) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported project schema {data.get('schema')!r} (expected {SCHEMA_VERSION})"
            )
        cut_plan_raw = as_optional_list(data, "cut_plan")
        return cls(
            source=Path(as_str(data, "source")),
            script=LessonScript.from_dict(as_list(data, "script")),
            captions=CaptionStyle.from_dict(as_dict(data, "captions")),
            cut_options=CutOptions.from_dict(as_dict(data, "cut_options")),
            audio_preset=as_str(data, "audio_preset", "teaching"),
            align_language=as_optional_str(data, "align_language"),
            gaze=GazeOptions.from_dict(as_dict(data, "gaze")),
            alignment=as_optional_list(data, "alignment"),
            cut_plan=CutPlan.from_dict(cut_plan_raw) if cut_plan_raw is not None else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> Project:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def create(cls, source: str | Path, script_text: str) -> Project:
        from openclip.engine.script import parse_script

        return cls(source=Path(source), script=parse_script(script_text))
