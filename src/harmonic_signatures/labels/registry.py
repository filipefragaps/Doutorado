from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_LEADING_DASH_RE = re.compile(r"^[\s\-]+")


def normalize_label(raw: str) -> str:
    return _LEADING_DASH_RE.sub("", str(raw)).strip()


@dataclass
class LabelRegistryUpdate:
    known: list[str]
    new: list[str]


class LabelRegistry:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._labels: set[str] = self._load()

    def _load(self) -> set[str]:
        if self.path.exists():
            return set(json.loads(self.path.read_text(encoding="utf-8")))
        return set()

    def update(self, labels) -> LabelRegistryUpdate:
        seen = {normalize_label(label) for label in labels}
        new = sorted(seen - self._labels)
        self._labels |= seen
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(sorted(self._labels), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return LabelRegistryUpdate(known=sorted(self._labels), new=new)
