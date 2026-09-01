from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TEMPLATE_PATH = Path(__file__).parent / "index.template.html"


def render_template(template_path: Path, scene_payload: dict) -> str:
    return template_path.read_text(encoding="utf-8").replace(
        "__SCENE_JSON__",
        json.dumps(scene_payload, separators=(",", ":")),
    )
