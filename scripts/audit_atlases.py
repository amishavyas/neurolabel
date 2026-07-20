#!/usr/bin/env python3
"""Audit bundled or explicitly named atlas images."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from neurolabel.core.audit import AtlasAudit, audit_atlas, audit_bundled
from neurolabel.core.specification import builtin_atlas_ids, load_specification


def main(argv: Sequence[str] | None = None) -> int:
    """Run atlas audits and emit a human or JSON report."""
    parser = argparse.ArgumentParser(
        description="Report factual NIfTI and discrete-label atlas observations."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="Built-in atlas IDs or NIfTI paths; defaults to all bundled atlases.",
    )
    parser.add_argument("--json", metavar="PATH", help="Write JSON report to PATH.")
    arguments = parser.parse_args(argv)

    audits = _audits(arguments.targets)
    if arguments.json:
        output = Path(arguments.json).expanduser()
        output.write_text(
            json.dumps([audit.to_dict() for audit in audits], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    else:
        print(_human_report(audits))
    return 0


def _audits(targets: Sequence[str]) -> tuple[AtlasAudit, ...]:
    if not targets:
        return audit_bundled()
    builtins = set(builtin_atlas_ids())
    audits: list[AtlasAudit] = []
    for target in targets:
        if target in builtins:
            specification = load_specification(target)
            audits.append(
                audit_atlas(specification.resolve_image_path(), specification)
            )
        else:
            audits.append(audit_atlas(Path(target)))
    return tuple(audits)


def _human_report(audits: Sequence[AtlasAudit]) -> str:
    sections: list[str] = []
    for audit in audits:
        lines = [f"[{audit.atlas_id}]"]
        lines.extend(f"{name}: {value}" for name, value in audit.to_dict().items())
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


if __name__ == "__main__":
    raise SystemExit(main())
