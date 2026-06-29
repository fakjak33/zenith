"""Append a research note to the CAS model registry (used by the /zenith-research
skill and available standalone).

    python scripts/add_research_note.py --family frm_ts_mom \
        --title "Factor Momentum Everywhere" --source https://... \
        --abstract "..." [--weight 1.1] [--status processed]

Run with the repo's virtualenv so ``zenith`` is importable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zenith.cas import registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, help="registry family key (see registry.DEFAULT_WEIGHTS)")
    ap.add_argument("--title", required=True)
    ap.add_argument("--abstract", default="")
    ap.add_argument("--source", default="", help="URL or filename")
    ap.add_argument("--status", default="", help="e.g. processed | pending-review")
    ap.add_argument("--weight", type=float, default=None, help="optional new family weight")
    a = ap.parse_args()

    reg = registry.load()
    if a.family not in reg["weights"]:
        print(f"WARNING: '{a.family}' is not an existing family. Known families:")
        print("  " + ", ".join(sorted(reg["weights"])))
    note = registry.add_note(a.family, a.title, a.abstract,
                             weight_adjustment=a.weight, source=a.source, status=a.status)
    print(f"Saved note against '{note['family']}' ({note['ts']}).")
    if a.weight is not None:
        print(f"Set weight -> {note.get('weight_set_to')}")


if __name__ == "__main__":
    main()
