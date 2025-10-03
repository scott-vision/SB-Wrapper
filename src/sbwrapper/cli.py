"""Command line helpers for the :mod:`sbwrapper` package."""

from __future__ import annotations

import argparse
import pydoc

from . import c_metadata_lib
from .sb_access import SBAccess


def _render_metadata_names() -> str:
    names = sorted(
        {
            name
            for name, value in vars(c_metadata_lib).items()
            if not name.startswith("_") and isinstance(value, type)
        }
    )
    if not names:
        return "No metadata classes are available."
    header = "Available metadata structures:\n" + "\n".join(
        f" - {name}" for name in names
    )
    return header


def main(argv: list[str] | None = None) -> int:
    """Run the ``sbwrapper`` command line interface."""

    parser = argparse.ArgumentParser(
        description="Utility helpers for inspecting the sbwrapper APIs."
    )
    parser.add_argument(
        "--doc",
        action="store_true",
        help="Render the documentation for the SBAccess client.",
    )
    parser.add_argument(
        "--list-metadata",
        action="store_true",
        help="List the metadata structures generated from SlideBook headers.",
    )

    args = parser.parse_args(argv)

    if args.doc:
        print(pydoc.render_doc(SBAccess, renderer=pydoc.plaintext))

    if args.list_metadata:
        print(_render_metadata_names())

    if not args.doc and not args.list_metadata:
        parser.print_help()

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI passthrough
    raise SystemExit(main())
