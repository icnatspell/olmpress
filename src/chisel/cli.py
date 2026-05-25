"""Command-line interface for chisel."""

from __future__ import annotations

import sys
from argparse import ArgumentParser, Namespace

from chisel import __version__


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="chisel",
        description="Model compression toolkit for PyTorch and Hugging Face Transformers.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run an Olive workflow with chisel passes loaded.")
    run_p.add_argument(
        "config",
        nargs="?",
        help="Path to an Olive workflow config file (positional alias for --config).",
    )
    run_p.add_argument(
        "--config",
        dest="config_flag",
        help="Path to an Olive workflow config file.",
    )

    sub.add_parser("list", help="List chisel-registered Olive passes and evaluators.")

    return parser


def _collect_pass_package_config() -> dict[str, dict[str, str]]:
    """Build the Olive package-config mapping for every chisel pass in chisel.passes.__all__."""
    from olive.passes import Pass

    from chisel import passes

    result: dict[str, dict[str, str]] = {}
    for name in passes.__all__:
        attr = getattr(passes, name)
        if isinstance(attr, type) and issubclass(attr, Pass):
            result[name] = {"module_path": f"{attr.__module__}.{attr.__name__}"}
    return result


def _run(config_path: str) -> None:
    from olive.package_config import OlivePackageConfig
    from olive.workflows import run as olive_run

    pkg = OlivePackageConfig.load_default_config().model_dump()
    pkg["passes"].update(_collect_pass_package_config())
    olive_run(config_path, package_config=pkg)


def _list() -> None:
    from olive.passes import Pass

    from chisel import evaluators, passes

    chisel_passes = sorted(
        name
        for name in passes.__all__
        if isinstance(getattr(passes, name), type) and issubclass(getattr(passes, name), Pass)
    )

    print(f"chisel {__version__}")
    print("\nRegistered Olive passes:")
    for name in chisel_passes:
        print(f"  - {name}")

    print("\nRegistered Olive evaluators:")
    for name in sorted(evaluators.__all__):
        print(f"  - {name}")


def _handle_run(parser: ArgumentParser, args: Namespace) -> None:
    config_path = args.config or args.config_flag
    if config_path is None:
        parser.error("the 'run' command requires a config path (positional or --config).")
    _run(config_path)


def main() -> None:
    """Entry point for the ``chisel`` CLI."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        _handle_run(parser, args)
    elif args.command == "list":
        _list()
    else:
        parser.print_help()
        sys.exit(0)
