"""chisel: LLM compression on top of Microsoft Olive."""

from chisel import evaluators as _evaluators  # noqa: F401
from chisel import passes as _passes  # noqa: F401
from chisel._olive_patches import apply as _apply_olive_patches

_apply_olive_patches()


def main() -> None:
    """Entry point for the ``chisel`` CLI."""
    import sys
    from argparse import ArgumentParser

    parser = ArgumentParser(
        prog="chisel",
        description="LLM compression on top of Microsoft Olive.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run an Olive workflow with chisel evaluators loaded.")
    run_p.add_argument("--config", required=True, help="Path to an Olive workflow config file.")

    args = parser.parse_args()

    if args.command == "run":
        # chisel is already imported at this point, so all @Registry.register
        # decorators have fired and chisel_degradation is visible to Olive.
        from olive.package_config import OlivePackageConfig
        from olive.workflows import run as olive_run

        pkg = OlivePackageConfig.load_default_config().model_dump()
        pkg["passes"]["TorchPruningPass"] = {
            "module_path": ("chisel.passes.pytorch.structured_pruning.TorchPruningPass")
        }
        olive_run(args.config, package_config=pkg)
    else:
        parser.print_help()
        sys.exit(0)
