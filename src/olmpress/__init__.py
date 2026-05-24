"""olmpress: LLM compression on top of Microsoft Olive."""

from olmpress import evaluators as _evaluators  # noqa: F401
from olmpress import passes as _passes  # noqa: F401
from olmpress._olive_patches import apply as _apply_olive_patches

_apply_olive_patches()


def main() -> None:
    """Entry point for the ``olmpress`` CLI."""
    import sys  # noqa: PLC0415
    from argparse import ArgumentParser  # noqa: PLC0415

    parser = ArgumentParser(
        prog="olmpress",
        description="LLM compression on top of Microsoft Olive.",
    )
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run an Olive workflow with olmpress evaluators loaded.")
    run_p.add_argument("--config", required=True, help="Path to an Olive workflow config file.")

    args = parser.parse_args()

    if args.command == "run":
        # olmpress is already imported at this point, so all @Registry.register
        # decorators have fired and olmpress_degradation is visible to Olive.
        from olive.package_config import OlivePackageConfig  # noqa: PLC0415
        from olive.workflows import run as olive_run  # noqa: PLC0415

        pkg = OlivePackageConfig.load_default_config().model_dump()
        pkg["passes"]["TorchPruningPass"] = {
            "module_path": (
                "olmpress.passes.pytorch.sparsification.structured_pruning.TorchPruningPass"
            )
        }
        olive_run(args.config, package_config=pkg)
    else:
        parser.print_help()
        sys.exit(0)
