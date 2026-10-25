"""Command line: ``python -m relyflow.diagnostics validate MODEL --out DIR``."""

import argparse
import json
import sys

from relyflow.diagnostics.validate import validate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m relyflow.diagnostics")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("validate", help="run the validation suite for one network")
    run.add_argument("network", help="model name (pretrained network) or training output directory")
    run.add_argument("--out", required=True)
    run.add_argument("--n-sbc", type=int, default=1000)
    run.add_argument("--n-draws", type=int, default=1000)
    run.add_argument("--n-ppc", type=int, default=200)
    run.add_argument("--n-mcmc", type=int, default=20)
    run.add_argument("--refine", choices=["importance"], default=None)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--workers", type=int, default=10)
    compare = commands.add_parser("compare", help="validate model comparison and the OOD check")
    compare.add_argument("comparison", help="model-comparison training output directory")
    compare.add_argument("--out", required=True)
    compare.add_argument(
        "--amortizers", default=None, help="directory of <model>/ posterior networks"
    )
    compare.add_argument("--n-calibration", type=int, default=3000)
    compare.add_argument("--n-crosscheck-per-model", type=int, default=20)
    compare.add_argument("--n-ood", type=int, default=150)
    compare.add_argument("--n-reference", type=int, default=500)
    compare.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "compare":
        from relyflow.diagnostics.comparison import evaluate

        metrics = evaluate(args.pop("comparison"), args.pop("out"), **args)
        print(json.dumps(metrics["ood"]["rates"], indent=2))
        return
    metrics = validate(args.pop("network"), args.pop("out"), **args)
    print(json.dumps(metrics["passed"], indent=2))
    sys.exit(0 if all(metrics["passed"].values()) else 1)


if __name__ == "__main__":
    main()
