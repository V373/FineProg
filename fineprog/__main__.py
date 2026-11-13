"""Command-line entry points for the Local training and progress pipeline."""

import argparse
from copy import deepcopy
from datetime import datetime

from .config import ROOT, get_task, load_settings, resolve_path


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task", required=True)
    common.add_argument("--config-dir", help="Directory containing tasks/training/progress YAML")
    common.add_argument("--device", help="cpu or cuda; defaults to the available device")
    common.add_argument("--output", help="New output directory (default: runs/<task>/<command>-<time>)")

    for command in ("train-lower", "train-upper", "train-stage2"):
        child = commands.add_parser(command, parents=[common])
        child.add_argument("--steps", type=int, help="Limit optimizer steps for a short run")
        if command == "train-stage2":
            child.add_argument("--labels-csv", required=True, help="Teacher labels for nonexpert training")
            child.add_argument("--checkpoint", help="Stage1 Local initialization checkpoint")

    child = commands.add_parser("label", parents=[common])
    child.add_argument("--split", default="fail")
    child.add_argument("--checkpoint", help="Upper model checkpoint (defaults to the task teacher)")

    for command in ("fit-progress", "predict"):
        child = commands.add_parser(command, parents=[common])
        child.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
        child.add_argument("--checkpoint", help="Local checkpoint matching the chosen stage")
        if command == "predict":
            child.add_argument("--progress-model", required=True)
            child.add_argument("--split", default="valid")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    import torch

    settings = load_settings(args.config_dir)
    task = get_task(settings, args.task)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = (resolve_path(args.output) if args.output else
              ROOT / "runs" / args.task / f"{args.command}-{datetime.now():%Y%m%d-%H%M%S-%f}")

    if args.command.startswith("train-"):
        from .training import train_lower, train_stage2, train_upper

        key = {"train-lower": "lower", "train-upper": "upper", "train-stage2": "stage2"}[args.command]
        cfg = deepcopy(settings["training"][key])
        if key == "stage2":
            cfg["query_labels"] = str(resolve_path(args.labels_csv))
            if args.checkpoint:
                task["weights"]["lower"] = resolve_path(args.checkpoint)
        function = {"lower": train_lower, "upper": train_upper, "stage2": train_stage2}[key]
        result = function(task, cfg, output, device=device, max_steps=args.steps)
    elif args.command == "label":
        from .training import predict_labels

        checkpoint = resolve_path(args.checkpoint) if args.checkpoint else task["weights"]["upper"]
        result = predict_labels(task, checkpoint, args.split, output, device=device)
    else:
        from .progress import fit_progress, predict_progress_file

        cfg = {**settings["progress"], **task["progress"][args.stage]}
        key = "lower" if args.stage == "stage1" else "stage2"
        checkpoint = resolve_path(args.checkpoint) if args.checkpoint else task["weights"][key]
        if args.command == "fit-progress":
            result = fit_progress(task, cfg, checkpoint, output, device=device)
        else:
            result = predict_progress_file(task, cfg, checkpoint, resolve_path(args.progress_model),
                                           args.split, output, device=device)
    print(result)


if __name__ == "__main__":
    main()
