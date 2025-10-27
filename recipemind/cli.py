#!/usr/bin/env python3
"""Unified command line interface for RecipeMind utilities.

This script merges the functionality that used to live in multiple shell and
Python entry points (``run.sh``, ``train.py`` and ``test.py``).  The interface is
organized into sub-commands so that common workflows – setting up the
environment, training, testing and figure generation – can be accessed from a
single file while reusing the existing model implementation modules.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_DIR = REPO_ROOT / "legacy"

import numpy as np
import torch
import wandb
import pickle
import setproctitle

from recipemind.config import *  # noqa: F401,F403 - re-exported constants are required.
from recipemind.models import *  # noqa: F401,F403 - used by training/testing logic.
from recipemind.pipeline import *  # noqa: F401,F403 - pipelines are reused during testing.
from recipemind.pipeline.trainer import *  # noqa: F401,F403 - training utilities and collate fns.

# Align thread usage with the original training script defaults.
torch.set_num_threads(1)
os.environ.setdefault("MKL_NUM_THREADS", "20")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "20")
os.environ.setdefault("OMP_NUM_THREADS", "20")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "20")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _info(message: str) -> None:
    print(f"[INFO] {message}")


def _run(cmd: Iterable[str], cwd: Path | None = None) -> None:
    """Run a subprocess command, raising a helpful message if it fails."""
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    if completed.returncode != 0:
        cmd_repr = " ".join(cmd)
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {cmd_repr}")


# ---------------------------------------------------------------------------
# Environment bootstrap (formerly run.sh)
# ---------------------------------------------------------------------------

def cmd_setup(args: argparse.Namespace) -> None:
    """Create a virtual environment, install dependencies and bootstrap folders."""
    python_candidates = []
    if args.python is not None:
        python_candidates.append(Path(args.python))
    python_candidates.append(Path("/opt/homebrew/bin/python3"))
    python_candidates.extend(Path(p) for p in os.environ.get("PATH", "").split(os.pathsep))

    chosen_python: Path | None = None
    for candidate in python_candidates:
        if candidate.is_dir():
            # If a directory was added from PATH splitting, append python3.
            candidate_path = candidate / "python3"
        else:
            candidate_path = candidate
        if candidate_path.exists() and os.access(candidate_path, os.X_OK):
            chosen_python = candidate_path
            break

    if chosen_python is None:
        raise RuntimeError("python3 not found. Please install Python 3 (e.g., 'brew install python').")

    version = subprocess.check_output([str(chosen_python), "-V"], text=True).strip()
    _info(f"Using Python: {version} at {chosen_python}")

    venv_dir = Path(args.venv_dir)
    if not venv_dir.exists():
        _info(f"Creating virtual environment at {venv_dir}")
        _run([str(chosen_python), "-m", "venv", str(venv_dir)])

    python_exe = venv_dir / "bin" / "python"
    if not python_exe.exists():
        raise RuntimeError("Failed to locate python executable in the virtual environment.")

    _info("Upgrading pip and installing dependencies")
    _run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(python_exe), "-m", "pip", "install", "-r", str(REPO_ROOT / "requirements.txt")])

    for directory in (REPO_ROOT / "data", REPO_ROOT / "figures", REPO_ROOT / "outputs"):
        directory.mkdir(parents=True, exist_ok=True)
    _info("Environment ready.")


# ---------------------------------------------------------------------------
# Figure generation utilities (formerly run.sh)
# ---------------------------------------------------------------------------

def cmd_figures(args: argparse.Namespace) -> None:
    mode = args.mode
    figures_dir = REPO_ROOT / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if mode == "default":
        _info("Generating default figures via Mypaperfiguretable.py")
        _run([sys.executable, str(LEGACY_DIR / "Mypaperfiguretable.py")])
    elif mode == "paper":
        _info("Generating full HerbMind publication figures")
        _run([
            sys.executable,
            str(LEGACY_DIR / "HerbMindFiguresMatched.py"),
            "--output-dir",
            str(figures_dir),
        ])
    elif mode == "pro":
        _info("Generating publication-quality figures (pro mode)")
        outputs_dir = REPO_ROOT / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, str(LEGACY_DIR / "Mypaperfiguretable_Pro.py")])
    else:
        raise ValueError(f"Unsupported figure mode: {mode}")
    _info("Done. Figures saved under ./figures/.")


# ---------------------------------------------------------------------------
# Training logic (formerly train.py)
# ---------------------------------------------------------------------------

def setup_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_vector_dimensions_train(args: argparse.Namespace) -> argparse.Namespace:
    lang_dim = {}
    dim_dict = {
        "reciptor": 600,
        "bert-base-uncased": 768,
        "flavorgraph": 300,
        "im2recipe": 300,
        "binary": 630,
    }

    lang_dim["J"] = dim_dict[args.initial_vectors_J]
    lang_dim["T"] = dim_dict[args.initial_vectors_T]
    lang_dim["R"] = dim_dict[args.initial_vectors_R]

    args.lang_dim = lang_dim
    return args


def baseline_arguments(args: argparse.Namespace) -> argparse.Namespace:
    if args.model_struct == "kitchenette":
        _info("Kitchenette Baseline Model")
        args.dataset_name = "recipemind_doublets"
        args.initial_vectors_J = "im2recipe"
        args.hidden_dim = 1024
        args.dropout_rate = 0.2
        args.learning_rate = 1e-4
        args.weight_decay = 1e-5
        args.num_epochs = 60
        args.batch_size = 32
    return args


def cmd_train(args: argparse.Namespace) -> None:
    args = baseline_arguments(args)
    if "wnd" in args.model_struct:
        args.batch_size = 32

    _info(f"Setting random seed {args.random_seed}")
    setup_seed(args.random_seed)

    _info("Computing vector dimensions")
    args = get_vector_dimensions_train(args)

    _info("Initialising Weights & Biases")
    wandb_init_args = {
        "project": args.project_name,
        "group": args.session_name,
        "name": f"training_{args.random_seed}",
        "config": args,
    }
    for key, value in wandb_init_args.items():
        print(key, value)
    wandb.init(**wandb_init_args)
    setproctitle.setproctitle(f"{args.session_name}")
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("valid/step")
    wandb.define_metric("valid/*", step_metric="valid/step")

    _info("Loading model, trainer and collate function")
    trainer = load_recipe_trainer(args)
    collate = CollateFn(args)
    model = load_recipe_model(args).cuda()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _info(f"Trainable parameters for {args.model_struct}: {num_params}")
    wandb.watch(model, log="gradients", log_freq=1000)
    pickle.dump(args, open(trainer.checkpoint_path + "model_config.pkl", "wb"))

    _info("Loading train/valid datasets")
    train_loader = get_train_loader(args, collate)
    valid_loader = get_valid_loader(args, collate)

    _info("Starting training")
    model = trainer.train_model(model, train_loader, valid_loader, args.fine_tuning)
    del train_loader
    torch.cuda.empty_cache()

    if not args.debug_mode:
        _info("Evaluating on the validation set")
        trainer.test_model(model, valid_loader, True)
    wandb.finish()


# ---------------------------------------------------------------------------
# Testing logic (formerly test.py)
# ---------------------------------------------------------------------------

def get_vector_dimensions_test(args: argparse.Namespace) -> argparse.Namespace:
    lang_dim = {}
    dim_dict = {
        "reciptor": 768,
        "bert-base-uncased": 768,
        "flavorgraph": 300,
        "im2recipe": 300,
    }

    if args.model_struct == "recipebowl":
        lang_dim["J"] = 300
        lang_dim["T"] = 630
        lang_dim["R"] = 600
        args.initial_vectors_J = "flavorgraph"
        args.initial_vectors_T = "binary"
        args.initial_vectors_R = "reciptor"
    else:
        lang_dim["J"] = dim_dict[args.initial_vectors_J]
        lang_dim["T"] = dim_dict[args.initial_vectors_T]
        lang_dim["R"] = dim_dict[args.initial_vectors_R]
        lang_dim["S"] = dim_dict[args.initial_vectors_J]
        lang_dim["H"] = dim_dict[args.initial_vectors_J]

    args.lang_dim = lang_dim
    return args


def cmd_test(args: argparse.Namespace) -> None:
    _info(f"Setting random seed {args.random_seed}")
    setup_seed(args.random_seed)

    _info("Loading saved training configuration")
    pn, sn = args.project_name, args.session_name
    downstream_task = args.downstream_task
    ideation_score = args.ideation_score
    batch_size = args.batch_size

    args_cfg = pickle.load(open(f"{OUT_PATH}{pn}_{sn}_{args.random_seed}/model_config.pkl", "rb"))
    args_cfg = get_vector_dimensions_test(args_cfg)
    args_cfg.batch_size = batch_size
    if "ars" in args_cfg.model_struct:
        args_cfg.hidden_dim = 128

    _info("Initialising Weights & Biases")
    wandb_init_args = {
        "project": args.project_name,
        "group": args.session_name,
        "name": f"{downstream_task}_{args.random_seed}",
        "config": args_cfg,
    }
    for key, value in wandb_init_args.items():
        print(key, value)
    wandb.init(**wandb_init_args)
    setproctitle.setproctitle(f"{args.session_name}")

    _info("Loading collate function")
    collate = CollateFn(args_cfg)

    _info("Loading model checkpoint and running the downstream pipeline")
    dataset_suffix = f"recipemind_{downstream_task.split('_')[-1]}_{ideation_score}"
    args_cfg.dataset_name = dataset_suffix
    model = load_recipe_model(args_cfg).cuda()
    checkpoint = torch.load(f"{OUT_PATH}{pn}_{sn}_{args.random_seed}/epoch_best.mdl")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    test_loader = get_test_loader(args_cfg, collate)

    if downstream_task == "scoring_subset2":
        pipeline = FoodPairingPipeline(
            model=model,
            collate=collate,
            session_name=f"{args_cfg.session_name}_{args.random_seed}",
            dataset_name=args_cfg.dataset_name,
        )
        results = pipeline(test_loader, "subset2")
    elif downstream_task in {
        "scoring_subset3",
        "scoring_subset4",
        "scoring_subset5",
        "scoring_subset6",
        "scoring_subset7",
    }:
        pipeline = NtupletScoringPipeline(
            model=model,
            collate=collate,
            session_name=f"{args_cfg.session_name}_{args.random_seed}",
            dataset_name=args_cfg.dataset_name,
        )
        subset_label = downstream_task.replace("scoring_", "")
        results = pipeline(test_loader, subset_label)
    else:
        raise ValueError(f"Unsupported downstream task: {downstream_task}")

    wandb.log(results)
    wandb.finish()


# ---------------------------------------------------------------------------
# Argument parser wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RecipeMind consolidated CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # setup
    parser_setup = subparsers.add_parser("setup", help="Create venv and install dependencies")
    parser_setup.add_argument("--python", help="Path to python3 executable to use", default=None)
    parser_setup.add_argument("--venv-dir", default=".venv", help="Virtual environment directory")
    parser_setup.set_defaults(func=cmd_setup)

    # figures
    parser_figures = subparsers.add_parser("figures", help="Generate publication figures")
    parser_figures.add_argument(
        "--mode",
        choices=["default", "paper", "pro"],
        default="default",
        help="Figure generation mode",
    )
    parser_figures.set_defaults(func=cmd_figures)

    # train
    parser_train = subparsers.add_parser("train", help="Train RecipeMind models")
    parser_train.add_argument("--project_name", "-pn", default="Test SonyAI", type=str)
    parser_train.add_argument("--session_name", "-sn", default="Test SonyAI", type=str)
    parser_train.add_argument("--random_seed", default=911012, type=int)
    parser_train.add_argument("--fine_tuning", default=None, type=str)
    parser_train.add_argument("--debug_mode", "-dm", default=False, action="store_true")
    parser_train.add_argument("--dataset_index", default="ver1", type=str)
    parser_train.add_argument("--dataset_version", default="211210", type=str)
    parser_train.add_argument("--dataset_name", default="recipemind_mixed_sPMId02", type=str)
    parser_train.add_argument("--initial_vectors_J", default="flavorgraph", type=str)
    parser_train.add_argument("--initial_vectors_T", default="bert-base-uncased", type=str)
    parser_train.add_argument("--initial_vectors_R", default="bert-base-uncased", type=str)
    parser_train.add_argument("--model_struct", default="recipemind", type=str)
    parser_train.add_argument("--model_analysis", default=False, action="store_true")
    parser_train.add_argument("--hidden_dim", default=128, type=int)
    parser_train.add_argument("--dropout_rate", default=0.025, type=float)
    parser_train.add_argument("--learning_rate", default=1e-4, type=float)
    parser_train.add_argument("--weight_decay", default=1e-5, type=float)
    parser_train.add_argument("--num_epochs", default=30, type=int)
    parser_train.add_argument("--batch_size", default=512, type=int)
    parser_train.add_argument("--loss_function", default="rmse", type=str)
    parser_train.add_argument("--grad_update", default="default", type=str)
    parser_train.add_argument("--train_eval", default=False, action="store_true")
    parser_train.add_argument("--hybrid_coef", default=0.5, type=float)
    parser_train.add_argument("--mc_dropout", default=False, action="store_true")
    parser_train.add_argument("--sab_num_aheads", default=8, type=int)
    parser_train.add_argument("--sab_num_blocks", default=3, type=int)
    parser_train.add_argument("--ars_num_hsets", default=256, type=int)
    parser_train.add_argument("--ars_num_helms", default=8, type=int)
    parser_train.add_argument("--pma_num_aheads", default=8, type=int)
    parser_train.add_argument("--pma_num_sdvecs", default=4, type=int)
    parser_train.add_argument("--pma_num_blocks", default=2, type=int)
    parser_train.add_argument("--multihead_sim", default="general_dot", type=str)
    parser_train.add_argument("--multihead_big", default=False, action="store_true")
    parser_train.set_defaults(func=cmd_train)

    # test
    parser_test = subparsers.add_parser("test", help="Run downstream evaluation")
    parser_test.add_argument("--project_name", "-pn", default="Test", type=str)
    parser_test.add_argument("--group_name", "-gn", default="Test", type=str)
    parser_test.add_argument("--session_name", "-sn", default="Test", type=str)
    parser_test.add_argument("--ideation_score", "-is", default="sPMId02", type=str)
    parser_test.add_argument("--random_seed", default=911012, type=int)
    parser_test.add_argument("--dataset_index", default=5, type=int)
    parser_test.add_argument("--dataset_version", default="211210", type=str)
    parser_test.add_argument("--dataset_name", default="recipemind", type=str)
    parser_test.add_argument("--initial_vectors_J", default="flavorgraph", type=str)
    parser_test.add_argument("--initial_vectors_T", default="nothing", type=str)
    parser_test.add_argument("--initial_vectors_R", default="nothing", type=str)
    parser_test.add_argument("--model_struct", default="recipemind", type=str)
    parser_test.add_argument("--model_analysis", default=False, action="store_true")
    parser_test.add_argument("--hidden_dim", default=1024, type=int)
    parser_test.add_argument("--dropout_rate", default=0.0, type=float)
    parser_test.add_argument("--learning_rate", default=0.001, type=float)
    parser_test.add_argument("--weight_decay", default=0.0, type=float)
    parser_test.add_argument("--batch_size", default=500, type=int)
    parser_test.add_argument("--downstream_task", "-dt", default="food_pairing", type=str)
    parser_test.set_defaults(func=cmd_test)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
