import argparse
import os

ROOT_PATH = "./data/"
OUT_PATH = "./saved/"
CSV_PATH = "./results/"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HerbMind configuration")
    parser.add_argument("--data_dir", default=ROOT_PATH, help="Path to the prescription CSV directory.")
    parser.add_argument("--cases", default="cases.yaml", help="YAML file defining case-study seeds.")
    parser.add_argument("--plot_dir", default="./plots/", help="Directory for analysis outputs.")
    parser.add_argument("--model_path", default="./model_best.pth", help="Checkpoint path for the trained model.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size used during training.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Optimiser learning rate.")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="L2 weight decay.")
    parser.add_argument("--grad_update", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--loss_function", default="mse", help="Training loss identifier.")
    parser.add_argument("--model_struct", default="herbmind_isab_pma_cat", help="Model architecture spec string.")
    parser.add_argument("--project_name", default="HerbMind", help="Project identifier for experiment tracking.")
    parser.add_argument("--session_name", default="default_session", help="Experiment session label.")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--mc_dropout", action="store_true", help="Enable Monte-Carlo dropout during inference.")
    parser.add_argument("--model_analysis", action="store_true", help="Store attention maps for later analysis.")
    parser.add_argument("--device", default="cpu", help="Torch device spec (cpu or cuda).")
    parser.add_argument("--cases_limit", type=int, default=None, help="Optional limit on number of cases to process.")
    return parser


parser = build_parser()


def ensure_directories() -> None:
    os.makedirs(ROOT_PATH, exist_ok=True)
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(CSV_PATH, exist_ok=True)
