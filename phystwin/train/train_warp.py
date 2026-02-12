from __future__ import annotations

import logging
import os
import pickle
import random
import sys
from pathlib import Path
from argparse import ArgumentParser

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from phystwin.train.trainer_warp import InvPhyTrainerWarp, load_yaml_config


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--config", type=str, default="phystwin/config/cloth.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-iter", type=int, default=0)
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--monitor-frames", type=int, default=0)
    parser.add_argument("--monitor-every", type=int, default=1)
    parser.add_argument("--optimal-path", type=str, default="")
    parser.add_argument("--no-optimal", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    set_all_seeds(42)

    data_path = f"{args.base_path}/{args.case_name}/final_data.pkl"
    phystwin_root = Path(__file__).resolve().parents[1]
    base_dir = str(phystwin_root / "experiments" / args.case_name)

    config = load_yaml_config(args.config)

    initial_params = None
    default_opt_path = phystwin_root / "experiments_optimization" / args.case_name / "optimal_params.pkl"
    optimal_path = args.optimal_path or str(default_opt_path)
    if not args.no_optimal and os.path.exists(optimal_path):
        with open(optimal_path, "rb") as f:
            initial_params = pickle.load(f)
        if isinstance(initial_params, dict):
            config.update(initial_params)

    monitor = args.monitor or bool(config.get("monitor", False))
    monitor_frames = args.monitor_frames
    if monitor_frames == 0 and "monitor_frames" in config:
        monitor_frames = int(config["monitor_frames"])
    monitor_every = args.monitor_every
    if monitor_every == 1 and "monitor_every" in config:
        monitor_every = int(config["monitor_every"])

    trainer = InvPhyTrainerWarp(
        data_path=data_path,
        base_dir=base_dir,
        train_frame=args.train_frame,
        config=config,
        device=args.device,
        monitor=monitor,
        monitor_frames=monitor_frames,
        monitor_every=monitor_every,
        initial_params=initial_params,
    )

    max_iter = args.max_iter if args.max_iter > 0 else None
    trainer.train(max_iter=max_iter)


if __name__ == "__main__":
    main()
