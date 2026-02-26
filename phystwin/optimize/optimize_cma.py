# The first stage to optimize the sparse parameters using CMA-ES (Newton port)
from __future__ import annotations

import logging
import sys
from pathlib import Path
from argparse import ArgumentParser

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from phystwin.optimize.cma_optimize_warp import OptimizerCMA, load_yaml_config


def set_all_seeds(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except Exception:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--train_frame", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--config", type=str, default="phystwin/config/cloth.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--monitor", action="store_true")
    parser.add_argument("--monitor-frames", type=int, default=0)
    parser.add_argument("--monitor-every", type=int, default=1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    seed = 42
    set_all_seeds(seed)

    data_path = f"{args.base_path}/{args.case_name}/final_data.pkl"
    phystwin_root = Path(__file__).resolve().parents[1]
    base_dir = str(phystwin_root / "experiments_optimization" / args.case_name)

    config = load_yaml_config(args.config)
    monitor = args.monitor or bool(config.get("monitor", False))
    monitor_frames = args.monitor_frames
    if monitor_frames == 0 and "monitor_frames" in config:
        monitor_frames = int(config["monitor_frames"])
    monitor_every = args.monitor_every
    if monitor_every == 1 and "monitor_every" in config:
        monitor_every = int(config["monitor_every"])

    optimizer = OptimizerCMA(
        data_path=data_path,
        base_dir=base_dir,
        train_frame=args.train_frame,
        config=config,
        device=args.device,
        monitor=monitor,
        monitor_frames=monitor_frames,
        monitor_every=monitor_every,
    )
    optimizer.optimize(max_iter=args.max_iter)


if __name__ == "__main__":
    main()
