import sys

import pandas as pd
import wandb

from petfinder_pawpularity.config import create_config, to_flat_dict
from petfinder_pawpularity.run import train_folds


def run_weight_decay(wandb_params, data, weight_decay=1e-1):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            optimizer=dict(params=dict(weight_decay=weight_decay)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run(wandb_entity, wandb_project, wandb_mode=None):
    data_path = "../input/petfinder-pawpularity-score/train"
    data = pd.read_csv(f"{data_path}.csv")
    wandb_params = dict(
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
    )
    for weight_decay in [1e-4]:
        run_weight_decay(wandb_params, data, weight_decay)


if __name__ == "__main__":
    wandb_entity = sys.argv[1] if len(sys.argv) > 1 else None
    wandb_project = sys.argv[2] if len(sys.argv) > 2 else None
    run(wandb_entity, wandb_project)
