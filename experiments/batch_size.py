import sys

import pandas as pd
import wandb

from petfinder_pawpularity.config import create_config, to_flat_dict
from petfinder_pawpularity.run import train_folds
from petfinder_pawpularity.util import add_path_to_data


def run_batch_size(wandb_params, data, batch_size=8):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            train=dict(batch_size=batch_size),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run(wandb_entity, wandb_project, wandb_mode=None):
    data_path = "../input/petfinder-pawpularity-score/train"
    data = add_path_to_data(pd.read_csv(f"{data_path}.csv"), data_path)

    wandb_params = dict(
        group=f"experiment-batch_size",
        job_type="train",
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
    )

    for batch_size in [4, 8, 16, 32]:
        run_batch_size(wandb_params, data, batch_size)


if __name__ == "__main__":
    wandb_entity = sys.argv[1] if len(sys.argv) > 1 else None
    wandb_project = sys.argv[2] if len(sys.argv) > 2 else None
    run(wandb_entity, wandb_project)
