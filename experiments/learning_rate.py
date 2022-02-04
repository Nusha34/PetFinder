import sys

import pandas as pd
import wandb
import torch

from petfinder_pawpularity.config import create_config, to_flat_dict
from petfinder_pawpularity.run import train_folds
from petfinder_pawpularity.util import add_path_to_data


def run_one(
    wandb_params,
    data,
    learning_rate=1e-5,
    batch_size=32,
    T_0=16,
    eta_min_divide=10,
    image_size=384,
    model_name="swin_large_patch4_window12_384",
):
    config = create_config(
        dict(
            train=dict(batch_size=batch_size),
            transform=dict(image_size=image_size),
            model_name=model_name,
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(
                params=dict(T_0=T_0, eta_min=learning_rate / eta_min_divide)
            ),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate1(wandb_params, data, learning_rate=1e-5):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=20, eta_min=learning_rate / 100)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate3(wandb_params, data, learning_rate=3e-6):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=10, eta_min=learning_rate / 10)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate3_restart(wandb_params, data, learning_rate=3e-5):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=5, eta_min=learning_rate / 10)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate5(wandb_params, data, learning_rate=5e-5):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=10, eta_min=learning_rate / 10)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate5_large_batch(wandb_params, data, learning_rate=5e-5):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            train=dict(batch_size=32),
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=10, eta_min=learning_rate / 10)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate5_slow_large_batch(
    wandb_params, data, learning_rate=5e-5
):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            train=dict(batch_size=32),
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=10, eta_min=learning_rate / 100)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate3_fast_large_batch(
    wandb_params, data, learning_rate=5e-5
):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            train=dict(batch_size=32),
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=10, eta_min=learning_rate / 10)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate3_slow_large_batch(
    wandb_params, data, learning_rate=5e-5
):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            train=dict(batch_size=32),
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=10, eta_min=learning_rate / 100)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate1_slow(wandb_params, data, learning_rate=1e-5):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=10, eta_min=learning_rate / 1000)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_learning_rate1_large_batch(wandb_params, data, learning_rate=1e-5):
    config = create_config(
        dict(
            model_name="swin_tiny_patch4_window7_224",
            train=dict(batch_size=32),
            optimizer=dict(params=dict(lr=learning_rate)),
            scheduler=dict(params=dict(T_0=10, eta_min=learning_rate / 1000)),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run(wandb_entity, wandb_project, wandb_mode=None):
    data_path = "../input/petfinder-pawpularity-score/train"
    data = add_path_to_data(pd.read_csv(f"{data_path}.csv"), data_path)

    wandb_params = dict(
        group=f"experiment-large_model_lr",
        job_type="train",
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
    )

    for learning_rate in [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]:
        run_one(
            wandb_params,
            data,
            learning_rate=learning_rate,
            batch_size=32,
            image_size=224,
            model_name="swin_large_patch4_window7_224",
        )


if __name__ == "__main__":
    wandb_entity = sys.argv[1] if len(sys.argv) > 1 else None
    wandb_project = sys.argv[2] if len(sys.argv) > 2 else None
    run(wandb_entity, wandb_project)
