import sys

import pandas as pd
import wandb

from petfinder_pawpularity.config import create_config, to_flat_dict
from petfinder_pawpularity.run import train_folds
from petfinder_pawpularity.util import add_path_to_data


def get_transforms(
    degrees=10,
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
    shear=0,
    brightness=0.1,
    contrast=0,
    saturation=0.1,
    hue=0,
    includes=None,
):
    augmented_transforms = [
        dict(name="RandomHorizontalFlip"),
        dict(name="RandomPosterize", params=dict(bits=7)),
        dict(name="RandomAdjustSharpness", params=dict(sharpness_factor=2)),
        dict(name="RandomAutocontrast"),
        dict(
            name="RandomAffine",
            params=dict(
                degrees=degrees, translate=translate, scale=scale, shear=shear
            ),
        ),
        dict(
            name="ColorJitter",
            params=dict(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
        ),
    ]
    augmented_transforms = [
        t for t in augmented_transforms if t["name"] in includes
    ]
    resize = dict(name="Resize")
    augmented_transforms.append(resize)
    normal_transforms = [resize]
    return augmented_transforms, normal_transforms


def run_folds(wandb_params, data, transformers, experiment_name, folds=None):
    augmented_transforms, normal_transforms = transformers
    config = create_config(
        dict(
            n_folds=10,
            n_epochs=5,
            transform=dict(image_size=384),
            model_name="swin_large_patch4_window12_384",
            train=dict(transforms=augmented_transforms),
            test=dict(transforms=normal_transforms),
            # optimizer=dict(params=dict(weight_decay=weight_decay)),
        )
    )

    train_folds(
        config,
        data,
        wandb_params={
            **dict(group=f"experiment-{experiment_name}", job_type="train"),
            **wandb_params,
        },
        folds=folds,
    )


def run(wandb_entity, wandb_project, wandb_mode=None):
    data_path = "../input/petfinder-pawpularity-score/train"
    data = add_path_to_data(pd.read_csv(f"{data_path}.csv"), data_path)

    wandb_params = dict(
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
    )

    run_folds(
        wandb_params,
        data,
        get_transforms(includes=["RandomHorizontalFlip", "RandomAffine"]),
        experiment_name="folds",
    )


if __name__ == "__main__":
    wandb_entity = sys.argv[1] if len(sys.argv) > 1 else None
    wandb_project = sys.argv[2] if len(sys.argv) > 2 else None
    run(wandb_entity, wandb_project)
