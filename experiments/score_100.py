import sys
import os

import pandas as pd
import numpy as np
import wandb
from cache.autoencoder import get_dataloader

from petfinder_pawpularity.config import create_config, to_flat_dict
from petfinder_pawpularity.run import (
    train_folds,
    eval_model,
    init_default_model,
)
from petfinder_pawpularity.util import add_path_to_data

import random
import torchvision.transforms.functional as F
from torchvision.utils import save_image

from sklearn.model_selection import train_test_split
from PIL import Image

import pytorch_lightning as pl

import torch
import torch.nn as nn


class PetfinderPawpularityCombined(pl.LightningModule):
    def __init__(self, classifier, regressor):
        super().__init__()
        self.classifier = classifier
        self.regressor = regressor

        def rmse(outs, targets):
            scores, classes = outs
            scores[classes > 0.5] = 1
            preds = scores * 100
            targets = targets * 100
            result = torch.sqrt(torch.mean((targets - preds) ** 2)).item()
            return result

        def positive(outs, targets):
            _, classes = outs
            return torch.sum((classes > 0.5))

        def true_positive(outs, targets):
            _, classes = outs
            return torch.sum((classes > 0.5) * (targets == 1))

        def false_positive(outs, targets):
            _, classes = outs
            return torch.sum((classes > 0.5) * (targets != 1))

        def precision(outs, targets):
            _, classes = outs
            fp = torch.sum((classes > 0.5) * (targets != 1))
            tp = torch.sum((classes > 0.5) * (targets == 1))
            return tp / (fp + tp + 1e-10)

        self.metrics = [
            positive,
            true_positive,
            false_positive,
            precision,
            rmse,
        ]

    def forward(self, x):
        classes = torch.sigmoid(self.classifier(x))
        scores = torch.sigmoid(self.regressor(x))
        return scores, classes

    def step(self, batch, batch_idx, stage):
        x, y = batch
        target = y.float().unsqueeze(1)
        scores, classes = self(x)
        return dict(loss=0, scores=scores, classes=classes, targets=target)

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def epoch_end(self, outs, stage):
        scores = torch.cat([o["scores"] for o in outs])
        classes = torch.cat([o["classes"] for o in outs])
        targets = torch.cat([o["targets"] for o in outs])
        print("classes", classes)
        for metric in self.metrics:
            self.log(
                f"{stage}_{metric.__name__}",
                metric((scores, classes), targets),
            )

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")


def get_transforms():
    Non = []
    Hflip = [dict(name="hflip", params=[])]

    augmented_transforms = dict(
        Rotate=dict(name="affine", params=[-15, (0, 0), 1, 0]),
        RotateCC=dict(name="affine", params=[15, (0, 0), 1, 0]),
        TranslateY=dict(name="affine", params=[0, (0, 0.1), 1, 0]),
        TranslateX=dict(name="affine", params=[0, (0.1, 0), 1, 0]),
        ScaleLarge=dict(name="affine", params=[0, (0, 0), 1.1, 0]),
        ScaleSmall=dict(name="affine", params=[0, (0, 0), 0.9, 0]),
        BrightnessPlus=dict(name="adjust_brightness", params=[1.1]),
        BrightnessMinus=dict(name="adjust_brightness", params=[0.9]),
        ContrastPlus=dict(name="adjust_contrast", params=[1.1]),
        ContrastMinus=dict(name="adjust_contrast", params=[0.9]),
        SharpnessPlus=dict(name="adjust_sharpness", params=[1.1]),
        SharpnessMinus=dict(name="adjust_sharpness", params=[0.9]),
        HuePlus=dict(name="adjust_hue", params=[0.1]),
        HueMinus=dict(name="adjust_hue", params=[-0.1]),
        SaturationPlus=dict(name="adjust_saturation", params=[1.1]),
        SaturationMinus=dict(name="adjust_saturation", params=[0.9]),
    )

    non_augmented_transforms = {
        f"Non{key}": Non + [val] for key, val in augmented_transforms.items()
    }

    hflip_augmented_transforms = {
        f"Hflip{key}": Hflip + [val]
        for key, val in augmented_transforms.items()
    }

    return {
        **(dict(Non=Non, Hflip=Hflip)),
        **non_augmented_transforms,
        **hflip_augmented_transforms,
    }


def get_more_100(data):
    data_more_100 = pd.DataFrame(
        columns=["Id", "Pawpularity", "Path", "Transform"]
    )
    if not os.path.exists("./output_images"):
        os.mkdir("./output_images")

    for t_name, t in get_transforms().items():
        for idx in range(len(data)):
            transformed = Image.open(data["Path"].iloc[idx])
            if len(t) > 0:
                transformed = getattr(F, t[0]["name"])(
                    transformed, *t[0]["params"]
                )
            if len(t) == 2:
                transformed = getattr(F, t[1]["name"])(
                    transformed, *t[1]["params"]
                )

            image_id = "%032x" % (random.getrandbits(128))
            image_pawpularity = data["Pawpularity"].iloc[idx]
            image_path = f"./output_images/{image_id}.jpg"
            image_transform = t_name

            save_image(
                F.to_tensor(F.resize(transformed, (224, 224))), image_path
            )
            data_more_100.loc[len(data_more_100)] = [
                image_id,
                image_pawpularity,
                image_path,
                image_transform,
            ]
    data_more_100.to_csv("more_100.csv", index=False)
    return data_more_100


def get_new_data(data):
    data = data.copy()
    data_classifier = pd.DataFrame(columns=["Id", "Pawpularity", "Path"])

    # less than 100 (downsample)
    data_lt_100 = data[data["Pawpularity"] < 100]
    data_lt_100.loc[:, "Pawpularity"] = 0
    data_lt_100 = data_lt_100.iloc[: int(len(data_lt_100) * 0.25)]
    data_classifier = data_classifier.append(data_lt_100, ignore_index=True)

    # more than 100 (upsample)
    data_more_100 = get_more_100(data[data["Pawpularity"] == 100])
    data_classifier = data_classifier.append(
        data_more_100[["Id", "Pawpularity", "Path"]], ignore_index=True
    )

    return data_classifier.sample(frac=1).reset_index(drop=True)


def run_100_combined(wandb_params, data, val_data):
    # Train Classifier
    config_classifier = create_config(
        dict(
            n_epochs=3,
            model_name="swin_tiny_patch4_window7_224",
            model_type="classifier",
        )
    )
    run_classifier = wandb.init(
        config=config_classifier, reinit=True, **wandb_params
    )

    data_classifier = get_new_data(data[["Id", "Pawpularity", "Path"]])

    def init_model(config):
        return init_default_model(
            config,
            metrics=[
                "true_positive",
                "positive",
                "false_positive",
                "precision",
            ],
        )

    model_classifier = train_folds(
        config_classifier, data_classifier, init_model=init_model
    )
    run_classifier.finish()
    val_data_classifier = val_data[["Id", "Pawpularity", "Path"]].copy()
    val_data_classifier.loc[
        val_data_classifier["Pawpularity"] < 100, "Pawpularity"
    ] = 0
    eval_model(config_classifier, model_classifier, val_data_classifier)

    # Train Regressor
    config_regressor = create_config(
        dict(model_name="swin_tiny_patch4_window7_224")
    )
    data_regressor = data[["Id", "Pawpularity", "Path"]]
    run_regressor = wandb.init(
        config=config_regressor, reinit=True, **wandb_params
    )
    model_regressor = train_folds(config_regressor, data_regressor)
    run_regressor.finish()

    # Eval Combind Model
    model_combined = PetfinderPawpularityCombined(
        classifier=model_classifier, regressor=model_regressor
    )
    config = create_config(dict(model_name="combined"))
    eval_model(config, model_combined, val_data)


def run_100(wandb_params, data, val_data):
    config = create_config(dict(model_name="swin_tiny_patch4_window7_224"))
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)

    data = data[["Id", "Pawpularity", "Path"]]
    data_more_100 = get_more_100(data[data["Pawpularity"] == 100])
    newdata = pd.DataFrame(columns=["Id", "Pawpularity", "Path"])
    newdata = newdata.append(
        data[data["Pawpularity"] < 100], ignore_index=True
    )
    newdata = newdata.append(
        data_more_100[["Id", "Pawpularity", "Path"]], ignore_index=True
    )

    newdata = newdata.sample(frac=1).reset_index(drop=True)
    model = train_folds(config, newdata)
    run.finish()
    eval_model(config, model, val_data)


def run_normal(wandb_params, data, val_data):
    config = create_config(dict(model_name="swin_tiny_patch4_window7_224"))
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    data = data[["Id", "Pawpularity", "Path"]]
    model = train_folds(config, data)
    run.finish()
    eval_model(config, model, val_data)


def run(wandb_entity, wandb_project, wandb_mode=None):
    wandb_params = dict(
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
    )
    data_path = "../input/petfinder-pawpularity-score/train"
    data = add_path_to_data(pd.read_csv(f"{data_path}.csv"), data_path)

    # data_index = np.arange(0, len(data))
    train_idx, val_idx = train_test_split(
        np.arange(0, len(data)),
        test_size=0.2,
        random_state=3,
    )
    train_data, val_data = data.iloc[train_idx], data.iloc[val_idx]
    run_normal(wandb_params, train_data, val_data)
    run_100(wandb_params, train_data, val_data)
    run_100_combined(wandb_params, train_data, val_data)


if __name__ == "__main__":
    wandb_entity = sys.argv[1] if len(sys.argv) > 1 else None
    wandb_project = sys.argv[2] if len(sys.argv) > 2 else None
    run(wandb_entity, wandb_project)
