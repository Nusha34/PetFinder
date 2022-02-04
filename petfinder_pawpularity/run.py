from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from torchvision.utils import save_image

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
import wandb

from petfinder_pawpularity.model import PetfinderPawpularityModel
from petfinder_pawpularity.transform import get_dataloader, get_transform


data_path = "../input/petfinder-pawpularity-score/train"


def init_default_model(
    config,
    metrics=[],
    modify_model=None,
    modify_loss_function=None,
    create_final_layer=None,
    **kwargs,
):
    if config.model_mode == "reuse_weights":
        arg_modify_model = modify_model

        def modify_model(model):
            for param in model.parameters():
                param.requires_grad = False
            if arg_modify_model is not None:
                arg_modify_model(model)
            return model

    model = PetfinderPawpularityModel(
        config.model_name,
        criterion=config.criterion,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        metrics=metrics,
        modify_model=modify_model,
        modify_loss_function=modify_loss_function,
        create_final_layer=create_final_layer,
        **kwargs,
    )
    return model


def train(
    config,
    model,
    train_dataframe,
    val_dataframe,
    return_best=True,
):
    print("train one fold", len(train_dataframe), len(val_dataframe))
    train_dataloader = get_dataloader(
        train_dataframe,
        get_transform(config.transform, config.train.transforms),
        batch_size=config.train.batch_size,
        shuffle=True,
    )
    val_dataloader = get_dataloader(
        val_dataframe,
        get_transform(config.transform, config.test.transforms),
        batch_size=config.test.batch_size,
    )

    model_checkpoint = ModelCheckpoint(
        filename="best_loss",
        monitor="val_rmse",
        save_last=False,
    )
    callbacks = [model_checkpoint]

    if config.early_stopping:
        callbacks.append(EarlyStopping(monitor="val_rmse"))

    trainer = Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        logger=WandbLogger(),
        max_epochs=config.n_epochs,
        callbacks=callbacks,
        precision=16,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    ckpt_path = model_checkpoint.best_model_path
    wandb.save(ckpt_path)
    if return_best:
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
    return model


def eval_model(config, model, dataframe):
    val_dataloader = get_dataloader(
        dataframe,
        get_transform(config.transform, config.test.transforms),
        batch_size=config.test.batch_size,
    )

    trainer = Trainer(
        gpus=(1 if torch.cuda.is_available() else 0),
        logger=WandbLogger(),
        callbacks=[],
    )
    trainer.validate(model, dataloaders=val_dataloader)
    # return loss


def train_folds(
    config,
    data,
    init_model=init_default_model,
    folds=None,
    wandb_params=None,
    **kwargs,
):
    use_size = int(len(data) * config.data_rate)
    if config.data_rate != 1:
        data = data.iloc[:use_size]
    data_index = np.arange(0, len(data))
    if config.n_folds == 0:
        seed_everything(config.seed)
        train_idx, test_idx = train_test_split(
            data_index,
            test_size=1 - config.train.rate,
            random_state=config.seed,
        )
        model = init_model(config, **kwargs)
        if wandb_params is not None:
            run = wandb.init(config=config, reinit=True, **wandb_params)
            model = train(
                config,
                model,
                data.iloc[train_idx],
                data.iloc[test_idx],
            )
            run.finish()
        else:
            model = train(
                config,
                model,
                data.iloc[train_idx],
                data.iloc[test_idx],
            )
        return model
    else:
        kf = KFold(n_splits=config.n_folds)
        models = []
        for i, (train_idx, test_idx) in enumerate(kf.split(data_index)):
            torch.cuda.empty_cache()
            if folds is not None and i not in folds:
                continue
            seed_everything(config.seed + i)
            model = init_model(config, **kwargs)
            if wandb_params is not None:
                run = wandb.init(config=config, reinit=True, **wandb_params)
                model = train(
                    config, model, data.iloc[train_idx], data.iloc[test_idx]
                )
                run.finish()
            else:
                model = train(
                    config, model, data.iloc[train_idx], data.iloc[test_idx]
                )
            models.append(model)
        return models
