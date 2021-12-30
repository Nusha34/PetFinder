import sys

import pandas as pd
import wandb
import torch

from petfinder_pawpularity.config import create_config, to_flat_dict
from petfinder_pawpularity.run import train_folds, init_default_model
from petfinder_pawpularity.util import add_path_to_data


def run_normal_model(wandb_params, model_name, data):
    config = create_config(dict(model_name=model_name))
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_inception_resnet_v2(wandb_params, data):
    # GoogLeNet
    config = create_config(
        dict(
            model_name="inception_resnet_v2",
            transform=dict(image_size=299),
        )
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data)
    run.finish()


def run_inception_v3(wandb_params, data):
    def init_model(config):
        def modify_model_with_aux(model):
            num_aux_input_features = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = torch.nn.Linear(num_aux_input_features, 1)
            return model

        def modify_criterion_with_aux(criterion):
            def loss_function(outs, y):
                outputs, aux_outputs = outs
                loss = criterion(outputs, y)
                if aux_outputs is None:
                    return loss
                return loss + 0.4 * criterion(aux_outputs, y)

            return loss_function

        return init_default_model(
            config,
            modify_model=modify_model_with_aux,
            modify_loss_function=modify_criterion_with_aux,
            aux_logits=True,
        )

    config = create_config(
        dict(model_name="inception_v2", transform=dict(image_size=299))
    )
    run = wandb.init(config=to_flat_dict(config), reinit=True, **wandb_params)
    train_folds(config, data, init_model=init_model)
    run.finish()


def run(wandb_entity=None, wandb_project=None, wandb_mode=None):
    data_path = "../input/petfinder-pawpularity-score/train"
    data = add_path_to_data(pd.read_csv(f"{data_path}.csv"), data_path)

    wandb_params = dict(
        entity=wandb_entity,
        project=wandb_project,
        mode=wandb_mode,
    )

    for model_name in [
        "resnet18",
        "resnet34",
        "resnet50",
        "efficientnet_b2",
        "efficientnet_b4",
        "vit_tiny_patch16_224",
        "vit_base_patch16_224",
    ]:
        run_normal_model(wandb_params, model_name, data)
    run_inception_resnet_v2(wandb_params, data)
    run_inception_v3(wandb_params, data)


if __name__ == "__main__":
    wandb_entity = sys.argv[1] if len(sys.argv) > 1 else None
    wandb_project = sys.argv[2] if len(sys.argv) > 2 else None
    run(wandb_entity, wandb_project)
