import pytorch_lightning as pl
from timm import create_model
import torch
import torch.nn as nn


class PetfinderPawpularityModel(pl.LightningModule):
    def __init__(
        self,
        name,
        criterion,
        optimizer,
        scheduler,
        pretrained=True,
        modify_model=None,
        modify_loss_function=None,
        create_final_layer=None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # create model
        backbone = create_model(
            name, pretrained=(pretrained == True), **kwargs
        )
        if isinstance(pretrained, str):
            backbone.load_state_dict(torch.load(pretrained))
        if modify_model is not None:
            backbone = modify_model(backbone)

        fc_name, fc = list(backbone.named_children())[-1]
        in_features = (
            fc.in_features
            if hasattr(fc, "in_features")
            else [
                c.in_features
                for c in fc.children()
                if hasattr(c, "in_features")
            ][0]
        )

        final_layer = (
            create_final_layer(in_features)
            if create_final_layer is not None
            else nn.Sequential(nn.Dropout(0.7), nn.Linear(in_features, 1))
        )

        setattr(backbone, fc_name, final_layer)
        self.backbone = backbone

        # config for trainning
        def rmse(preds, targets):
            if criterion == "BCEWithLogitsLoss":
                preds = torch.sigmoid(preds)
            else:
                preds[preds > 1] = 1
            preds = preds * 100
            targets = targets * 100
            result = torch.sqrt(torch.mean((targets - preds) ** 2)).item()
            return result

        loss_function = getattr(torch.nn, criterion)()
        if modify_loss_function is not None:
            loss_function = modify_loss_function(loss_function)

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.metrics = [rmse]

    def forward(self, x):
        x = self.backbone(x)
        return x

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad == True]
        optimizer_func = getattr(torch.optim, self.optimizer.name)
        optimizer = optimizer_func(params, **self.optimizer.params)
        scheduler_func = getattr(torch.optim.lr_scheduler, self.scheduler.name)
        scheduler = scheduler_func(optimizer, **self.scheduler.params)
        return [optimizer], [scheduler]

    def step(self, batch, batch_idx, stage):
        x, y = batch
        target = y.float().unsqueeze(1)
        outs = self(x)
        loss = self.loss_function(outs, target)
        preds = (outs[0] if type(outs) is tuple else outs).detach()
        return {"loss": loss, "preds": preds, "targets": target}

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def epoch_end(self, outs, stage):
        preds = torch.cat([o["preds"] for o in outs])
        targets = torch.cat([o["targets"] for o in outs])
        loss = sum(
            [(o["loss"] * o["preds"].size(0)).item() for o in outs]
        ) / preds.size(0)
        self.log(f"{stage}_loss", loss)

        for metric in self.metrics:
            self.log(f"{stage}_{metric.__name__}", metric(preds, targets))

    def training_epoch_end(self, outputs):
        self.epoch_end(outputs, "train")

    def validation_epoch_end(self, outputs):
        self.epoch_end(outputs, "val")

    def test_epoch_end(self, outputs):
        self.epoch_end(outputs, "test")
