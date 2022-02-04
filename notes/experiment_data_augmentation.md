# Data Augmentation

## Status

DONE

## Motivation

hyperparameter tuning

## Result

- [RandomPosterize, 18.1134 = (18.47+18.357+17.803+17.908+18.029)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2qftuhpc)
- [RandomAdjustSharpness, 18.1044 = (18.434+18.375+17.815+17.878+18.02)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2rgym9on)
- [ColorJitter, 18.0984 = (17.885+18.361+18.424+17.824+17.998)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/groups/experiment-ColorJitter/workspace?workspace=user-wangyashuu)
- [RandomAffine, 17.978 = (18.345+17.869+17.677+17.741+18.257)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/groups/experiment-RandomAffine/workspace)
- [RandomAutocontrast, 18.1118 = (17.888+18.473+18.353+17.847+17.998)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/groups/experiment-RandomAutocontrast/workspace)

- [None, 18.0946 = (17.892+18.419+18.375+17.994+17.793)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/groups/experiment-None/workspace)

- [RandomPosterize-RandomAdjustSharpness-RandomAutocontrast 18.104 = (17.994+17.881+17.835+18.36+18.45)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/groups/experiment-RandomPosterize-RandomAdjustSharpness-RandomAutocontrast)

- [RandomAdjustSharpness-RandomAutocontrast 18.1056 = (18.441+18.349+17.852+17.902+17.984)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/groups/experiment-RandomAdjustSharpness-RandomAutocontrast/workspace)

## Conclusion

no big different [horizontal_flip + ShiftScaleRotate] vs [horizontal_flip + RandomResizedCrop]
