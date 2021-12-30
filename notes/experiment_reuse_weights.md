# Reuse Weights

## Status

DONE

## Desciption

experiments on different models, reuse the pretrained weight, only modify last layer

## Result

- [resnet18, 21.626](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/7yyq7n9o)
- [resnet34, 21.530](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1be7hnni)
- [resnet50, 19.866](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3s1imhj9)
- [inception v3, 18.493](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/kaakgyjd)
- [inception_resnet v2, 19.718](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/25xhs438)
- [efficientb2, 20.494](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3dsgeoko)
- [efficientb4, 20.773](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/vfgzv71f)
- [swin, 20.52](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1z2xb7z9)
- [vit base, 32.578](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2bz5te8u)
- [vit tiny, 28.852](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2n1imp78)

# Conclusion:

- all models perform bad in Reuse Weights. The reason might be that to less parameters to memory the pattern between image and pawpularity score.

- run n_folds for inception v3
