# Score 100

## Status

DONE

## Motivation

The number of samples with a score of 100 is high and does not conform to a normal distribution.

## Desciption

experiments

- normal vs upsample 100
- normal vs (pre classify 100 images + normal model)

## Result

### data with more 100

- normal [eval 18.21](https://wandb.ai/wangyashuu/uncategorized/runs/2lk7he5q), [train](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/262tfa46).
- with more 100 [eval 18.45](https://wandb.ai/wangyashuu/uncategorized/runs/txpjd9t6), [train](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/qsi22duh)

### classifier, unsample =100, downsample <100

- normal [eval, 18.57055](https://wandb.ai/wangyashuu/uncategorized/runs/3eqe087c), [train](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/ma9zxc54)
- with classifier [eval, 18.66165](https://wandb.ai/wangyashuu/uncategorized/runs/36qo91iy), [train classifier](https://wandb.ai/wangyashuu/uncategorized/runs/5539vwk3), [train regressor](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/17d4im3k)

### classifier, less epochs for classifier & less data

- normal [eval, 18.57055](https://wandb.ai/wangyashuu/uncategorized/runs/5k97fmos), [train](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2brzf1wb)
- with classifier [eval, 18.79646](https://wandb.ai/wangyashuu/uncategorized/runs/26iu5uiw), [train classifier](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3obha5ov), [train](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3vdzmfwc)

## Conclusion
