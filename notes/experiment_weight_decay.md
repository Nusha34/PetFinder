# Weight Decay

## Status

DONE

## Motivation

try to handle overfitting, weight decay can be add easily

## Desciption

experiments on with weight decay

## Result

### Swin

- [no weight decay, 18.021](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/36t2rnlj)
- [weight decay=1e-5, 18.019](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/22ctqjvk)
- [weight decay=1e-4, 18.019](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/18d0zt53)
- [weight decay=1e-3, 18.019](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/v56pmfw2)
- [weight decay=1e-2, 18.021](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2e8hvmcb)
- [weight decay=1e-1, 18.025](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/ioj70h26)

### Inception

- [no weight decay, 18.455](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3f9uu1e7)
- [weight decay=1e-5, 18.481](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/16env8g5)
- [weight decay=1e-4, 18.473](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3txfrevo)
- [weight decay=1e-3, 18.493](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/9kjxynya)
- [weight decay=1e-2, 18.508](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1e70zft2)
- [weight decay=1e-1, 18.457](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/34959mjx)
- [weight decay=1, 18.487](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2aj0ssi0)
- [weight decay=10, 18.369](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2ozu3z7h)
- [weight decay=100, 19.244](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2cz1xtx8)

### Inception with batch

- [0, 18.5548 = (18.696+18.695+18.529+18.331+18.523)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/16evehsu)
- [0.1, 18.565 = (18.716+18.708+18.535+18.33+18.536)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1l3zz89i)
- [0.3, 18.6674 = (18.898+18.909+18.629+18.321+18.58)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3a4y8pf0)
- [1, 18.679 = (18.923+18.913+18.639+18.328+18.592)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2erx6bln)
- [3, 18.6686 = (18.875+18.913+18.672+18.294+18.589)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/ieq8gvsf)
- [10, 18.5434 = (18.665+18.685+18.515+18.366+18.486)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/jucr76hx)
- [30, 18.6762 = (18.782+18.912+18.686+18.362+18.639)/5](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/t0tvwazz)

## Conclusion

It does not seem to help. weight decay can be

- swin 1e-4
- inception 10
