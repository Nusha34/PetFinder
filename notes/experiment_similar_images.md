# Similar Images

## Motivation

similar images but diffierent pawpularity might confuse our model. consider similar images as low quality source, remove them when training but not during eval.

## Desciption

experiments on different source of similar images

- [average hash](https://www.kaggle.com/annachechulina/data-cleaning)
- [combined hash](https://www.kaggle.com/schulta/petfinder-identify-duplicates-and-share-findings)
- [cnn + cos_similarity](https://www.kaggle.com/burakbekci/petfinder-finding-duplicates-with-cnn)
- [mobilenet](https://www.kaggle.com/showeed/annoy-similar-images-edit2)

## Result

### all

Train

- [18.431](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1p50nyuj)
- [18.615](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/ftik3hka)
- [18.56](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2p0eaw4y)
- [17.596](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/31w8kkjn)
- [17.579](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1lm7bto5)

Val 18.252 = (18.421+18.271+18.271+18.181+18.12)/5

- [18.421](https://wandb.ai/wangyashuu/uncategorized/runs/13mt00vy)
- [18.271](https://wandb.ai/wangyashuu/uncategorized/runs/h2feq0hp)
- [18.12](https://wandb.ai/wangyashuu/uncategorized/runs/3ol4h5o0)
- [18.271](https://wandb.ai/wangyashuu/uncategorized/runs/22r78hez)
- [18.181](https://wandb.ai/wangyashuu/uncategorized/runs/1cicw26w)

### average hash

Train

- [18.326](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1n53onxu)
- [18.494](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1h1rcjzj)
- [18.544](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2jm0s3w3)
- [17.554](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/13z4enm1)
- [17.441](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1cnx9wlc)

Val 18.254 = (18.266+18.24+18.087+18.419+18.26)/5

- [18.266](https://wandb.ai/wangyashuu/uncategorized/runs/3und4bog)
- [18.24](https://wandb.ai/wangyashuu/uncategorized/runs/23r5x6p7)
- [18.087](https://wandb.ai/wangyashuu/uncategorized/runs/369cl7d0)
- [18.419](https://wandb.ai/wangyashuu/uncategorized/runs/3i8czu9h)
- [18.26](https://wandb.ai/wangyashuu/uncategorized/runs/1frf13kk)

### combined hash

- [18.411](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/30jre1ly)
- [18.555](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/o41qh0xp)
- [18.424](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2yrb9jkx)
- [17.605](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/tel7sq23)
- [17.53](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3cic522m)

Val: 18.234 = (18.275+18.277+18.074+18.27+18.277)/5

- [18.275](https://wandb.ai/wangyashuu/uncategorized/runs/18eu1huj)
- [18.424](https://wandb.ai/wangyashuu/uncategorized/runs/2rq8saj4)
- [18.074](https://wandb.ai/wangyashuu/uncategorized/runs/97pbpgw0)
- [18.27](https://wandb.ai/wangyashuu/uncategorized/runs/1y6cj9jg)
- [18.277](https://wandb.ai/wangyashuu/uncategorized/runs/38fftz06)

### cnn + cos_similarty

Train:

- [18.371](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1luj8peb)
- [18.684](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2qnrq223)
- [18.486](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/7tm044dq)
- [17.609](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/36t83y2o)
- [17.62](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/es37oh3b)

Val: 18.233 = (18.199+18.257+18.274+18.143+18.294)/5

- [18.199](https://wandb.ai/wangyashuu/uncategorized/runs/110h5uyj)
- [18.257](https://wandb.ai/wangyashuu/uncategorized/runs/1kcajnpu)
- [18.274](https://wandb.ai/wangyashuu/uncategorized/runs/1w99gil1)
- [18.143](https://wandb.ai/wangyashuu/uncategorized/runs/3s970b4x)
- [18.294](https://wandb.ai/wangyashuu/uncategorized/runs/3vd6a5nn)

###

Train

- [18.349](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/2j6vr05e)
- [18.605](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/15239jsc)
- [18.499](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1htvvur4)
- [17.453](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/1t94ush4)
- [17.447](https://wandb.ai/wangyashuu/PetfinderPawpularity/runs/3a2gqe5b)

Val: 18.21 = (18.152+18.206+18.175+18.189+18.328)/5

- [18.328](https://wandb.ai/wangyashuu/uncategorized/runs/32f1z6ie)
- [18.189](https://wandb.ai/wangyashuu/uncategorized/runs/l0jweqjv)
- [18.175](https://wandb.ai/wangyashuu/uncategorized/runs/2tfihkpx)
- [18.206](https://wandb.ai/wangyashuu/uncategorized/runs/176xhefd)
- [18.152](https://wandb.ai/wangyashuu/uncategorized/runs/1c916xb9)

## Conclusions

It seems similar images does not help to much
