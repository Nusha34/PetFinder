from torch.utils.data import DataLoader
from torchvision import transforms
import albumentations
from albumentations.pytorch import ToTensorV2


from petfinder_pawpularity.dataset import PetfinderPawpularityDataset


normalize_params = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
size_transforms = ["Resize", "RandomResizedCrop", "CenterCrop", "RandomCrop"]


def get_transform(conf_transform, conf_transforms, normalized=True):
    image_size = conf_transform.image_size
    use_albumentations = conf_transform.lib == "albumentations"

    lib = albumentations if use_albumentations else transforms
    transformers = []
    for t in conf_transforms:
        if hasattr(t, "params"):
            params = t.params
        else:
            if t.name in size_transforms:
                params = (
                    dict(width=image_size, height=image_size)
                    if use_albumentations
                    else dict(size=[image_size, image_size])
                )
            elif t.name == "SmallestMaxSize":
                params = dict(max_size=image_size)
            else:
                params = dict()

        transformers.append(getattr(lib, t.name)(**params))
    if normalized:
        transformers += (
            [lib.Normalize(**normalize_params), ToTensorV2()]
            if use_albumentations
            else [lib.ToTensor(), lib.Normalize(**normalize_params)]
        )
    else:
        transformers += (
            [ToTensorV2()] if use_albumentations else [lib.ToTensor()]
        )
    return lib.Compose(transformers)


# transform from 0-100 to 0-1 for label y
def divide100(y):
    return y / 100


def target_inverse_transform(y):
    return y * 100


def get_dataloader(
    data, transform, batch_size, shuffle=False, target_transform=divide100
):
    dataset = PetfinderPawpularityDataset(
        data,
        transform=transform,
        target_transform=target_transform,
    )
    return DataLoader(
        dataset,
        shuffle=shuffle,
        num_workers=2,
        batch_size=batch_size,
    )
