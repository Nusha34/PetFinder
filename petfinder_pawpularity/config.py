from box import Box

mode = ["reuse_architecture", "reuse_weights", "fine_tuning"][1]


def merge(*args):
    result = dict()
    for d in args:
        for k in d:
            if (
                k in result
                and isinstance(result[k], dict)
                and isinstance(d[k], dict)
            ):
                result[k] = merge(result[k], d[k])
            else:
                result[k] = d[k]
    return result


def create_config(custom_config=dict()):
    default_config = dict(
        seed=2021,
        data_rate=1,
        n_epochs=10,
        n_folds=0,
        optimizer=dict(
            name="AdamW",
            params=dict(lr=1e-5),
        ),
        scheduler=dict(
            name="CosineAnnealingWarmRestarts",
            params=dict(T_0=10, eta_min=1e-5 / 100),
        ),
        criterion="BCEWithLogitsLoss",
        transform=dict(
            image_size=224,
            lib="transforms",
        ),
        train=dict(
            rate=0.9,
            transforms=[dict(name="Resize")],
            batch_size=16,
        ),
        test=dict(
            batch_size=64,
            transforms=[dict(name="Resize")],
        ),
        model_mode="fine_tuning",
        early_stopping=True,
    )
    config = merge(default_config, custom_config)
    return Box(config)


def get_transforms_config(
    degrees=10,
    translate=(0.1, 0.1),
    scale=(0.9, 1.1),
    shear=0,
    brightness=0.1,
    contrast=0,
    saturation=0.1,
    hue=0,
    includes=[],
):
    augmented_transforms = [
        dict(name="RandomHorizontalFlip"),
        dict(name="RandomPosterize", params=dict(bits=7)),
        dict(name="RandomAdjustSharpness", params=dict(sharpness_factor=2)),
        dict(name="RandomAutocontrast"),
        dict(
            name="RandomAffine",
            params=dict(
                degrees=degrees, translate=translate, scale=scale, shear=shear
            ),
        ),
        dict(
            name="ColorJitter",
            params=dict(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
        ),
    ]
    augmented_transforms = [
        t for t in augmented_transforms if t["name"] in includes
    ]
    resize = dict(name="Resize")
    augmented_transforms.append(resize)

    return augmented_transforms


def get_simple_transform_config():
    resize = dict(name="Resize")
    return [resize]


def to_flat_dict(box):
    result = dict()
    if not hasattr(box, "items"):
        return result
    for key, val in box.items():
        if type(val) in [float, int, bool, str]:
            result[key] = val
        else:
            for sub_key, sub_val in to_flat_dict(val).items():
                result[f"{key}_{sub_key}"] = sub_val
    return result


albumentations_aug = [
    dict(name="SmallestMaxSize"),
    dict(name="HorizontalFlip"),
    dict(
        name="ShiftScaleRotate",
        params=dict(
            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
        ),
    ),
    dict(
        name="HueSaturationValue",
        params=dict(
            hue_shift_limit=0.2,
            sat_shift_limit=0.2,
            val_shift_limit=0.2,
            p=0.5,
        ),
    ),
    dict(
        name="RGBShift",
        params=dict(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.5),
    ),
    dict(
        name="RandomBrightnessContrast",
        params=dict(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
    ),
    dict(name="RandomResizedCrop"),
]


transforms_aug = [
    dict(name="Resize"),
    dict(name="RandomHorizontalFlip"),
    dict(name="RandomRotation", params=dict(degress=(-5, +5))),
    dict(name="RandomAdjustSharpness", params=dict(sharpness_factor=2)),
    dict(name="RandomAutocontrast"),
    dict(name="RandomCrop"),
]

normal_transforms = [dict(name="Resize")]
