"""
Data augmentation pipeline for EarVN1.0.
Matches paper: flip, rotate ±15°, color jitter, random erasing, Gaussian blur.
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 224, aug_cfg: dict = None) -> transforms.Compose:
    """
    Training augmentation pipeline.

    From paper:
        - Random horizontal flip
        - Random rotation (±15°)
        - Color jitter (brightness=0.2, contrast=0.2)
        - Random erasing (p=0.25)
        - Gaussian blur (sigma=[0.1, 2.0])
        - Resize to 224x224, bicubic
        - ImageNet normalization
    """
    if aug_cfg is None:
        aug_cfg = {}

    rotation = aug_cfg.get("rotation_degrees", 15)
    cj = aug_cfg.get("color_jitter", {"brightness": 0.2, "contrast": 0.2})
    erase_prob = aug_cfg.get("random_erasing_prob", 0.25)
    blur_sigma = aug_cfg.get("gaussian_blur_sigma", [0.1, 2.0])

    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=rotation),
        transforms.ColorJitter(
            brightness=cj.get("brightness", 0.2),
            contrast=cj.get("contrast", 0.2),
        ),
        transforms.GaussianBlur(kernel_size=5, sigma=tuple(blur_sigma)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
        transforms.RandomErasing(p=erase_prob, scale=(0.02, 0.33)),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
