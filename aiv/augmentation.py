
from typing import Dict
from torchvision import transforms


def get_transforms(config: Dict[str, Dict[str, Dict[str, any]]]) -> transforms.Compose:
    """
    주어진 설정(config)에 따라 이미지 변환 파이프라인을 생성합니다.

    Args:
        config (Dict[str, Dict[str, Dict[str, any]]]): 이미지 변환에 대한 설정 정보.

    Returns:
        transforms.Compose: 설정된 변환을 포함한 이미지 변환 파이프라인.
    """

    transform_list = []

    # Resize
    if config['augmentation']['resize']['enabled']:
        image_size = config['input_size']
        transform_list.append(transforms.Resize((image_size[0], image_size[1])))

    # RandomHorizontalFlip
    if config['augmentation']['random_horizontal_flip']['enabled']:
        p = config['augmentation']['random_horizontal_flip']['p']
        transform_list.append(transforms.RandomHorizontalFlip(p=p))

    # RandomRotation
    if config['augmentation']['random_rotation']['enabled']:
        degrees = config['augmentation']['random_rotation']['degrees']
        transform_list.append(transforms.RandomRotation(degrees=degrees))

    # RandomCrop
    if config['augmentation']['random_crop']['enabled']:
        size = config['augmentation']['random_crop']['size']
        transform_list.append(transforms.RandomCrop(size))

    # brightness, contrast, saturation, hue
    if config['augmentation']['color_jitter']['enabled']:
        brightness = config['augmentation']['color_jitter']['brightness']
        contrast = config['augmentation']['color_jitter']['contrast']
        saturation = config['augmentation']['color_jitter']['saturation']
        hue = config['augmentation']['color_jitter']['hue']
        transform_list.append(transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        ))

    # GaussianBlur
    if config['augmentation']['gaussian_blur']['enabled']:
        kernel_size = config['augmentation']['gaussian_blur']['kernel_size']
        sigma = config['augmentation']['gaussian_blur']['sigma']
        transform_list.append(transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma))

    # ToTensor
    if config['augmentation']['to_tensor']['enabled']:
        transform_list.append(transforms.ToTensor())

    # Normalize
    if config['augmentation']['normalize']['enabled']:
        mean = config['augmentation']['normalize']['mean']
        std = config['augmentation']['normalize']['std']
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    # RandomErasing
    if config['augmentation']['random_erasing']['enabled']:
        p = config['augmentation']['random_erasing']['p']
        scale = config['augmentation']['random_erasing']['scale']
        ratio = config['augmentation']['random_erasing']['ratio']
        value = config['augmentation']['random_erasing']['value']
        transform_list.append(transforms.RandomErasing(p=p, scale=scale, ratio=ratio, value=value))

    return transforms.Compose(transform_list)
