import os
import cv2
import torch
import numpy as np
import random
from copy import deepcopy
from PIL import Image, ImageEnhance

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_AUG_PARAMS = {
    "noise_level": 0.05,
    "noise_area_ratio": 0.1,
    "brightness_factor_range": (0.8, 1.2),
    "adjustment_area_ratio": 0.1,
    "amplitude": 5,
    "wavelength": 20,
    "artifact_intensity": 0.5,
    "flare_intensity": 0.5,
}


def _normalize_aug_params(params=None):
    """Merge user-specified augmentation params with safe defaults."""
    merged = deepcopy(DEFAULT_AUG_PARAMS)
    if params:
        for key, value in params.items():
            if value is not None:
                merged[key] = value

    # Ensure brightness range stored as tuple for random.uniform
    if isinstance(merged["brightness_factor_range"], list):
        merged["brightness_factor_range"] = tuple(merged["brightness_factor_range"])

    return merged

# 高效的噪声处理 (基于NumPy，避免逐像素操作)
def add_random_noise(image, noise_level=0.05, noise_area_ratio=0.1):
    """添加泊松噪声到灰度图像"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    h, w = image.shape
    # 使用泊松噪声而不是随机点操作
    noise = np.random.poisson(lam=noise_level * 10, size=(h, w)).astype(np.float32)
    noise = noise - noise.mean()  # 居中
    
    # 仅在指定比例区域应用噪声
    if noise_area_ratio < 1.0:
        mask = np.random.rand(h, w) < noise_area_ratio
        noise = noise * mask
    
    noisy_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_image

# 全局亮度调整 (更稳定)
def random_brightness_adjustment(image, brightness_factor_range=(0.8, 1.2), adjustment_area_ratio=0.1):
    """全局亮度缩放，避免复杂的区域操作"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    brightness_factor = random.uniform(*brightness_factor_range)
    adjusted = np.clip(image.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
    return adjusted

# 简单的对比度缩放
def ripple_distortion(image, amplitude=5, wavelength=20):
    """对比度缩放 (用对比度代替复杂的波纹失真)"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    contrast_factor = random.uniform(0.8, 1.2)
    # 对比度缩放: I' = (I - 128) * factor + 128
    adjusted = (image.astype(np.float32) - 128) * contrast_factor + 128
    return np.clip(adjusted, 0, 255).astype(np.uint8)

# 添加亮度随机偏差
def add_metallic_artifacts(image, artifact_intensity=0.5):
    """添加局部亮度扰动 (模拟伪影效果)"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    h, w = image.shape
    # 生成随机亮度偏差图
    deviation = np.random.randn(h, w) * (artifact_intensity * 20)
    artifact = np.clip(image.astype(np.float32) + deviation, 0, 255).astype(np.uint8)
    return artifact

# 添加高频噪声
def add_lens_flare(image, flare_intensity=0.5, flare_position=None):
    """添加高斯噪声 (模拟光学效果)"""
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    h, w = image.shape
    # 高斯噪声
    noise = np.random.randn(h, w) * (flare_intensity * 15)
    flare_image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return flare_image

def augment_image(image, aug_params=None, **kwargs):
    """应用所有增强操作到灰度图 (NumPy 原生,无GPU转移)"""
    # Allow callers to provide either a dict or keyword overrides
    param_source = {}
    if aug_params:
        param_source.update(aug_params)
    param_source.update(kwargs)
    params = _normalize_aug_params(param_source)

    # 应用增强序列 (均在 NumPy 中操作)
    result = image
    if not isinstance(result, np.ndarray):
        result = cv2.imread(result, cv2.IMREAD_GRAYSCALE) if isinstance(result, str) else np.array(result)
    
    result = add_random_noise(result, params['noise_level'], params['noise_area_ratio'])
    result = random_brightness_adjustment(result, params['brightness_factor_range'], params['adjustment_area_ratio'])
    result = ripple_distortion(result, params['amplitude'], params['wavelength'])
    result = add_metallic_artifacts(result, params['artifact_intensity'])
    result = add_lens_flare(result, params['flare_intensity'])
    
    return result

def augment_single_image(image, num_augments=1, **kwargs):
    """生成单张图像的多个增强版本"""
    # 加载图像
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to load image from {image}")
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Invalid image input")

    params = _normalize_aug_params(kwargs)
    augmented_results = []
    for _ in range(num_augments):
        aug_img = augment_image(image, params)
        augmented_results.append(aug_img)

    return augmented_results
