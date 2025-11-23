"""
统一预处理流水线
实现: 彩色 → 超分辨率 → 重塑 → 去色 → 一对多增强
"""

import os
import yaml
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from .super_resolution import SuperResolution
from .img_process_train import augment_single_image


class PreprocessPipeline:
    """
    自动预处理流水线
    检查images_gray是否存在 → 不存在则生成 → 验证格式
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 读取参数
        self.num_augments = self.config.get('num_augments', 3)
        self.sr_enabled = self.config.get('sr_enabled', True)
        self.sr_scale = self.config.get('sr_scale_factor', 2)
        self.target_size = (self.config.get('img_width', 640), self.config.get('img_height', 640))
        self.regenerate_gray = self.config.get('regenerate_gray', False)
        
        # 增强参数
        self.aug_params = self.config.get('augmentation_params', {})
        
        # 超分辨率模块
        self.sr_module = SuperResolution(scale_factor=self.sr_scale) if self.sr_enabled else None
        
        print("✓ 预处理流水线初始化完成")
        print(f"  - 多版本增强: {self.num_augments}x")
        print(f"  - 超分辨率: {'启用 ({}x)'.format(self.sr_scale) if self.sr_enabled else '禁用'}")
        print(f"  - 目标尺寸: {self.target_size}")
    
    def preprocess_single(self, color_image):
        """
        单张图像预处理: 彩色 → 超分 → 重塑 → 去色 → 多增强
        
        Args:
            color_image: numpy array [H, W, 3] (BGR)
            
        Returns:
            gray_augmented_list: List of [H, W] numpy arrays
        """
        # 1. 超分辨率
        if self.sr_enabled:
            color_image = self.sr_module.upsample_numpy(color_image.astype(np.float32) / 255.0)
            color_image = (color_image * 255).astype(np.uint8)
        
        # 2. 重塑到目标尺寸
        H, W = color_image.shape[:2]
        if (H, W) != self.target_size:
            color_image = cv2.resize(color_image, self.target_size, interpolation=cv2.INTER_AREA)
        
        # 3. 去色 (使用Rec.601权重)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        
        # 4. 一对多增强
        gray_augmented_list = []
        for i in range(self.num_augments):
            # 使用img_process_train.py的augment_single_image
            augmented = augment_single_image(
                gray_image, 
                num_augments=1,
                **self.aug_params
            )[0]  # 取第一个结果
            
            gray_augmented_list.append(augmented)
        
        return gray_augmented_list
    
    def preprocess_folder(self, images_folder, output_folder):
        """
        批量处理整个文件夹
        
        Args:
            images_folder: 彩色图像文件夹路径
            output_folder: 灰度图像输出文件夹路径
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\n开始批量预处理: {len(image_files)} 张图像")
        print(f"输出文件夹: {output_folder}")
        
        for img_file in tqdm(image_files, desc="预处理进度"):
            # 读取彩色图像
            img_path = os.path.join(images_folder, img_file)
            color_img = cv2.imread(img_path)
            
            if color_img is None:
                print(f"⚠ 跳过无法读取的文件: {img_file}")
                continue
            
            # 预处理生成多版本
            gray_list = self.preprocess_single(color_img)
            
            # 保存多版本
            base_name = os.path.splitext(img_file)[0]
            for aug_idx, gray_aug in enumerate(gray_list):
                output_name = f"{base_name}_aug{aug_idx}.png"
                output_path = os.path.join(output_folder, output_name)
                cv2.imwrite(output_path, gray_aug)
        
        print(f"✓ 预处理完成: 生成 {len(image_files) * self.num_augments} 张灰度图像")
    
    def validate_gray_dataset(self, images_folder, images_gray_folder):
        """
        验证images_gray格式是否符合要求
        
        Args:
            images_folder: 彩色图像文件夹
            images_gray_folder: 灰度图像文件夹
            
        Returns:
            valid: bool
            errors: List[str]
        """
        errors = []
        
        # 检查文件夹是否存在
        if not os.path.exists(images_gray_folder):
            errors.append(f"images_gray文件夹不存在: {images_gray_folder}")
            return False, errors
        
        # 统计彩色图像数量
        color_files = [f for f in os.listdir(images_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        expected_gray_count = len(color_files) * self.num_augments
        
        # 统计灰度图像数量
        gray_files = [f for f in os.listdir(images_gray_folder) 
                     if f.lower().endswith('.png')]
        actual_gray_count = len(gray_files)
        
        # 检查数量
        if actual_gray_count != expected_gray_count:
            errors.append(
                f"灰度图像数量不匹配: 期望 {expected_gray_count} "
                f"(彩色{len(color_files)} × 增强{self.num_augments})，实际 {actual_gray_count}"
            )
        
        # 检查尺寸 (抽样5张)
        sample_size = min(5, len(gray_files))
        for gray_file in gray_files[:sample_size]:
            gray_path = os.path.join(images_gray_folder, gray_file)
            gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
            
            if gray_img is None:
                errors.append(f"无法读取灰度图像: {gray_file}")
                continue
            
            H, W = gray_img.shape
            if (W, H) != self.target_size:
                errors.append(
                    f"尺寸不匹配: {gray_file} 实际尺寸 ({W}, {H})，期望 {self.target_size}"
                )
        
        # 检查命名格式
        naming_errors = []
        for color_file in color_files[:10]:  # 抽样10个
            base_name = os.path.splitext(color_file)[0]
            for aug_idx in range(self.num_augments):
                expected_name = f"{base_name}_aug{aug_idx}.png"
                expected_path = os.path.join(images_gray_folder, expected_name)
                
                if not os.path.exists(expected_path):
                    naming_errors.append(f"缺失文件: {expected_name}")
        
        if naming_errors:
            errors.append(f"命名格式错误 (抽样检查): {naming_errors[:3]}")
        
        return len(errors) == 0, errors
    
    def check_and_generate(self, images_folder, images_gray_folder):
        """
        检查images_gray文件夹 → 不存在/无效则生成 → 验证
        
        Args:
            images_folder: 彩色图像文件夹
            images_gray_folder: 灰度图像文件夹
            
        Returns:
            success: bool
        """
        print("\n" + "="*60)
        print("开始预处理检查")
        print("="*60)
        
        # 1. 检查是否需要生成
        need_generate = False
        
        if not os.path.exists(images_gray_folder):
            print("✗ images_gray文件夹不存在")
            need_generate = True
        elif self.regenerate_gray:
            print("⚠ regenerate_gray=True，强制重新生成")
            need_generate = True
        else:
            # 验证现有数据
            print("✓ images_gray文件夹已存在，开始验证...")
            valid, errors = self.validate_gray_dataset(images_folder, images_gray_folder)
            
            if valid:
                print("✓ 验证通过，使用现有灰度数据")
                return True
            else:
                print("✗ 验证失败:")
                for error in errors:
                    print(f"  - {error}")
                need_generate = True
        
        # 2. 生成灰度数据
        if need_generate:
            print("\n开始生成灰度数据...")
            
            # 删除旧文件夹
            if os.path.exists(images_gray_folder):
                import shutil
                shutil.rmtree(images_gray_folder)
                print(f"✓ 清理旧数据: {images_gray_folder}")
            
            # 生成新数据
            self.preprocess_folder(images_folder, images_gray_folder)
            
            # 验证生成结果
            valid, errors = self.validate_gray_dataset(images_folder, images_gray_folder)
            
            if valid:
                print("\n✓ 灰度数据生成并验证成功")
                return True
            else:
                print("\n✗ 生成后验证失败:")
                for error in errors:
                    print(f"  - {error}")
                return False
        
        return True


def test_pipeline():
    """测试流水线"""
    # 创建临时测试数据
    test_color = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    
    # 创建临时配置
    temp_config = {
        'num_augments': 3,
        'sr_enabled': True,
        'sr_scale_factor': 2,
        'img_width': 640,
        'img_height': 640,
        'augmentation_params': {
            'noise_level': 0.05,
            'brightness_factor_range': [0.8, 1.2]
        }
    }
    
    with open('temp_config.yaml', 'w') as f:
        yaml.dump(temp_config, f)
    
    # 测试流水线
    pipeline = PreprocessPipeline('temp_config.yaml')
    gray_list = pipeline.preprocess_single(test_color)
    
    print(f"\n✓ 单张测试: 输入 {test_color.shape} → 输出 {len(gray_list)} 张灰度图")
    print(f"  - 灰度图尺寸: {gray_list[0].shape}")
    
    # 清理
    os.remove('temp_config.yaml')
    
    print("\n✅ 流水线测试通过")


if __name__ == '__main__':
    test_pipeline()
