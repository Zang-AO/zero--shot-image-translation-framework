"""
ZSXT 推理脚本
功能：
1. 读取目标域图像
2. 超分辨率(可选) + 灰度化(无降噪)
3. 加载ZSXT模型推理
4. 保存到指定输出目录
5. 复制labels文件夹(方便ultralytics检测)

使用示例:
    python inference.py --input CLC_extract/images --output CLC_extract_zsxt/images --checkpoint checkpoints/gen_best.pth
"""

import os
import sys
import argparse
import shutil
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

from src.model import GeneratorUNet
from src.super_resolution import SuperResolution


class ZSXTInference:
    """ZSXT推理器"""
    
    def __init__(self, checkpoint_path, config_path='config.yaml', device='cuda'):
        """
        Args:
            checkpoint_path: 模型权重路径 (e.g., checkpoints/gen_best.pth)
            config_path: 配置文件路径
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f) or {}
        
        # 推理配置
        inference_cfg = self.config.get('inference', {})
        self.sr_enabled = inference_cfg.get('sr_enabled', True)
        self.sr_scale = inference_cfg.get('sr_scale_factor', 2)
        self.sr_model_path = inference_cfg.get('sr_model_path', None)
        self.target_size = (
            inference_cfg.get('img_width', 640),
            inference_cfg.get('img_height', 640)
        )
        self.decolor_only = inference_cfg.get('decolor_only', True)  # 推理时不加噪声
        
        # 初始化超分辨率模块
        if self.sr_enabled:
            self.sr_module = SuperResolution(scale_factor=self.sr_scale, device=self.device, model_path=self.sr_model_path)
            print(f"✓ 超分辨率已启用 ({self.sr_scale}x)")
        else:
            self.sr_module = None
            print("✓ 超分辨率已禁用")
        
        # 加载生成器模型
        print(f"[Model] 加载检查点: {checkpoint_path}")
        self.generator = GeneratorUNet(input_channels=1, output_channels=3).to(self.device)
        
        # 兼容不同checkpoint格式，支持常见键名并处理 DataParallel 前缀
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # 从 checkpoint 中提取模型权重字典
        state_dict = None
        if isinstance(checkpoint, dict):
            # 常见字段
            for key in ('state_dict', 'model_state_dict', 'gen_state_dict', 'generator_state_dict', 'generator', 'gen'):
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            # 如果没有常见字段，可能 checkpoint 本身就是 state_dict
            if state_dict is None:
                # 判断字典的键是否看起来像参数名 (含有点号)
                sample_keys = list(checkpoint.keys())[:10]
                if sample_keys and all(isinstance(k, str) and '.' in k for k in sample_keys):
                    state_dict = checkpoint
        else:
            state_dict = checkpoint

        if state_dict is None:
            raise RuntimeError(f'无法从checkpoint中找到模型权重: {checkpoint_path}')

        # 处理 DataParallel 保存的 'module.' 前缀
        new_state = {}
        for k, v in state_dict.items():
            new_key = k
            if k.startswith('module.'):
                new_key = k[len('module.'):]
            new_state[new_key] = v

        # 尝试加载，使用 strict=False 以兼容部分键不匹配的检查点
        load_res = self.generator.load_state_dict(new_state, strict=False)
        # 打印提示信息
        missing = load_res.missing_keys if hasattr(load_res, 'missing_keys') else []
        unexpected = load_res.unexpected_keys if hasattr(load_res, 'unexpected_keys') else []
        if missing:
            print(f"⚠ 加载checkpoint时缺失键 ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
        if unexpected:
            print(f"⚠ 加载checkpoint时多余键 ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
        
        self.generator.eval()
        print(f"✓ 模型加载成功 (设备: {self.device})")
        print(f"✓ 目标尺寸: {self.target_size}")
    
    def preprocess_image(self, image_path):
        """
        预处理单张图像: 超分辨率 + 灰度化
        
        Args:
            image_path: 输入图像路径
            
        Returns:
            gray_tensor: [1, 1, H, W] 范围 [-1, 1]
            original_size: (W, H) 原始尺寸
        """
        # 读取彩色图像
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        original_size = (img.shape[1], img.shape[0])  # (W, H)
        
        # 步骤1: 超分辨率 (可选)
        if self.sr_enabled:
            img = self.sr_module.upsample_numpy(img.astype(np.float32) / 255.0)
            img = (img * 255).astype(np.uint8)
        
        # 步骤2: 重塑到目标尺寸
        if (img.shape[1], img.shape[0]) != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        
        # 步骤3: 灰度化 (使用Rec.601标准，无噪声)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 转为tensor [-1, 1]
        gray_tensor = torch.from_numpy(gray).float() / 255.0
        gray_tensor = gray_tensor.unsqueeze(0).unsqueeze(0) * 2 - 1  # [1, 1, H, W]
        
        return gray_tensor.to(self.device), original_size
    
    def infer_single(self, gray_tensor):
        """
        单张图像推理
        
        Args:
            gray_tensor: [1, 1, H, W] 范围 [-1, 1]
            
        Returns:
            rgb_np: [H, W, 3] numpy array, 范围 [0, 255], uint8
        """
        with torch.no_grad():
            fake_rgb = self.generator(gray_tensor)  # [1, 3, H, W] 范围 [-1, 1]
        
        # 转为numpy [0, 255]
        rgb_np = fake_rgb.squeeze(0).cpu().detach().numpy()  # [3, H, W]
        rgb_np = (rgb_np + 1) / 2 * 255  # [-1,1] → [0,255]
        rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
        rgb_np = rgb_np.transpose(1, 2, 0)  # [H, W, 3]
        
        # BGR格式 (OpenCV)
        rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        
        return rgb_np
    
    def infer_folder(self, input_folder, output_folder, copy_labels=True):
        """
        批量推理文件夹
        
        Args:
            input_folder: 输入图像文件夹 (e.g., CLC_extract/images)
            output_folder: 输出图像文件夹 (e.g., CLC_extract_zsxt/images)
            copy_labels: 是否复制labels文件夹
        """
        # 创建输出目录
        os.makedirs(output_folder, exist_ok=True)
        
        # 获取所有图像文件
        image_files = [f for f in os.listdir(input_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not image_files:
            print(f"✗ 未找到图像文件: {input_folder}")
            return
        
        print(f"\n{'='*70}")
        print(f"开始推理")
        print(f"{'='*70}")
        print(f"输入路径: {input_folder}")
        print(f"输出路径: {output_folder}")
        print(f"图像数量: {len(image_files)}")
        print(f"{'='*70}\n")
        
        # 批量推理
        for img_file in tqdm(image_files, desc="推理进度"):
            input_path = os.path.join(input_folder, img_file)
            output_path = os.path.join(output_folder, img_file)
            
            try:
                # 预处理
                gray_tensor, original_size = self.preprocess_image(input_path)
                
                # 推理
                rgb_img = self.infer_single(gray_tensor)
                
                # 保存 (保持目标推理分辨率 self.target_size)
                target_w, target_h = self.target_size
                if rgb_img.shape[:2] != (target_h, target_w):
                    rgb_img = cv2.resize(rgb_img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                cv2.imwrite(output_path, rgb_img)
                
            except Exception as e:
                print(f"\n✗ 处理失败: {img_file} - {e}")
                continue
        
        print(f"\n✓ 推理完成: {len(image_files)} 张图像")
        print(f"✓ 输出路径: {output_folder}")
        
        # 复制labels文件夹 (方便ultralytics)
        if copy_labels:
            input_base = Path(input_folder).parent  # CLC_extract
            output_base = Path(output_folder).parent  # CLC_extract_zsxt
            
            labels_src = input_base / 'labels'
            labels_dst = output_base / 'labels'
            
            if labels_src.exists():
                if labels_dst.exists():
                    shutil.rmtree(labels_dst)
                shutil.copytree(labels_src, labels_dst)
                print(f"✓ 已复制labels: {labels_src} → {labels_dst}")
            else:
                print(f"⚠ 未找到labels文件夹: {labels_src}")
        
        print(f"\n{'='*70}")
        print(f"推理任务完成！")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='ZSXT推理脚本')
    parser.add_argument('--input', type=str,
                        default='D:\MyProg\ZSXT_Article\_code\datasets\Target_domain\EDS_test\PID_extract\images',
                        help='输入图像文件夹路径 (默认: datasets/Target_domain/KDXray_test/CLC_extract/images)')
    parser.add_argument('--output', type=str,
                        default='D:\MyProg\ZSXT_Article\_code\datasets\Target_domain\EDS_test\PID_extract_ZSXT\images',
                        help='输出图像文件夹路径 (默认: datasets/Target_domain/KDXray_test/CLC_extract_KD/images)')
    parser.add_argument('--checkpoint', type=str, default='D:\MyProg\ZSXT_Article\_code\EDS_results\checkpoints_EDS\gen_best.pth',
                        help='模型权重路径 (默认: checkpoints/gen_best.pth)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='配置文件路径 (默认: config.yaml)')
    parser.add_argument('--sr-model', type=str, default=None,
                       help='可选深度超分模型权重路径，优先于配置文件')
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu, 默认: cuda)')
    parser.add_argument('--no-copy-labels', action='store_true',
                       help='不复制labels文件夹')
    
    args = parser.parse_args()
    
    # 初始化推理器
    inferencer = ZSXTInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device
    )

    # 如果命令行传入深度SR模型路径，优先使用它
    if args.sr_model:
        inferencer.sr_module = SuperResolution(scale_factor=inferencer.sr_scale, device=inferencer.device, model_path=args.sr_model)
    
    # 执行推理
    inferencer.infer_folder(
        input_folder=args.input,
        output_folder=args.output,
        copy_labels=not args.no_copy_labels
    )


if __name__ == '__main__':
    main()
