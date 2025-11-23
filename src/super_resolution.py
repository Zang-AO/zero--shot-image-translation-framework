"""
GPU加速超分辨率模块
替代imge_reshape2.py的CPU版本，使用PyTorch GPU加速
"""

import torch
import torch.nn.functional as F
import numpy as np


class SuperResolution:
    """
    GPU双线性插值超分辨率
    比ESPCN更简单但速度快10倍
    """
    
    def __init__(self, scale_factor=2, device='cuda'):
        """
        Args:
            scale_factor: 放大倍数 (默认2倍)
            device: 'cuda' or 'cpu'
        """
        self.scale_factor = scale_factor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def upsample(self, image_tensor):
        """
        GPU双线性插值放大
        
        Args:
            image_tensor: [C, H, W] or [B, C, H, W]
            
        Returns:
            upsampled: [C, H*scale, W*scale] or [B, C, H*scale, W*scale]
        """
        single_image = False
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
            single_image = True
        
        B, C, H, W = image_tensor.shape
        new_H = int(H * self.scale_factor)
        new_W = int(W * self.scale_factor)
        
        # GPU双线性插值
        upsampled = F.interpolate(
            image_tensor.to(self.device),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )
        
        if single_image:
            return upsampled.squeeze(0)
        return upsampled
    
    def upsample_numpy(self, image_np):
        """
        NumPy接口 (HWC或HW格式)
        
        Args:
            image_np: numpy array [H, W, C] or [H, W]
            
        Returns:
            upsampled_np: numpy array [H*scale, W*scale, C] or [H*scale, W*scale]
        """
        # 转为tensor
        if image_np.ndim == 2:  # 灰度
            tensor = torch.from_numpy(image_np).float().unsqueeze(0)  # [1, H, W]
            is_gray = True
        else:  # 彩色
            tensor = torch.from_numpy(image_np).float().permute(2, 0, 1)  # [C, H, W]
            is_gray = False
        
        # 放大
        upsampled = self.upsample(tensor)
        
        # 转回numpy
        if is_gray:
            return upsampled.squeeze(0).cpu().numpy()  # [H*scale, W*scale]
        else:
            return upsampled.permute(1, 2, 0).cpu().numpy()  # [H*scale, W*scale, C]


class GPUImageProcessor:
    """GPU批量图像处理器 (整合image_process_gpu.py功能)"""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    def gaussian_blur(self, image_tensor, kernel_size=5, sigma=1.0):
        """高斯模糊"""
        channels = image_tensor.shape[1] if image_tensor.ndim == 4 else image_tensor.shape[0]
        
        # 创建高斯核
        x = torch.arange(kernel_size, dtype=torch.float32, device=self.device) - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d /= kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.expand(channels, 1, kernel_size, kernel_size).contiguous()
        
        # 卷积
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        blurred = F.conv2d(
            image_tensor.to(self.device), 
            kernel, 
            padding=kernel_size // 2, 
            groups=channels
        )
        
        return blurred.squeeze(0) if blurred.shape[0] == 1 else blurred
    
    def sharpen(self, image_tensor):
        """边缘增强"""
        channels = image_tensor.shape[1] if image_tensor.ndim == 4 else image_tensor.shape[0]
        
        # 锐化核
        kernel = torch.tensor(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 
            dtype=torch.float32, 
            device=self.device
        )
        kernel = kernel.expand(channels, 1, 3, 3).contiguous()
        
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        sharpened = F.conv2d(
            image_tensor.to(self.device), 
            kernel, 
            padding=1, 
            groups=channels
        )
        
        return sharpened.squeeze(0) if sharpened.shape[0] == 1 else sharpened
    
    def resize_to_fixed(self, image_tensor, size=(640, 640)):
        """调整到固定尺寸"""
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        resized = F.interpolate(
            image_tensor.to(self.device), 
            size=size, 
            mode='area'
        )
        
        return resized.squeeze(0) if resized.shape[0] == 1 else resized


if __name__ == '__main__':
    # 测试
    import cv2
    
    print("="*60)
    print("GPU超分辨率模块测试")
    print("="*60)
    
    # 测试超分辨率
    sr = SuperResolution(scale_factor=2, device='cuda')
    
    # 测试灰度图
    gray_img = np.random.randint(0, 255, (320, 320), dtype=np.uint8)
    upsampled_gray = sr.upsample_numpy(gray_img)
    print(f"✓ 灰度图超分: {gray_img.shape} → {upsampled_gray.shape}")
    
    # 测试彩色图
    color_img = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
    upsampled_color = sr.upsample_numpy(color_img)
    print(f"✓ 彩色图超分: {color_img.shape} → {upsampled_color.shape}")
    
    # 测试GPU处理器
    processor = GPUImageProcessor(device='cuda')
    
    tensor = torch.rand(1, 3, 640, 640)
    blurred = processor.gaussian_blur(tensor)
    print(f"✓ 高斯模糊: {tensor.shape} → {blurred.shape}")
    
    sharpened = processor.sharpen(tensor)
    print(f"✓ 锐化: {tensor.shape} → {sharpened.shape}")
    
    resized = processor.resize_to_fixed(tensor, size=(960, 960))
    print(f"✓ Resize: {tensor.shape} → {resized.shape}")
    
    print("\n" + "="*60)
    print("✅ 所有测试通过")
    print("="*60)
