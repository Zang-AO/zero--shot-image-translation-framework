"""
统一训练脚本 - 整合预处理流水线
特性:
- 自动检查/生成images_gray
- 单域训练
- 可视化和日志系统
- 参考train_zsxt_improved.py设计
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import yaml
import cv2
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# 加载配置
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f) or {}

from src.model import GeneratorUNet, PatchGANDiscriminator, weights_init_normal
from src.losses import CombinedLoss
from src.preprocess_pipeline import PreprocessPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==================== 评估指标计算 ====================

def safe_ssim(img1, img2):
    """安全的SSIM计算 (处理小图像)"""
    try:
        h, w = img1.shape[:2]
        min_side = min(h, w)
        win_size = max(3, min(11, min_side if min_side % 2 == 1 else min_side - 1))
        return ssim(img1, img2, channel_axis=2, win_size=win_size, data_range=1.0)
    except Exception as e:
        print(f"[Warning] SSIM计算失败: {e}")
        return 0.0


def calculate_mae(fake_img, real_img):
    """计算MAE (平均绝对误差) - 像素准确性
    
    Args:
        fake_img: 生成图 [B,C,H,W] 范围[-1,1]
        real_img: 真实图 [B,C,H,W] 范围[-1,1]
    
    Returns:
        mae: 平均绝对误差
    """
    with torch.no_grad():
        # 归一化到[0,1]用于计算
        fake_norm = (fake_img + 1) / 2
        real_norm = (real_img + 1) / 2
        
        # 计算逐像素绝对差异
        mae = torch.mean(torch.abs(fake_norm - real_norm))
        
    return mae.item()


def calculate_color_kl(fake_img, real_img, bins=64, eps=1e-8):
    """计算颜色直方图KL散度 - 颜色分布准确性
    
    Args:
        fake_img: 生成图 [B,C,H,W] 范围[-1,1]
        real_img: 真实图 [B,C,H,W] 范围[-1,1]
        bins: 直方图bins数
        eps: 防零频参数
    
    Returns:
        kl_div: KL散度 (平均三个通道)
    """
    with torch.no_grad():
        # 归一化到[0,1]
        fake_norm = (fake_img + 1) / 2
        real_norm = (real_img + 1) / 2
        
        # 计算RGB各通道的直方图 (CPU计算更稳定)
        fake_np = fake_norm.cpu().numpy()  # [B,C,H,W]
        real_np = real_norm.cpu().numpy()
        
        kl_divs = []
        
        for c in range(fake_np.shape[1]):  # 遍历RGB三个通道
            # 计算直方图
            fake_hist, _ = np.histogram(fake_np[:, c, :, :].flatten(), bins=bins, range=(0, 1))
            real_hist, _ = np.histogram(real_np[:, c, :, :].flatten(), bins=bins, range=(0, 1))
            
            # 归一化 + 防零频
            fake_hist = (fake_hist + eps) / (fake_hist.sum() + bins * eps)
            real_hist = (real_hist + eps) / (real_hist.sum() + bins * eps)
            
            # 计算KL散度: sum(P(x) * log(P(x) / Q(x)))
            kl = np.sum(real_hist * (np.log(real_hist) - np.log(fake_hist)))
            kl_divs.append(kl)
        
        # 返回三通道平均KL散度
        return np.mean(kl_divs)


def calculate_lpips(fake_img, real_img, model_vgg, device):
    """计算LPIPS (感知距离) - 使用VGG特征层
    
    Args:
        fake_img: 生成图 [B,C,H,W]
        real_img: 真实图 [B,C,H,W]
        model_vgg: 预训练VGG19模型 (特征提取器)
        device: 计算设备 (cuda/cpu)
    
    Returns:
        lpips: 感知距离分数 (0.0-0.2 正常范围)
    """
    with torch.no_grad():
        # 归一化到[0,1]
        fake_norm = (fake_img + 1) / 2
        real_norm = (real_img + 1) / 2
        
        # VGG19标准化 (ImageNet)
        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        fake_vgg = (fake_norm - vgg_mean) / vgg_std
        real_vgg = (real_norm - vgg_mean) / vgg_std
        
        # 提取特征 [B, 512, H', W']
        feat_fake = model_vgg(fake_vgg)
        feat_real = model_vgg(real_vgg)
        
        # 计算空间维度上的L2距离 (保留空间信息更稳定)
        # feat_diff: [B, C, H', W']
        feat_diff = (feat_fake - feat_real) ** 2
        
        # 沿着空间维度求平均，再求通道均值
        # 结果: [B]
        lpips_per_batch = torch.sqrt(torch.mean(feat_diff, dim=[1, 2, 3]))
        
        # 返回批次平均值
        lpips = torch.mean(lpips_per_batch)
        
    return lpips.item()


def calculate_fid(fake_features, real_features):
    """计算FID (Fréchet Inception Distance)
    
    Args:
        fake_features: 生成图特征 [N, D]
        real_features: 真实图特征 [N, D]
    
    Returns:
        fid: FID分数
    """
    try:
        from scipy.linalg import sqrtm
        
        # 确保至少有2个样本计算协方差
        if len(fake_features) < 2 or len(real_features) < 2:
            print(f"[Warning] FID样本不足: fake={len(fake_features)}, real={len(real_features)}")
            return 999.0
        
        mu_fake = np.mean(fake_features, axis=0)
        mu_real = np.mean(real_features, axis=0)
        
        sigma_fake = np.cov(fake_features.T)
        sigma_real = np.cov(real_features.T)
        
        # 处理1D情况 (当D=1时np.cov返回标量)
        if sigma_fake.ndim == 0:
            sigma_fake = np.array([[sigma_fake]])
        if sigma_real.ndim == 0:
            sigma_real = np.array([[sigma_real]])
        
        diff = mu_fake - mu_real
        cov_sqrt = sqrtm(sigma_fake @ sigma_real)
        
        # 处理复数情况
        if np.iscomplexobj(cov_sqrt):
            cov_sqrt = cov_sqrt.real
        
        fid = np.linalg.norm(diff) + np.trace(sigma_fake + sigma_real - 2 * cov_sqrt)
        return float(fid)
    except Exception as e:
        print(f"[Warning] FID计算失败: {e}")
        return 999.0

# 从config提取超参数
batch_size = config.get('batch_size', 8)
num_epochs = config.get('num_epochs', 50)
learning_rate = config.get('learning_rate', 0.0002)
img_width = config.get('img_width', 640)
img_height = config.get('img_height', 640)
save_dir = "generated_images"
os.makedirs(save_dir, exist_ok=True)


# ==================== 数据集 ====================

class ImagePairDataset(Dataset):
    """
    图像对数据集 (灰度, 彩色)
    支持一对多映射: 一张彩色对应多张增强后的灰度
    """

    def __init__(self, gray_dir, rgb_dir, image_size=(640, 640)):
        self.gray_dir = gray_dir
        self.rgb_dir = rgb_dir
        self.image_size = image_size

        # 获取灰度图列表 (含aug后缀)
        all_gray_files = [f for f in os.listdir(gray_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 建立映射: gray_aug0.png → 对应的彩色图 (支持多种格式)
        self.image_pairs = []
        for gray_file in all_gray_files:
            # 提取基础名 (去除_augN后缀)
            if '_aug' in gray_file:
                base_name = gray_file.split('_aug')[0]
            else:
                base_name = os.path.splitext(gray_file)[0]  # 去除扩展名
            
            # 尝试多种文件格式
            rgb_candidates = [
                os.path.join(rgb_dir, base_name + '.jpg'),
                os.path.join(rgb_dir, base_name + '.png'),
                os.path.join(rgb_dir, base_name + '.jpeg'),
            ]
            
            rgb_path = None
            for candidate in rgb_candidates:
                if os.path.exists(candidate):
                    rgb_path = candidate
                    break
            
            if rgb_path:
                self.image_pairs.append((gray_file, os.path.basename(rgb_path)))
        
        print(f"[Dataset] 加载 {len(self.image_pairs)} 张图像对")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        gray_file, rgb_file = self.image_pairs[idx]

        # 灰度图
        gray_path = os.path.join(self.gray_dir, gray_file)
        gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
        if gray_img is None:
            raise ValueError(f"Failed to load {gray_path}")
        
        # 尺寸检查 (自动resize)
        if gray_img.shape[:2] != self.image_size[::-1]:
            gray_img = cv2.resize(gray_img, self.image_size)

        # 彩色图
        rgb_path = os.path.join(self.rgb_dir, rgb_file)
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise ValueError(f"Failed to load {rgb_path}")
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        
        if rgb_img.shape[:2] != self.image_size[::-1]:
            rgb_img = cv2.resize(rgb_img, self.image_size)

        # 转为tensor [-1, 1]
        gray_tensor = torch.from_numpy(gray_img).float() / 255.0
        gray_tensor = gray_tensor.unsqueeze(0) * 2 - 1
        
        rgb_tensor = torch.from_numpy(rgb_img).float() / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1) * 2 - 1

        return {
            'gray': gray_tensor,
            'rgb': rgb_tensor,
            'filename': gray_file
        }


# ==================== 训练器 ====================

class Trainer:
    """统一的训练器"""
    
    def __init__(self, dataset_name='EDS', pretrained_gen=None, pretrained_disc=None):
        self.device = device
        self.dataset_name = dataset_name
        
        # 训练历史
        self.history = {
            'loss_G': [],
            'loss_D': [],
            'loss_L1': [],
            'loss_GAN': [],
            'loss_Perc': [],
            'loss_Color': []
        }
        
        # 初始化模型
        print(f"[Model] 构建模型...")
        self.gen = GeneratorUNet().to(device)
        self.disc = PatchGANDiscriminator().to(device)
        
        self.gen.apply(weights_init_normal)
        self.disc.apply(weights_init_normal)
        
        # 加载预训练权重 (如提供)
        if pretrained_gen and os.path.exists(pretrained_gen):
            print(f"[Model] 加载预训练生成器: {pretrained_gen}")
            try:
                checkpoint = torch.load(pretrained_gen, map_location=device)
                
                # 判断是完整checkpoint还是单纯的state_dict
                if isinstance(checkpoint, dict):
                    if 'gen_state_dict' in checkpoint:
                        # 完整checkpoint格式
                        state_dict = checkpoint['gen_state_dict']
                        print(f"  → 检测到完整checkpoint格式，提取生成器权重")
                    elif 'state_dict' in checkpoint:
                        # 另一种checkpoint格式
                        state_dict = checkpoint['state_dict']
                        print(f"  → 检测到state_dict格式")
                    else:
                        # 纯state_dict格式
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # 加载权重
                self.gen.load_state_dict(state_dict)
                print(f"  ✓ 生成器权重加载成功")
            except Exception as e:
                print(f"  ✗ 生成器权重加载失败: {e}")
                print(f"  → 将使用随机初始化权重重新开始")
        
        if pretrained_disc and os.path.exists(pretrained_disc):
            print(f"[Model] 加载预训练判别器: {pretrained_disc}")
            try:
                checkpoint = torch.load(pretrained_disc, map_location=device)
                
                # 判断是完整checkpoint还是单纯的state_dict
                if isinstance(checkpoint, dict):
                    if 'disc_state_dict' in checkpoint:
                        # 完整checkpoint格式
                        state_dict = checkpoint['disc_state_dict']
                        print(f"  → 检测到完整checkpoint格式，提取判别器权重")
                    elif 'state_dict' in checkpoint:
                        # 另一种checkpoint格式
                        state_dict = checkpoint['state_dict']
                        print(f"  → 检测到state_dict格式")
                    else:
                        # 纯state_dict格式
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint
                
                # 加载权重
                self.disc.load_state_dict(state_dict)
                print(f"  ✓ 判别器权重加载成功")
            except Exception as e:
                print(f"  ✗ 判别器权重加载失败: {e}")
                print(f"  → 将使用随机初始化权重重新开始")
        
        print(f"  Generator params: {sum(p.numel() for p in self.gen.parameters())/1e6:.2f}M")
        print(f"  Discriminator params: {sum(p.numel() for p in self.disc.parameters())/1e6:.2f}M")
        
        # 初始化损失函数 (动态三阶段权重,由config.yaml控制)
        total_epochs = config.get('num_epochs', num_epochs)
        loss_weights_config = config.get('loss_weights', {})
        # 若用户提供单值，则扩展为三阶段数组
        def expand(key, default_triplet):
            val = loss_weights_config.get(key)
            if val is None:
                return default_triplet
            if isinstance(val, (list, tuple)) and len(val) == 3:
                return list(val)
            return [val, val, val]
        custom_weights = {
            'lambda_L1': expand('l1', [1.5, 1.5, 1.5]),       # L1基础(稳定)
            'lambda_GAN': expand('gan', [0.5, 1.0, 0.8]),     # GAN平衡(中期强，后期降低)
            'lambda_Perc': expand('perceptual', [1.5, 1.8, 2.0]),  # Perceptual特征
            'lambda_Color': loss_weights_config.get('color', 30.0),  # Color颜色
            # 'lambda_Physical': 0.0,   # 未实现 - 物理约束损失 (预留)
            # 'lambda_NCE': 0.0         # 未实现 - NCE对比学习损失 (预留)
        }
        self.criterion = CombinedLoss(total_epochs=total_epochs, custom_weights=custom_weights).to(device)
        self.total_epochs = total_epochs
        print("[Loss] 动态权重配置 (早→中→后 或 固定):")
        for k,v in custom_weights.items():
            print(f"  {k}: {v}")
        
        # 优化器
        self.opt_G = optim.Adam(self.gen.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        self.opt_D = optim.Adam(self.disc.parameters(), lr=learning_rate/2, betas=(0.5, 0.999))
        
        # 最佳模型追踪
        self.best_loss_G = float('inf')
        self.best_mae = float('inf')
        
        # 初始化VGG19用于LPIPS计算
        try:
            from torchvision.models import vgg19, VGG19_Weights
            vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            self.vgg_extractor = nn.Sequential(*list(vgg.children())[:16]).to(device)
            self.vgg_extractor.eval()
            for param in self.vgg_extractor.parameters():
                param.requires_grad = False
            self.use_lpips = True
        except Exception as e:
            print(f"[Warning] VGG19加载失败，将跳过LPIPS计算: {e}")
            self.vgg_extractor = None
            self.use_lpips = False
        
        # 评估指标历史 (新四个指标: 对应四个损失函数)
        self.history_metrics = {
            'mae': [],        # 像素准确性 (对应L1损失)
            'fid': [],        # 生成真实感 (对应GAN损失)
            'lpips': [],      # 感知质量 (对应感知损失)
            'color_kl': []    # 颜色分布 (对应颜色损失)
        }
    
    def evaluate_metrics(self, train_loader, num_batches=5):
        """计算MAE、FID、LPIPS、Color-KL评估指标 (对应四个损失函数维度)"""
        self.gen.eval()
        
        mae_scores = []
        color_kl_scores = []
        lpips_scores = []
        
        fake_features_list = []
        real_features_list = []
        
        with torch.no_grad():
            batch_count = 0
            for batch in train_loader:
                if batch_count >= num_batches:
                    break
                
                gray_img = batch['gray'].to(device)
                rgb_img = batch['rgb'].to(device)
                
                # 生成图像
                fake_rgb = self.gen(gray_img)
                
                # 1. 计算MAE (像素准确性 - 对应L1损失)
                mae = calculate_mae(fake_rgb, rgb_img)
                mae_scores.append(mae)
                
                # 2. 计算Color-KL (颜色分布准确性 - 对应颜色损失)
                try:
                    kl = calculate_color_kl(fake_rgb, rgb_img)
                    color_kl_scores.append(kl)
                except Exception as e:
                    print(f"[Warning] Color-KL计算失败: {e}")
                
                # 3. 计算LPIPS (感知质量 - 对应感知损失)
                if self.use_lpips and fake_rgb.shape[0] > 0:
                    try:
                        lpips = calculate_lpips(fake_rgb, rgb_img, self.vgg_extractor, device)
                        lpips_scores.append(lpips)
                    except Exception as e:
                        print(f"[Warning] LPIPS计算失败: {e}")
                
                # 4. 提取特征用于FID (真实感 - 对应GAN损失)
                if self.vgg_extractor is not None:
                    try:
                        fake_norm = (fake_rgb + 1) / 2
                        real_norm = (rgb_img + 1) / 2
                        
                        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
                        vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
                        
                        fake_vgg = (fake_norm - vgg_mean) / vgg_std
                        real_vgg = (real_norm - vgg_mean) / vgg_std
                        
                        feat_fake = self.vgg_extractor(fake_vgg)
                        feat_real = self.vgg_extractor(real_vgg)
                        
                        feat_fake_avg = torch.mean(feat_fake, dim=[2, 3])
                        feat_real_avg = torch.mean(feat_real, dim=[2, 3])
                        
                        fake_features_list.append(feat_fake_avg.cpu().numpy())
                        real_features_list.append(feat_real_avg.cpu().numpy())
                    except Exception as e:
                        print(f"[Warning] FID特征提取失败: {e}")
                
                batch_count += 1
        
        # 计算平均指标
        metrics = {}
        
        # MAE (平均绝对误差)
        if mae_scores:
            metrics['mae'] = np.mean(mae_scores)
        else:
            metrics['mae'] = 0.0
        
        # Color-KL (颜色直方图KL散度)
        if color_kl_scores:
            metrics['color_kl'] = np.mean(color_kl_scores)
        else:
            metrics['color_kl'] = 0.0
        
        # LPIPS (感知距离)
        if lpips_scores:
            metrics['lpips'] = np.mean(lpips_scores)
        else:
            metrics['lpips'] = 0.0
        
        # FID (生成质量)
        if fake_features_list and real_features_list:
            fake_feat = np.concatenate(fake_features_list, axis=0)
            real_feat = np.concatenate(real_features_list, axis=0)
            metrics['fid'] = calculate_fid(fake_feat, real_feat)
        else:
            metrics['fid'] = 999.0
        
        self.gen.train()
        return metrics
    
    def save_sample_images(self, train_loader, epoch, num_samples=4):
        """保存样例图像对比"""
        self.gen.eval()
        
        # 随机采样
        sample_batch = next(iter(train_loader))
        gray_img = sample_batch['gray'][:num_samples].to(device)
        rgb_img = sample_batch['rgb'][:num_samples].to(device)
        
        with torch.no_grad():
            fake_rgb = self.gen(gray_img)
        
        # 转为numpy [0, 255]
        def tensor_to_img(tensor):
            img = tensor.cpu().detach().numpy()
            img = (img + 1) / 2 * 255
            img = np.clip(img, 0, 255).astype(np.uint8)
            return img
        
        # 绘制对比图
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        
        for i in range(num_samples):
            # 灰度输入 (squeeze掉通道维度)
            gray = tensor_to_img(gray_img[i]).squeeze()
            axes[i, 0].imshow(gray, cmap='gray')
            axes[i, 0].set_title('Input (Gray)')
            axes[i, 0].axis('off')
            
            # 生成彩色
            fake = tensor_to_img(fake_rgb[i]).transpose(1, 2, 0)
            axes[i, 1].imshow(fake)
            axes[i, 1].set_title('Generated (RGB)')
            axes[i, 1].axis('off')
            
            # 真实彩色
            real = tensor_to_img(rgb_img[i]).transpose(1, 2, 0)
            axes[i, 2].imshow(real)
            axes[i, 2].set_title('Ground Truth (RGB)')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        sample_path = os.path.join(save_dir, f'epoch_{epoch+1}_samples.png')
        plt.savefig(sample_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 样例图像已保存: {sample_path}")
        
        self.gen.train()
    
    def save_training_curves(self):
        """保存训练曲线 (损失 + 评估指标)"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        epochs = range(1, len(self.history['loss_G']) + 1)
        
        # 总损失
        axes[0, 0].plot(epochs, self.history['loss_G'], label='Generator', color='blue')
        axes[0, 0].plot(epochs, self.history['loss_D'], label='Discriminator', color='red')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 分项损失
        loss_keys = ['loss_L1', 'loss_GAN', 'loss_Perc', 'loss_Color']
        colors = ['green', 'orange', 'brown', 'pink']
        
        for idx, (key, color) in enumerate(zip(loss_keys, colors)):
            row = (idx + 1) // 3
            col = (idx + 1) % 3
            axes[row, col].plot(epochs, self.history[key], color=color)
            axes[row, col].set_title(key.replace('loss_', '').upper() + ' Loss')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].grid(True)
        
        plt.tight_layout()
        curve_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(curve_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 训练曲线已保存: {curve_path}")
        
        # 保存评估指标曲线 (新的四个指标)
        if any(self.history_metrics.values()):
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            metric_epochs = range(1, len(self.history_metrics['mae']) + 1)
            
            # MAE (像素准确性)
            axes[0, 0].plot(metric_epochs, self.history_metrics['mae'], color='blue', marker='o')
            axes[0, 0].set_title('MAE↓ (Pixel Accuracy)')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('MAE')
            axes[0, 0].grid(True)
            
            # FID (生成真实感)
            axes[0, 1].plot(metric_epochs, self.history_metrics['fid'], color='orange', marker='s')
            axes[0, 1].set_title('FID↓ (Generation Quality)')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('FID')
            axes[0, 1].grid(True)
            
            # LPIPS (感知质量)
            axes[1, 0].plot(metric_epochs, self.history_metrics['lpips'], color='green', marker='^')
            axes[1, 0].set_title('LPIPS↓ (Perceptual Quality)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('LPIPS')
            axes[1, 0].grid(True)
            
            # Color-KL (颜色分布)
            axes[1, 1].plot(metric_epochs, self.history_metrics['color_kl'], color='red', marker='d')
            axes[1, 1].set_title('Color-KL↓ (Color Distribution)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Color-KL')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            metric_curve_path = os.path.join(save_dir, 'evaluation_metrics_curves.png')
            plt.savefig(metric_curve_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"✓ 评估指标曲线已保存: {metric_curve_path}")
    
    def train_epoch(self, train_loader, epoch):
        """训练一个epoch (包含动态权重设置)"""
        self.criterion.set_epoch(epoch)
        self.gen.train()
        self.disc.train()
        
        loss_G_total = 0.0
        loss_D_total = 0.0
        loss_components = {k: 0.0 for k in ['L1', 'GAN', 'Perc', 'Color']}
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.total_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            gray_img = batch['gray'].to(device)
            rgb_img = batch['rgb'].to(device)
            
            # ===== 判别器更新 =====
            self.opt_D.zero_grad()
            
            fake_rgb = self.gen(gray_img)
            
            # 拼接为4通道输入
            real_input = torch.cat([gray_img, rgb_img], dim=1)
            fake_input = torch.cat([gray_img, fake_rgb.detach()], dim=1)
            
            pred_real = self.disc(real_input)
            pred_fake = self.disc(fake_input)
            
            loss_D = self.criterion.forward_discriminator(pred_real, pred_fake)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(self.disc.parameters(), max_norm=config.get('gradient_clip', 1.0))
            self.opt_D.step()
            
            # ===== 生成器更新 =====
            self.opt_G.zero_grad()
            
            fake_rgb = self.gen(gray_img)
            fake_input = torch.cat([gray_img, fake_rgb], dim=1)
            pred_fake = self.disc(fake_input)
            
            loss_G, loss_dict = self.criterion.forward_generator(
                fake=fake_rgb,
                real=rgb_img,
                pred_fake=pred_fake
            )
            
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.gen.parameters(), max_norm=config.get('gradient_clip', 1.0))
            self.opt_G.step()
            
            # 累积损失
            loss_G_total += loss_G.item()
            loss_D_total += loss_D.item()
            for k in loss_components.keys():
                loss_components[k] += loss_dict.get(k, 0.0)
            
            pbar.set_postfix({
                'L_G': f"{loss_G.item():.4f}",
                'L_D': f"{loss_D.item():.4f}",
                'L1': f"{loss_dict['L1']:.4f}",
                'GAN': f"{loss_dict['GAN']:.4f}"
            })
        
        # 计算平均值
        avg_loss_G = loss_G_total / len(train_loader)
        avg_loss_D = loss_D_total / len(train_loader)
        
        # 记录历史
        self.history['loss_G'].append(avg_loss_G)
        self.history['loss_D'].append(avg_loss_D)
        for k in loss_components.keys():
            self.history[f'loss_{k}'].append(loss_components[k] / len(train_loader))
        
        gap = abs(avg_loss_D - avg_loss_G)
        weights_current = loss_dict.get('weights', {})
        print(f"\n[Epoch {epoch+1}/{self.total_epochs}] Loss_G: {avg_loss_G:.4f} | Loss_D: {avg_loss_D:.4f} | Gap: {gap:.4f}")
        print("  Weights:")
        print("   L1={lambda_L1:.2f} GAN={lambda_GAN:.2f} Perc={lambda_Perc:.2f} Color={lambda_Color:.2f}".format(**weights_current))
        print("  Ratios: L1={ratio_L1:.1f}% Perc={ratio_Perc:.1f}% GAN={ratio_GAN:.1f}% Color={ratio_Color:.1f}%".format(**loss_dict))
        
        return avg_loss_G, avg_loss_D
    
    def train(self, gray_dir, rgb_dir):
        """完整训练流程"""
        print(f"\n{'='*70}")
        print(f"开始训练: {self.dataset_name}")
        print(f"轮次: {self.total_epochs}, 批大小: {batch_size}, 图像尺寸: ({img_width}, {img_height})")
        print(f"设备: {device}")
        print(f"{'='*70}\n")
        
        # 加载数据
        dataset = ImagePairDataset(gray_dir, rgb_dir, image_size=(img_width, img_height))
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        # 训练循环
        for epoch in range(num_epochs):
            avg_loss_G, avg_loss_D = self.train_epoch(train_loader, epoch)
            
            # 计算评估指标 (每个epoch都评估)
            print(f"\n[评估中...] 计算MAE↓ FID↓ LPIPS↓ Color-KL↓")
            metrics = self.evaluate_metrics(train_loader, num_batches=min(5, len(train_loader)))
            
            self.history_metrics['mae'].append(metrics['mae'])
            self.history_metrics['fid'].append(metrics['fid'])
            self.history_metrics['lpips'].append(metrics['lpips'])
            self.history_metrics['color_kl'].append(metrics['color_kl'])
            
            print(f"  MAE↓:       {metrics['mae']:.6f}")
            print(f"  FID↓:       {metrics['fid']:.4f}")
            print(f"  LPIPS↓:     {metrics['lpips']:.6f}")
            print(f"  Color-KL↓:  {metrics['color_kl']:.6f}")
                
            
            # 最佳模型 (基于MAE)
            if not hasattr(self, 'best_mae'):
                self.best_mae = float('inf')
            
            if metrics['mae'] < self.best_mae:
                self.best_mae = metrics['mae']
                os.makedirs("checkpoints", exist_ok=True)
                best_path = "checkpoints/gen_best_mae.pth"
                torch.save(self.gen.state_dict(), best_path)
                print(f"  ✓ 最佳MAE模型已保存: {best_path}")
            
            # 保存样例图像
            sample_interval = config.get('sample_interval', 5)
            if (epoch + 1) % sample_interval == 0 or (epoch + 1) == num_epochs:
                self.save_sample_images(train_loader, epoch)
            
            # 保存检查点
            save_interval = config.get('save_interval', 10)
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = f"checkpoints/gen_epoch_{epoch+1}.pth"
                torch.save(self.gen.state_dict(), ckpt_path)
                print(f"✓ 检查点已保存: {ckpt_path}")
            
            # 保存最佳模型
            if avg_loss_G < self.best_loss_G:
                self.best_loss_G = avg_loss_G
                best_path = "checkpoints/gen_best.pth"
                torch.save(self.gen.state_dict(), best_path)
                print(f"✓ 最佳模型已更新: {best_path} (Loss_G: {avg_loss_G:.4f})")
        
        # 保存最终模型
        final_path = "checkpoints/gen_final.pth"
        torch.save(self.gen.state_dict(), final_path)
        print(f"✓ 最终模型已保存: {final_path}")
        
        # 保存训练曲线
        self.save_training_curves()
        
        print(f"\n{'='*70}")
        print(f"✓ 训练完成!")
        print(f"  最佳Loss_G: {self.best_loss_G:.4f}")
        if self.best_mae < float('inf'):
            print(f"  最佳MAE↓: {self.best_mae:.6f}")
        print(f"  总迭代数: {len(train_loader) * self.total_epochs}")
        print(f"{'='*70}\n")


# ==================== 主函数 ====================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("单域训练脚本 - 整合预处理流水线")
    print("="*70 + "\n")
    
    # ===== Step 1: 路径和配置 =====
    data_root = config.get('data_root', 'datasets/Source_domain/EDS')
    images_folder_rel = config.get('images_folder', 'train/images')
    images_gray_folder_rel = config.get('images_gray_folder', 'train/images_gray')
    
    # 构建绝对路径
    images_folder = os.path.join(data_root, images_folder_rel)
    images_gray_folder = os.path.join(data_root, images_gray_folder_rel)
    
    # 目标尺寸
    target_width = config.get('img_width', 640)
    target_height = config.get('img_height', 640)
    target_size = (target_width, target_height)
    
    print(f"[Path] 彩色图像: {images_folder}")
    print(f"[Path] 灰度图像: {images_gray_folder}")
    print(f"[Config] 目标尺寸: {target_size}")
    
    # ===== Step 2: 智能检测 - 是否需要重新预处理 =====
    need_preprocess = True
    
    if os.path.exists(images_gray_folder):
        gray_files = [f for f in os.listdir(images_gray_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(gray_files) > 0:
            print(f"\n[检测] 发现 {len(gray_files)} 张灰度图像，验证尺寸...")
            
            # 抽样检查前5张图像的尺寸
            sample_size = min(5, len(gray_files))
            size_match_count = 0
            
            for i, gray_file in enumerate(gray_files[:sample_size]):
                gray_path = os.path.join(images_gray_folder, gray_file)
                try:
                    gray_img = cv2.imread(gray_path, cv2.IMREAD_GRAYSCALE)
                    if gray_img is not None:
                        h, w = gray_img.shape
                        if (w, h) == target_size:
                            size_match_count += 1
                            print(f"  ✓ {gray_file}: 尺寸 ({w}, {h}) 匹配")
                        else:
                            print(f"  ✗ {gray_file}: 尺寸 ({w}, {h}) ≠ 目标 {target_size}")
                    else:
                        print(f"  ✗ {gray_file}: 无法读取")
                except Exception as e:
                    print(f"  ✗ {gray_file}: 错误 - {e}")
            
            # 如果5张样本中全部匹配，则跳过预处理
            if size_match_count == sample_size:
                print(f"\n✓ 灰度数据集尺寸验证通过，跳过预处理")
                need_preprocess = False
            else:
                print(f"\n⚠ 灰度数据集尺寸不匹配或损坏，将重新预处理")
                need_preprocess = True
        else:
            print(f"\n[检测] 灰度图像文件夹为空，需要预处理")
            need_preprocess = True
    else:
        print(f"\n[检测] 灰度图像文件夹不存在，需要创建并预处理")
        need_preprocess = True
    
    # ===== Step 3: 预处理（如需要） =====
    if need_preprocess:
        print(f"\n{'='*70}")
        print("开始预处理灰度数据...")
        print(f"{'='*70}\n")
        
        pipeline = PreprocessPipeline('config.yaml')
        success = pipeline.check_and_generate(images_folder, images_gray_folder)
        
        if not success:
            print("\n❌ 预处理失败，无法继续训练")
            sys.exit(1)
    
    # ===== Step 4: 加载预训练权重（由config.yaml控制） =====
    pretrained_gen = config.get('pretrained_gen')
    pretrained_disc = config.get('pretrained_disc')
    
    # 检查路径有效性并显示加载信息
    if pretrained_gen:
        if os.path.exists(pretrained_gen):
            print(f"[配置] 生成器预训练权重: {pretrained_gen}")
        else:
            print(f"[警告] 生成器权重文件不存在: {pretrained_gen}，将重新初始化")
            pretrained_gen = None
    
    if pretrained_disc:
        if os.path.exists(pretrained_disc):
            print(f"[配置] 判别器预训练权重: {pretrained_disc}")
        else:
            print(f"[警告] 判别器权重文件不存在: {pretrained_disc}，将重新初始化")
            pretrained_disc = None
    
    # ===== Step 5: 开始训练 =====
    trainer = Trainer(dataset_name='EDS', pretrained_gen=pretrained_gen, pretrained_disc=pretrained_disc)
    trainer.train(images_gray_folder, images_folder)
