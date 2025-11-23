"""
ZSXT: 多任务组合损失模块 (改进版)

参照 LOSS_FUNCTION_DESIGN / LOSS_FUNCTION_VISUALIZATION 文档，将 5 核心组件构建为医学优先级：
    1. L1Loss          像素对齐基础 (主导70-75%, 早期强→后期弱)
    2. GANLoss         视觉真实感 (平衡5-10%, 早期弱→后期强)
    3. PerceptualLoss  语义/医学特征 (辅助15-20%, 逐步增强)
    4. ColorLoss       全局颜色物理分布 (轻量2-3%, 固定约束)
    5. (可选)PhysicalLoss 灰度守恒约束 (默认关闭)

三阶段动态权重策略 (progress ∈ [0,1]):
    progress < 0.30        早期 (L1强势保证收敛)
    0.30 <= p < 0.70       中期 (平衡过渡)
    p >= 0.70              后期 (对抗增强真实感)

权重函数表 (早→中→后):
    λ_L1:       10.0 → 8.0 → 6.0   (主导递减: 保留像素完整性,逐步释放)
    λ_GAN:      0.1 → 0.2 → 0.3    (平衡递增: 防止Mode Collapse,后期增强)
    λ_Perc:     3.0 → 4.0 → 5.0    (辅助递增: 逐步精化特征)
    λ_Color:    0.3 固定            (轻量约束: 防止颜色偏移)

目标监控指标:
    Loss_G ≈ 1.00 → 0.90 → <0.85
    Gap = |Loss_D - Loss_G| < 0.15 稳定对抗
    L1占比 > 70% 全程保证
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import VGG19_Weights
from typing import Tuple, List, Dict


##################################
# 1. L1损失 (λ_L1 = 10)
##################################

class L1Loss(nn.Module):
    """L1像素级损失 - 核心创新点"""
    def __init__(self, reduction='mean'):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(reduction=reduction)
    
    def forward(self, fake: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        论文方程(9):
        L_L1 = ||G(I_aug) - I_s||_1
        
        与CycleGAN循环一致性的核心差异:
        - 循环: 多对多映射，全局平均导致混淆
        - 像素L1: 单射映射，每灰度→唯一源色，保护微细结构
        
        Args:
            fake: 生成的彩色图 [B, 3, H, W]
            target: 目标图(源域原始彩色) [B, 3, H, W]
        
        Returns:
            loss: 标量损失
        """
        return self.loss(fake, target)


##################################
# 2. GAN损失 (λ_GAN = 1)
##################################

class GANLoss(nn.Module):
    """GAN对抗损失 - PatchGAN判别器"""
    def __init__(self, use_lsgan=False, reduction='mean'):
        super(GANLoss, self).__init__()
        self.use_lsgan = use_lsgan
        self.reduction = reduction
        
        if use_lsgan:
            # LS-GAN: 更稳定的训练
            self.loss_real = nn.MSELoss(reduction=reduction)
            self.loss_fake = nn.MSELoss(reduction=reduction)
        else:
            # Binary Cross Entropy: 标准GAN
            self.loss = nn.BCEWithLogitsLoss(reduction=reduction)
    
    def forward_discriminator(self, pred_real: torch.Tensor, 
                            pred_fake: torch.Tensor) -> torch.Tensor:
        """
        判别器损失
        
        Args:
            pred_real: 真实数据判别结果
            pred_fake: 生成数据判别结果
        
        Returns:
            D_loss
        """
        if self.use_lsgan:
            loss_real = self.loss_real(pred_real, torch.ones_like(pred_real))
            loss_fake = self.loss_fake(pred_fake, torch.zeros_like(pred_fake))
            return (loss_real + loss_fake) * 0.5
        else:
            loss_real = self.loss(pred_real, torch.ones_like(pred_real))
            loss_fake = self.loss(pred_fake, torch.zeros_like(pred_fake))
            return (loss_real + loss_fake) * 0.5
    
    def forward_generator(self, pred_fake: torch.Tensor) -> torch.Tensor:
        """
        生成器GAN损失 (带缩放因子保证与L1量级一致)
        
        Args:
            pred_fake: 生成数据的判别结果
        
        Returns:
            G_loss (缩放后,使权重能真正起作用)
        """
        if self.use_lsgan:
            loss = self.loss_fake(pred_fake, torch.ones_like(pred_fake))
        else:
            loss = self.loss(pred_fake, torch.ones_like(pred_fake))
        # 缩放因子: 使GAN损失数值范围与L1/Perceptual一致 (0.05-0.2)
        return loss * 0.1


##################################
# 3. 感知损失 (动态权重核心组件)
##################################

class PerceptualLoss(nn.Module):
    """VGG19 感知损失 (特征层 L1)

    - 失败回退: 若预训练权重不可用则使用随机初始化并警告, 防止训练中断
    - 输入范围: 期望 [-1,1] → 归一化到 torchvision 标准 ImageNet 分布
    """
    def __init__(self, use_layers: List[int] = None):
        super().__init__()
        if use_layers is None:
            # 取前 36 层(包含 relu_5_1 前)的中间特征用于多尺度感知
            use_layers = [4, 9, 18, 27, 34]  # relu_2_1, relu_3_1, relu_4_1, relu_5_1 等近似
        self.use_layers = set(use_layers)
        try:
            vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
            self.failed = False
        except Exception:
            vgg = models.vgg19(weights=None).features
            self.failed = True
            print("[PerceptualLoss] ⚠️ 预训练权重加载失败，使用随机初始化，效果会受影响。")
        # 截取需要的层, 保留整个顺序但只提取指定输出
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()
        # 标准化参数 (ImageNet)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # 输入是 [-1,1], 转 [0,1] 再标准化
        x = (x + 1.0) * 0.5
        return (x - self.mean) / self.std

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        fake_n = self._normalize(fake)
        real_n = self._normalize(real)
        feats_fake = []
        feats_real = []
        for idx, layer in enumerate(self.vgg):
            fake_n = layer(fake_n)
            real_n = layer(real_n)
            if idx in self.use_layers:
                feats_fake.append(fake_n)
                feats_real.append(real_n)
        loss = 0.0
        for f_fake, f_real in zip(feats_fake, feats_real):
            loss = loss + self.criterion(f_fake, f_real)
        
        # 平均多层特征
        loss = loss / max(len(feats_fake), 1)
        
        # 关键: 感知损失数值缩放因子
        # VGG特征的L1范围通常是0.1-1之间，与像素L1一致
        # 这里使用0.1作为缩放因子，使权重系数能够真正起作用
        loss = loss * 0.1
        
        return loss


##################################
# 4. 颜色分布损失 (KL 直方图)
##################################

class ColorLoss(nn.Module):
    """颜色分布损失 - 轻量全局颜色物理约束

    对 R/G/B 分别统计 64-bin 直方图, 使用 KL(P_real || P_fake).
    """
    def __init__(self, bins: int = 64):
        super().__init__()
        self.bins = bins

    def _hist(self, x: torch.Tensor) -> torch.Tensor:
        # x:[B,3,H,W] in [-1,1] → [0,1]
        x = (x + 1.0) * 0.5
        B, C, H, W = x.shape
        hists = []
        for c in range(C):
            channel = x[:, c].contiguous().view(B, -1)
            # 逐 batch 统计再平均 (减少内存)
            batch_hist = []
            for b in range(B):
                hist = torch.histc(channel[b], bins=self.bins, min=0.0, max=1.0)
                hist = hist / (hist.sum() + 1e-8)
                batch_hist.append(hist.unsqueeze(0))
            batch_hist = torch.cat(batch_hist, dim=0).mean(dim=0)  # [bins]
            hists.append(batch_hist)
        return torch.cat(hists, dim=0)  # [3*bins]

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        pf = torch.clamp(self._hist(fake), 1e-8, 1.0)
        pr = torch.clamp(self._hist(real), 1e-8, 1.0)
        # KL(pr || pf)
        return (pr * (pr.log() - pf.log())).sum() / pr.numel()


##################################
# (可选) 物理灰度约束 / NCE - 兼容旧代码
##################################

class PhysicalConstraintLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('w', torch.tensor([0.2989,0.5870,0.1140]).view(1,3,1,1))
        self.l1 = nn.L1Loss()
    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        return self.l1((fake*self.w).sum(1,keepdim=True), (real*self.w).sum(1,keepdim=True))


##################################
# 组合损失 (动态权重)
##################################

class CombinedLoss(nn.Module):
    """多任务组合损失 (动态权重)

    total = Σ λ_i(progress) * L_i
    progress = current_epoch / total_epochs

    支持附加: enable_physical / enable_nce (默认关闭)
    """
    def __init__(self, total_epochs: int = 50,
                 enable_physical: bool = False,
                 enable_nce: bool = False,
                 custom_weights: Dict = None):
        super().__init__()
        self.total_epochs = max(total_epochs, 1)
        self.enable_physical = enable_physical
        self.enable_nce = enable_nce
        # 基础损失实例
        self.l1_loss = L1Loss()
        self.gan_loss = GANLoss(use_lsgan=False)
        self.perc_loss = PerceptualLoss()
        self.color_loss = ColorLoss()
        self.physical_loss = PhysicalConstraintLoss() if enable_physical else None
        # 简化 NCE (如果启用以保持兼容)
        self.nce = nn.MSELoss() if enable_nce else None

        # 提供自定义覆盖 (dict: key -> [early,mid,late] or scalar)
        # 若 custom_weights 为 None, 使用代码默认值(应由 train.py 从 config.yaml 传入)
        self.weight_cfg = custom_weights or {
            'lambda_L1': [1.5, 1.5, 1.5],       # L1稳定(30%)
            'lambda_GAN': [0.5, 1.0, 0.8],      # GAN平衡(20%)
            'lambda_Perc': [1.5, 1.8, 2.0],     # Perceptual特征(20%)
            'lambda_Color': 30.0,                # Color平衡(20%)
            'lambda_Physical': 0.0 if not enable_physical else [0.5, 0.5, 0.5],
            'lambda_NCE': 0.0 if not enable_nce else [1.0, 1.0, 1.0]
        }
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch

    def _interp(self, trio):
        if not isinstance(trio, (list, tuple)):
            return float(trio)
        early, mid, late = trio
        p = self.current_epoch / self.total_epochs
        if p < 0.30:
            return early
        if p < 0.70:
            # 线性插值 early→mid
            alpha = (p - 0.30) / 0.40
            return early + (mid - early) * alpha
        # 线性插值 mid→late
        alpha = (p - 0.70) / 0.30
        return mid + (late - mid) * alpha

    def _weights(self) -> Dict[str,float]:
        return {k: self._interp(v) for k, v in self.weight_cfg.items()}

    def forward_generator(self,
                          fake: torch.Tensor,
                          real: torch.Tensor,
                          pred_fake: torch.Tensor = None,
                          feat_fake: torch.Tensor = None,
                          feat_real: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        w = self._weights()
        L1 = self.l1_loss(fake, real)
        L_GAN = self.gan_loss.forward_generator(pred_fake) if pred_fake is not None else fake.new_tensor(0.0)
        L_Perc = self.perc_loss(fake, real)
        L_Color = self.color_loss(fake, real)
        L_Physical = self.physical_loss(fake, real) if self.physical_loss else fake.new_tensor(0.0)
        if self.nce:
            if feat_fake is None: feat_fake = fake
            if feat_real is None: feat_real = real
            L_NCE = self.nce(feat_fake, feat_real)
        else:
            L_NCE = fake.new_tensor(0.0)
        # 总损失 (仅使用四个核心损失函数)
        total = (w['lambda_L1']*L1 + w['lambda_GAN']*L_GAN + w['lambda_Perc']*L_Perc +
                  w['lambda_Color']*L_Color)
        # 占比 (仅核心组件)
        denom = total.detach().item() + 1e-8
        ratios = {
            'ratio_L1': (w['lambda_L1']*L1.detach().item())/denom*100,
            'ratio_Perc': (w['lambda_Perc']*L_Perc.detach().item())/denom*100,
            'ratio_GAN': (w['lambda_GAN']*L_GAN.detach().item())/denom*100,
            'ratio_Color': (w['lambda_Color']*L_Color.detach().item())/denom*100,
        }
        loss_dict = {
            'L1': L1.item(), 'GAN': L_GAN.item(), 'Perc': L_Perc.item(), 'Color': L_Color.item(),
            'Physical': L_Physical.item(), 'NCE': L_NCE.item(), 'Total': total.item(),
            **ratios,
            'epoch': self.current_epoch,
            'weights': {k: round(v,4) for k,v in w.items()}
        }
        return total, loss_dict

    def forward(self, pred_color: torch.Tensor, target_color: torch.Tensor, disc_output: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        return self.forward_generator(fake=pred_color, real=target_color, pred_fake=disc_output,
                                      feat_fake=pred_color, feat_real=target_color)

    def forward_discriminator(self, pred_real: torch.Tensor, pred_fake: torch.Tensor) -> torch.Tensor:
        return self.gan_loss.forward_discriminator(pred_real, pred_fake)


