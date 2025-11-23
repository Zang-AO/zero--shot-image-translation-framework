import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

##################################
# 定义 U-Net 生成器
##################################

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_down=True, use_batchnorm=True):
        super(UNetBlock, self).__init__()
        if is_down:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)

        self.batchnorm = nn.BatchNorm2d(out_channels) if use_batchnorm else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.batchnorm(x)
        x = self.relu(x)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=3):
        super(GeneratorUNet, self).__init__()

        # 编码器部分
        self.down1 = UNetBlock(input_channels, 64, is_down=True, use_batchnorm=False)
        self.down2 = UNetBlock(64, 128, is_down=True)
        self.down3 = UNetBlock(128, 256, is_down=True)
        self.down4 = UNetBlock(256, 512, is_down=True)
        self.down5 = UNetBlock(512, 512, is_down=True)
        self.down6 = UNetBlock(512, 512, is_down=True)
        self.down7 = UNetBlock(512, 512, is_down=True)
        self.down8 = UNetBlock(512, 512, is_down=True)

        # 解码器部分
        self.up1 = UNetBlock(512, 512, is_down=False)
        self.up2 = UNetBlock(1024, 512, is_down=False)
        self.up3 = UNetBlock(1024, 512, is_down=False)
        self.up4 = UNetBlock(1024, 512, is_down=False)
        self.up5 = UNetBlock(1024, 256, is_down=False)
        self.up6 = UNetBlock(512, 128, is_down=False)
        self.up7 = UNetBlock(256, 64, is_down=False)

        # 最后输出3通道的彩色图像
        self.final = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, return_features=False):
        # 编码器部分
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # 解码器部分
        u1 = self.up1(d8)
        u1 = self._crop_and_concat(u1, d7)
        u2 = self.up2(u1)
        u2 = self._crop_and_concat(u2, d6)
        u3 = self.up3(u2)
        u3 = self._crop_and_concat(u3, d5)
        u4 = self.up4(u3)
        u4 = self._crop_and_concat(u4, d4)
        u5 = self.up5(u4)
        u5 = self._crop_and_concat(u5, d3)
        u6 = self.up6(u5)
        u6 = self._crop_and_concat(u6, d2)
        u7 = self.up7(u6)

        output = torch.tanh(self.final(torch.cat([u7, d1], dim=1)))

        if return_features:
            # 返回特征图（用于 PatchNCE）
            return [d1, d2, d3, d4, d5, d6, d7, d8]

        return output

    def _crop_and_concat(self, upsampled, bypass):
        _, _, h, w = bypass.size()
        upsampled = F.interpolate(upsampled, size=(h, w), mode='bilinear', align_corners=True)
        return torch.cat([upsampled, bypass], dim=1)

##################################
# PatchGAN 判别器
##################################

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=4, ndf=64):
        super(PatchGANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)  # 输出真假预测
        )

    def forward(self, x):
        return self.model(x)

##################################
# PatchNCE Loss (对比损失)
##################################

class PatchNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(PatchNCELoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, feat_q, feat_k, num_patches=256, num_negative=1024):
        batch_size, feat_dim, h, w = feat_q.size()

        # Flatten and normalize features
        feat_q = feat_q.view(batch_size, feat_dim, -1)  # [batch_size, feat_dim, h * w]
        feat_k = feat_k.view(batch_size, feat_dim, -1)
        feat_q = F.normalize(feat_q, dim=1)
        feat_k = F.normalize(feat_k, dim=1)

        # Sampling
        num_patches = min(num_patches, h * w)
        num_negative = min(num_negative, h * w)

        pos_indices = torch.randint(0, h * w, (batch_size, num_patches), device=feat_q.device)
        neg_indices = torch.randint(0, h * w, (batch_size, num_negative), device=feat_q.device)

        # Positive samples
        feat_q_sampled = torch.gather(feat_q, 2, pos_indices.unsqueeze(1).expand(-1, feat_dim, -1))
        feat_k_pos_sampled = torch.gather(feat_k, 2, pos_indices.unsqueeze(1).expand(-1, feat_dim, -1))

        # Negative samples
        feat_k_neg_sampled = torch.gather(feat_k, 2, neg_indices.unsqueeze(1).expand(-1, feat_dim, -1))

        # Compute similarities
        l_pos = torch.sum(feat_q_sampled * feat_k_pos_sampled, dim=1, keepdim=True)  # [batch_size, 1, num_patches]
        l_neg = torch.bmm(feat_q_sampled.permute(0, 2, 1), feat_k_neg_sampled)  # [batch_size, num_patches, num_negative]

        # Combine logits and targets
        logits = torch.cat([l_pos.permute(0, 2, 1), l_neg], dim=-1)  # [batch_size, num_patches, 1 + num_negative]
        targets = torch.zeros(batch_size * num_patches, dtype=torch.long, device=feat_q.device)

        # Reshape for loss computation
        logits = logits.view(-1, logits.size(-1))

        # Apply temperature scaling
        logits /= self.temperature

        # Compute loss
        loss = self.cross_entropy_loss(logits, targets)

        return loss



##################################
# 感知损失 (VGG19)
##################################

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(weights='VGG19_Weights.IMAGENET1K_V1').features
        self.slice1 = nn.Sequential(*vgg[:4])   # Conv1_1 to ReLU1_1
        self.slice2 = nn.Sequential(*vgg[4:9])  # Conv2_1 to ReLU2_1
        self.slice3 = nn.Sequential(*vgg[9:16]) # Conv3_1 to ReLU3_1

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        # Ensure inputs are normalized appropriately for VGG19
        # VGG19 expects images normalized with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # If your images are normalized differently, adjust accordingly

        # Pass through first slice
        h_x = self.slice1(x)
        h_y = self.slice1(y)
        loss = F.mse_loss(h_x, h_y)

        # Pass through second slice
        h_x = self.slice2(h_x)  # Use output from previous slice
        h_y = self.slice2(h_y)
        loss += F.mse_loss(h_x, h_y)

        # Pass through third slice
        h_x = self.slice3(h_x)
        h_y = self.slice3(h_y)
        loss += F.mse_loss(h_x, h_y)

        return loss


##################################
# 权重初始化
##################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
