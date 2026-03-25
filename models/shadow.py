from timm import create_model
from safetensors.torch import load_file
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random






class DinoEncoder(nn.Module):
    def __init__(self,extract_layers=[2, 5, 8, 11]):
        super().__init__()
        self.model = create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False, img_size=416)
        state_dict = load_file(
            "/root/autodl-tmp/code/models/dinov3.safetensors")
        self.model.load_state_dict(state_dict,strict=False)
        self.extract_layers = extract_layers
        self.num_prefix_tokens = self.model.num_prefix_tokens

        # 存储中间特征
        self.features = []
        self.hooks = []

        # 注册 forward hooks
        self._register_hooks()

    def _register_hooks(self):
        """注册 hook 来提取中间层特征"""

        def get_activation(name):
            def hook(module, input, output):
                # output 是 [B, 677, 384] (包含 CLS token)
                # 去掉 CLS token 并 reshape
                feat = output[:, self.num_prefix_tokens:]  # [B, 676, 384]

                B, N, C = feat.shape
                H = W = int(math.sqrt(N))
                feat = feat.transpose(1, 2).reshape(B, C, H, W)

                self.features.append(feat)

            return hook

        # 为指定层注册 hook
        for i in self.extract_layers:
            hook = self.model.blocks[i].register_forward_hook(
                get_activation(f'block_{i}')
            )
            self.hooks.append(hook)

    def forward(self, x):
        """
        Args:
            x: [B, 3, 416, 416]

        Returns:
            features: List of [B, 384, 26, 26]
        """
        # 清空之前的特征
        self.features = []

        # 直接调用原始的 forward_features
        # hooks 会自动捕获中间层输出
        _ = self.model.forward_features(x)

        # 返回捕获的特征
        return self.features.copy()


class DINODecoder(nn.Module):
    """
    改进版解码器，针对阴影检测优化：
    1. 渐进式上采样，保留边界细节
    2. 注意力机制增强关键区域
    3. 残差连接保留特征
    """

    def __init__(self, in_dim=384, out_dim=1):
        super().__init__()

        self.fusion1 = nn.Sequential(
            nn.Conv2d(384 + 384, 384, 3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )

        # ===== 渐进式上采样路径 =====
        # Stage 1: 26x26 -> 52x52
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        self.fusion2 = nn.Sequential(
            nn.Conv2d(192 + 192+ 192, 192, 3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )


        # Stage 2: 52x52 -> 104x104
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        self.fusion3 = nn.Sequential(
            nn.Conv2d(96 + 96+ 96, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )

        # Stage 3: 104x104 -> 208x208
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.fusion4 = nn.Sequential(
            nn.Conv2d(48 + 48+ 48, 48, 3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # ===== 最终预测头 =====
        self.final = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, out_dim, 1)
        )

    def forward(self, feats,diff_feat, target_size):
        # x: [B, 384, 26, 26]

        x=torch.cat([feats[-1], diff_feat[-1]], dim=1)
        x=self.fusion1(x)

        x = self.up1(x)
        x = torch.cat([x,feats[-2], diff_feat[-2]], dim=1)
        x = self.fusion2(x)

        x = self.up2(x)
        x = torch.cat([x,feats[-3], diff_feat[-3]], dim=1)
        x = self.fusion3(x)

        x = self.up3(x)
        x = torch.cat([x,feats[-4], diff_feat[-4]], dim=1)
        x = self.fusion4(x)


        # 4. 最终预测
        x = self.final(x)  # [B, 1, 208, 208]

        # 5. 最后一次上采样到目标尺寸
        if x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x


class MultiLayerDiffWithUpsample(nn.Module):
    """
    改进的多层 Diff：每层 diff 都有独立的上采样路径
    """

    def __init__(self, dim=384):
        super().__init__()

        self.upsample_layer2 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, 2, stride=2),
            nn.BatchNorm2d(dim//2), nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim//2, 3, padding=1),
            nn.BatchNorm2d(dim//2), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(dim//2, dim//4, 2, stride=2),
            nn.BatchNorm2d(dim//4), nn.ReLU(inplace=True),
            nn.Conv2d(dim//4, dim//4, 3, padding=1),
            nn.BatchNorm2d(dim//4), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(dim//4, dim//8, 2, stride=2),
            nn.BatchNorm2d(dim//8), nn.ReLU(inplace=True),
            nn.Conv2d(dim//8, dim//8, 3, padding=1),
            nn.BatchNorm2d(dim//8), nn.ReLU(inplace=True)
        )


        # Layer 5: 26 → 52 → 104 (2次上采样)
        self.upsample_layer5 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, 2, stride=2),
            nn.BatchNorm2d(dim//2), nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim//2, 3, padding=1),
            nn.BatchNorm2d(dim//2), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(dim//2, dim//4, 2, stride=2),
            nn.BatchNorm2d(dim//4), nn.ReLU(inplace=True),
            nn.Conv2d(dim//4, dim//4, 3, padding=1),
            nn.BatchNorm2d(dim//4), nn.ReLU(inplace=True)
        )

        # Layer 8: 26 → 52 (1次上采样)
        self.upsample_layer8 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, 2, stride=2),
            nn.BatchNorm2d(dim//2), nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim//2, 3, padding=1),
            nn.BatchNorm2d(dim//2), nn.ReLU(inplace=True)
        )

        # Layer 11: 不上采样 (已经在 decoder input)

    def forward(self, diff_feats):
        """
        Returns:
            diff_dict: {
                'layer2':  [B, 384, 208, 208],  ← 已上采样
                'layer5':  [B, 384, 104, 104],  ← 已上采样
                'layer8':  [B, 384, 52, 52],    ← 已上采样
                'layer11': [B, 384, 26, 26]     ← 原始尺寸
            }
        """
        diff_feats[0]=self.upsample_layer2(diff_feats[0])
        diff_feats[1]=self.upsample_layer5(diff_feats[1])
        diff_feats[2]=self.upsample_layer8(diff_feats[2])

        return diff_feats


class VarianceSpatialSmoothing(nn.Module):
    """
    Variance Spatial Refinement

    核心思想:
    - 阴影: variance 在空间上连续 → 原始 variance 可信
    - 噪声: variance 空间跳变 → 用平滑版本替代
    """


    def __init__(self, dim=384):
        super().__init__()

        # 平滑算子（低频）
        self.smoother = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        # 置信度映射（把 smoothness 映射到 [0,1]）
        self.confidence = nn.Sequential(
            nn.Conv2d(1, 1, 3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, variance):
        """
        Args:
            variance: [B, C, H, W]
        Returns:
            refined_variance: [B, C, H, W]
        """

        # 1. 平滑版本（低频、连续）
        smoothed = self.smoother(variance)

        # 2. 空间不连续度（高频）
        diff = torch.abs(variance - smoothed)

        # 3. 空间连续性评分（小 diff → 高可信）
        #   [B,1,H,W]
        smoothness = diff.mean(dim=1, keepdim=True)

        # 4. 连续性 → 置信度 (0~1)
        #   高置信度 → 信原始
        weight = self.confidence(-smoothness)

        # 5. 物理一致的加权融合
        refined = weight * variance + (1.0 - weight) * smoothed

        return refined


# ---------------------- Shadow Detection Net ----------------------
class ShadowDetectionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DinoEncoder()
        self.diff_up1=MultiLayerDiffWithUpsample()
        self.diff_up2=MultiLayerDiffWithUpsample()
        self.decoder = DINODecoder()
        self.gamma_list = [0.4, 0.6, 0.8, 1.0]
        self.spatial_smoothing = nn.ModuleList([
            VarianceSpatialSmoothing(dim=384) for _ in range(4)
        ])



    def forward(self, rgb):
        # 原始特征（作为主干）
        feats = self.encoder(rgb)

        # ==============================
        #  Illumination Sensitivity Modeling
        # ==============================

        feats_gamma = []

        for g in self.gamma_list:
            rgb_g = (rgb + 1e-6).pow(g)
            feats_gamma.append(self.encoder(rgb_g))

        diff_feat = []

        for i in range(4):
            # [K, B, C, H, W]
            stack = torch.stack([fg[i] for fg in feats_gamma], dim=0)

            # ⭐ 光照不稳定性（比 diff 更物理）
            instability = stack.var(dim=0)

            smooth_var = self.spatial_smoothing[i](instability)

            diff_feat.append(smooth_var)

        # 多层上采样
        diff_feat = self.diff_up1(diff_feat)
        feats = self.diff_up2(feats)

        # 主干 + illumination instability
        mask = self.decoder(feats, diff_feat, (416, 416))



        return mask




# ---------------------- Test ----------------------
if __name__ == '__main__':
    model = ShadowDetectionNet()
    # 总参数量
    total = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total / 1e6:.2f} M")

    # 可训练参数量
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数量: {trainable / 1e6:.2f} M")
    x1 = torch.randn(1, 3, 416, 416)
    y = model(x1)
    print(y.shape)  # [1,1,256,256]
