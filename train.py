import torch
import torch.nn as nn
from tqdm import tqdm
import os
import torch.nn.functional as F
# 请确保这些模块在你的路径中存在

from models.shadow import ShadowDetectionNet
from dataset.dataset import load_data


from sklearn.metrics import confusion_matrix
import numpy as np
import random

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_ber(outputs, labels):
    predictions = torch.sigmoid(outputs)
    predicted_classes = (predictions > 0.5).float()
    predicted_classes = predicted_classes.view(-1).cpu().numpy().astype(int)
    labels = labels.view(-1).cpu().numpy().astype(int)
    cm = confusion_matrix(labels, predicted_classes, labels=[0, 1])

    if cm.shape != (2, 2): return 100.0

    TP, TN, FP, FN = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
    Np, Nn = TP + FN, TN + FP
    if Np == 0 or Nn == 0: return 100.0
    BER = (1 - (1 / 2) * ((TP / Np) + (TN / Nn))) * 100
    return BER


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, alpha=None, reduction='mean'):

        if alpha == None:
            alpha = self.alpha

        eps = 1e-8
        # print(logits.shape, target.shape)
        logits = logits.flatten(1)
        target = target.flatten(1)
        pt = torch.sigmoid(logits)
        loss = -alpha * (1 - pt) ** self.gamma * target * torch.log(pt + eps) - (1 - alpha) * pt ** self.gamma * (
                    1 - target) * torch.log(1 - pt + eps)
        # print(loss.shape)
        # print('===> ', alpha, target.min().item(), target.max().item(), pt.min().item(), pt.max().item(), loss.min().item(), loss.max().item())


        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss



def train(model, train_loader, test_loader, device, num_epochs=100, lr=6e-5, save_path='checkpoints_baseline'):
    os.makedirs(save_path, exist_ok=True)

    # 1. 优化器配置 (AdamW 是 Transformer 的标准选择)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,  # 峰值学习率
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01  # 防止过拟合
    )

    # 2. [核心修改] 组合调度器: Warmup + Cosine Decay
    warmup_epochs = 5  # 前5轮热身

    # 阶段A: Warmup (从 lr*0.01 线性增加到 lr)
    scheduler_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )

    # 阶段B: Cosine Decay (从 lr 降到 1e-6)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6
    )

    # 串联调度器: 0-5轮用Warmup, 5-100轮用Cosine
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cosine],
        milestones=[warmup_epochs]
    )
    criterion = FocalLoss(8/9,2.0)
    
    model.to(device)
    best_ber = float('inf')

    print(f"Start training: Epochs={num_epochs}, Base LR={lr}")
    print(f"Strategy: Linear Warmup ({warmup_epochs} epochs) -> Cosine Decay")

    for epoch in range(num_epochs):
        model.train()
        shadow_loss=0.0
        total_samples = 0

        # 获取当前实际学习率用于打印
        current_lr = scheduler.get_last_lr()[0]

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [LR={current_lr:.2e}]', ncols=110)

        for images, labels, name in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()


            shadow_pred = model(images)


            loss_seg = criterion(shadow_pred, labels)
            total_loss = loss_seg

            total_loss.backward()
            optimizer.step()

            shadow_loss += loss_seg.item() * images.size(0)
            total_samples += images.size(0)



            pbar.set_postfix({'loss': f"{total_loss.item():.4f}"})

        # 每个 Epoch 结束后更新学习率
        scheduler.step()

        avg_shadow_loss = shadow_loss / total_samples
        print(f"shadow Loss: {avg_shadow_loss:.4f}  ")

        # --- 验证过程 ---
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, name in tqdm(test_loader, desc='Validating', ncols=100, leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                shadow_pred= model(inputs)

                orig_h, orig_w = labels.shape[-2], labels.shape[-1]

                # 使用双线性插值恢复尺寸
                shadow_pred = F.interpolate(
                    shadow_pred,
                    size=(orig_h, orig_w),
                    mode='bilinear',
                    align_corners=False
                )

                all_preds.append(shadow_pred.view(-1).detach().cpu().half())
                all_labels.append(labels.view(-1).detach().cpu().half())


        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_ber = calculate_ber(all_preds, all_labels)

        print(f"Validation BER: {val_ber:.4f}")

        # 保存检查点
        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, f'epoch_{epoch + 1}.pth'))

        if val_ber < best_ber:
            best_ber = val_ber
            torch.save(model.state_dict(), os.path.join(save_path, 'best.pth'))
            print(f"🔥 Best model saved! BER: {val_ber:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 请根据实际情况修改路径
    train_image_dir = '../datasets/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages'
    train_label_dir = '../datasets/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks'
    test_image_dir = '../datasets/SBU-shadow/SBU-Test/ShadowImages'
    test_label_dir = '../datasets/SBU-shadow/SBU-Test/ShadowMasks'


    train_loader, test_loader = load_data(
        train_image_dir=train_image_dir,
        train_label_dir=train_label_dir,
        test_image_dir=test_image_dir,
        test_label_dir=test_label_dir,
        batch_size=8
    )

    model = ShadowDetectionNet()
    #model = nn.DataParallel(model, device_ids=[0,1])
    # 使用调优后的参数
    train(model,
          train_loader,
          test_loader,
          device,
          num_epochs=50,
          lr=6e-5,  # 设定为 6e-5，配合 Warmup 使用效果最佳
          save_path='sbu')


if __name__ == "__main__":
    set_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    main()
