from models.shadow import ShadowDetectionNet
from dataset.dataset import load_data
from torchvision.utils import save_image
import torch
import torch.nn.functional as F  # 引入 functional 用于 resize
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix



def calculate_ber(outputs, labels):
    """
    计算 BER 指标
    outputs: 展平后的预测概率向量 [N_pixels]
    labels: 展平后的标签向量 [N_pixels]
    """
    # 已经是展平的向量了，直接二值化
    # outputs 已经是 sigmoid 后的概率，或者在这里 sigmoid
    # 注意：在 test 函数里我们已经 sigmoid 过了，这里直接判断
    predicted_classes = (outputs > 0.5).float()

    # 转 numpy
    predicted_classes = predicted_classes.cpu().numpy().astype(int)
    labels = labels.cpu().numpy().astype(int)

    # 计算混淆矩阵
    cm = confusion_matrix(labels, predicted_classes, labels=[0, 1])
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    Np = TP + FN
    Nn = TN + FP

    if Np == 0 or Nn == 0:
        return 100.0

    # 阴影区域错误率
    shadow_ber = (FN / Np) * 100
    # 非阴影区域错误率
    non_shadow_ber = (FP / Nn) * 100
    # 全局BER
    BER = 0.5 * (shadow_ber + non_shadow_ber)
    return BER, shadow_ber, non_shadow_ber

def test(model, test_loader, device, save_dir=None):
    """测试函数"""
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    # 用列表存储展平后的 1D 向量
    all_preds_flat = []
    all_labels_flat = []

    with torch.no_grad():
        for images, labels, name in tqdm(test_loader, desc="Testing", ncols=100):
            images = images.to(device)
            # labels 不需要 to(device) 用于计算，除非想在 GPU 上算混淆矩阵
            # labels 现在的尺寸是原图尺寸 [1, 1, H_orig, W_orig]

            # 1. 模型推理 (输出固定尺寸 416x416)
            shadow_pred = model(images)  # [1, 1, 416, 416]

            # 2. [核心] 将预测结果 Resize 回原图尺寸
            # labels.shape[-2:] 获取原图的 (H, W)
            orig_h, orig_w = labels.shape[-2], labels.shape[-1]

            # 使用双线性插值恢复尺寸
            shadow_pred_resized = F.interpolate(
                shadow_pred,
                size=(orig_h, orig_w),
                mode='bilinear',
                align_corners=False
            )

            # 3. Sigmoid 归一化
            preds = torch.sigmoid(shadow_pred_resized)
            preds = (preds > 0.5).float()

            # 4. 收集数据用于计算 BER
            # 因为每张图尺寸不一样，不能直接 stack 图像
            # 我们将每张图展平为 1D 向量，然后存入列表
            all_preds_flat.append(preds.view(-1).cpu())
            all_labels_flat.append(labels.view(-1).cpu())

            # 5. 保存预测图
            base = os.path.splitext(name[0])[0].split('/')[-1]
            save_path = os.path.join(save_dir, f"{base}.png")
            save_image(preds, save_path)

    # ---- 计算 BER ----
    print("Calculating Global BER...")
    # 将所有像素拼接成一个巨大的 1D 向量
    all_preds_flat = torch.cat(all_preds_flat)
    all_labels_flat = torch.cat(all_labels_flat)

    ber, shadow_ber, non_shadow_ber = calculate_ber(all_preds_flat, all_labels_flat)
    print(f"BER:            {ber:.4f}%")
    print(f"Shadow BER:     {shadow_ber:.4f}%")
    print(f"Non-Shadow BER: {non_shadow_ber:.4f}%")
    return ber, shadow_ber, non_shadow_ber

# -------------------------------
# 主函数
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_image_dir = 'datasets/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowImages'
    train_label_dir = 'datasets/SBU-shadow/SBUTrain4KRecoveredSmall/ShadowMasks'
    test_image_dir = 'datasets/SBU-shadow/SBU-Test/ShadowImages'
    test_label_dir = 'datasets/SBU-shadow/SBU-Test/ShadowMasks'



    # 加载数据
    # 注意：load_data 内部已经强制设置 test batch_size=1
    train_loader, test_loader = load_data(
        train_image_dir=train_image_dir,
        train_label_dir=train_label_dir,
        test_image_dir=test_image_dir,
        test_label_dir=test_label_dir,
        batch_size=1  # 这个只影响训练集
    )

    # 初始化模型
    model = ShadowDetectionNet()
    weight_path = 'weights/best.pth'
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: Weights not found at {weight_path}")

    model.to(device)

    # 测试
    save_dir = 'results'  # 保存到新目录以区分
    test(model, test_loader, device, save_dir=save_dir)


if __name__ == "__main__":
    main()
