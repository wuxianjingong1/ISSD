import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np
import random


class ShadowDataset(Dataset):
    def __init__(self, image_dir, label_dir, train=True, size=416):
        """
        Args:
            image_dir (str): 图像文件夹路径
            label_dir (str): 标签文件夹路径
            train (bool): 是否为训练模式（True则开启数据增强）
            size (int): 目标图像尺寸
        """
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.label_paths = [os.path.join(label_dir, fname) for fname in os.listdir(label_dir) if
                            fname.endswith(('.png', '.jpg', '.jpeg'))]

        self.image_paths.sort()
        self.label_paths.sort()

        self.train = train
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def transform(self, image, mask):
        # 目标尺寸
        target_size = (self.size, self.size)

        if self.train:
            # --- 训练增强策略 ---

            # 1. Resize 到稍大尺寸 (例如 size + 30)，以便进行随机裁剪
            #    策略: 先放大一点，再裁剪，可以保留更多局部细节，增加多样性
            pad = 30
            resize_h = self.size + pad
            resize_w = self.size + pad

            # Image 使用双线性插值，Mask 必须使用最近邻插值(NEAREST)以保持 0/1 标签
            image = TF.resize(image, (resize_h, resize_w), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (resize_h, resize_w), interpolation=Image.NEAREST)

            # 2. Random Crop (随机裁剪)
            # get_params 会返回随机生成的裁剪参数 (i, j, h, w)
            # 我们将这组参数同时应用到 image 和 mask 上
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=target_size)
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # 3. Random Horizontal Flip (随机水平翻转)
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        else:
            # --- 测试/验证策略 ---
            # 直接 Resize 到目标尺寸，不做其他变换
            image = TF.resize(image, target_size, interpolation=Image.BILINEAR)

        # 4. 转为 Tensor (归一化到 [0, 1])
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask

    def __getitem__(self, idx):
        # 加载 RGB 图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        # 加载 Mask (转为单通道灰度)
        label_image = Image.open(self.label_paths[idx]).convert('L')

        # 应用联合变换
        img_tensor, label_tensor = self.transform(image, label_image)

        img_name = self.image_paths[idx]

        return img_tensor, label_tensor, img_name


def worker_init_fn(worker_id):
    np.random.seed(42 + worker_id)
    torch.manual_seed(42 + worker_id)


def load_data(train_image_dir, train_label_dir, test_image_dir, test_label_dir, batch_size=16):
    """
    Args:
        train_image_dir (str): 训练集图像路径
        train_label_dir (str): 训练集标签路径
        test_image_dir (str): 测试集图像路径
        test_label_dir (str): 测试集标签路径
        batch_size (int): Batch Size
    """

    # 创建训练集 (train=True, 开启增强)
    train_dataset = ShadowDataset(
        image_dir=train_image_dir,
        label_dir=train_label_dir,
        train=True,
        size=416
    )

    # 创建测试集 (train=False, 关闭增强)
    test_dataset = ShadowDataset(
        image_dir=test_image_dir,
        label_dir=test_label_dir,
        train=False,
        size=416
    )

    g = torch.Generator()
    g.manual_seed(42)

    # 建议开启 num_workers 多线程读取以加速训练
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=g,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader


