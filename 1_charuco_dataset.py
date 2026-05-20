import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import json
import os
from typing import List, Tuple, Dict, Optional
import random
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

class CharucoDataset(Dataset):
    """
    ChArUco数据集类，用于深度学习训练
    """
    
    def __init__(self, data_folder: str, mode: str = 'train', 
                 train_split: float = 0.8, random_seed: int = 42,
                 image_size: Tuple[int, int] = (640, 480),
                 max_corners: int = 35):
        """
        初始化数据集
        
        Args:
            data_folder: 增强数据文件夹路径
            mode: 模式 ('train' 或 'val')
            train_split: 训练集比例
            random_seed: 随机种子
            image_size: 图像尺寸 (width, height)
            max_corners: 最大角点数量
        """
        self.data_folder = data_folder
        self.mode = mode
        self.image_size = image_size
        self.max_corners = max_corners
        
        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # 加载元数据
        self.metadata = self._load_metadata()
        
        # 分割训练/验证集
        self.samples = self._split_dataset(train_split)
        
        # 定义图像变换
        self.transform = self._get_transforms()
        
        print(f"✅ {mode.upper()}数据集初始化完成")
        print(f"📊 样本数量: {len(self.samples)}")
        print(f"📐 图像尺寸: {self.image_size}")
        print(f"🎯 最大角点数: {self.max_corners}")

    def _load_metadata(self) -> Dict:
        """加载元数据文件"""
        metadata_file = os.path.join(self.data_folder, "augmented_metadata.json")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"元数据文件不存在: {metadata_file}")
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        print(f"📂 加载元数据: {len(metadata['samples'])} 个样本")
        return metadata

    def _split_dataset(self, train_split: float) -> List[Dict]:
        """分割训练/验证数据集"""
        all_samples = self.metadata['samples']
        
        # 提取原始图像样本（用于分割基准）
        original_samples = []
        augmented_samples = []
        
        for sample in all_samples:
            if sample['augmentation_params'].get('original', False):
                original_samples.append(sample)
            else:
                augmented_samples.append(sample)
        
        # 按原始图像分割，确保同一原始图像的增强样本不会同时出现在训练和验证集
        train_originals, val_originals = train_test_split(
            original_samples, 
            train_size=train_split, 
            random_state=42
        )
        
        # 获取原始图像的文件名前缀
        train_prefixes = set()
        val_prefixes = set()
        
        for sample in train_originals:
            prefix = sample['filename'].split('_')[1]  # 从 'original_001.jpg' 提取 '001'
            train_prefixes.add(prefix)
        
        for sample in val_originals:
            prefix = sample['filename'].split('_')[1]
            val_prefixes.add(prefix)
        
        # 分配增强样本
        train_samples = train_originals.copy()
        val_samples = val_originals.copy()
        
        for sample in augmented_samples:
            # 从 'augmented_001_0001.jpg' 提取 '001'
            prefix = sample['filename'].split('_')[1]
            
            if prefix in train_prefixes:
                train_samples.append(sample)
            elif prefix in val_prefixes:
                val_samples.append(sample)
        
        if self.mode == 'train':
            selected_samples = train_samples
        else:
            selected_samples = val_samples
        
        print(f"📊 数据集分割完成:")
        print(f"  训练集: {len(train_samples)} 样本")
        print(f"  验证集: {len(val_samples)} 样本")
        print(f"  当前模式: {self.mode} ({len(selected_samples)} 样本)")
        
        return selected_samples

    def _get_transforms(self) -> transforms.Compose:
        """获取图像变换"""
        transform_list = [
            transforms.ToPILImage(),
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])  # 灰度图像标准化
        ]
        
        # 训练时添加额外的数据增强
        if self.mode == 'train':
            transform_list.insert(-2, transforms.RandomApply([
                transforms.ColorJitter(brightness=0.1, contrast=0.1)
            ], p=0.3))
        
        return transforms.Compose(transform_list)

    def _process_corners(self, corners_data: List[List[float]], 
                        ids_data: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理角点数据
        
        Returns:
            corners: 角点坐标张量 [max_corners, 2]
            corner_ids: 角点ID张量 [max_corners]
            valid_mask: 有效角点掩码 [max_corners]
        """
        corners = np.array(corners_data, dtype=np.float32)
        corner_ids = np.array(ids_data, dtype=np.int32)
        
        # 标准化角点坐标到 [0, 1] 范围
        corners[:, 0] /= self.image_size[0]  # x坐标 / width
        corners[:, 1] /= self.image_size[1]  # y坐标 / height
        
        # 创建固定长度的数组
        padded_corners = np.zeros((self.max_corners, 2), dtype=np.float32)
        padded_ids = np.zeros(self.max_corners, dtype=np.int32)
        valid_mask = np.zeros(self.max_corners, dtype=np.bool_)
        
        # 填充有效数据
        num_corners = min(len(corners), self.max_corners)
        padded_corners[:num_corners] = corners[:num_corners]
        padded_ids[:num_corners] = corner_ids[:num_corners]
        valid_mask[:num_corners] = True
        
        # 转换为PyTorch张量
        corners_tensor = torch.from_numpy(padded_corners)
        ids_tensor = torch.from_numpy(padded_ids)
        mask_tensor = torch.from_numpy(valid_mask)
        
        return corners_tensor, ids_tensor, mask_tensor

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取单个样本"""
        sample = self.samples[idx]
        
        # 读取图像
        image_path = os.path.join(self.data_folder, sample['filename'])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")
        
        # 应用图像变换
        image_tensor = self.transform(image)
        
        # 处理角点数据
        corners, corner_ids, valid_mask = self._process_corners(
            sample['corners_data'], 
            sample['ids_data']
        )
        
        # 创建样本字典
        sample_dict = {
            'image': image_tensor,
            'corners': corners,
            'corner_ids': corner_ids,
            'valid_mask': valid_mask,
            'num_corners': torch.tensor(sample['corners_count'], dtype=torch.long),
            'quality_score': torch.tensor(sample['quality_score'], dtype=torch.float32),
            'filename': sample['filename']
        }
        
        return sample_dict


def create_data_loaders(data_folder: str, 
                       batch_size: int = 16,
                       train_split: float = 0.8,
                       num_workers: int = 4,
                       random_seed: int = 42) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器
    
    Args:
        data_folder: 数据文件夹路径
        batch_size: 批大小
        train_split: 训练集比例
        num_workers: 数据加载进程数
        random_seed: 随机种子
    
    Returns:
        train_loader, val_loader: 训练和验证数据加载器
    """
    # 创建数据集
    train_dataset = CharucoDataset(
        data_folder=data_folder,
        mode='train',
        train_split=train_split,
        random_seed=random_seed
    )
    
    val_dataset = CharucoDataset(
        data_folder=data_folder,
        mode='val',
        train_split=train_split,
        random_seed=random_seed
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def visualize_sample(sample: Dict[str, torch.Tensor], save_path: str = None):
    """
    可视化单个样本
    
    Args:
        sample: 样本字典
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    # 获取图像和角点
    image = sample['image'].squeeze().numpy()
    corners = sample['corners'].numpy()
    valid_mask = sample['valid_mask'].numpy()
    corner_ids = sample['corner_ids'].numpy()
    
    # 反标准化图像
    image = image * 0.229 + 0.485
    image = np.clip(image, 0, 1)
    
    # 反标准化角点坐标
    corners[:, 0] *= 640  # width
    corners[:, 1] *= 480  # height
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    
    # 绘制有效角点
    valid_corners = corners[valid_mask]
    valid_ids = corner_ids[valid_mask]
    
    plt.scatter(valid_corners[:, 0], valid_corners[:, 1], 
               c='red', s=50, alpha=0.7)
    
    # 标注角点ID
    for i, (corner, corner_id) in enumerate(zip(valid_corners, valid_ids)):
        plt.text(corner[0] + 5, corner[1] + 5, str(corner_id), 
                color='yellow', fontsize=8)
    
    plt.title(f"文件: {sample['filename']}\n"
              f"角点数量: {sample['num_corners'].item()}\n"
              f"质量分数: {sample['quality_score'].item():.1f}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📸 可视化保存到: {save_path}")
    
    plt.show()


def analyze_dataset(data_folder: str):
    """
    分析数据集统计信息
    
    Args:
        data_folder: 数据文件夹路径
    """
    # 创建数据集
    train_dataset = CharucoDataset(data_folder, mode='train')
    val_dataset = CharucoDataset(data_folder, mode='val')
    
    print("\n📊 数据集分析报告")
    print("=" * 50)
    
    # 基本统计
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"总样本数: {len(train_dataset) + len(val_dataset)}")
    
    # 角点数量统计
    train_corners = [sample['num_corners'].item() for sample in train_dataset]
    val_corners = [sample['num_corners'].item() for sample in val_dataset]
    
    print(f"\n角点数量统计:")
    print(f"  训练集: 平均 {np.mean(train_corners):.1f}, 范围 {min(train_corners)}-{max(train_corners)}")
    print(f"  验证集: 平均 {np.mean(val_corners):.1f}, 范围 {min(val_corners)}-{max(val_corners)}")
    
    # 质量分数统计
    train_quality = [sample['quality_score'].item() for sample in train_dataset]
    val_quality = [sample['quality_score'].item() for sample in val_dataset]
    
    print(f"\n质量分数统计:")
    print(f"  训练集: 平均 {np.mean(train_quality):.1f}, 范围 {min(train_quality):.1f}-{max(train_quality):.1f}")
    print(f"  验证集: 平均 {np.mean(val_quality):.1f}, 范围 {min(val_quality):.1f}-{max(val_quality):.1f}")


def test_data_loader():
    """
    测试数据加载器功能
    """
    print("🧪 测试数据加载器...")
    
    # 查找数据文件夹
    data_folders = [f for f in os.listdir('.') if f.startswith('augmented_data_')]
    if not data_folders:
        print("❌ 未找到增强数据文件夹")
        return
    
    # 使用最新的数据文件夹
    data_folder = sorted(data_folders)[-1]
    print(f"📁 使用数据文件夹: {data_folder}")
    
    try:
        # 创建数据加载器
        train_loader, val_loader = create_data_loaders(
            data_folder=data_folder,
            batch_size=4,
            num_workers=0  # Windows下设置为0
        )
        
        print("✅ 数据加载器创建成功")
        
        # 测试训练集
        print("\n🔄 测试训练集...")
        for i, batch in enumerate(train_loader):
            print(f"  批次 {i+1}:")
            print(f"    图像形状: {batch['image'].shape}")
            print(f"    角点形状: {batch['corners'].shape}")
            print(f"    有效掩码: {batch['valid_mask'].sum(dim=1).tolist()}")
            
            if i >= 2:  # 只测试前3个批次
                break
        
        # 测试验证集
        print("\n🔄 测试验证集...")
        for i, batch in enumerate(val_loader):
            print(f"  批次 {i+1}:")
            print(f"    图像形状: {batch['image'].shape}")
            print(f"    角点形状: {batch['corners'].shape}")
            print(f"    有效掩码: {batch['valid_mask'].sum(dim=1).tolist()}")
            
            if i >= 2:  # 只测试前3个批次
                break
        
        # 可视化样本
        print("\n📸 可视化训练样本...")
        sample = train_loader.dataset[0]
        visualize_sample(sample, f"sample_visualization_{data_folder}.png")
        
        # 数据集分析
        analyze_dataset(data_folder)
        
        print("\n🎉 数据加载器测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_data_loader() 