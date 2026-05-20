import cv2
import numpy as np
import json
import os
import glob
from datetime import datetime
import random
from typing import List, Tuple, Dict, Optional
import copy

class DataAugmentation:
    def __init__(self, config: Dict = None):
        """
        初始化数据增强器
        """
        self.config = config or self.get_default_config()
        self.image_size = (640, 480)  # (width, height)
        self.augmented_count = 0
        self.valid_count = 0
        self.invalid_count = 0
        
        print("✅ 数据增强器初始化成功！")
        print(f"📊 配置参数: {self.config}")

    def get_default_config(self) -> Dict:
        """
        获取默认配置
        """
        return {
            'rotation': {'min': -15, 'max': 15, 'step': 5},
            'scaling': {'min': 0.8, 'max': 1.2, 'step': 0.1},
            'translation': {'range': 30, 'step': 15},
            'brightness': {'min': 0.7, 'max': 1.3, 'step': 0.2},
            'contrast': {'min': 0.8, 'max': 1.2, 'step': 0.1},
            'blur': {'kernel_sizes': [3, 5], 'probability': 0.3},
            'noise': {'gaussian_std': [5, 10, 15], 'probability': 0.2},
            'combination': {'light_prob': 0.6, 'heavy_prob': 0.2}
        }

    def rotate_image_and_corners(self, image: np.ndarray, corners: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        旋转图像和角点
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 创建旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 旋转图像
        rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
        
        # 旋转角点
        if corners.size > 0:
            # 转换角点格式用于变换
            corners_homogeneous = np.column_stack([corners, np.ones(corners.shape[0])])
            rotated_corners = (rotation_matrix @ corners_homogeneous.T).T
            
            return rotated_image, rotated_corners
        
        return rotated_image, corners

    def scale_image_and_corners(self, image: np.ndarray, corners: np.ndarray, scale: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        缩放图像和角点
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 计算缩放后的尺寸
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        scaled_image = cv2.resize(image, (new_w, new_h))
        
        # 创建画布并将缩放后的图像居中放置
        canvas = np.zeros((h, w), dtype=image.dtype)
        
        # 计算放置位置
        start_x = max(0, (w - new_w) // 2)
        start_y = max(0, (h - new_h) // 2)
        end_x = min(w, start_x + new_w)
        end_y = min(h, start_y + new_h)
        
        # 计算在缩放图像中的裁剪区域
        crop_start_x = max(0, (new_w - w) // 2)
        crop_start_y = max(0, (new_h - h) // 2)
        crop_end_x = crop_start_x + (end_x - start_x)
        crop_end_y = crop_start_y + (end_y - start_y)
        
        canvas[start_y:end_y, start_x:end_x] = scaled_image[crop_start_y:crop_end_y, crop_start_x:crop_end_x]
        
        # 缩放角点
        if corners.size > 0:
            # 角点坐标变换
            scaled_corners = corners * scale
            # 调整到居中位置
            offset_x = (w - new_w) // 2 - crop_start_x
            offset_y = (h - new_h) // 2 - crop_start_y
            scaled_corners[:, 0] += offset_x
            scaled_corners[:, 1] += offset_y
            
            return canvas, scaled_corners
        
        return canvas, corners

    def translate_image_and_corners(self, image: np.ndarray, corners: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        平移图像和角点
        """
        h, w = image.shape[:2]
        
        # 创建平移矩阵
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        
        # 平移图像
        translated_image = cv2.warpAffine(image, translation_matrix, (w, h))
        
        # 平移角点
        if corners.size > 0:
            translated_corners = corners + np.array([dx, dy])
            return translated_image, translated_corners
        
        return translated_image, corners

    def apply_brightness(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整图像亮度
        """
        adjusted = image.astype(np.float32) * factor
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def apply_contrast(self, image: np.ndarray, factor: float) -> np.ndarray:
        """
        调整图像对比度
        """
        mean = np.mean(image)
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def apply_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        """
        应用高斯模糊
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def apply_noise(self, image: np.ndarray, noise_type: str, strength: float) -> np.ndarray:
        """
        添加噪声
        """
        if noise_type == 'gaussian':
            noise = np.random.normal(0, strength, image.shape).astype(np.float32)
            noisy_image = image.astype(np.float32) + noise
            return np.clip(noisy_image, 0, 255).astype(np.uint8)
        elif noise_type == 'salt_pepper':
            noisy_image = image.copy()
            prob = strength / 1000.0  # 调整概率
            
            # 椒噪声
            coords = np.random.random(image.shape[:2]) < prob
            noisy_image[coords] = 0
            
            # 盐噪声
            coords = np.random.random(image.shape[:2]) < prob
            noisy_image[coords] = 255
            
            return noisy_image
        
        return image

    def validate_corners(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """
        验证角点是否有效
        """
        if corners.size == 0:
            return False
        
        h, w = image_shape
        
        # 检查角点是否在图像边界内
        valid_corners = np.sum(
            (corners[:, 0] >= 0) & (corners[:, 0] < w) &
            (corners[:, 1] >= 0) & (corners[:, 1] < h)
        )
        
        # 至少需要8个有效角点
        return valid_corners >= 8

    def check_image_quality(self, image: np.ndarray) -> bool:
        """
        检查图像质量
        """
        # 检查图像是否过暗或过亮
        mean_intensity = np.mean(image)
        if mean_intensity < 20 or mean_intensity > 235:
            return False
        
        # 检查图像是否过度模糊
        laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
        if laplacian_var < 50:  # 模糊阈值
            return False
        
        return True

    def generate_augmentation_params(self) -> List[Dict]:
        """
        生成增强参数组合
        """
        params_list = []
        
        # 1. 单一变换
        # 旋转
        angles = np.arange(self.config['rotation']['min'], 
                          self.config['rotation']['max'] + self.config['rotation']['step'], 
                          self.config['rotation']['step'])
        for angle in angles:
            if angle != 0:  # 跳过原始图像
                params_list.append({'rotation': angle})
        
        # 缩放
        scales = np.arange(self.config['scaling']['min'], 
                          self.config['scaling']['max'] + self.config['scaling']['step'], 
                          self.config['scaling']['step'])
        for scale in scales:
            if abs(scale - 1.0) > 0.01:  # 跳过原始尺寸
                params_list.append({'scaling': scale})
        
        # 平移
        trans_range = self.config['translation']['range']
        trans_step = self.config['translation']['step']
        for dx in range(-trans_range, trans_range + trans_step, trans_step):
            for dy in range(-trans_range, trans_range + trans_step, trans_step):
                if dx != 0 or dy != 0:  # 跳过原始位置
                    params_list.append({'translation': (dx, dy)})
        
        # 亮度
        brightness_values = np.arange(self.config['brightness']['min'], 
                                     self.config['brightness']['max'] + self.config['brightness']['step'], 
                                     self.config['brightness']['step'])
        for brightness in brightness_values:
            if abs(brightness - 1.0) > 0.01:  # 跳过原始亮度
                params_list.append({'brightness': brightness})
        
        # 对比度
        contrast_values = np.arange(self.config['contrast']['min'], 
                                   self.config['contrast']['max'] + self.config['contrast']['step'], 
                                   self.config['contrast']['step'])
        for contrast in contrast_values:
            if abs(contrast - 1.0) > 0.01:  # 跳过原始对比度
                params_list.append({'contrast': contrast})
        
        # 2. 组合变换
        # 轻度组合 (2-3种变换)
        light_combinations = []
        for i in range(len(params_list)):
            for j in range(i + 1, min(i + 50, len(params_list))):  # 限制组合数量
                if random.random() < self.config['combination']['light_prob']:
                    combined_params = {}
                    combined_params.update(params_list[i])
                    combined_params.update(params_list[j])
                    light_combinations.append(combined_params)
        
        params_list.extend(light_combinations[:100])  # 限制组合数量
        
        # 3. 添加模糊和噪声
        blur_noise_combinations = []
        for params in params_list[:50]:  # 对部分样本添加模糊和噪声
            if random.random() < self.config['blur']['probability']:
                new_params = params.copy()
                new_params['blur'] = random.choice(self.config['blur']['kernel_sizes'])
                blur_noise_combinations.append(new_params)
            
            if random.random() < self.config['noise']['probability']:
                new_params = params.copy()
                new_params['noise'] = ('gaussian', random.choice(self.config['noise']['gaussian_std']))
                blur_noise_combinations.append(new_params)
        
        params_list.extend(blur_noise_combinations)
        
        print(f"🔄 生成了 {len(params_list)} 个增强参数组合")
        return params_list

    def augment_single_image(self, image: np.ndarray, corners: np.ndarray, params: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        对单张图像应用增强
        """
        try:
            augmented_image = image.copy()
            augmented_corners = corners.copy()
            
            # 应用几何变换
            if 'rotation' in params:
                augmented_image, augmented_corners = self.rotate_image_and_corners(
                    augmented_image, augmented_corners, params['rotation']
                )
            
            if 'scaling' in params:
                augmented_image, augmented_corners = self.scale_image_and_corners(
                    augmented_image, augmented_corners, params['scaling']
                )
            
            if 'translation' in params:
                dx, dy = params['translation']
                augmented_image, augmented_corners = self.translate_image_and_corners(
                    augmented_image, augmented_corners, dx, dy
                )
            
            # 应用图像质量变换
            if 'brightness' in params:
                augmented_image = self.apply_brightness(augmented_image, params['brightness'])
            
            if 'contrast' in params:
                augmented_image = self.apply_contrast(augmented_image, params['contrast'])
            
            if 'blur' in params:
                augmented_image = self.apply_blur(augmented_image, params['blur'])
            
            if 'noise' in params:
                noise_type, strength = params['noise']
                augmented_image = self.apply_noise(augmented_image, noise_type, strength)
            
            # 验证结果
            if not self.validate_corners(augmented_corners, augmented_image.shape):
                return None, None
            
            if not self.check_image_quality(augmented_image):
                return None, None
            
            return augmented_image, augmented_corners
            
        except Exception as e:
            print(f"⚠️ 增强失败: {e}")
            return None, None

    def process_original_data(self, data_folder: str) -> List[Dict]:
        """
        处理原始数据
        """
        print(f"\n📂 处理原始数据: {data_folder}")
        
        # 读取元数据
        metadata_file = os.path.join(data_folder, "collection_metadata.json")
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        original_samples = []
        
        for image_data in metadata['images']:
            filename = image_data['filename']
            image_path = os.path.join(data_folder, filename)
            
            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"⚠️ 无法读取图像: {filename}")
                continue
            
            # 提取角点数据
            corners_data = image_data['corners_data']
            corners = np.array(corners_data, dtype=np.float32).reshape(-1, 2)
            
            # 提取角点ID
            ids_data = image_data['ids_data']
            corner_ids = np.array(ids_data, dtype=np.int32).flatten()
            
            sample = {
                'filename': filename,
                'image': image,
                'corners': corners,
                'corner_ids': corner_ids,
                'corners_count': len(corners),
                'quality_score': image_data['quality_score']
            }
            
            original_samples.append(sample)
        
        print(f"✅ 成功处理 {len(original_samples)} 个原始样本")
        return original_samples

    def run_augmentation(self, input_folder: str, output_folder: str):
        """
        运行数据增强
        """
        print("🚀 开始数据增强...")
        
        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)
        
        # 处理原始数据
        original_samples = self.process_original_data(input_folder)
        
        # 生成增强参数
        augmentation_params = self.generate_augmentation_params()
        
        # 保存增强后的数据
        augmented_samples = []
        
        print(f"\n🔄 开始增强处理...")
        
        for i, sample in enumerate(original_samples):
            print(f"\n处理原始图像 {i+1}/{len(original_samples)}: {sample['filename']}")
            
            # 保存原始图像
            original_output = {
                'filename': f"original_{i+1:03d}.jpg",
                'image': sample['image'],
                'corners': sample['corners'],
                'corner_ids': sample['corner_ids'],
                'corners_count': sample['corners_count'],
                'quality_score': sample['quality_score'],
                'augmentation_params': {'original': True}
            }
            augmented_samples.append(original_output)
            
            # 应用增强
            for j, params in enumerate(augmentation_params):
                augmented_image, augmented_corners = self.augment_single_image(
                    sample['image'], sample['corners'], params
                )
                
                if augmented_image is not None and augmented_corners is not None:
                    self.valid_count += 1
                    
                    # 保存增强后的样本
                    augmented_output = {
                        'filename': f"augmented_{i+1:03d}_{j+1:04d}.jpg",
                        'image': augmented_image,
                        'corners': augmented_corners,
                        'corner_ids': sample['corner_ids'],
                        'corners_count': len(augmented_corners),
                        'quality_score': 85.0,  # 假设增强后的质量分数
                        'augmentation_params': params
                    }
                    augmented_samples.append(augmented_output)
                else:
                    self.invalid_count += 1
                
                self.augmented_count += 1
                
                # 每1000个样本显示进度
                if self.augmented_count % 1000 == 0:
                    print(f"  已处理: {self.augmented_count}, 有效: {self.valid_count}, 无效: {self.invalid_count}")
        
        # 保存增强后的数据
        self.save_augmented_data(augmented_samples, output_folder)
        
        print(f"\n🎉 数据增强完成!")
        print(f"📊 统计信息:")
        print(f"  原始样本: {len(original_samples)}")
        print(f"  尝试增强: {self.augmented_count}")
        print(f"  有效增强: {self.valid_count}")
        print(f"  无效增强: {self.invalid_count}")
        print(f"  总样本数: {len(augmented_samples)}")
        print(f"  输出文件夹: {output_folder}")

    def save_augmented_data(self, augmented_samples: List[Dict], output_folder: str):
        """
        保存增强后的数据
        """
        print(f"\n💾 保存增强数据到: {output_folder}")
        
        # 保存图像文件
        for sample in augmented_samples:
            image_path = os.path.join(output_folder, sample['filename'])
            cv2.imwrite(image_path, sample['image'])
        
        # 创建元数据
        metadata = {
            'augmentation_info': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(augmented_samples),
                'augmentation_config': self.config
            },
            'samples': []
        }
        
        for sample in augmented_samples:
            sample_metadata = {
                'filename': sample['filename'],
                'corners_count': int(sample['corners_count']),  # 转换为Python int
                'quality_score': float(sample['quality_score']),  # 转换为Python float
                'corners_data': sample['corners'].astype(float).tolist(),  # 确保为float类型
                'ids_data': sample['corner_ids'].astype(int).tolist(),  # 确保为int类型
                'augmentation_params': sample['augmentation_params']
            }
            metadata['samples'].append(sample_metadata)
        
        # 保存元数据
        metadata_file = os.path.join(output_folder, "augmented_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 保存完成:")
        print(f"  图像文件: {len(augmented_samples)} 张")
        print(f"  元数据文件: {metadata_file}")

def find_latest_data_folder():
    """
    查找最新的数据文件夹
    """
    pattern = "calibration_data_*"
    folders = glob.glob(pattern)
    if not folders:
        return None
    
    # 按创建时间排序，返回最新的
    folders.sort(key=os.path.getctime, reverse=True)
    return folders[0]

def main():
    print("=" * 60)
    print("🔄 ChArUco 数据增强系统")
    print("=" * 60)
    
    # 查找输入数据文件夹
    input_folder = find_latest_data_folder()
    if input_folder is None:
        print("❌ 错误：没有找到标定数据文件夹")
        return
    
    print(f"📁 输入数据文件夹: {input_folder}")
    
    # 创建输出文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"augmented_data_{timestamp}"
    
    # 创建数据增强器
    augmenter = DataAugmentation()
    
    # 确认开始增强
    print(f"\n准备开始数据增强...")
    print(f"输入: {input_folder}")
    print(f"输出: {output_folder}")
    print("\n按 Enter 键开始增强...")
    input()
    
    # 运行数据增强
    augmenter.run_augmentation(input_folder, output_folder)

if __name__ == "__main__":
    main() 