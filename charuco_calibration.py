import cv2
import numpy as np
import os
import json
import glob
from datetime import datetime

class CharucoCalibrator:
    def __init__(self, data_folder):
        """
        初始化ChArUco标定器
        """
        self.data_folder = data_folder
        
        # 标定板参数 (与数据收集时一致)
        self.squares_x = 7
        self.squares_y = 5
        self.square_length = 0.0335  # 3.35cm
        self.marker_length = 0.0335 * 0.75
        
        # 创建ArUco字典和ChArUco标定板
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )
        
        # 创建检测器
        self.detector_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # 标定结果
        self.calibration_result = {}
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
        
        print(f"✅ ChArUco标定器初始化成功！")
        print(f"📁 数据文件夹: {data_folder}")
        print(f"📏 标定板尺寸: {self.squares_x}x{self.squares_y}")
        print(f"📦 格子大小: {self.square_length*100:.2f}cm")

    def load_calibration_data(self):
        """
        加载标定数据
        """
        # 读取元数据
        metadata_file = os.path.join(self.data_folder, "collection_metadata.json")
        if not os.path.exists(metadata_file):
            print(f"❌ 错误：找不到元数据文件 {metadata_file}")
            return False
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        # 获取图像文件列表
        image_pattern = os.path.join(self.data_folder, "calibration_image_*.jpg")
        self.image_files = sorted(glob.glob(image_pattern))
        
        print(f"📊 找到 {len(self.image_files)} 张标定图像")
        print(f"📋 元数据包含 {len(self.metadata['images'])} 条记录")
        
        if len(self.image_files) == 0:
            print("❌ 错误：没有找到标定图像")
            return False
        
        return True

    def detect_charuco_in_images(self):
        """
        从元数据中加载ChArUco角点数据（而不是重新检测）
        """
        print("\n🔍 从元数据中加载ChArUco角点数据...")
        
        # 存储所有检测到的角点
        all_corners = []
        all_ids = []
        all_image_sizes = []
        valid_images = []
        
        # 从元数据中获取角点数据
        for i, image_data in enumerate(self.metadata['images']):
            filename = image_data['filename']
            corners_count = image_data['corners_count']
            
            print(f"处理图像 {i+1}/{len(self.metadata['images'])}: {filename}")
            print(f"  📊 角点数量: {corners_count}")
            
            if corners_count >= 8:  # 至少需要8个角点
                # 转换角点数据格式
                corners_data = image_data['corners_data']
                ids_data = image_data['ids_data']
                
                # 转换为numpy数组
                charuco_corners = np.array(corners_data, dtype=np.float32)
                charuco_ids = np.array(ids_data, dtype=np.int32)
                
                # 获取图像尺寸
                image_path = os.path.join(self.data_folder, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    image_size = (image.shape[1], image.shape[0])  # (width, height)
                    
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
                    all_image_sizes.append(image_size)
                    valid_images.append(image_path)
                    
                    print(f"  ✅ 成功加载 {corners_count} 个角点")
                else:
                    print(f"  ❌ 无法读取图像文件")
            else:
                print(f"  ❌ 角点数量不足 ({corners_count} < 8)")
        
        self.all_corners = all_corners
        self.all_ids = all_ids
        self.all_image_sizes = all_image_sizes
        self.valid_images = valid_images
        
        print(f"\n📊 角点加载结果:")
        print(f"有效图像数量: {len(valid_images)}")
        print(f"总角点集合: {len(all_corners)}")
        
        if len(all_corners) < 3:
            print("❌ 错误：有效图像太少，无法进行标定")
            return False
        
        return True

    def calibrate_camera(self):
        """
        执行相机标定
        """
        print("\n🎯 开始相机标定...")
        
        if len(self.all_corners) == 0:
            print("❌ 错误：没有有效的角点数据")
            return False
        
        # 使用第一张图像的尺寸作为标准
        image_size = self.all_image_sizes[0]
        print(f"📏 图像尺寸: {image_size[0]} x {image_size[1]}")
        
        try:
            # 执行ChArUco标定
            calibration_flags = (cv2.CALIB_RATIONAL_MODEL + 
                               cv2.CALIB_THIN_PRISM_MODEL + 
                               cv2.CALIB_TILTED_MODEL)
            
            print("🔄 正在计算相机参数...")
            
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                self.all_corners, self.all_ids, self.charuco_board, image_size, None, None
            )
            
            if ret:
                self.camera_matrix = camera_matrix
                self.dist_coeffs = dist_coeffs
                self.rvecs = rvecs
                self.tvecs = tvecs
                self.calibration_error = ret
                
                print(f"✅ 标定成功！")
                print(f"📊 重投影误差: {ret:.4f} 像素")
                
                # 分析标定结果
                self.analyze_calibration_result(image_size)
                return True
            else:
                print("❌ 标定失败")
                return False
                
        except Exception as e:
            print(f"❌ 标定过程出错: {e}")
            return False

    def analyze_calibration_result(self, image_size):
        """
        分析标定结果
        """
        print("\n📋 标定结果分析:")
        
        # 相机内参数矩阵
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        print(f"📐 相机内参数:")
        print(f"  焦距 fx: {fx:.2f} 像素")
        print(f"  焦距 fy: {fy:.2f} 像素")
        print(f"  光心 cx: {cx:.2f} 像素")
        print(f"  光心 cy: {cy:.2f} 像素")
        print(f"  长宽比: {fx/fy:.4f}")
        
        # 畸变系数
        print(f"\n🔍 畸变系数:")
        if len(self.dist_coeffs) >= 5:
            k1, k2, p1, p2, k3 = self.dist_coeffs[0][:5]
            print(f"  径向畸变 k1: {k1:.6f}")
            print(f"  径向畸变 k2: {k2:.6f}")
            print(f"  径向畸变 k3: {k3:.6f}")
            print(f"  切向畸变 p1: {p1:.6f}")
            print(f"  切向畸变 p2: {p2:.6f}")
        
        # 计算视场角
        width, height = image_size
        fov_x = 2 * np.arctan(width / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(height / (2 * fy)) * 180 / np.pi
        
        print(f"\n📷 相机属性:")
        print(f"  水平视场角: {fov_x:.1f}°")
        print(f"  垂直视场角: {fov_y:.1f}°")
        print(f"  图像中心偏移: ({cx-width/2:.1f}, {cy-height/2:.1f}) 像素")
        
        # 误差分析
        print(f"\n📊 精度评估:")
        print(f"  重投影误差: {self.calibration_error:.4f} 像素")
        if self.calibration_error < 0.5:
            print("  ✅ 优秀 (< 0.5 像素)")
        elif self.calibration_error < 1.0:
            print("  ✅ 良好 (< 1.0 像素)")
        elif self.calibration_error < 2.0:
            print("  ⚠️ 一般 (< 2.0 像素)")
        else:
            print("  ❌ 较差 (≥ 2.0 像素)")

    def save_calibration_result(self):
        """
        保存标定结果
        """
        print("\n💾 保存标定结果...")
        
        # 准备保存数据
        calibration_data = {
            'calibration_info': {
                'timestamp': datetime.now().isoformat(),
                'board_size': (self.squares_x, self.squares_y),
                'square_length': self.square_length,
                'marker_length': self.marker_length,
                'num_images_used': len(self.valid_images),
                'total_images': len(self.image_files),
                'reprojection_error': float(self.calibration_error)
            },
            'camera_matrix': self.camera_matrix.tolist(),
            'distortion_coefficients': self.dist_coeffs.tolist(),
            'image_size': self.all_image_sizes[0],
            'valid_images': [os.path.basename(img) for img in self.valid_images]
        }
        
        # 保存JSON格式的标定结果
        calibration_file = os.path.join(self.data_folder, "camera_calibration.json")
        with open(calibration_file, 'w', encoding='utf-8') as f:
            json.dump(calibration_data, f, indent=2, ensure_ascii=False)
        
        # 保存NumPy格式的参数（便于后续使用）
        np.savez(os.path.join(self.data_folder, "camera_parameters.npz"),
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                rvecs=self.rvecs,
                tvecs=self.tvecs,
                reprojection_error=self.calibration_error)
        
        print(f"✅ 标定结果已保存:")
        print(f"  📄 JSON格式: {calibration_file}")
        print(f"  📦 NumPy格式: camera_parameters.npz")

    def test_undistortion(self):
        """
        测试去畸变效果
        """
        print("\n🧪 测试去畸变效果...")
        
        if len(self.valid_images) == 0:
            print("❌ 没有有效图像进行测试")
            return
        
        # 选择第一张图像进行测试
        test_image_path = self.valid_images[0]
        print(f"📸 使用测试图像: {os.path.basename(test_image_path)}")
        
        # 读取图像
        original = cv2.imread(test_image_path)
        if original is None:
            print("❌ 无法读取测试图像")
            return
        
        # 去畸变
        undistorted = cv2.undistort(original, self.camera_matrix, self.dist_coeffs)
        
        # 保存对比图像
        comparison = np.hstack((original, undistorted))
        
        # 添加标签
        cv2.putText(comparison, "Original", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(comparison, "Undistorted", (original.shape[1] + 50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存结果
        comparison_file = os.path.join(self.data_folder, "undistortion_comparison.jpg")
        cv2.imwrite(comparison_file, comparison)
        
        undistorted_file = os.path.join(self.data_folder, "test_undistorted.jpg")
        cv2.imwrite(undistorted_file, undistorted)
        
        print(f"✅ 去畸变测试完成:")
        print(f"  📸 对比图像: {comparison_file}")
        print(f"  📸 去畸变图像: {undistorted_file}")

    def run_calibration(self):
        """
        运行完整的标定流程
        """
        print("🚀 开始ChArUco相机标定流程")
        
        # 1. 加载数据
        if not self.load_calibration_data():
            return False
        
        # 2. 检测角点
        if not self.detect_charuco_in_images():
            return False
        
        # 3. 执行标定
        if not self.calibrate_camera():
            return False
        
        # 4. 保存结果
        self.save_calibration_result()
        
        # 5. 测试去畸变
        self.test_undistortion()
        
        print("\n🎉 相机标定完成！")
        print("📊 标定结果已保存到数据文件夹中")
        return True

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
    print("🎯 ChArUco 项目 - 相机标定")
    print("=" * 60)
    
    # 查找数据文件夹
    data_folder = find_latest_data_folder()
    
    if data_folder is None:
        print("❌ 错误：没有找到标定数据文件夹")
        print("请先运行 charuco_data_collection.py 收集标定数据")
        return
    
    print(f"📁 使用数据文件夹: {data_folder}")
    
    # 创建标定器
    calibrator = CharucoCalibrator(data_folder)
    
    # 确认开始标定
    print("\n按 Enter 键开始相机标定...")
    input()
    
    # 运行标定
    success = calibrator.run_calibration()
    
    if success:
        print("\n✅ 恭喜！相机标定成功完成")
        print("🔧 你现在可以使用标定结果进行：")
        print("  - 图像去畸变")
        print("  - 3D测量")
        print("  - 增强现实应用")
        print("  - 机器视觉项目")
    else:
        print("\n❌ 标定失败，请检查数据质量")

if __name__ == "__main__":
    main() 