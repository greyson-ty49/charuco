import cv2
import numpy as np
import os
from datetime import datetime
import json

class CharucoDataCollector:
    def __init__(self):
        """
        初始化ChArUco数据收集器
        """
        # 标定板参数
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
        
        # 数据收集设置 - 专业级54张标定
        self.target_images = 54  # 目标图像数量 (专业级标定质量)
        self.min_corners_required = 8  # 最少需要的角点数 (降低要求)
        self.quality_threshold = 30.0  # 质量阈值（百分比）(降低阈值)
        
        # 创建数据文件夹
        self.data_folder = f"calibration_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.data_folder, exist_ok=True)
        
        # 收集状态
        self.collected_images = []
        self.collection_metadata = {
            'board_size': (self.squares_x, self.squares_y),
            'square_length': self.square_length,
            'marker_length': self.marker_length,
            'target_images': self.target_images,
            'min_corners_required': self.min_corners_required,
            'quality_threshold': self.quality_threshold,
            'images': []
        }
        
        # 距离覆盖建议 (参考用，不强制)
        self.distance_ranges = [
            "25cm-35cm: 极近距离 (重点覆盖)",
            "40cm-60cm: 近距离 (核心使用范围)",
            "70cm-90cm: 中距离 (标准拍摄)",
            "100cm-130cm: 远距离 (完整覆盖)",
            "140cm-150cm: 超远距离 (边界测试)"
        ]
        
        print(f"✅ ChArUco数据收集器初始化成功！")
        print(f"📁 数据保存目录: {self.data_folder}")
        print(f"🎯 目标图像数量: {self.target_images}")
        print(f"📊 最少角点要求: {self.min_corners_required}")
        print(f"🎨 质量阈值: {self.quality_threshold}%")

    def detect_charuco(self, frame):
        """
        检测ChArUco标定板
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测ArUco标记
        marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        charuco_corners = None
        charuco_ids = None
        
        if marker_ids is not None and len(marker_ids) > 0:
            try:
                # 尝试新API
                num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.charuco_board
                )
                if num_corners <= 0:
                    charuco_corners, charuco_ids = None, None
            except:
                try:
                    # 尝试旧API
                    charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, self.charuco_board
                    )
                except:
                    charuco_corners, charuco_ids = None, None
        
        return marker_corners, marker_ids, charuco_corners, charuco_ids

    def evaluate_image_quality(self, charuco_corners, charuco_ids, frame_shape):
        """
        评估图像质量
        """
        if charuco_corners is None or not hasattr(charuco_corners, '__len__'):
            return 0.0, "No corners"
        
        corner_count = len(charuco_corners)
        if corner_count < self.min_corners_required:
            return 0.0, f"Need {self.min_corners_required}+ corners"
        
        # 计算检测质量 (简化评估)
        max_possible_corners = (self.squares_x - 1) * (self.squares_y - 1)  # 24个
        detection_quality = (corner_count / max_possible_corners) * 100
        
        # 简化分布检查
        if corner_count >= self.min_corners_required:
            # 只要角点数量足够，就认为质量可以
            if detection_quality >= 30:  # 如果检测到30%以上的角点
                return detection_quality, "Good"
            else:
                return detection_quality, "OK"
        
        return detection_quality, "Poor"

    def draw_collection_interface(self, frame, marker_corners, marker_ids, charuco_corners, charuco_ids):
        """
        绘制数据收集界面
        """
        height, width = frame.shape[:2]
        
        # 绘制ArUco标记
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        
        # 绘制ChArUco角点
        if charuco_corners is not None and hasattr(charuco_corners, '__len__') and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
        
        # 计算检测信息
        marker_count = len(marker_ids) if marker_ids is not None else 0
        corner_count = len(charuco_corners) if charuco_corners is not None and hasattr(charuco_corners, '__len__') else 0
        
        # 评估图像质量
        quality_score, quality_text = self.evaluate_image_quality(charuco_corners, charuco_ids, frame.shape)
        
        # 主信息面板
        panel_width = 450
        panel_height = 200
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (panel_width, panel_height), (255, 255, 255), 2)
        
        # 显示摄像头信息
        cv2.putText(frame, "iPhone Camera - ChArUco Data Collection", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示收集进度 (使用英文避免编码问题)
        progress = len(self.collected_images)
        cv2.putText(frame, f"Progress: {progress}/{self.target_images}", 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 显示检测信息
        cv2.putText(frame, f"ArUco Markers: {marker_count}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"ChArUco Corners: {corner_count}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 显示质量评估
        quality_color = (0, 255, 0) if quality_score >= self.quality_threshold else (0, 255, 255) if quality_score >= 30 else (0, 0, 255)
        cv2.putText(frame, f"Quality: {quality_score:.1f}% ({quality_text})", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 2)
        
        # 显示进度
        progress = len(self.collected_images)
        progress_percent = (progress / self.target_images) * 100
        cv2.putText(frame, f"Progress: {progress}/{self.target_images} ({progress_percent:.1f}%)", 
                   (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 显示拍摄状态
        if progress >= self.target_images:
            cv2.putText(frame, "COMPLETE! Press 'q' to finish", 
                       (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif quality_score >= self.quality_threshold:
            cv2.putText(frame, "READY! Press SPACE to capture", 
                       (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Adjust board position", 
                       (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示自由拍摄模式 (无固定建议)
        cv2.putText(frame, "Free shooting mode - Cover 25cm-150cm range", 
                   (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.putText(frame, "Vary angles & distances for best results", 
                   (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # 显示调试信息（右上角）
        debug_x = width - 300
        cv2.rectangle(frame, (debug_x, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.rectangle(frame, (debug_x, 10), (width - 10, 120), (255, 255, 255), 1)
        
        cv2.putText(frame, "DEBUG INFO:", (debug_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Min corners: {self.min_corners_required}", (debug_x + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Quality threshold: {self.quality_threshold}%", (debug_x + 10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Current corners: {corner_count}", (debug_x + 10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, f"Current quality: {quality_score:.1f}%", (debug_x + 10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 操作提示
        cv2.putText(frame, "SPACE:capture | q:quit | r:restart", 
                   (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, quality_score

    def save_image(self, frame, charuco_corners, charuco_ids, quality_score):
        """
        保存标定图像
        """
        image_id = len(self.collected_images) + 1
        filename = f"calibration_image_{image_id:02d}.jpg"
        filepath = os.path.join(self.data_folder, filename)
        
        # 保存图像
        cv2.imwrite(filepath, frame)
        
        # 记录元数据
        image_metadata = {
            'filename': filename,
            'image_id': image_id,
            'corners_count': len(charuco_corners) if charuco_corners is not None else 0,
            'quality_score': quality_score,
            'timestamp': datetime.now().isoformat(),
            'corners_data': charuco_corners.tolist() if charuco_corners is not None else None,
            'ids_data': charuco_ids.tolist() if charuco_ids is not None else None,
            'image_resolution': f"{frame.shape[1]}x{frame.shape[0]}",
            'camera_type': 'iPhone (DroidCam)'
        }
        
        self.collected_images.append(image_metadata)
        self.collection_metadata['images'].append(image_metadata)
        
        # 保存元数据文件
        metadata_file = os.path.join(self.data_folder, "collection_metadata.json")
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.collection_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 保存图像 {image_id}: {filename} (质量: {quality_score:.1f}%, 分辨率: {frame.shape[1]}x{frame.shape[0]})")
        return image_id

    def run_collection(self):
        """
        运行数据收集
        """
        cap = cv2.VideoCapture('http://192.168.161.221:4747/video')
        
        if not cap.isOpened():
            print("❌ 错误：无法打开iPhone摄像头")
            print("请检查：")
            print("1. DroidCam是否正在iPhone上运行")
            print("2. 网络连接是否正常 (http://192.168.161.221:4747)")
            print("3. 防火墙是否阻止了连接")
            return False
        
        # 获取iPhone摄像头的原始分辨率（保持不变以保证数据质量）
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算显示窗口尺寸（一半大小）
        display_width = width // 2
        display_height = height // 2
        
        print(f"📱 iPhone摄像头分辨率: {width} x {height}")
        print(f"🖥️ 显示窗口尺寸: {display_width} x {display_height} (原始尺寸的50%)")
        
        # 更新元数据中的分辨率信息
        self.collection_metadata['camera_info'] = {
            'camera_type': 'iPhone (DroidCam)',
            'resolution': f"{width}x{height}",
            'display_resolution': f"{display_width}x{display_height}"
        }
        
        print()
        print("🎯 ChArUco专业级数据收集开始！(iPhone摄像头)")
        print("📋 操作说明:")
        print("- 目标：54张图像，覆盖25cm-150cm距离范围")
        print("- 自由拍摄模式，您可以自由选择角度和距离")
        print("- 当质量达到要求时，按空格键拍摄")
        print("- 按 'q' 键退出")
        print("- 按 'r' 键重新开始收集")
        print("- iPhone摄像头质量更好，应该更容易达到质量要求")
        print()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 错误：无法读取摄像头画面")
                break
            
            # 检测ChArUco
            marker_corners, marker_ids, charuco_corners, charuco_ids = self.detect_charuco(frame)
            
            # 绘制界面
            frame, quality_score = self.draw_collection_interface(
                frame, marker_corners, marker_ids, charuco_corners, charuco_ids
            )
            
            # 缩放图像到一半大小用于显示（保持原始分辨率质量）
            display_frame = cv2.resize(frame, (display_width, display_height))
            
            # 显示缩放后的画面
            cv2.imshow('ChArUco Data Collection (iPhone)', display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 退出数据收集...")
                break
            elif key == ord(' '):  # 空格键拍摄
                if quality_score >= self.quality_threshold:
                    image_id = self.save_image(frame, charuco_corners, charuco_ids, quality_score)
                    
                    # 检查是否完成收集
                    if len(self.collected_images) >= self.target_images:
                        print()
                        print("🎉 数据收集完成！")
                        print(f"📊 成功收集 {len(self.collected_images)} 张图像")
                        print(f"📁 数据保存在: {self.data_folder}")
                        break
                else:
                    print("⚠️ 图像质量不符合要求，请调整标定板位置")
            elif key == ord('r'):  # 重新开始
                print("🔄 重新开始数据收集...")
                self.collected_images = []
                self.collection_metadata['images'] = []
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 显示收集结果
        print()
        print("📊 iPhone摄像头数据收集统计:")
        print(f"目标图像数量: {self.target_images}")
        print(f"实际收集数量: {len(self.collected_images)}")
        print(f"数据保存目录: {self.data_folder}")
        print(f"📱 摄像头分辨率: {width}x{height}")
        print(f"🖥️ 显示窗口尺寸: {display_width}x{display_height}")
        
        if len(self.collected_images) >= 54:
            print("✅ 专业级标定数据收集完成！")
            print("✅ 54张高质量图像应该能提供极佳的标定效果")
        elif len(self.collected_images) >= 30:
            print("✅ 收集的图像数量足够进行高质量标定")
            print("✅ iPhone摄像头的高质量图像应该能提供很好的标定效果")
        elif len(self.collected_images) >= 10:
            print("✅ 收集的图像数量足够进行基础标定")
            print("💡 建议收集更多图像以获得更好的标定效果")
        else:
            print("⚠️ 建议收集更多图像以获得更好的标定效果")
        
        return len(self.collected_images) > 0

def main():
    print("=" * 60)
    print("📸 ChArUco 项目 - 专业级数据收集 (iPhone摄像头)")
    print("=" * 60)
    
    print(f"📱 iPhone摄像头地址: http://192.168.161.221:4747/video")
    print(f"🎯 目标图像数量: 54张 (专业级标定)")
    print(f"📏 距离覆盖范围: 25cm - 150cm")
    print(f"📐 自由拍摄模式: 您可以自由选择角度和距离")
    
    # 创建数据收集器
    collector = CharucoDataCollector()
    
    # 开始收集
    print("\n按 Enter 键开始专业级iPhone摄像头数据收集...")
    input()
    
    collector.run_collection()

if __name__ == "__main__":
    main() 