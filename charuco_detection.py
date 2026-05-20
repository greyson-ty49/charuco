import cv2
import numpy as np

class CharucoDetector:
    def __init__(self):
        """
        初始化ChArUco检测器
        """
        # 读取标定板参数
        self.squares_x = 7  # 棋盘格数量
        self.squares_y = 5
        self.square_length = 0.0335  # 实际格子大小 3.35cm = 0.0335m
        self.marker_length = 0.0335 * 0.75  # ArUco标记大小 (格子大小的75%)
        
        # 创建ArUco字典和ChArUco标定板
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.charuco_board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length,
            self.marker_length,
            self.aruco_dict
        )
        
        # 创建检测器参数
        self.detector_params = cv2.aruco.DetectorParameters()
        
        # 优化检测参数
        self.detector_params.adaptiveThreshWinSizeMin = 3
        self.detector_params.adaptiveThreshWinSizeMax = 23
        self.detector_params.adaptiveThreshWinSizeStep = 10
        self.detector_params.minMarkerPerimeterRate = 0.03
        self.detector_params.maxMarkerPerimeterRate = 4.0
        
        # 创建ArUco检测器
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # 统计信息
        self.stats = {
            'total_frames': 0,
            'frames_with_detection': 0,
            'max_corners_detected': 0,
            'max_markers_detected': 0
        }
        
        print("✅ ChArUco检测器初始化成功！")
        print(f"📏 标定板尺寸: {self.squares_x}x{self.squares_y}")
        print(f"📦 格子实际大小: {self.square_length*100:.2f}cm")
        print(f"🔍 ArUco字典: DICT_4X4_50")

    def detect_charuco(self, frame):
        """
        检测ChArUco标定板
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测ArUco标记
        marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        charuco_corners = None
        charuco_ids = None
        
        # 如果检测到ArUco标记，继续检测ChArUco角点
        if marker_ids is not None and len(marker_ids) > 0:
            # 插值ChArUco角点 - 修复OpenCV兼容性问题
            try:
                # 使用新的API调用方式 (返回3个值)
                num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, self.charuco_board
                )
                
                # 检查返回值
                if num_corners <= 0:
                    charuco_corners, charuco_ids = None, None
                    
            except Exception as e:
                # 尝试旧的API调用方式 (返回2个值)
                try:
                    charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, self.charuco_board
                    )
                except Exception as e2:
                    charuco_corners, charuco_ids = None, None
        
        return marker_corners, marker_ids, charuco_corners, charuco_ids

    def draw_detections(self, frame, marker_corners, marker_ids, charuco_corners, charuco_ids):
        """
        在图像上绘制检测结果
        """
        height, width = frame.shape[:2]
        
        # 绘制ArUco标记
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        
        # 绘制ChArUco角点
        if charuco_corners is not None and hasattr(charuco_corners, '__len__') and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
        
        # 添加检测信息
        marker_count = len(marker_ids) if marker_ids is not None else 0
        corner_count = len(charuco_corners) if charuco_corners is not None and hasattr(charuco_corners, '__len__') else 0
        
        # 信息面板背景
        cv2.rectangle(frame, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 140), (255, 255, 255), 2)
        
        # 显示摄像头信息
        cv2.putText(frame, "iPhone Camera - ChArUco Detection", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 显示检测信息
        cv2.putText(frame, f"ArUco Markers: {marker_count}", 
                   (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"ChArUco Corners: {corner_count}", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 检测质量评估
        max_possible_corners = (self.squares_x - 1) * (self.squares_y - 1)  # 6*4 = 24
        detection_quality = (corner_count / max_possible_corners) * 100 if max_possible_corners > 0 else 0
        
        color = (0, 255, 0) if detection_quality > 50 else (0, 255, 255) if detection_quality > 20 else (0, 0, 255)
        cv2.putText(frame, f"Quality: {detection_quality:.1f}%", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # 显示状态
        if corner_count > 0:
            status = "GOOD DETECTION" if detection_quality > 50 else "PARTIAL DETECTION"
            cv2.putText(frame, status, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            cv2.putText(frame, "NO DETECTION", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 更新统计信息
        self.stats['total_frames'] += 1
        if corner_count > 0:
            self.stats['frames_with_detection'] += 1
        self.stats['max_corners_detected'] = max(self.stats['max_corners_detected'], corner_count)
        self.stats['max_markers_detected'] = max(self.stats['max_markers_detected'], marker_count)
        
        return frame

    def run_detection(self):
        """
        运行实时检测
        """
        cap = cv2.VideoCapture('http://192.168.98.239:4747/video')
        
        if not cap.isOpened():
            print("❌ 错误：无法打开iPhone摄像头")
            print("请检查：")
            print("1. DroidCam是否正在iPhone上运行")
            print("2. 网络连接是否正常 (http://192.168.161.221:4747)")
            print("3. 防火墙是否阻止了连接")
            return False
        
        # 获取iPhone摄像头的原始分辨率（保持不变以保证检测精度）
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算显示窗口尺寸（一半大小）
        display_width = width // 2
        display_height = height // 2
        
        print(f"📱 iPhone摄像头分辨率: {width} x {height}")
        print(f"🖥️ 显示窗口尺寸: {display_width} x {display_height} (原始尺寸的50%)")
        
        print()
        print("🔍 ChArUco检测器已启动！(iPhone摄像头)")
        print("📋 操作说明:")
        print("- 将打印的ChArUco标定板放在iPhone摄像头前面")
        print("- 尝试不同的距离和角度")
        print("- 按 'q' 键退出程序")
        print("- 按 's' 键保存当前检测结果")
        print("- 确保光线充足，避免反光")
        print("- iPhone摄像头质量更好，检测应该更准确")
        print()
        
        saved_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 错误：无法读取摄像头画面")
                break
            
            # 检测ChArUco
            marker_corners, marker_ids, charuco_corners, charuco_ids = self.detect_charuco(frame)
            
            # 绘制检测结果
            frame = self.draw_detections(frame, marker_corners, marker_ids, charuco_corners, charuco_ids)
            
            # 添加操作提示
            cv2.putText(frame, "Press 'q' to quit, 's' to save detection", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 缩放图像到一半大小用于显示（保持原始分辨率质量）
            display_frame = cv2.resize(frame, (display_width, display_height))
            
            # 显示缩放后的画面
            cv2.imshow('ChArUco Detection (iPhone)', display_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 退出检测程序...")
                break
            elif key == ord('s'):
                if charuco_corners is not None and hasattr(charuco_corners, '__len__') and len(charuco_corners) > 0:
                    saved_count += 1
                    filename = f"charuco_detection_{saved_count}.jpg"
                    cv2.imwrite(filename, frame)  # 保存原始分辨率的图像
                    print(f"💾 检测结果已保存为: {filename} (原始分辨率 {width}x{height})")
                else:
                    print("⚠️ 没有检测到ChArUco特征，无法保存")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 显示统计信息
        print()
        print("📊 iPhone摄像头检测统计:")
        print(f"总帧数: {self.stats['total_frames']}")
        print(f"成功检测帧数: {self.stats['frames_with_detection']}")
        success_rate = (self.stats['frames_with_detection'] / self.stats['total_frames']) * 100 if self.stats['total_frames'] > 0 else 0
        print(f"检测成功率: {success_rate:.1f}%")
        print(f"最大角点数: {self.stats['max_corners_detected']}")
        print(f"最大标记数: {self.stats['max_markers_detected']}")
        print(f"保存图像数: {saved_count}")
        print(f"📱 摄像头分辨率: {width}x{height}")
        print(f"🖥️ 显示窗口尺寸: {display_width}x{display_height}")
        if saved_count > 0:
            print("✅ 检测结果已保存，可用于后续标定")
        print("🎯 iPhone摄像头检测程序结束")
        
        return True

def main():
    print("=" * 60)
    print("🎯 ChArUco 项目 - 标定板检测 (iPhone摄像头)")
    print("=" * 60)
    
    # 检查OpenCV版本
    print(f"🔧 OpenCV版本: {cv2.__version__}")
    print(f"📱 iPhone摄像头地址: http://192.168.161.221:4747/video")
    
    # 创建检测器
    detector = CharucoDetector()
    
    # 开始检测
    print("\n按 Enter 键开始iPhone摄像头检测...")
    input()
    
    detector.run_detection()

if __name__ == "__main__":
    main() 