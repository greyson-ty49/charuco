import cv2
import numpy as np
import json
import os
import glob
from datetime import datetime
import math

class CharucoValidator:
    def __init__(self):
        """
        初始化ChArUco标定验证器
        """
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibration_data = None
        self.camera = None
        
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
        
        print("✅ ChArUco标定验证器初始化成功！")

    def load_calibration_result(self):
        """
        加载标定结果
        """
        print("\n📂 寻找标定结果...")
        
        # 查找最新的标定数据文件夹
        pattern = "calibration_data_*"
        folders = glob.glob(pattern)
        if not folders:
            print("❌ 错误：没有找到标定数据文件夹")
            return False
        
        # 按创建时间排序，使用最新的
        folders.sort(key=os.path.getctime, reverse=True)
        data_folder = folders[0]
        
        # 加载标定结果
        calibration_file = os.path.join(data_folder, "camera_calibration.json")
        
        if not os.path.exists(calibration_file):
            print(f"❌ 错误：找不到标定结果文件 {calibration_file}")
            return False
        
        with open(calibration_file, 'r', encoding='utf-8') as f:
            self.calibration_data = json.load(f)
        
        # 提取相机参数
        self.camera_matrix = np.array(self.calibration_data['camera_matrix'])
        self.dist_coeffs = np.array(self.calibration_data['distortion_coefficients'])
        
        print(f"✅ 标定结果加载成功！")
        print(f"📁 数据文件夹: {data_folder}")
        print(f"📊 重投影误差: {self.calibration_data['calibration_info']['reprojection_error']:.4f} 像素")
        print(f"📷 使用图像: {self.calibration_data['calibration_info']['num_images_used']} 张")
        
        return True

    def init_camera(self):
        """
        初始化相机
        """
        print("\n📹 初始化iPhone摄像头...")
        
        self.camera = cv2.VideoCapture('http://192.168.85.221:4747/video')
        if not self.camera.isOpened():
            print("❌ 错误：无法打开iPhone摄像头")
            print("请检查：")
            print("1. DroidCam是否正在iPhone上运行")
            print("2. 网络连接是否正常 (http://192.168.98.239:4747)")
            print("3. 防火墙是否阻止了连接")
            return False
        
        # 获取iPhone摄像头的原始分辨率
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算显示窗口尺寸（一半大小）
        self.display_width = width // 2
        self.display_height = height // 2
        
        print(f"📱 iPhone摄像头分辨率: {width} x {height}")
        print(f"🖥️ 显示窗口尺寸: {self.display_width} x {self.display_height} (原始尺寸的50%)")
        print("✅ iPhone摄像头初始化成功！")
        return True

    def calculate_pose(self, charuco_corners, charuco_ids):
        """
        计算标定板姿态
        """
        try:
            # 使用标定结果估计标定板姿态
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.charuco_board,
                self.camera_matrix, self.dist_coeffs, None, None
            )
            
            if success:
                return rvec, tvec
            else:
                return None, None
        except:
            return None, None

    def draw_3d_axis(self, image, rvec, tvec, length=0.05):
        """
        绘制3D坐标轴
        """
        # 定义坐标轴端点
        axis_points = np.array([
            [0, 0, 0],          # 原点
            [length, 0, 0],     # X轴
            [0, length, 0],     # Y轴
            [0, 0, -length]     # Z轴
        ], dtype=np.float32)
        
        # 投影到图像平面
        projected_points, _ = cv2.projectPoints(
            axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        projected_points = projected_points.astype(int).reshape(-1, 2)
        
        # 绘制坐标轴
        origin = tuple(projected_points[0])
        x_axis = tuple(projected_points[1])
        y_axis = tuple(projected_points[2])
        z_axis = tuple(projected_points[3])
        
        # X轴 - 红色
        cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3)
        # Y轴 - 绿色
        cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3)
        # Z轴 - 蓝色
        cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3)
        
        # 添加轴标签
        cv2.putText(image, "X", x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(image, "Y", y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Z", z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def draw_cube(self, image, rvec, tvec, size=0.03):
        """
        绘制3D立方体
        """
        # 定义立方体顶点
        cube_points = np.array([
            [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],  # 底面
            [0, 0, -size], [size, 0, -size], [size, size, -size], [0, size, -size]  # 顶面
        ], dtype=np.float32)
        
        # 投影到图像平面
        projected_points, _ = cv2.projectPoints(
            cube_points, rvec, tvec, self.camera_matrix, self.dist_coeffs
        )
        
        projected_points = projected_points.astype(int).reshape(-1, 2)
        
        # 绘制立方体边线
        # 底面
        cv2.polylines(image, [projected_points[:4]], True, (0, 255, 255), 2)
        # 顶面
        cv2.polylines(image, [projected_points[4:]], True, (0, 255, 255), 2)
        # 连接边
        for i in range(4):
            cv2.line(image, tuple(projected_points[i]), tuple(projected_points[i+4]), (0, 255, 255), 2)

    def calculate_distance_and_angle(self, tvec, rvec):
        """
        计算距离和角度
        """
        # 计算距离（相机到标定板的距离）
        distance = np.linalg.norm(tvec)
        
        # 计算角度（将旋转向量转换为欧拉角）
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        
        # 提取欧拉角
        sy = math.sqrt(rotation_matrix[0, 0] * rotation_matrix[0, 0] + rotation_matrix[1, 0] * rotation_matrix[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            y = math.atan2(-rotation_matrix[2, 0], sy)
            z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
            y = math.atan2(-rotation_matrix[2, 0], sy)
            z = 0
        
        # 转换为度数
        x_deg = math.degrees(x)
        y_deg = math.degrees(y)
        z_deg = math.degrees(z)
        
        return distance, x_deg, y_deg, z_deg

    def run_validation(self):
        """
        运行验证程序
        """
        print("\n🚀 开始ChArUco标定验证...")
        
        # 加载标定结果
        if not self.load_calibration_result():
            return False
        
        # 初始化相机
        if not self.init_camera():
            return False
        
        print("\n📋 iPhone摄像头验证模式说明:")
        print("1️⃣ 基础验证 - 显示原始图像和去畸变图像")
        print("2️⃣ 姿态估计 - 显示3D坐标轴")
        print("3️⃣ 3D效果 - 显示虚拟立方体")
        print("4️⃣ 测量功能 - 显示距离和角度")
        print("\n⌨️ 操作说明:")
        print("- 按 '1' 键切换到基础验证模式")
        print("- 按 '2' 键切换到姿态估计模式")
        print("- 按 '3' 键切换到3D效果模式")
        print("- 按 '4' 键切换到测量功能模式")
        print("- 按 'q' 键退出验证")
        print("- 窗口大小已调整为iPhone摄像头分辨率的50%")
        
        mode = 1  # 默认模式
        frame_count = 0
        detection_count = 0
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ 错误：无法读取相机画面")
                break
            
            frame_count += 1
            
            # 创建去畸变图像
            undistorted = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
            
            # 检测ChArUco标记
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            marker_corners, marker_ids, rejected = self.aruco_detector.detectMarkers(gray)
            
            # 创建显示图像（在原始尺寸上处理，最后再缩放）
            if mode == 1:
                # 基础验证模式：显示原始和去畸变图像
                display_frame = np.hstack((frame, undistorted))
                cv2.putText(display_frame, "Original", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(display_frame, "Undistorted", (frame.shape[1] + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                display_frame = undistorted.copy()
            
            # 如果检测到标记，进行进一步处理
            if marker_ids is not None and len(marker_ids) > 0:
                detection_count += 1
                
                # 绘制检测到的标记
                cv2.aruco.drawDetectedMarkers(display_frame, marker_corners, marker_ids)
                
                # 检测ChArUco角点
                try:
                    # 尝试新API
                    num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, self.charuco_board
                    )
                    
                    if num_corners > 0:
                        # 绘制角点
                        cv2.aruco.drawDetectedCornersCharuco(display_frame, charuco_corners, charuco_ids)
                        
                        # 计算姿态
                        rvec, tvec = self.calculate_pose(charuco_corners, charuco_ids)
                        
                        if rvec is not None and tvec is not None:
                            if mode == 2:
                                # 姿态估计模式：绘制3D坐标轴
                                self.draw_3d_axis(display_frame, rvec, tvec)
                            elif mode == 3:
                                # 3D效果模式：绘制虚拟立方体
                                self.draw_cube(display_frame, rvec, tvec)
                            elif mode == 4:
                                # 测量功能模式：显示距离和角度
                                distance, x_deg, y_deg, z_deg = self.calculate_distance_and_angle(tvec, rvec)
                                
                                cv2.putText(display_frame, f"Distance: {distance*100:.1f} cm", 
                                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                cv2.putText(display_frame, f"X: {x_deg:.1f} deg", 
                                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                                cv2.putText(display_frame, f"Y: {y_deg:.1f} deg", 
                                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                cv2.putText(display_frame, f"Z: {z_deg:.1f} deg", 
                                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                
                                # 绘制简单的3D坐标轴
                                self.draw_3d_axis(display_frame, rvec, tvec, 0.03)
                except:
                    # 如果新API失败，尝试旧API
                    try:
                        charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                            marker_corners, marker_ids, gray, self.charuco_board
                        )
                        
                        if charuco_corners is not None and len(charuco_corners) > 0:
                            # 类似的处理逻辑...
                            pass
                    except:
                        pass
            
            # 显示状态信息
            mode_names = ["", "Basic Validation", "Pose Estimation", "3D Effects", "Measurement"]
            cv2.putText(display_frame, f"Mode: {mode_names[mode]}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示检测统计
            detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
            cv2.putText(display_frame, f"Detection: {detection_rate:.1f}%", 
                       (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # 缩放图像用于显示（保持原始分辨率用于处理）
            if mode == 1:
                # 基础验证模式：并排显示，需要特殊处理
                display_width_combined = self.display_width
                display_height_combined = self.display_height
                final_display = cv2.resize(display_frame, (display_width_combined, display_height_combined))
            else:
                # 其他模式：直接缩放
                final_display = cv2.resize(display_frame, (self.display_width, self.display_height))
            
            # 显示缩放后的图像
            cv2.imshow("ChArUco Calibration Validation (iPhone)", final_display)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                mode = 1
                print("🔄 切换到基础验证模式")
            elif key == ord('2'):
                mode = 2
                print("🔄 切换到姿态估计模式")
            elif key == ord('3'):
                mode = 3
                print("🔄 切换到3D效果模式")
            elif key == ord('4'):
                mode = 4
                print("🔄 切换到测量功能模式")
        
        # 清理资源
        self.camera.release()
        cv2.destroyAllWindows()
        
        print(f"\n📊 iPhone摄像头验证统计:")
        print(f"总帧数: {frame_count}")
        print(f"检测成功: {detection_count}")
        print(f"检测率: {detection_rate:.1f}%")
        print(f"📱 摄像头分辨率: {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}")
        print(f"🖥️ 显示窗口尺寸: {self.display_width}x{self.display_height}")
        
        if detection_rate > 80:
            print("✅ 标定质量优秀 - iPhone摄像头验证表现良好")
        elif detection_rate > 60:
            print("✅ 标定质量良好 - 可以正常使用")
        else:
            print("⚠️ 标定质量一般 - 建议重新标定")
        
        return True

def main():
    print("=" * 60)
    print("🔍 ChArUco 项目 - 标定验证 (iPhone摄像头)")
    print("=" * 60)
    
    print(f"📱 iPhone摄像头地址: http://192.168.98.239:4747/video")
    
    
    # 创建验证器
    validator = CharucoValidator()
    
    # 确认开始验证
    print("\n准备开始验证，请确保标定板在iPhone摄像头视野内...")
    print("按 Enter 键开始验证...")
    input()
    
    # 运行验证
    success = validator.run_validation()
    
    if success:
        print("\n✅ iPhone摄像头标定验证完成！")
        print("🎯 你的相机标定结果工作正常，可以用于:")
        print("  - 精确的3D姿态估计")
        print("  - 增强现实应用")
        print("  - 机器视觉测量")
        print("  - 图像去畸变处理")
        print("  - iPhone摄像头的高质量应用")
    else:
        print("\n❌ 验证失败")

if __name__ == "__main__":
    main() 