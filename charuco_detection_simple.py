import cv2
import numpy as np

def test_charuco_detection():
    """
    简化版ChArUco检测测试
    """
    # 标定板参数
    squares_x = 7
    squares_y = 5
    square_length = 0.0335  # 3.35cm
    marker_length = 0.0335 * 0.75  # 75% of square size
    
    # 创建ArUco字典和ChArUco标定板
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    charuco_board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )
    
    # 创建检测器
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
    
    # 打开iPhone摄像头
    cap = cv2.VideoCapture('http://192.168.161.221:4747/video')
    if not cap.isOpened():
        print("❌ 无法打开iPhone摄像头")
        print("请检查：")
        print("1. DroidCam是否正在iPhone上运行")
        print("2. 网络连接是否正常 (http://192.168.161.221:4747)")
        print("3. 防火墙是否阻止了连接")
        return
    
    # 获取iPhone摄像头的原始分辨率（保持不变以保证检测精度）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 计算显示窗口尺寸（一半大小）
    display_width = width // 2
    display_height = height // 2
    
    print(f"📱 iPhone摄像头分辨率: {width} x {height}")
    print(f"🖥️ 显示窗口尺寸: {display_width} x {display_height} (原始尺寸的50%)")
    
    print("🎯 简化版ChArUco检测器启动 (iPhone摄像头)")
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
            break
            
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 检测ArUco标记
        marker_corners, marker_ids, rejected = aruco_detector.detectMarkers(gray)
        
        # 绘制ArUco标记
        if marker_ids is not None and len(marker_ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        
        # 尝试检测ChArUco角点
        charuco_corners = None
        charuco_ids = None
        
        if marker_ids is not None and len(marker_ids) > 0:
            # 尝试不同的API调用方式
            try:
                # 方法1: 新API (返回3个值)
                num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners, marker_ids, gray, charuco_board
                )
                if num_corners <= 0:
                    charuco_corners, charuco_ids = None, None
            except:
                try:
                    # 方法2: 旧API (返回2个值)
                    charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        marker_corners, marker_ids, gray, charuco_board
                    )
                except Exception as e:
                    # 静默处理错误，避免控制台输出过多
                    charuco_corners, charuco_ids = None, None
        
        # 绘制ChArUco角点
        if charuco_corners is not None and hasattr(charuco_corners, '__len__') and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
        
        # 显示信息
        marker_count = len(marker_ids) if marker_ids is not None else 0
        corner_count = len(charuco_corners) if charuco_corners is not None and hasattr(charuco_corners, '__len__') else 0
        
        # 信息面板
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 100), (255, 255, 255), 2)
        
        # 显示摄像头信息
        cv2.putText(frame, "iPhone Camera - ChArUco Detection", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"ArUco Markers: {marker_count}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"ChArUco Corners: {corner_count}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 状态
        if corner_count > 0:
            cv2.putText(frame, "DETECTION OK!", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO CORNERS", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 缩放图像到一半大小用于显示（保持原始分辨率质量）
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # 显示缩放后的画面
        cv2.imshow('ChArUco Detection - Simple (iPhone)', display_frame)
        
        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if corner_count > 0:
                saved_count += 1
                filename = f"charuco_simple_{saved_count}.jpg"
                cv2.imwrite(filename, frame)  # 保存原始分辨率的图像
                print(f"💾 保存图像: {filename} (原始分辨率 {width}x{height})")
            else:
                print("⚠️ 没有检测到角点，无法保存")
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎯 简化版ChArUco检测器测试完成 (iPhone摄像头)")
    print(f"💾 保存图片数: {saved_count}")
    if saved_count > 0:
        print(f"📱 图片分辨率: {width}x{height}")
        print("✅ 检测结果已保存，可用于后续标定")

if __name__ == "__main__":
    test_charuco_detection() 