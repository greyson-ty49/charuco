import cv2
import numpy as np

def test_camera():
    """
    测试摄像头功能
    """
    print("🎥 正在初始化摄像头...")
    
    # 打开摄像头 (使用iPhone通过droidcam的HTTP流)
    cap = cv2.VideoCapture('http://192.168.161.221:4747/video')
    
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("❌ 错误：无法打开摄像头")
        print("请检查：")
        print("1. 摄像头是否被其他程序占用")
        print("2. 摄像头权限是否开启")
        print("3. 摄像头是否正常连接")
        return False
    
    # 获取iPhone摄像头的原始分辨率（保持不变以保证图像质量）
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📱 iPhone摄像头分辨率: {width} x {height}")
    
    # 计算显示窗口尺寸（一半大小）
    display_width = width // 2
    display_height = height // 2
    print(f"🖥️ 显示窗口尺寸: {display_width} x {display_height} (原始尺寸的50%)")
    
    print(f"✅ 摄像头初始化成功！")
    print(f"📏 摄像头分辨率: {width} x {height}")
    print(f"🖥️ 显示窗口尺寸: {display_width} x {display_height}")
    print(f"🎞️ 帧率: {fps:.1f} FPS")
    print()
    print("📋 操作说明:")
    print("- 按 'q' 键退出程序")
    print("- 按 'c' 键捕获当前画面")
    print("- 按 's' 键保存当前画面")
    print("- 确保画面清晰，光线充足")
    print()
    
    # 计数器
    frame_count = 0
    saved_count = 0
    
    while True:
        # 读取摄像头画面
        ret, frame = cap.read()
        
        if not ret:
            print("❌ 错误：无法读取摄像头画面")
            break
        
        frame_count += 1
        
        # 在画面上添加信息
        # 添加标题
        cv2.putText(frame, "iPhone Camera Test - ChArUco Project", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加操作提示
        cv2.putText(frame, "Press 'q' to quit, 'c' to capture, 's' to save", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加帧数信息
        cv2.putText(frame, f"Frame: {frame_count}", 
                   (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        
        # 在画面中央添加十字线（帮助对焦）
        center_x, center_y = width // 2, height // 2
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
        
        # 缩放图像到一半大小用于显示（保持原始分辨率质量）
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # 显示缩放后的画面
        cv2.imshow('iPhone Camera Test', display_frame)
        
        # 检查按键
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("👋 退出程序...")
            break
        elif key == ord('c'):
            print(f"📸 捕获画面 #{frame_count}")
        elif key == ord('s'):
            saved_count += 1
            filename = f"camera_test_{saved_count}.jpg"
            cv2.imwrite(filename, frame)  # 保存原始分辨率的图像
            print(f"💾 画面已保存为: {filename} (原始分辨率 {width}x{height})")
    
    # 释放摄像头并关闭窗口
    cap.release()
    cv2.destroyAllWindows()
    
    print()
    print("✅ iPhone摄像头测试完成！")
    print(f"📊 总处理帧数: {frame_count}")
    print(f"💾 保存图片数: {saved_count}")
    print(f"📱 摄像头分辨率: {width}x{height}")
    print(f"🖥️ 显示窗口尺寸: {display_width}x{display_height}")
    
    return True

def check_camera_info():
    """
    检查可用的摄像头
    """
    print("🔍 正在检查iPhone摄像头连接...")
    
    # 测试iPhone摄像头连接
    cap = cv2.VideoCapture('http://192.168.161.221:4747/video')
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        display_width = width // 2
        display_height = height // 2
        
        print(f"📹 iPhone摄像头分辨率: {width}x{height}")
        print(f"🖥️ 显示窗口尺寸: {display_width}x{display_height} (50%缩放)")
        
        cap.release()
        print("✅ iPhone摄像头连接成功")
        return True
    else:
        print("❌ 无法连接到iPhone摄像头")
        print("请检查：")
        print("1. DroidCam是否正在iPhone上运行")
        print("2. 网络连接是否正常 (http://192.168.161.221:4747)")
        print("3. 防火墙是否阻止了连接")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("🎥 ChArUco 项目 - iPhone摄像头测试 (DroidCam)")
    print("=" * 50)
    
    # 检查OpenCV版本
    print(f"🔧 OpenCV版本: {cv2.__version__}")
    print(f"📱 iPhone摄像头地址: http://192.168.161.221:4747/video")
    
    # 检查可用摄像头
    if check_camera_info():
        print()
        input("按 Enter 键开始iPhone摄像头测试...")
        test_camera()
    else:
        print("❌ 无法继续，请检查iPhone摄像头连接") 