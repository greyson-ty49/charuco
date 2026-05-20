import cv2
import numpy as np
import os

def generate_charuco_board():
    """
    生成ChArUco标定板并保存为图片
    """
    # ChArUco标定板参数
    # 棋盘格数量 (内部角点数量)
    squares_x = 7  # 水平方向格子数
    squares_y = 5  # 垂直方向格子数
    
    # 每个格子的大小 (像素)
    square_length = 100  # 格子边长，像素
    
    # ArUco标记的大小 (相对于格子的比例)
    marker_length = 75  # ArUco标记边长，像素
    
    # ArUco字典 (使用4x4字典，包含50个标记)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # 创建ChArUco标定板
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),    # 棋盘格数量
        square_length,             # 格子大小
        marker_length,             # 标记大小
        aruco_dict                 # ArUco字典
    )
    
    # 生成标定板图像
    # 计算图像大小 (留一些边距)
    img_size = (squares_x * square_length + 200, squares_y * square_length + 200)
    
    # 创建标定板图像
    board_img = board.generateImage(img_size)
    
    # 保存图像
    output_filename = "charuco_board_7x5.png"
    cv2.imwrite(output_filename, board_img)
    
    print(f"✅ ChArUco标定板已生成: {output_filename}")
    print(f"📏 图像尺寸: {img_size[0]} x {img_size[1]} 像素")
    print(f"🎯 棋盘格数量: {squares_x} x {squares_y}")
    print(f"📦 每个格子大小: {square_length} x {square_length} 像素")
    print(f"🔍 ArUco标记大小: {marker_length} x {marker_length} 像素")
    print()
    print("📋 打印说明:")
    print("1. 用A4纸打印这个图片")
    print("2. 打印时选择'适合页面大小'或'缩放到适合'")
    print("3. 打印后用尺子测量一个格子的实际大小(厘米)")
    print("4. 记录下实际测量的格子大小，后面会用到")
    print()
    
    # 同时保存标定板参数到文件
    params_filename = "charuco_board_params.txt"
    with open(params_filename, 'w', encoding='utf-8') as f:
        f.write("ChArUco标定板参数\n")
        f.write("================\n")
        f.write(f"棋盘格数量: {squares_x} x {squares_y}\n")
        f.write(f"格子大小(像素): {square_length}\n")
        f.write(f"ArUco标记大小(像素): {marker_length}\n")
        f.write(f"ArUco字典: DICT_4X4_50\n")
        f.write("\n")
        f.write("打印后请测量:\n")
        f.write("实际格子大小(厘米): _____ cm\n")
        f.write("(请用尺子测量打印后的格子大小并填写)\n")
    
    print(f"📄 参数文件已保存: {params_filename}")
    
    return output_filename

if __name__ == "__main__":
    # 检查OpenCV是否安装
    try:
        cv2_version = cv2.__version__
        print(f"🔧 OpenCV版本: {cv2_version}")
    except:
        print("❌ 错误: 未安装OpenCV")
        print("请先安装: pip install opencv-python")
        exit(1)
    
    # 生成ChArUco标定板
    generate_charuco_board() 