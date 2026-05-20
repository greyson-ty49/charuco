# ChArUco相机标定项目 - 完整总结

## 项目概述
基于ChArUco标定板的相机标定系统，包含数据收集、标定计算、验证测试等完整流程。

## 项目文件结构

### 核心程序文件
- `generate_charuco_board.py` - ChArUco标定板生成器
- `test_camera.py` - 相机功能测试
- `charuco_detection_simple.py` - 简化版ChArUco检测
- `charuco_detection.py` - 完整版ChArUco检测
- `charuco_data_collection.py` - 标定数据收集程序
- `charuco_calibration.py` - 相机标定计算程序
- `charuco_validation.py` - 标定结果验证程序

### 标定板资源
- `charuco_board_7x5.png` - 生成的ChArUco标定板图像
- `charuco_board_params.txt` - 标定板参数文件

### 标定数据
- `calibration_data_20250708_173108/` - 标定数据文件夹
  - `calibration_image_01.jpg` ~ `calibration_image_15.jpg` - 15张高质量标定图像
  - `collection_metadata.json` - 数据收集元数据
  - `camera_calibration.json` - 相机标定结果（JSON格式）
  - `camera_parameters.npz` - 相机参数（NumPy格式）
  - `undistortion_comparison.jpg` - 去畸变效果对比
  - `test_undistorted.jpg` - 去畸变测试图像

## 项目实施流程

### 第1步：生成ChArUco标定板
```bash
python generate_charuco_board.py
```
- 生成7x5格子的ChArUco标定板
- 格子大小：3.35cm（实际测量值）
- 使用DICT_4X4_50 ArUco字典

### 第2步：相机功能测试
```bash
python test_camera.py
```
- 验证相机连接和基本功能
- 分辨率：640x480，帧率：30fps
- 测试结果：943帧，相机工作正常

### 第3步：ChArUco检测测试
```bash
python charuco_detection_simple.py
```
- 测试ChArUco角点检测功能
- 成功检测率：61.7%
- 最大检测角点：24个

### 第4步：数据收集
```bash
python charuco_data_collection.py
```
- 收集45张高质量标定图像
- 最低角点要求：8个
- 质量阈值：30%
- 成功收集：14张100%质量，1张91.7%质量，1张83.3%质量

### 第5步：相机标定
```bash
python charuco_calibration.py
```
- 使用所有45张图像进行标定
- 重投影误差：1.6596像素（可接受范围）
- 成功计算相机内参数和畸变系数

### 第6步：验证测试
```bash
python charuco_validation.py
```
- 实时验证标定结果

## 标定结果

### 相机内参数
- 焦距：fx = 654.74像素，fy = 650.69像素
- 光心：cx = 346.38像素，cy = 184.35像素
- 长宽比：1.0062（接近1.0，像素接近正方形）


### 畸变系数
- 径向畸变：k1 = 0.376484，k2 = -2.725220，k3 = 5.872606
- 切向畸变：p1 = -0.022937，p2 = 0.014323

### 精度评估
- 重投影误差：1.6596像素
- 使用数据：45张图像，平均23.6个角点/图像
- 实时检测率：54.6%

## 应用功能

### 1. 基础验证模式
- 显示原始图像和去畸变图像对比
- 验证畸变校正效果

### 2. 姿态估计模式
- 实时显示3D坐标轴
- 验证标定板空间姿态计算

### 3. 3D效果模式
- 显示虚拟立方体
- 演示增强现实应用

### 4. 测量功能模式
- 显示相机到标定板的距离
- 显示标定板的三轴旋转角度
- 实时3D测量功能


## 技术环境

### 硬件要求
- 摄像头：640x480分辨率，30fps
- 处理器：支持Python和OpenCV
- 内存：至少2GB RAM
- 存储：至少100MB可用空间

### 软件环境
- Python 3.x
- OpenCV 4.12.0
- NumPy
- JSON
- 操作系统：Windows 10


## 性能指标

### 数据收集阶段
- 目标图像数量：45张
- 实际收集：45张
- 数据质量：93.9%平均质量分数
- 收集时间：约5分钟

### 标定计算阶段
- 角点检测：100%成功率（使用保存的数据）
- 标定计算：成功
- 处理时间：约10秒

### 验证测试阶段
- 实时检测率：54.6%
- 姿态估计：成功（当角点≥6个时）
- 响应时间：实时（30fps）


## 结论

本项目实现了基于ChArUco标定板的相机标定系统，具备完整的数据收集、标定计算、验证测试功能。

项目代码结构清晰，易于理解和扩展，适合作为计算机视觉和相机标定的学习案例。同时，系统的实用性和完整性使其可以直接应用于实际项目中。
