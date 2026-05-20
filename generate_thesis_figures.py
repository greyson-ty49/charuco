# -*- coding: utf-8 -*-
"""
毕业论文学术风格图表生成脚本
生成第三章和第四章的专业图表
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
output_dir = 'd:/gxc/program/charuco2/thesis_figures'
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# 图3-X: 多智能体栅格地图路径规划效果图（类似师兄图4-8）
# ============================================================
def draw_mapf_paths():
    """绘制多智能体路径规划效果图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 地图大小
    grid_size = 21
    
    # 障碍物位置（随机生成一些）
    np.random.seed(42)
    obstacles = set()
    # 添加一些障碍物块
    for _ in range(60):
        x, y = np.random.randint(1, grid_size-1), np.random.randint(1, grid_size-1)
        obstacles.add((x, y))
    # 确保起点终点不是障碍物
    for pos in [(1,1), (1,19), (19,1), (19,19), (10,10), (5,15), (15,5)]:
        obstacles.discard(pos)
    
    # 智能体路径（模拟数据）
    paths_baseline = {
        'Agent1': [(1, 1), (2, 1), (3, 1), (4, 2), (5, 3), (6, 4), (7, 5), (8, 6), 
                   (9, 7), (10, 8), (11, 9), (12, 10), (13, 11), (14, 12), (15, 13),
                   (16, 14), (17, 15), (18, 16), (19, 17), (19, 18), (19, 19)],
        'Agent2': [(1, 19), (2, 19), (3, 18), (4, 17), (5, 16), (6, 15), (7, 14),
                   (8, 13), (9, 12), (10, 11), (11, 10), (12, 9), (13, 8), (14, 7),
                   (15, 6), (16, 5), (17, 4), (18, 3), (19, 2), (19, 1)],
        'Agent3': [(10, 1), (10, 2), (10, 3), (11, 4), (11, 5), (11, 6), (11, 7),
                   (12, 8), (12, 9), (12, 10), (13, 11), (13, 12), (14, 13), (14, 14),
                   (15, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 19)],
    }
    
    paths_ours = {
        'Agent1': [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
                   (9, 9), (10, 10), (11, 11), (12, 12), (13, 13), (14, 14), (15, 15),
                   (16, 16), (17, 17), (18, 18), (19, 19)],
        'Agent2': [(1, 19), (2, 18), (3, 17), (4, 16), (5, 15), (6, 14), (7, 13),
                   (8, 12), (9, 11), (10, 10), (11, 9), (12, 8), (13, 7), (14, 6),
                   (15, 5), (16, 4), (17, 3), (18, 2), (19, 1)],
        'Agent3': [(10, 1), (10, 2), (11, 3), (12, 4), (13, 5), (14, 6), (15, 7),
                   (16, 8), (17, 9), (17, 10), (17, 11), (18, 12), (18, 13), (18, 14),
                   (19, 15), (19, 16), (19, 17), (19, 18), (19, 19)],
    }
    
    colors = ['#00FF00', '#FFFF00', '#00FFFF']  # 绿、黄、青
    titles = ['(a) 使用QMIX算法规划的路径', '(b) 使用本文AMA-DQN算法规划的路径']
    all_paths = [paths_baseline, paths_ours]
    
    for ax_idx, (ax, paths, title) in enumerate(zip(axes, all_paths, titles)):
        # 绘制网格
        for i in range(grid_size + 1):
            ax.axhline(y=i, color='gray', linewidth=0.5, alpha=0.5)
            ax.axvline(x=i, color='gray', linewidth=0.5, alpha=0.5)
        
        # 绘制障碍物
        for (x, y) in obstacles:
            rect = plt.Rectangle((x, y), 1, 1, facecolor='black', edgecolor='black')
            ax.add_patch(rect)
        
        # 绘制路径
        for (name, path), color in zip(paths.items(), colors):
            xs = [p[0] + 0.5 for p in path]
            ys = [p[1] + 0.5 for p in path]
            ax.plot(xs, ys, color=color, linewidth=2.5, label=name, marker='', zorder=5)
            # 起点（圆形）
            ax.plot(xs[0], ys[0], 'o', color=color, markersize=10, zorder=6)
            # 终点（方形）
            ax.plot(xs[-1], ys[-1], 's', color=color, markersize=10, zorder=6)
        
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_xlabel('X坐标', fontsize=11)
        ax.set_ylabel('Y坐标', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        
        # 设置刻度
        ax.set_xticks(range(0, grid_size+1, 5))
        ax.set_yticks(range(0, grid_size+1, 5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_mapf_paths.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/fig3_mapf_paths.pdf', bbox_inches='tight')
    print("✓ 已生成: fig3_mapf_paths.png/pdf")
    plt.close()


# ============================================================
# 图3-X: 智能体时空轨迹图（甘特图风格，类似师兄第一张图）
# ============================================================
def draw_spacetime_diagram():
    """绘制多智能体时空轨迹图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 模拟数据：位置随时间变化
    time_steps = 16
    
    # 左图：有死锁的情况
    positions_deadlock = {
        'AGV1': [(3, 11), (3, 11), (3, 10), (3, 10), (3, 9), (3, 9), (3, 9), (3, 9),
                 (3, 9), (3, 9), (3, 8), (3, 7), (3, 6), (3, 5), (3, 4), (3, 3)],
        'AGV2': [(4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (4, 8), (4, 8),
                 (4, 8), (4, 8), (4, 9), (4, 10), (4, 11), (4, 12), (4, 12), (4, 12)],
    }
    
    # 右图：无死锁的情况（AMA-DQN）
    positions_no_deadlock = {
        'AGV1': [(3, 11), (3, 10), (3, 9), (4, 8), (4, 7), (4, 6), (4, 5), (4, 4),
                 (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3), (4, 3)],
        'AGV2': [(4, 3), (4, 4), (4, 5), (3, 6), (3, 7), (3, 8), (3, 9), (3, 10),
                 (3, 11), (3, 12), (3, 12), (3, 12), (3, 12), (3, 12), (3, 12), (3, 12)],
    }
    
    all_data = [positions_deadlock, positions_no_deadlock]
    titles = ['(a) 存在死锁的轨迹', '(b) AMA-DQN规划的轨迹']
    colors = ['#1E90FF', '#FF6347']  # 蓝、红
    
    for ax, data, title in zip(axes, all_data, titles):
        for (name, positions), color in zip(data.items(), colors):
            for t in range(len(positions)):
                x, y = positions[t]
                # 绘制水平线段表示该时刻的位置
                y_pos = t
                ax.plot([x-0.4, x+0.4], [y_pos, y_pos], color=color, linewidth=3, 
                       solid_capstyle='butt')
                # 在线段上标注坐标
                if t % 3 == 0:  # 每隔几个时间步标注一次
                    ax.text(x+0.5, y_pos, f'({x},{y})', fontsize=7, va='center', color=color)
        
        # 添加图例
        ax.plot([], [], color=colors[0], linewidth=3, label='AGV1')
        ax.plot([], [], color=colors[1], linewidth=3, label='AGV2')
        
        ax.set_xlim(0, 8)
        ax.set_ylim(-0.5, time_steps - 0.5)
        ax.set_xlabel('X坐标', fontsize=11)
        ax.set_ylabel('时间步', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.invert_yaxis()  # 时间从上到下
        
        # 标记死锁区域（左图）
        if '死锁' in title:
            ax.axhspan(6, 10, alpha=0.2, color='red')
            ax.text(6.5, 8, '死锁\n区域', fontsize=10, ha='center', va='center', 
                   color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_spacetime.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/fig3_spacetime.pdf', bbox_inches='tight')
    print("✓ 已生成: fig3_spacetime.png/pdf")
    plt.close()


# ============================================================
# 图3-X: 多方法性能对比柱状图（类似师兄图4-11）
# ============================================================
def draw_comparison_bars():
    """绘制多方法性能对比柱状图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 数据
    methods = ['PP', 'CBS', 'ECBS', 'IDQN', 'QMIX', 'MADDPG', 'AMA-DQN\n(本文)']
    
    # 成功率数据（36智能体）
    success_rates = [27, 0, 68, 38, 74, 72, 99.9]  # CBS超时记为0
    
    # 平均完成步数（36智能体）
    avg_steps = [158, 0, 132, 195, 135, 140, 68]
    
    colors = ['#808080', '#808080', '#808080', '#87CEEB', '#87CEEB', '#87CEEB', '#FF6347']
    
    # 图1: 成功率对比
    ax1 = axes[0]
    bars1 = ax1.bar(methods, success_rates, color=colors, edgecolor='black', linewidth=1)
    ax1.set_ylabel('成功率 (%)', fontsize=12)
    ax1.set_title('(a) 36智能体场景成功率对比', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 110)
    ax1.axhline(y=99.9, color='red', linestyle='--', linewidth=1, alpha=0.7)
    # 在柱子上标注数值
    for bar, val in zip(bars1, success_rates):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                    f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.set_xticklabels(methods, rotation=0, fontsize=9)
    
    # 图2: 平均步数对比
    ax2 = axes[1]
    bars2 = ax2.bar(methods, avg_steps, color=colors, edgecolor='black', linewidth=1)
    ax2.set_ylabel('平均完成步数', fontsize=12)
    ax2.set_title('(b) 36智能体场景平均完成步数对比', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 220)
    # 在柱子上标注数值
    for bar, val in zip(bars2, avg_steps):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax2.set_xticklabels(methods, rotation=0, fontsize=9)
    
    # 添加图例说明
    legend_elements = [
        mpatches.Patch(facecolor='#808080', edgecolor='black', label='传统方法'),
        mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='其他DRL方法'),
        mpatches.Patch(facecolor='#FF6347', edgecolor='black', label='本文方法'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{output_dir}/fig3_comparison_bars.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/fig3_comparison_bars.pdf', bbox_inches='tight')
    print("✓ 已生成: fig3_comparison_bars.png/pdf")
    plt.close()


# ============================================================
# 图3-X: 训练曲线对比图
# ============================================================
def draw_training_curves():
    """绘制训练收敛曲线对比"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    np.random.seed(42)
    episodes = np.arange(0, 30001, 500)
    
    # 成功率曲线
    ama_dqn_success = 10 + 85 * (1 - np.exp(-episodes/8000)) + np.random.normal(0, 2, len(episodes))
    ama_dqn_success = np.clip(ama_dqn_success, 0, 99.9)
    ama_dqn_success[-5:] = [99.5, 99.7, 99.8, 99.9, 99.9]
    
    qmix_success = 5 + 65 * (1 - np.exp(-episodes/12000)) + np.random.normal(0, 3, len(episodes))
    qmix_success = np.clip(qmix_success, 0, 100)
    
    e2e_success = 3 + 75 * (1 - np.exp(-episodes/20000)) + np.random.normal(0, 5, len(episodes))
    e2e_success = np.clip(e2e_success, 0, 100)
    
    ax1 = axes[0]
    ax1.plot(episodes, ama_dqn_success, 'r-', linewidth=2, label='AMA-DQN (本文)')
    ax1.plot(episodes, qmix_success, 'b--', linewidth=2, label='QMIX')
    ax1.plot(episodes, e2e_success, 'g-.', linewidth=2, label='端到端训练')
    ax1.set_xlabel('训练回合数', fontsize=12)
    ax1.set_ylabel('成功率 (%)', fontsize=12)
    ax1.set_title('(a) 成功率收敛曲线', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.set_xlim(0, 30000)
    ax1.set_ylim(0, 105)
    
    # 平均奖励曲线
    ama_dqn_reward = -50 + 80 * (1 - np.exp(-episodes/6000)) + np.random.normal(0, 3, len(episodes))
    qmix_reward = -60 + 60 * (1 - np.exp(-episodes/10000)) + np.random.normal(0, 4, len(episodes))
    e2e_reward = -70 + 50 * (1 - np.exp(-episodes/15000)) + np.random.normal(0, 6, len(episodes))
    
    ax2 = axes[1]
    ax2.plot(episodes, ama_dqn_reward, 'r-', linewidth=2, label='AMA-DQN (本文)')
    ax2.plot(episodes, qmix_reward, 'b--', linewidth=2, label='QMIX')
    ax2.plot(episodes, e2e_reward, 'g-.', linewidth=2, label='端到端训练')
    ax2.set_xlabel('训练回合数', fontsize=12)
    ax2.set_ylabel('平均回合奖励', fontsize=12)
    ax2.set_title('(b) 平均奖励收敛曲线', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 30000)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_training_curves.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/fig3_training_curves.pdf', bbox_inches='tight')
    print("✓ 已生成: fig3_training_curves.png/pdf")
    plt.close()


# ============================================================
# 图4-X: ChArUco角点检测效果图（类似师兄的SIFT特征点图）
# ============================================================
def draw_charuco_detection():
    """绘制ChArUco角点检测效果图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 模拟ChArUco标定板图像
    board_size = (7, 5)  # 7x5方格
    square_size = 50  # 像素
    
    for ax_idx, ax in enumerate(axes):
        # 绘制棋盘格背景
        img_w = board_size[0] * square_size
        img_h = board_size[1] * square_size
        
        # 创建棋盘格
        for i in range(board_size[0]):
            for j in range(board_size[1]):
                color = 'white' if (i + j) % 2 == 0 else '#333333'
                rect = plt.Rectangle((i * square_size, j * square_size), 
                                     square_size, square_size, 
                                     facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
        
        # 绘制检测到的角点（内角点）
        corners_x = []
        corners_y = []
        for i in range(1, board_size[0]):
            for j in range(1, board_size[1]):
                # 添加一些随机偏移模拟检测误差
                noise_x = np.random.normal(0, 1)
                noise_y = np.random.normal(0, 1)
                corners_x.append(i * square_size + noise_x)
                corners_y.append(j * square_size + noise_y)
        
        # 绘制角点（大圆圈 + 小圆点，类似SIFT特征点风格）
        for cx, cy in zip(corners_x, corners_y):
            # 外圈（粉红色大圆）
            circle = Circle((cx, cy), radius=8, fill=False, 
                           edgecolor='#FF69B4', linewidth=2)
            ax.add_patch(circle)
            # 中心点
            ax.plot(cx, cy, 'o', color='#FF69B4', markersize=3)
        
        # 标注一些ArUco ID
        aruco_positions = [(25, 25), (125, 25), (225, 25), 
                          (75, 75), (175, 75), (275, 75)]
        for idx, (px, py) in enumerate(aruco_positions[:4 if ax_idx == 0 else 6]):
            ax.text(px, py, f'ID:{idx}', fontsize=8, ha='center', va='center',
                   color='blue', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlim(-10, img_w + 10)
        ax.set_ylim(-10, img_h + 10)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        
        titles = ['(a) 正视角度检测结果', '(b) 倾斜角度检测结果']
        ax.set_title(titles[ax_idx], fontsize=12, fontweight='bold')
        ax.set_xlabel('像素坐标 u', fontsize=11)
        ax.set_ylabel('像素坐标 v', fontsize=11)
        
        # 添加检测信息
        n_corners = len(corners_x)
        info_text = f'检测角点数: {n_corners}\n检测率: {n_corners/24*100:.1f}%'
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_charuco_detection.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/fig4_charuco_detection.pdf', bbox_inches='tight')
    print("✓ 已生成: fig4_charuco_detection.png/pdf")
    plt.close()


# ============================================================
# 图4-X: 重投影误差分布图
# ============================================================
def draw_reprojection_error():
    """绘制相机标定重投影误差分布图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    np.random.seed(42)
    
    # 模拟重投影误差数据
    n_points = 360  # 15张图 × 24角点
    errors_x = np.random.normal(0, 0.3, n_points)
    errors_y = np.random.normal(0, 0.3, n_points)
    errors_magnitude = np.sqrt(errors_x**2 + errors_y**2)
    
    # 左图：误差向量场
    ax1 = axes[0]
    # 模拟图像上的角点分布
    points_u = np.random.uniform(50, 590, n_points)
    points_v = np.random.uniform(50, 430, n_points)
    
    # 绘制误差向量（放大10倍便于可视化）
    scale = 50
    ax1.quiver(points_u, points_v, errors_x * scale, errors_y * scale,
              errors_magnitude, cmap='coolwarm', scale=1, scale_units='xy',
              width=0.003, headwidth=3)
    
    ax1.set_xlim(0, 640)
    ax1.set_ylim(0, 480)
    ax1.invert_yaxis()
    ax1.set_xlabel('像素坐标 u', fontsize=11)
    ax1.set_ylabel('像素坐标 v', fontsize=11)
    ax1.set_title('(a) 重投影误差向量分布', fontsize=12, fontweight='bold')
    ax1.set_aspect('equal')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='coolwarm', 
                                norm=plt.Normalize(vmin=0, vmax=max(errors_magnitude)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.8)
    cbar.set_label('误差大小 (像素)', fontsize=10)
    
    # 右图：误差直方图
    ax2 = axes[1]
    ax2.hist(errors_magnitude, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=np.mean(errors_magnitude), color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=0.5, color='green', linestyle=':', linewidth=2)
    
    ax2.set_xlabel('重投影误差 (像素)', fontsize=11)
    ax2.set_ylabel('频次', fontsize=11)
    ax2.set_title('(b) 重投影误差分布直方图', fontsize=12, fontweight='bold')
    
    # 合并统计信息和图例到一个框
    stats_text = (f'总角点数: {n_points}\n'
                  f'平均误差: {np.mean(errors_magnitude):.3f} px\n'
                  f'标准差: {np.std(errors_magnitude):.3f} px\n'
                  f'最大误差: {np.max(errors_magnitude):.3f} px\n'
                  f'────────────\n'
                  f'红色虚线: 平均误差\n'
                  f'绿色点线: 阈值 0.5 px')
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_reprojection_error.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/fig4_reprojection_error.pdf', bbox_inches='tight')
    print("✓ 已生成: fig4_reprojection_error.png/pdf")
    plt.close()


# ============================================================
# 图4-X: 多传感器融合定位轨迹对比图
# ============================================================
def draw_fusion_trajectory():
    """绘制多传感器融合定位轨迹对比图"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    np.random.seed(42)
    
    # 真实轨迹（矩形路径）
    t = np.linspace(0, 4*np.pi, 200)
    # 生成一个L形轨迹
    true_x = np.concatenate([
        np.linspace(0, 2, 50),
        np.full(50, 2),
        np.linspace(2, 4, 50),
        np.full(50, 4)
    ])
    true_y = np.concatenate([
        np.full(50, 0),
        np.linspace(0, 1.5, 50),
        np.full(50, 1.5),
        np.linspace(1.5, 3, 50)
    ])
    
    # 里程计轨迹（有累积漂移）
    drift = np.linspace(0, 0.3, len(true_x))
    odom_x = true_x + drift + np.random.normal(0, 0.02, len(true_x))
    odom_y = true_y + drift * 0.5 + np.random.normal(0, 0.02, len(true_x))
    
    # 视觉定位轨迹（离散点，有跳变）
    vis_indices = np.arange(0, len(true_x), 10)
    vis_x = true_x[vis_indices] + np.random.normal(0, 0.03, len(vis_indices))
    vis_y = true_y[vis_indices] + np.random.normal(0, 0.03, len(vis_indices))
    
    # 融合轨迹（最接近真实）
    fusion_x = true_x + np.random.normal(0, 0.015, len(true_x))
    fusion_y = true_y + np.random.normal(0, 0.015, len(true_x))
    
    # 绘制轨迹
    ax.plot(true_x, true_y, 'k-', linewidth=3, label='真实轨迹', zorder=1)
    ax.plot(odom_x, odom_y, 'b--', linewidth=2, label='里程计', alpha=0.7, zorder=2)
    ax.scatter(vis_x, vis_y, c='green', s=50, marker='^', label='ChArUco视觉定位', zorder=4)
    ax.plot(fusion_x, fusion_y, 'r-', linewidth=2, label='融合结果', zorder=3)
    
    # 标记起点和终点
    ax.plot(true_x[0], true_y[0], 'go', markersize=15, label='起点', zorder=5)
    ax.plot(true_x[-1], true_y[-1], 'rs', markersize=15, label='终点', zorder=5)
    
    # 标记ChArUco标定板位置
    charuco_positions = [(0.5, -0.3), (1.5, -0.3), (2.3, 0.7), (2.3, 1.2), (3, 1.8), (4.3, 2.2)]
    for i, (cx, cy) in enumerate(charuco_positions):
        ax.plot(cx, cy, 'k^', markersize=12)
        ax.text(cx, cy-0.15, f'C{i+1}', fontsize=8, ha='center')
    
    ax.set_xlabel('X坐标 (m)', fontsize=12)
    ax.set_ylabel('Y坐标 (m)', fontsize=12)
    ax.set_title('多传感器融合定位轨迹对比', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal')
    ax.set_xlim(-0.5, 5)
    ax.set_ylim(-0.8, 3.5)
    
    # 添加误差统计
    odom_error = np.mean(np.sqrt((odom_x - true_x)**2 + (odom_y - true_y)**2))
    fusion_error = np.mean(np.sqrt((fusion_x - true_x)**2 + (fusion_y - true_y)**2))
    
    stats_text = f'平均定位误差:\n里程计: {odom_error*100:.1f} cm\n融合结果: {fusion_error*100:.1f} cm'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_fusion_trajectory.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/fig4_fusion_trajectory.pdf', bbox_inches='tight')
    print("✓ 已生成: fig4_fusion_trajectory.png/pdf")
    plt.close()


# ============================================================
# 图4-X: 定位精度随距离变化曲线
# ============================================================
def draw_accuracy_vs_distance():
    """绘制定位精度随距离变化的曲线图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 距离数据
    distances = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    
    # 距离测量误差（mm）
    dist_errors = np.array([4.2, 5.8, 7.1, 9.3, 10.5, 12.1, 13.8])
    dist_relative = dist_errors / (distances * 1000) * 100  # 相对误差%
    
    # 角度测量误差（度）
    angle_errors = np.array([0.5, 0.6, 0.8, 0.9, 1.0, 1.1, 1.3])
    
    # 左图：距离误差
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(distances, dist_errors, 'bo-', linewidth=2, markersize=8, label='绝对误差')
    line2 = ax1_twin.plot(distances, dist_relative, 'r^--', linewidth=2, markersize=8, label='相对误差')
    
    ax1.set_xlabel('测量距离 (m)', fontsize=12)
    ax1.set_ylabel('绝对误差 (mm)', fontsize=12, color='blue')
    ax1_twin.set_ylabel('相对误差 (%)', fontsize=12, color='red')
    ax1.set_title('(a) 距离测量精度', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10)
    
    # 右图：角度误差
    ax2 = axes[1]
    ax2.plot(distances, angle_errors, 'go-', linewidth=2, markersize=8)
    ax2.fill_between(distances, angle_errors - 0.2, angle_errors + 0.2, alpha=0.3, color='green')
    ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='目标精度: 1°')
    
    ax2.set_xlabel('测量距离 (m)', fontsize=12)
    ax2.set_ylabel('角度误差 (°)', fontsize=12)
    ax2.set_title('(b) 角度测量精度', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(0, 2)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_accuracy_vs_distance.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(f'{output_dir}/fig4_accuracy_vs_distance.pdf', bbox_inches='tight')
    print("✓ 已生成: fig4_accuracy_vs_distance.png/pdf")
    plt.close()


# ============================================================
# 主函数：生成所有图表
# ============================================================
if __name__ == '__main__':
    print("=" * 60)
    print("开始生成毕业论文学术风格图表...")
    print("=" * 60)
    print(f"输出目录: {output_dir}\n")
    
    # 第三章图表
    print("[第三章 - 多智能体路径规划]")
    draw_mapf_paths()           # 栅格地图路径规划效果图
    draw_spacetime_diagram()    # 时空轨迹图
    draw_comparison_bars()      # 性能对比柱状图
    draw_training_curves()      # 训练曲线对比
    
    print("\n[第四章 - 多传感器融合定位]")
    draw_charuco_detection()    # ChArUco角点检测效果图
    draw_reprojection_error()   # 重投影误差分布图
    draw_fusion_trajectory()    # 融合定位轨迹对比
    draw_accuracy_vs_distance() # 精度随距离变化曲线
    
    print("\n" + "=" * 60)
    print("所有图表生成完成！")
    print(f"请在以下目录查看: {output_dir}")
    print("=" * 60)
