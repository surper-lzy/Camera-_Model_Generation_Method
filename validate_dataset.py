#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证和可视化生成的训练数据

功能：
1. 验证生成的数据集完整性
2. 可视化采样的相机参数分布
3. 展示样本图像（可选）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False


def validate_dataset(output_root: str):
    """验证数据集完整性"""
    output_path = Path(output_root)
    csv_path = output_path / "labels.csv"
    images_dir = output_path / "images"

    print("=" * 60)
    print("数据集验证报告")
    print("=" * 60)

    # 检查 CSV 文件
    if not csv_path.exists():
        print("❌ 错误: labels.csv 不存在!")
        return None

    df = pd.read_csv(csv_path)
    print(f"✅ CSV 文件加载成功: {len(df)} 条记录")

    # 检查图像文件
    if not images_dir.exists():
        print("❌ 错误: images 目录不存在!")
        return None

    missing_files = []
    for filename in df['filename']:
        if not (images_dir / filename).exists():
            missing_files.append(filename)

    if missing_files:
        print(f"⚠️  警告: 发现 {len(missing_files)} 个缺失的图像文件")
        print(f"   前5个: {missing_files[:5]}")
    else:
        print(f"✅ 所有图像文件存在: {len(df)} 个文件")

    print("=" * 60)
    return df


def plot_parameter_distributions(df: pd.DataFrame, save_path: str = None):
    """绘制参数分布图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('相机参数分布分析', fontsize=16, fontweight='bold')

    # 1. 俯仰角分布
    ax = axes[0, 0]
    ax.hist(df['pitch'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='水平线')
    ax.set_xlabel('俯仰角 Pitch (度)', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('俯仰角分布', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. 翻滚角分布
    ax = axes[0, 1]
    ax.hist(df['roll'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
    ax.set_xlabel('翻滚角 Roll (度)', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('翻滚角分布', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # 3. 偏航角分布
    ax = axes[0, 2]
    ax.hist(df['yaw'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
    ax.set_xlabel('偏航角 Yaw (度)', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('偏航角分布', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    # 4. 视场角分布（分投影类型）
    ax = axes[1, 0]
    df_pinhole = df[~df['is_fisheye']]
    df_fisheye = df[df['is_fisheye']]
    ax.hist(df_pinhole['fov'], bins=30, color='blue', alpha=0.5, label='针孔投影', edgecolor='black')
    ax.hist(df_fisheye['fov'], bins=30, color='orange', alpha=0.5, label='鱼眼投影', edgecolor='black')
    ax.set_xlabel('视场角 FoV (度)', fontsize=12)
    ax.set_ylabel('样本数量', fontsize=12)
    ax.set_title('视场角分布（按投影类型）', fontsize=14)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 5. 投影类型饼图
    ax = axes[1, 1]
    projection_counts = df['is_fisheye'].value_counts()
    labels = ['针孔投影', '鱼眼投影']
    colors = ['#66b3ff', '#ff9999']
    ax.pie(projection_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('投影类型分布', fontsize=14)

    # 6. 俯仰角 vs 视场角 散点图
    ax = axes[1, 2]
    scatter_pinhole = ax.scatter(
        df_pinhole['pitch'], df_pinhole['fov'],
        c='blue', alpha=0.3, s=10, label='针孔投影'
    )
    scatter_fisheye = ax.scatter(
        df_fisheye['pitch'], df_fisheye['fov'],
        c='orange', alpha=0.3, s=10, label='鱼眼投影'
    )
    ax.set_xlabel('俯仰角 Pitch (度)', fontsize=12)
    ax.set_ylabel('视场角 FoV (度)', fontsize=12)
    ax.set_title('俯仰角 vs 视场角', fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 分布图已保存到: {save_path}")

    plt.show()


def print_statistics(df: pd.DataFrame):
    """打印统计信息"""
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)

    print(f"\n📊 总体统计:")
    print(f"   总样本数: {len(df)}")
    print(f"   唯一全景图数: {df['source_pano'].nunique()}")
    print(f"   每个全景图平均样本数: {len(df) / df['source_pano'].nunique():.1f}")

    print(f"\n📐 角度统计:")
    print(f"   俯仰角范围: [{df['pitch'].min():.2f}°, {df['pitch'].max():.2f}°]")
    print(f"   俯仰角均值: {df['pitch'].mean():.2f}° (标准差: {df['pitch'].std():.2f}°)")
    print(f"   翻滚角范围: [{df['roll'].min():.2f}°, {df['roll'].max():.2f}°]")
    print(f"   偏航角范围: [{df['yaw'].min():.2f}°, {df['yaw'].max():.2f}°]")

    print(f"\n🔭 视场角统计:")
    print(f"   整体范围: [{df['fov'].min():.2f}°, {df['fov'].max():.2f}°]")
    print(f"   针孔投影: [{df[~df['is_fisheye']]['fov'].min():.2f}°, {df[~df['is_fisheye']]['fov'].max():.2f}°]")
    print(f"   鱼眼投影: [{df[df['is_fisheye']]['fov'].min():.2f}°, {df[df['is_fisheye']]['fov'].max():.2f}°]")

    print(f"\n📷 投影类型:")
    fisheye_count = df['is_fisheye'].sum()
    pinhole_count = len(df) - fisheye_count
    print(f"   针孔投影: {pinhole_count} ({pinhole_count/len(df)*100:.1f}%)")
    print(f"   鱼眼投影: {fisheye_count} ({fisheye_count/len(df)*100:.1f}%)")

    print(f"\n⚠️  黑色区域统计:")
    print(f"   顶部黑色比例均值: {df['top_black_ratio'].mean():.2%}")
    print(f"   底部黑色比例均值: {df['bottom_black_ratio'].mean():.2%}")

    print("=" * 60)


def visualize_samples(df: pd.DataFrame, images_dir: str, num_samples: int = 9):
    """可视化样本图像"""
    images_path = Path(images_dir)

    # 随机选择样本
    samples = df.sample(n=min(num_samples, len(df)))

    rows = int(np.ceil(np.sqrt(num_samples)))
    cols = int(np.ceil(num_samples / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    axes = axes.flatten() if num_samples > 1 else [axes]

    for idx, (_, row) in enumerate(samples.iterrows()):
        if idx >= len(axes):
            break

        # 读取图像
        img_path = images_path / row['filename']
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 显示图像
            axes[idx].imshow(img)

            # 设置标题
            proj_type = "鱼眼" if row['is_fisheye'] else "针孔"
            title = (f"{proj_type} | Pitch: {row['pitch']:.1f}°\n"
                    f"Roll: {row['roll']:.1f}° | FoV: {row['fov']:.1f}°")
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')

    # 隐藏多余的子图
    for idx in range(len(samples), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.suptitle('随机样本展示', fontsize=16, y=1.00)
    plt.show()


def main():
    """主函数"""
    # 配置路径
    output_root = r"C:\document\Stanford2D3D\output_dataset"

    # 1. 验证数据集
    df = validate_dataset(output_root)

    if df is None:
        print("❌ 数据集验证失败，退出!")
        return

    # 2. 打印统计信息
    print_statistics(df)

    # 3. 绘制参数分布图
    save_path = Path(output_root) / "parameter_distributions.png"
    plot_parameter_distributions(df, save_path=str(save_path))

    # 4. 可视化样本（可选，如果不想显示图像可以注释掉）
    visualize_choice = input("\n是否显示随机样本图像？(y/n): ").strip().lower()
    if visualize_choice == 'y':
        images_dir = Path(output_root) / "images"
        visualize_samples(df, str(images_dir), num_samples=9)

    print("\n✅ 验证完成！")


if __name__ == "__main__":
    main()

