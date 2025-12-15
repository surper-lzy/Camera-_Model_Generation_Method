#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成图像检查工具

用于检查和验证生成的训练图像质量：
- 检测黑色区域/空洞
- 分析边缘拉伸情况
- 验证不同模型的差异
- 生成可视化报告
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageInspector:
    """图像质量检查器"""

    def __init__(self, output_root: str):
        self.output_root = Path(output_root)
        self.images_dir = self.output_root / "images"
        self.labels_path = self.output_root / "labels.csv"
        self.report_dir = self.output_root / "inspection_report"
        self.report_dir.mkdir(exist_ok=True)

        # 加载标签
        if self.labels_path.exists():
            self.df = pd.read_csv(self.labels_path)
            logger.info(f"加载了 {len(self.df)} 条记录")
        else:
            logger.error(f"未找到标签文件: {self.labels_path}")
            self.df = None

    def check_black_regions(self, img: np.ndarray) -> Dict:
        """检测图像中的黑色区域"""
        black_mask = np.all(img < 10, axis=2)
        total_pixels = img.shape[0] * img.shape[1]
        black_pixels = np.sum(black_mask)
        black_ratio = black_pixels / total_pixels

        # 检测是否有大块连续黑色区域
        contours, _ = cv2.findContours(
            black_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        large_holes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > total_pixels * 0.01:  # 超过1%的区域
                large_holes.append(area / total_pixels)

        return {
            'black_ratio': black_ratio,
            'large_holes_count': len(large_holes),
            'largest_hole_ratio': max(large_holes) if large_holes else 0.0
        }

    def analyze_edge_stretch(self, img: np.ndarray) -> Dict:
        """分析边缘拉伸情况"""
        h, w = img.shape[:2]

        # 计算边缘区域（外围20%）的像素梯度
        edge_width = int(min(h, w) * 0.2)

        # 上边缘
        top_edge = img[:edge_width, :, :]
        # 下边缘
        bottom_edge = img[-edge_width:, :, :]
        # 左边缘
        left_edge = img[:, :edge_width, :]
        # 右边缘
        right_edge = img[:, -edge_width:, :]

        # 计算梯度强度（拉伸会导致梯度降低）
        def calc_gradient_strength(region):
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            return np.mean(gradient)

        return {
            'top_edge_gradient': calc_gradient_strength(top_edge),
            'bottom_edge_gradient': calc_gradient_strength(bottom_edge),
            'left_edge_gradient': calc_gradient_strength(left_edge),
            'right_edge_gradient': calc_gradient_strength(right_edge),
            'avg_edge_gradient': np.mean([
                calc_gradient_strength(top_edge),
                calc_gradient_strength(bottom_edge),
                calc_gradient_strength(left_edge),
                calc_gradient_strength(right_edge)
            ])
        }

    def check_image_quality(self, img_path: Path) -> Dict:
        """检查单张图像的质量"""
        img = cv2.imread(str(img_path))
        if img is None:
            return {'error': 'Failed to load image'}

        results = {
            'filename': img_path.name,
            'shape': img.shape,
            'mean_brightness': np.mean(img),
            'std_brightness': np.std(img),
        }

        # 黑色区域检测
        black_info = self.check_black_regions(img)
        results.update(black_info)

        # 边缘拉伸分析
        edge_info = self.analyze_edge_stretch(img)
        results.update(edge_info)

        # 异常判断
        results['is_abnormal'] = (
            black_info['black_ratio'] > 0.5 or  # 超过50%黑色
            black_info['largest_hole_ratio'] > 0.3 or  # 有超过30%的大洞
            results['mean_brightness'] < 10  # 平均亮度过低
        )

        return results

    def inspect_all_images(self, max_samples: int = None) -> pd.DataFrame:
        """检查所有图像"""
        if self.df is None:
            logger.error("无标签数据，无法检查")
            return None

        samples = self.df.head(max_samples) if max_samples else self.df

        results = []
        logger.info(f"开始检查 {len(samples)} 张图像...")

        for idx, row in samples.iterrows():
            img_path = self.images_dir / row['filename']
            if not img_path.exists():
                logger.warning(f"图像不存在: {img_path}")
                continue

            quality = self.check_image_quality(img_path)

            # 合并标签信息
            combined = {**row.to_dict(), **quality}
            results.append(combined)

            if (idx + 1) % 50 == 0:
                logger.info(f"已检查 {idx + 1}/{len(samples)} 张图像")

        results_df = pd.DataFrame(results)

        # 保存结果
        report_csv = self.report_dir / "quality_report.csv"
        results_df.to_csv(report_csv, index=False)
        logger.info(f"质量报告已保存: {report_csv}")

        return results_df

    def generate_model_comparison(self, num_samples: int = 5):
        """生成不同模型的对比图"""
        if self.df is None:
            return

        logger.info("生成模型对比图...")

        # 获取每种模型的样本
        fisheye_df = self.df[self.df['is_fisheye'] == True]

        models = ['equidistant', 'equisolid', 'orthographic', 'stereographic', 'kannala_brandt']

        for model in models:
            model_df = fisheye_df[fisheye_df['fisheye_model'] == model]
            if len(model_df) == 0:
                continue

            # 选择样本
            samples = model_df.head(min(num_samples, len(model_df)))

            # 创建对比图
            fig, axes = plt.subplots(1, len(samples), figsize=(4*len(samples), 4))
            if len(samples) == 1:
                axes = [axes]

            fig.suptitle(f'模型: {model}', fontsize=16)

            for idx, (_, row) in enumerate(samples.iterrows()):
                img_path = self.images_dir / row['filename']
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[idx].imshow(img_rgb)
                    axes[idx].set_title(
                        f"FoV: {row['fov']:.1f}°\n"
                        f"Pitch: {row['pitch']:.1f}°",
                        fontsize=10
                    )
                    axes[idx].axis('off')

            plt.tight_layout()
            output_path = self.report_dir / f"comparison_{model}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.info(f"保存对比图: {output_path}")

    def generate_summary_report(self, results_df: pd.DataFrame):
        """生成汇总报告"""
        logger.info("生成汇总报告...")

        # 创建多个子图
        fig = plt.figure(figsize=(16, 12))

        # 1. 黑色区域分布
        ax1 = plt.subplot(3, 3, 1)
        ax1.hist(results_df['black_ratio'], bins=50, edgecolor='black')
        ax1.set_xlabel('Black Ratio')
        ax1.set_ylabel('Count')
        ax1.set_title('黑色区域比例分布')
        ax1.axvline(0.5, color='r', linestyle='--', label='异常阈值')
        ax1.legend()

        # 2. 边缘梯度分布
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(results_df['avg_edge_gradient'], bins=50, edgecolor='black')
        ax2.set_xlabel('Average Edge Gradient')
        ax2.set_ylabel('Count')
        ax2.set_title('边缘梯度分布（拉伸检测）')

        # 3. 亮度分布
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(results_df['mean_brightness'], bins=50, edgecolor='black')
        ax3.set_xlabel('Mean Brightness')
        ax3.set_ylabel('Count')
        ax3.set_title('平均亮度分布')

        # 4. 不同模型的黑色区域对比
        ax4 = plt.subplot(3, 3, 4)
        fisheye_results = results_df[results_df['is_fisheye'] == True]
        if len(fisheye_results) > 0:
            model_black_ratios = fisheye_results.groupby('fisheye_model')['black_ratio'].mean()
            model_black_ratios.plot(kind='bar', ax=ax4)
            ax4.set_ylabel('Average Black Ratio')
            ax4.set_title('各模型平均黑色区域比例')
            ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

        # 5. 不同模型的边缘梯度对比
        ax5 = plt.subplot(3, 3, 5)
        if len(fisheye_results) > 0:
            model_gradients = fisheye_results.groupby('fisheye_model')['avg_edge_gradient'].mean()
            model_gradients.plot(kind='bar', ax=ax5, color='green')
            ax5.set_ylabel('Average Edge Gradient')
            ax5.set_title('各模型平均边缘梯度（拉伸程度）')
            ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45, ha='right')

        # 6. FoV vs 黑色区域
        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(results_df['fov'], results_df['black_ratio'], alpha=0.3, s=10)
        ax6.set_xlabel('FoV (degrees)')
        ax6.set_ylabel('Black Ratio')
        ax6.set_title('视场角 vs 黑色区域')

        # 7. Pitch vs 黑色区域
        ax7 = plt.subplot(3, 3, 7)
        ax7.scatter(results_df['pitch'], results_df['black_ratio'], alpha=0.3, s=10)
        ax7.set_xlabel('Pitch (degrees)')
        ax7.set_ylabel('Black Ratio')
        ax7.set_title('俯仰角 vs 黑色区域')

        # 8. 异常图像统计
        ax8 = plt.subplot(3, 3, 8)
        abnormal_count = results_df['is_abnormal'].sum()
        normal_count = len(results_df) - abnormal_count
        ax8.pie([normal_count, abnormal_count],
                labels=['正常', '异常'],
                autopct='%1.1f%%',
                colors=['green', 'red'])
        ax8.set_title(f'图像质量分布\n(异常: {abnormal_count}/{len(results_df)})')

        # 9. 文字统计摘要
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')

        summary_text = f"""
质量检查摘要
{'='*30}
总样本数: {len(results_df)}
异常样本数: {abnormal_count} ({abnormal_count/len(results_df)*100:.1f}%)

黑色区域统计:
  平均: {results_df['black_ratio'].mean():.2%}
  最大: {results_df['black_ratio'].max():.2%}
  >50%: {(results_df['black_ratio'] > 0.5).sum()} 张

边缘梯度统计:
  平均: {results_df['avg_edge_gradient'].mean():.2f}
  标准差: {results_df['avg_edge_gradient'].std():.2f}

亮度统计:
  平均: {results_df['mean_brightness'].mean():.1f}
  最小: {results_df['mean_brightness'].min():.1f}
        """

        ax9.text(0.1, 0.5, summary_text,
                transform=ax9.transAxes,
                fontsize=10,
                verticalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        # 保存报告
        report_path = self.report_dir / "summary_report.png"
        plt.savefig(report_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"汇总报告已保存: {report_path}")

    def find_abnormal_images(self, results_df: pd.DataFrame, top_n: int = 10):
        """找出最异常的图像并可视化"""
        logger.info("查找异常图像...")

        # 按黑色区域比例排序
        abnormal_by_black = results_df.nlargest(top_n, 'black_ratio')

        # 按边缘梯度排序（最小的可能有问题）
        abnormal_by_gradient = results_df.nsmallest(top_n, 'avg_edge_gradient')

        # 可视化最异常的图像
        fig, axes = plt.subplots(2, min(5, top_n), figsize=(20, 8))
        fig.suptitle('最异常的图像', fontsize=16)

        for idx in range(min(5, top_n)):
            # 黑色区域最多
            if idx < len(abnormal_by_black):
                row = abnormal_by_black.iloc[idx]
                img_path = self.images_dir / row['filename']
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[0, idx].imshow(img_rgb)
                    axes[0, idx].set_title(
                        f"黑色: {row['black_ratio']:.1%}\n"
                        f"{row['fisheye_model']}\n"
                        f"FoV: {row['fov']:.1f}°",
                        fontsize=9
                    )
                    axes[0, idx].axis('off')

            # 边缘梯度最小
            if idx < len(abnormal_by_gradient):
                row = abnormal_by_gradient.iloc[idx]
                img_path = self.images_dir / row['filename']
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[1, idx].imshow(img_rgb)
                    axes[1, idx].set_title(
                        f"梯度: {row['avg_edge_gradient']:.1f}\n"
                        f"{row['fisheye_model']}\n"
                        f"FoV: {row['fov']:.1f}°",
                        fontsize=9
                    )
                    axes[1, idx].axis('off')

        axes[0, 0].set_ylabel('黑色区域最多', fontsize=12)
        axes[1, 0].set_ylabel('边缘梯度最小\n(可能过度拉伸)', fontsize=12)

        plt.tight_layout()

        abnormal_path = self.report_dir / "abnormal_images.png"
        plt.savefig(abnormal_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"异常图像报告已保存: {abnormal_path}")


def main():
    """主函数"""
    print("=" * 60)
    print("Stanford2D3D 生成图像检查工具")
    print("=" * 60)

    # 初始化检查器
    output_root = r"C:\document\Stanford2D3D\output_dataset"
    inspector = ImageInspector(output_root)

    # 检查所有图像（或指定数量）
    print("\n[1/4] 检查图像质量...")
    results_df = inspector.inspect_all_images(max_samples=200)  # 可以调整数量

    if results_df is None or len(results_df) == 0:
        print("没有图像可检查！")
        return

    # 生成模型对比图
    print("\n[2/4] 生成模型对比图...")
    inspector.generate_model_comparison(num_samples=5)

    # 生成汇总报告
    print("\n[3/4] 生成汇总报告...")
    inspector.generate_summary_report(results_df)

    # 查找异常图像
    print("\n[4/4] 查找异常图像...")
    inspector.find_abnormal_images(results_df, top_n=10)

    # 打印关键统计
    print("\n" + "=" * 60)
    print("检查完成！关键统计：")
    print("=" * 60)
    print(f"总样本数: {len(results_df)}")
    print(f"异常样本数: {results_df['is_abnormal'].sum()} ({results_df['is_abnormal'].sum()/len(results_df)*100:.1f}%)")
    print(f"\n黑色区域统计:")
    print(f"  平均: {results_df['black_ratio'].mean():.2%}")
    print(f"  最大: {results_df['black_ratio'].max():.2%}")
    print(f"  >50%的样本: {(results_df['black_ratio'] > 0.5).sum()} 张")
    print(f"\n边缘梯度统计（值越小 = 拉伸越严重）:")
    print(f"  平均: {results_df['avg_edge_gradient'].mean():.2f}")
    print(f"  最小: {results_df['avg_edge_gradient'].min():.2f}")
    print(f"  最大: {results_df['avg_edge_gradient'].max():.2f}")

    # 按模型统计
    fisheye_df = results_df[results_df['is_fisheye'] == True]
    if len(fisheye_df) > 0:
        print(f"\n各鱼眼模型边缘梯度（越小 = 拉伸越明显）:")
        for model in fisheye_df['fisheye_model'].unique():
            model_df = fisheye_df[fisheye_df['fisheye_model'] == model]
            avg_grad = model_df['avg_edge_gradient'].mean()
            print(f"  {model:20s}: {avg_grad:.2f}")

    print("\n" + "=" * 60)
    print(f"详细报告已保存到: {inspector.report_dir}")
    print("=" * 60)

    # 给出建议
    print("\n💡 建议:")
    if results_df['black_ratio'].mean() > 0.3:
        print("  ⚠️  黑色区域较多，建议检查全景图质量或调整采样参数")
    if results_df['is_abnormal'].sum() > len(results_df) * 0.1:
        print("  ⚠️  异常样本超过10%，建议查看异常图像报告")

    # 边缘拉伸建议
    if len(fisheye_df) > 0:
        min_gradient = fisheye_df['avg_edge_gradient'].min()
        if min_gradient < 5:
            print("  ℹ️  部分图像边缘梯度很低，这是正常的鱼眼畸变特性")
            print("     stereographic 模型通常边缘拉伸最明显")

    print("\n✅ 检查完成！")


if __name__ == "__main__":
    main()

