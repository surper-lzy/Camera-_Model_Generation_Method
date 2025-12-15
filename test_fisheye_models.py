#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试脚本：验证多种鱼眼模型的实现

用于验证所有 5 种鱼眼模型生成的图像是否正常
"""

import cv2
import numpy as np
from pathlib import Path
from generate_training_dataset import Config, ViewGenerator, create_perspective_map
import matplotlib.pyplot as plt

def create_test_panorama():
    """创建一个简单的测试全景图（带网格和文字）"""
    pano = np.zeros((2048, 4096, 3), dtype=np.uint8)

    # 填充渐变背景
    for i in range(pano.shape[0]):
        pano[i, :, 0] = int(255 * (i / pano.shape[0]))  # 蓝色渐变
        pano[i, :, 1] = int(128)  # 绿色固定
        pano[i, :, 2] = int(255 * (1 - i / pano.shape[0]))  # 红色渐变

    # 绘制网格线
    grid_spacing = 256
    for i in range(0, pano.shape[0], grid_spacing):
        cv2.line(pano, (0, i), (pano.shape[1], i), (255, 255, 255), 2)
    for j in range(0, pano.shape[1], grid_spacing):
        cv2.line(pano, (j, 0), (j, pano.shape[0]), (255, 255, 255), 2)

    # 添加文字标记
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(pano, 'TOP (Zenith)', (1800, 200), font, 3, (255, 255, 0), 5)
    cv2.putText(pano, 'BOTTOM (Nadir)', (1600, 1900), font, 3, (255, 255, 0), 5)
    cv2.putText(pano, 'FRONT (0deg)', (1800, 1024), font, 3, (0, 255, 255), 5)

    return pano

def test_all_fisheye_models():
    """测试所有鱼眼模型"""
    print("=" * 60)
    print("测试所有鱼眼模型")
    print("=" * 60)

    # 创建测试全景图
    pano = create_test_panorama()
    print(f"创建测试全景图: {pano.shape}")

    # 配置参数
    config = Config()
    output_dir = Path("test_fisheye_output")
    output_dir.mkdir(exist_ok=True)

    # 保存原始全景图
    cv2.imwrite(str(output_dir / "test_panorama.jpg"), pano)
    print(f"保存测试全景图: {output_dir / 'test_panorama.jpg'}")

    # 测试参数
    test_params_base = {
        'yaw': 0.0,
        'pitch': 30.0,  # 向上看30度
        'roll': 0.0,
        'fov': 160.0,
        'is_fisheye': True
    }

    # 测试所有鱼眼模型
    fisheye_models = ['equidistant', 'equisolid', 'orthographic', 'stereographic', 'kannala_brandt']

    for model in fisheye_models:
        print(f"\n测试模型: {model}")

        # 准备参数
        params = test_params_base.copy()
        params['fisheye_model'] = model

        # 如果是 KB 模型，添加系数
        if model == 'kannala_brandt':
            params['kb_coeffs'] = {'k2': 0.01, 'k3': -0.005, 'k4': 0.002}
            print(f"  KB 系数: k2={params['kb_coeffs']['k2']}, k3={params['kb_coeffs']['k3']}, k4={params['kb_coeffs']['k4']}")

        # 创建映射
        map_x, map_y = create_perspective_map(
            pano_shape=(pano.shape[0], pano.shape[1]),
            output_size=config.output_image_size,
            **params
        )

        # 生成视图
        view = cv2.remap(
            pano, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 保存图像
        output_path = output_dir / f"fisheye_{model}.jpg"
        cv2.imwrite(str(output_path), view)
        print(f"  保存视图: {output_path}")

    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"输出目录: {output_dir.absolute()}")
    print("=" * 60)

def test_kb_coefficient_variation():
    """测试 KB 模型的系数变化"""
    print("\n" + "=" * 60)
    print("测试 Kannala-Brandt 模型系数变化")
    print("=" * 60)

    pano = create_test_panorama()
    config = Config()
    output_dir = Path("test_fisheye_output") / "kb_variations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 基础参数
    base_params = {
        'yaw': 0.0,
        'pitch': 0.0,
        'roll': 0.0,
        'fov': 170.0,
        'is_fisheye': True,
        'fisheye_model': 'kannala_brandt'
    }

    # 测试不同的系数组合
    kb_variations = [
        {'k2': 0.0, 'k3': 0.0, 'k4': 0.0, 'name': 'no_distortion'},
        {'k2': 0.03, 'k3': 0.0, 'k4': 0.0, 'name': 'positive_k2'},
        {'k2': -0.03, 'k3': 0.0, 'k4': 0.0, 'name': 'negative_k2'},
        {'k2': 0.01, 'k3': 0.01, 'k4': 0.0, 'name': 'positive_k2_k3'},
        {'k2': -0.02, 'k3': -0.01, 'k4': 0.005, 'name': 'complex_distortion'}
    ]

    for kb_var in kb_variations:
        name = kb_var.pop('name')
        params = base_params.copy()
        params['kb_coeffs'] = kb_var

        print(f"\n测试 {name}: k2={kb_var['k2']:.3f}, k3={kb_var['k3']:.3f}, k4={kb_var['k4']:.3f}")

        map_x, map_y = create_perspective_map(
            pano_shape=(pano.shape[0], pano.shape[1]),
            output_size=config.output_image_size,
            **params
        )

        view = cv2.remap(
            pano, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        output_path = output_dir / f"kb_{name}.jpg"
        cv2.imwrite(str(output_path), view)
        print(f"  保存: {output_path}")

    print("\n" + "=" * 60)
    print("KB 系数变化测试完成！")
    print(f"输出目录: {output_dir.absolute()}")
    print("=" * 60)

def test_model_distribution():
    """测试模型采样分布"""
    print("\n" + "=" * 60)
    print("测试模型采样分布")
    print("=" * 60)

    config = Config()
    generator = ViewGenerator(config)
    generator.set_seed(42)

    # 模拟采样 1000 次
    n_samples = 1000
    model_counts = {}
    kb_count = 0

    safety_flags = {
        'unsafe_upward': False,
        'unsafe_downward': False,
        'top_black_ratio': 0.0,
        'bottom_black_ratio': 0.0
    }

    for _ in range(n_samples):
        params = generator.sample_camera_params(safety_flags)

        if params['is_fisheye']:
            model = params['fisheye_model']
            model_counts[model] = model_counts.get(model, 0) + 1

            if model == 'kannala_brandt' and params['kb_coeffs'] is not None:
                kb_count += 1

    # 打印统计
    print(f"\n采样 {n_samples} 次:")
    print(f"  鱼眼样本: {sum(model_counts.values())} ({sum(model_counts.values())/n_samples*100:.1f}%)")
    print(f"  针孔样本: {n_samples - sum(model_counts.values())} ({(n_samples - sum(model_counts.values()))/n_samples*100:.1f}%)")

    print("\n鱼眼模型分布:")
    total_fisheye = sum(model_counts.values())
    for model in config.fisheye_models:
        count = model_counts.get(model, 0)
        actual_pct = count / total_fisheye * 100 if total_fisheye > 0 else 0
        expected_pct = config.fisheye_model_weights[config.fisheye_models.index(model)] * 100
        print(f"  {model:20s}: {count:4d} ({actual_pct:5.1f}%) [期望: {expected_pct:5.1f}%]")

    print(f"\nKB 模型带系数样本: {kb_count}")
    print("=" * 60)

if __name__ == "__main__":
    # 运行所有测试
    test_all_fisheye_models()
    test_kb_coefficient_variation()
    test_model_distribution()

    print("\n✅ 所有测试完成！")

