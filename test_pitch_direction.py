#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本：验证俯仰角方向是否正确
"""

import cv2
import numpy as np
from pathlib import Path
import sys

# 导入主脚本中的函数
sys.path.insert(0, str(Path(__file__).parent))
from generate_training_dataset import create_perspective_map, euler_to_rotation_matrix

def test_pitch_direction():
    """测试俯仰角方向"""
    print("=" * 60)
    print("测试俯仰角方向（上下是否颠倒）")
    print("=" * 60)

    # 创建一个简单的测试全景图
    # 上半部分是蓝色（天空），下半部分是绿色（地面）
    pano_h, pano_w = 2048, 4096
    test_pano = np.zeros((pano_h, pano_w, 3), dtype=np.uint8)

    # 上半部分 - 蓝色（天空）
    test_pano[:pano_h//2, :] = [255, 0, 0]  # BGR: 蓝色

    # 下半部分 - 绿色（地面）
    test_pano[pano_h//2:, :] = [0, 255, 0]  # BGR: 绿色

    # 中间画一条白线（地平线）
    test_pano[pano_h//2-5:pano_h//2+5, :] = [255, 255, 255]

    print("\n创建测试全景图：")
    print("  - 上半部分（Y < 1024）：蓝色（天空）")
    print("  - 下半部分（Y > 1024）：绿色（地面）")
    print("  - 中间白线：地平线")

    # 测试三个俯仰角
    test_cases = [
        (45, "向上看45°", "应该看到更多蓝色（天空）"),
        (0, "水平看0°", "应该看到白线（地平线）在中间"),
        (-45, "向下看-45°", "应该看到更多绿色（地面）"),
    ]

    output_size = (320, 320)

    for pitch, name, expected in test_cases:
        # 生成映射
        map_x, map_y = create_perspective_map(
            pano_shape=(pano_h, pano_w),
            output_size=output_size,
            yaw=0,
            pitch=pitch,
            roll=0,
            fov=90,
            is_fisheye=False
        )

        # 生成视图
        view = cv2.remap(
            test_pano,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        # 分析颜色
        blue_ratio = np.sum(view[:, :, 0] > 200) / (output_size[0] * output_size[1])
        green_ratio = np.sum(view[:, :, 1] > 200) / (output_size[0] * output_size[1])
        white_ratio = np.sum(np.all(view > 200, axis=2)) / (output_size[0] * output_size[1])

        print(f"\n{name} (pitch={pitch}°):")
        print(f"  期望: {expected}")
        print(f"  实际: 蓝色={blue_ratio:.1%}, 绿色={green_ratio:.1%}, 白色={white_ratio:.1%}")

        # 检查是否符合预期
        if pitch > 0:
            if blue_ratio > green_ratio:
                print("  ✅ 正确：向上看时看到更多天空（蓝色）")
            else:
                print("  ❌ 错误：向上看时应该看到更多天空，但实际看到更多地面")
        elif pitch < 0:
            if green_ratio > blue_ratio:
                print("  ✅ 正确：向下看时看到更多地面（绿色）")
            else:
                print("  ❌ 错误：向下看时应该看到更多地面，但实际看到更多天空")
        else:
            if white_ratio > 0.05:
                print("  ✅ 正确：水平看时能看到地平线（白线）")
            else:
                print("  ⚠️  警告：水平看时应该看到明显的地平线")

        # 保存图像用于手动检查
        output_dir = Path(__file__).parent / "test_output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"pitch_{pitch:+03d}.jpg"
        cv2.imwrite(str(output_path), view)
        print(f"  保存到: {output_path}")

    print("\n" + "=" * 60)
    print("测试完成！请检查 test_output/ 目录中的图像")
    print("=" * 60)


def test_rotation_matrix():
    """测试旋转矩阵的正确性"""
    print("\n" + "=" * 60)
    print("测试旋转矩阵")
    print("=" * 60)

    # 测试向上看45度
    R = euler_to_rotation_matrix(0, 45, 0)

    # 前方向量 (0, 0, 1)
    forward = np.array([0, 0, 1])
    rotated = R @ forward

    print(f"\n向上看45°:")
    print(f"  原始向量（前方）: {forward}")
    print(f"  旋转后向量: {rotated}")
    print(f"  Y分量: {rotated[1]:.3f}")

    if rotated[1] < 0:
        print("  ✅ 正确：Y分量为负，表示向上")
    else:
        print("  ❌ 错误：Y分量应该为负（向上）")


if __name__ == "__main__":
    test_pitch_direction()
    test_rotation_matrix()

