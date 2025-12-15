#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试：验证投影数学函数的正确性
"""

import numpy as np
import sys
from pathlib import Path

# 导入主脚本中的函数
sys.path.insert(0, str(Path(__file__).parent))
from generate_training_dataset import euler_to_rotation_matrix, create_perspective_map


def test_euler_to_rotation_matrix():
    """测试欧拉角到旋转矩阵的转换"""
    print("测试 1: 欧拉角 -> 旋转矩阵")
    print("-" * 50)

    # 测试 1: 零旋转
    R = euler_to_rotation_matrix(0, 0, 0)
    expected = np.eye(3)
    assert np.allclose(R, expected), "零旋转应该是单位矩阵"
    print("✅ 零旋转测试通过")

    # 测试 2: 旋转矩阵的正交性
    R = euler_to_rotation_matrix(45, 30, 15)
    should_be_identity = R @ R.T
    assert np.allclose(should_be_identity, np.eye(3)), "旋转矩阵应该是正交的"
    print("✅ 正交性测试通过")

    # 测试 3: 行列式应该为 1
    det = np.linalg.det(R)
    assert np.allclose(det, 1.0), "旋转矩阵的行列式应该为 1"
    print("✅ 行列式测试通过")

    # 测试 4: 90度偏航旋转
    R_yaw_90 = euler_to_rotation_matrix(90, 0, 0)
    test_vector = np.array([1, 0, 0])  # X轴方向
    rotated = R_yaw_90 @ test_vector
    expected = np.array([0, 0, -1])  # 应该旋转到 -Z 方向
    assert np.allclose(rotated, expected, atol=1e-10), "90度偏航旋转验证失败"
    print("✅ 90度偏航旋转测试通过")

    print("\n所有欧拉角测试通过！\n")


def test_projection_symmetry():
    """测试投影的对称性"""
    print("测试 2: 投影对称性")
    print("-" * 50)

    pano_shape = (2048, 4096)
    output_size = (320, 320)

    # 测试：零旋转，图像中心应该对应全景图的前方中心
    map_x, map_y = create_perspective_map(
        pano_shape=pano_shape,
        output_size=output_size,
        yaw=0, pitch=0, roll=0,
        fov=90,
        is_fisheye=False
    )

    # 检查图像中心点
    center_x = output_size[1] // 2
    center_y = output_size[0] // 2

    # 中心点应该映射到全景图的中心附近
    mapped_x = map_x[center_y, center_x]
    mapped_y = map_y[center_y, center_x]

    # 前方中心应该是 (pano_w/2, pano_h/2)
    expected_x = pano_shape[1] / 2
    expected_y = pano_shape[0] / 2

    print(f"图像中心 ({center_x}, {center_y}) -> 全景图 ({mapped_x:.1f}, {mapped_y:.1f})")
    print(f"期望位置: ({expected_x:.1f}, {expected_y:.1f})")

    assert abs(mapped_x - expected_x) < 10, "X坐标映射不正确"
    assert abs(mapped_y - expected_y) < 10, "Y坐标映射不正确"
    print("✅ 零旋转中心点映射测试通过")

    print("\n投影对称性测试通过！\n")


def test_fisheye_vs_pinhole():
    """测试鱼眼和针孔投影的差异"""
    print("测试 3: 鱼眼 vs 针孔投影")
    print("-" * 50)

    pano_shape = (2048, 4096)
    output_size = (320, 320)

    # 针孔投影
    map_x_pin, map_y_pin = create_perspective_map(
        pano_shape=pano_shape,
        output_size=output_size,
        yaw=0, pitch=0, roll=0,
        fov=90,
        is_fisheye=False
    )

    # 鱼眼投影
    map_x_fish, map_y_fish = create_perspective_map(
        pano_shape=pano_shape,
        output_size=output_size,
        yaw=0, pitch=0, roll=0,
        fov=150,
        is_fisheye=True
    )

    # 检查边缘点的差异
    edge_x = output_size[1] - 1
    edge_y = output_size[0] // 2

    pin_edge_x = map_x_pin[edge_y, edge_x]
    fish_edge_x = map_x_fish[edge_y, edge_x]

    print(f"边缘点映射差异:")
    print(f"  针孔投影 (FoV=90°): X = {pin_edge_x:.1f}")
    print(f"  鱼眼投影 (FoV=150°): X = {fish_edge_x:.1f}")

    # 鱼眼投影应该覆盖更广的范围
    assert fish_edge_x != pin_edge_x, "鱼眼和针孔投影应该产生不同的映射"
    print("✅ 鱼眼与针孔投影差异测试通过")

    print("\n投影类型差异测试通过！\n")


def test_pitch_rotation():
    """测试俯仰角旋转"""
    print("测试 4: 俯仰角旋转")
    print("-" * 50)

    pano_shape = (2048, 4096)
    output_size = (320, 320)

    # 向上看 (正俯仰角)
    map_x_up, map_y_up = create_perspective_map(
        pano_shape=pano_shape,
        output_size=output_size,
        yaw=0, pitch=45, roll=0,  # 向上45度
        fov=90,
        is_fisheye=False
    )

    # 向下看 (负俯仰角)
    map_x_down, map_y_down = create_perspective_map(
        pano_shape=pano_shape,
        output_size=output_size,
        yaw=0, pitch=-45, roll=0,  # 向下45度
        fov=90,
        is_fisheye=False
    )

    # 检查中心点的Y坐标
    center_x = output_size[1] // 2
    center_y = output_size[0] // 2

    y_up = map_y_up[center_y, center_x]
    y_down = map_y_down[center_y, center_x]

    print(f"俯仰角效果:")
    print(f"  向上 45°: Y = {y_up:.1f}")
    print(f"  水平 0°:  Y = {pano_shape[0]/2:.1f}")
    print(f"  向下 45°: Y = {y_down:.1f}")

    # 修正后的坐标系统：
    # - 向上看（正俯仰角）会看到更多天空区域（全景图中Y值较小的区域，即顶部）
    # - 向下看（负俯仰角）会看到更多地面区域（全景图中Y值较大的区域，即底部）
    assert y_up < pano_shape[0] / 2, "向上看应该映射到全景图上半部分（天空区域）"
    assert y_down > pano_shape[0] / 2, "向下看应该映射到全景图下半部分（地面区域）"
    print("✅ 俯仰角旋转测试通过")

    print("\n俯仰角测试通过！\n")


def test_yaw_rotation():
    """测试偏航角旋转"""
    print("测试 5: 偏航角旋转")
    print("-" * 50)

    pano_shape = (2048, 4096)
    output_size = (320, 320)

    # 向前看
    map_x_0, _ = create_perspective_map(
        pano_shape=pano_shape,
        output_size=output_size,
        yaw=0, pitch=0, roll=0,
        fov=90,
        is_fisheye=False
    )

    # 向右转90度
    map_x_90, _ = create_perspective_map(
        pano_shape=pano_shape,
        output_size=output_size,
        yaw=90, pitch=0, roll=0,
        fov=90,
        is_fisheye=False
    )

    # 向后看180度
    map_x_180, _ = create_perspective_map(
        pano_shape=pano_shape,
        output_size=output_size,
        yaw=180, pitch=0, roll=0,
        fov=90,
        is_fisheye=False
    )

    center_x = output_size[1] // 2
    center_y = output_size[0] // 2

    x_0 = map_x_0[center_y, center_x]
    x_90 = map_x_90[center_y, center_x]
    x_180 = map_x_180[center_y, center_x]

    print(f"偏航角效果:")
    print(f"  0°:   X = {x_0:.1f}")
    print(f"  90°:  X = {x_90:.1f}")
    print(f"  180°: X = {x_180:.1f}")

    # 基本的方向性检查
    assert abs(x_0 - pano_shape[1]/2) < 50, "0度应该指向前方"
    print("✅ 偏航角旋转测试通过")

    print("\n偏航角测试通过！\n")


def run_all_tests():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 12 + "数学函数单元测试 - 完整版" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    try:
        test_euler_to_rotation_matrix()
        test_projection_symmetry()
        test_fisheye_vs_pinhole()
        test_pitch_rotation()
        test_yaw_rotation()

        print("╔" + "=" * 58 + "╗")
        print("║" + " " * 15 + "✅ 所有测试通过！" + " " * 15 + "║")
        print("║" + " " * 10 + "数学实现正确，可以安全使用" + " " * 10 + "║")
        print("╚" + "=" * 58 + "╝")
        print()
        return True

    except AssertionError as e:
        print("\n╔" + "=" * 58 + "╗")
        print("║  ❌ 测试失败: " + str(e)[:40].ljust(40) + " ║")
        print("╚" + "=" * 58 + "╝")
        return False
    except Exception as e:
        print("\n╔" + "=" * 58 + "╗")
        print("║  ❌ 测试出错: " + str(e)[:40].ljust(40) + " ║")
        print("╚" + "=" * 58 + "╝")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

