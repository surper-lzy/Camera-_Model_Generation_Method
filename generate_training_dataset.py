#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stanford2D3D Dataset Processing Pipeline
用于从全景图生成相机参数回归训练数据

作者: AI Assistant
日期: 2025-12-06
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
@dataclass
class Config:
    """全局配置参数"""
    # 输入输出路径
    input_root: str = r"C:\document\Stanford2D3D"
    output_root: str = r"C:\document\Stanford2D3D\output_dataset"

    # 生成参数
    samples_per_pano: int = 100
    output_image_size: Tuple[int, int] = (320, 320)

    # 黑色区域检测阈值
    black_pixel_threshold: float = 0.20  # 20%
    check_zone_ratio: float = 0.10  # 检查顶部/底部的10%区域

    # 角度采样分布
    pitch_extreme_up_range: Tuple[float, float] = (60.0, 89.0)
    pitch_extreme_down_range: Tuple[float, float] = (-89.0, -60.0)
    pitch_normal_range: Tuple[float, float] = (-30.0, 30.0)
    roll_range: Tuple[float, float] = (-20.0, 20.0)
    yaw_range: Tuple[float, float] = (0.0, 360.0)

    # FoV 和投影类型
    fov_rectilinear_range: Tuple[float, float] = (60.0, 100.0)
    fov_fisheye_range: Tuple[float, float] = (140.0, 180.0)
    fisheye_probability: float = 0.7

    # 鱼眼投影模型配置
    # 支持的模型: equidistant, equisolid, orthographic, stereographic, kannala_brandt
    fisheye_models: Tuple[str, ...] = (
        'equidistant',      # 等距投影 (最常见)
        'equisolid',        # 等立体角投影
        'orthographic',     # 正交投影
        'stereographic',    # 体视投影
        'kannala_brandt'    # Kannala-Brandt 模型 (OpenCV)
    )

    # 各模型的采样权重 (权重之和应为1.0)
    fisheye_model_weights: Tuple[float, ...] = (
        0.25,  # equidistant
        0.25,  # equisolid
        0.15,  # orthographic
        0.15,  # stereographic
        0.20   # kannala_brandt (最常用，权重最高)
    )

    # Kannala-Brandt 模型畸变系数的随机扰动范围
    # r(θ) = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷
    # k1 会根据 FoV 动态计算，k2-k4 添加随机扰动以模拟不同的镜头特性
    kb_k2_range: Tuple[float, float] = (-0.05, 0.05)
    kb_k3_range: Tuple[float, float] = (-0.02, 0.02)
    kb_k4_range: Tuple[float, float] = (-0.005, 0.005)

    # 采样分布概率
    pitch_extreme_up_prob: float = 0.30
    pitch_extreme_down_prob: float = 0.30
    pitch_normal_prob: float = 0.40

    # 多进程参数
    num_workers: int = max(1, os.cpu_count() - 1)

    # 日志级别
    log_level: int = logging.INFO


# ==================== 日志配置 ====================
def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger("Stanford2D3D")
    logger.setLevel(level)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()


# ==================== 数学工具函数 ====================
def euler_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    将欧拉角（偏航、俯仰、翻滚）转换为3D旋转矩阵

    坐标系统约定：
    - X轴：右方向
    - Y轴：下方向（图像坐标系）
    - Z轴：前方向（相机光轴）

    旋转顺序：Yaw (Y轴) -> Pitch (X轴) -> Roll (Z轴)

    Args:
        yaw: 偏航角（度），绕Y轴旋转（水平转向）
        pitch: 俯仰角（度），绕X轴旋转（上下倾斜）
        roll: 翻滚角（度），绕Z轴旋转（画面倾斜）

    Returns:
        3x3 旋转矩阵
    """
    # 转换为弧度
    yaw_rad = np.deg2rad(yaw)
    pitch_rad = np.deg2rad(pitch)
    roll_rad = np.deg2rad(roll)

    # Yaw 旋转矩阵 (绕Y轴)
    R_yaw = np.array([
        [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
        [0, 1, 0],
        [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]
    ])

    # Pitch 旋转矩阵 (绕X轴)
    # 注意：在图像坐标系中Y轴向下为正
    # 正俯仰角应该使相机向上看（看到天空），即Y分量减小（变负）
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
        [0, np.sin(pitch_rad), np.cos(pitch_rad)]
    ])

    # Roll 旋转矩阵 (绕Z轴)
    R_roll = np.array([
        [np.cos(roll_rad), -np.sin(roll_rad), 0],
        [np.sin(roll_rad), np.cos(roll_rad), 0],
        [0, 0, 1]
    ])

    # 组合旋转: R = R_roll @ R_pitch @ R_yaw
    R = R_roll @ R_pitch @ R_yaw

    return R


def create_perspective_map(
    pano_shape: Tuple[int, int],
    output_size: Tuple[int, int],
    yaw: float,
    pitch: float,
    roll: float,
    fov: float,
    is_fisheye: bool = False,
    fisheye_model: str = 'equidistant',
    kb_coeffs: Dict[str, float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建从透视投影到等距柱状投影的映射（支持多种鱼眼模型）

    核心算法流程：
    1. 生成输出图像的像素网格 (u, v)
    2. 根据投影类型（针孔/鱼眼），将像素坐标转换为相机坐标系下的3D射线
       - 支持 5 种鱼眼模型：equidistant, equisolid, orthographic, stereographic, kannala_brandt
    3. 应用旋转矩阵，将相机射线转换到全局坐标系
    4. 将3D射线转换为球面坐标 (phi, theta)
    5. 将球面坐标映射到等距柱状投影的像素坐标 (x, y)

    鱼眼投影模型说明：
    - Equidistant (等距):       r = f * θ
    - Equisolid (等立体角):     r = 2f * sin(θ/2)
    - Orthographic (正交):      r = f * sin(θ)
    - Stereographic (体视):     r = 2f * tan(θ/2)
    - Kannala-Brandt (KB):      r = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷

    Args:
        pano_shape: 全景图尺寸 (height, width)
        output_size: 输出图像尺寸 (height, width)
        yaw, pitch, roll: 相机姿态（度）
        fov: 视场角（度）
        is_fisheye: 是否使用鱼眼投影
        fisheye_model: 鱼眼模型名称
        kb_coeffs: Kannala-Brandt 模型系数 {'k2': float, 'k3': float, 'k4': float}

    Returns:
        map_x, map_y: cv2.remap 使用的映射矩阵
    """
    pano_h, pano_w = pano_shape
    out_h, out_w = output_size

    # 步骤1: 创建输出图像的像素网格
    u, v = np.meshgrid(np.arange(out_w), np.arange(out_h))

    # 归一化：使用短边进行归一化，确保鱼眼投影是圆形的
    norm_radius = min(out_w, out_h) / 2.0
    u_norm = (u - out_w / 2.0) / norm_radius
    v_norm = (v - out_h / 2.0) / norm_radius

    # 步骤2: 像素坐标 -> 3D射线（相机坐标系）
    if is_fisheye:
        # 计算归一化的径向距离
        r = np.sqrt(u_norm**2 + v_norm**2)
        r = np.clip(r, 0, 1)  # 限制在有效范围 [0, 1]

        # 最大入射角
        theta_max = np.deg2rad(fov / 2.0)

        # 根据不同的鱼眼模型计算入射角 theta
        if fisheye_model == 'equidistant':
            # 等距投影: r = k * theta
            theta = r * theta_max

        elif fisheye_model == 'equisolid':
            # 等立体角投影: r = k * sin(theta/2)
            # 反解: theta = 2 * arcsin(r * sin(theta_max/2))
            sin_half_theta_max = np.sin(theta_max / 2.0)
            theta = 2.0 * np.arcsin(np.clip(r * sin_half_theta_max, -1, 1))

        elif fisheye_model == 'orthographic':
            # 正交投影: r = k * sin(theta)
            # 反解: theta = arcsin(r * sin(theta_max))
            sin_theta_max = np.sin(theta_max)
            theta = np.arcsin(np.clip(r * sin_theta_max, -1, 1))

        elif fisheye_model == 'stereographic':
            # 体视投影: r = k * tan(theta/2)
            # 反解: theta = 2 * arctan(r * tan(theta_max/2))
            tan_half_theta_max = np.tan(theta_max / 2.0)
            theta = 2.0 * np.arctan(r * tan_half_theta_max)

        elif fisheye_model == 'kannala_brandt' and kb_coeffs is not None:
            # Kannala-Brandt 模型: r = k1*θ + k2*θ³ + k3*θ⁵ + k4*θ⁷
            k2 = kb_coeffs.get('k2', 0.0)
            k3 = kb_coeffs.get('k3', 0.0)
            k4 = kb_coeffs.get('k4', 0.0)

            # 动态计算 k1，使得 r(theta_max) = 1
            # 1 = k1*theta_max + k2*theta_max³ + k3*theta_max⁵ + k4*theta_max⁷
            theta_max_powers = theta_max**np.arange(1, 8, 2)  # [θ, θ³, θ⁵, θ⁷]
            k1 = (1.0 - k2 * theta_max_powers[1] - k3 * theta_max_powers[2] - k4 * theta_max_powers[3]) / theta_max_powers[0]

            # 使用牛顿迭代法求解 theta
            # 初始猜测：使用等距模型
            theta = r * theta_max

            # 牛顿迭代（5次足够收敛）
            for iteration in range(5):
                # 计算 theta 的幂次
                theta_2 = theta * theta
                theta_3 = theta_2 * theta
                theta_5 = theta_3 * theta_2
                theta_7 = theta_5 * theta_2

                # f(theta) = k1*theta + k2*theta³ + k3*theta⁵ + k4*theta⁷ - r
                f_theta = k1 * theta + k2 * theta_3 + k3 * theta_5 + k4 * theta_7 - r

                # f'(theta) = k1 + 3*k2*theta² + 5*k3*theta⁴ + 7*k4*theta⁶
                theta_4 = theta_2 * theta_2
                theta_6 = theta_4 * theta_2
                f_prime_theta = k1 + 3.0 * k2 * theta_2 + 5.0 * k3 * theta_4 + 7.0 * k4 * theta_6

                # 防止除零
                f_prime_theta = np.where(np.abs(f_prime_theta) < 1e-10, 1e-10, f_prime_theta)

                # 牛顿更新: theta_new = theta - f(theta) / f'(theta)
                theta_new = theta - f_theta / f_prime_theta

                # 限制在有效范围内
                theta = np.clip(theta_new, 0, theta_max)
        else:
            # 默认回退到等距模型
            theta = r * theta_max

        # 确保 theta 在有效范围内
        theta = np.clip(theta, 0, np.pi / 2)

        # 方位角
        phi = np.arctan2(v_norm, u_norm)

        # 3D射线方向（球坐标转笛卡尔坐标）
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

    else:
        # 针孔投影（直线透视）
        # 焦距: f = 0.5 / tan(fov/2)
        f = 0.5 / np.tan(np.deg2rad(fov / 2.0))

        # 3D射线方向
        x = (u - out_w / 2.0) / (out_w * f)
        y = (v - out_h / 2.0) / (out_h * f)
        z = np.ones_like(x)

    # 归一化射线向量
    norm = np.sqrt(x**2 + y**2 + z**2)
    norm = np.where(norm < 1e-10, 1e-10, norm)  # 防止除零
    x, y, z = x / norm, y / norm, z / norm

    # 步骤3: 应用旋转矩阵（相机坐标系 -> 全局坐标系）
    R = euler_to_rotation_matrix(yaw, pitch, roll)

    # 将射线堆叠为 (3, H*W) 矩阵
    rays = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

    # 应用旋转: R @ rays
    rotated_rays = R @ rays

    # 重新整形为 (H, W)
    x_rot = rotated_rays[0].reshape(out_h, out_w)
    y_rot = rotated_rays[1].reshape(out_h, out_w)
    z_rot = rotated_rays[2].reshape(out_h, out_w)

    # 步骤4: 3D射线 -> 球面坐标
    # phi: 方位角（经度）[-π, π]
    # theta: 极角（纬度）[-π/2, π/2]
    theta_sphere = np.arctan2(y_rot, np.sqrt(x_rot**2 + z_rot**2))
    phi_sphere = np.arctan2(x_rot, z_rot)

    # 步骤5: 球面坐标 -> 等距柱状投影像素坐标
    # phi: [-π, π] -> [0, pano_w]
    # theta: [-π/2, π/2] -> [0, pano_h] (正值在上，负值在下)
    map_x = (phi_sphere + np.pi) / (2.0 * np.pi) * pano_w
    map_y = (np.pi / 2.0 + theta_sphere) / np.pi * pano_h

    # 转换为 float32（cv2.remap 要求）
    return map_x.astype(np.float32), map_y.astype(np.float32)


# ==================== 图像处理函数 ====================
class PanoramaAnalyzer:
    """全景图分析器"""

    @staticmethod
    def detect_black_regions(
        pano: np.ndarray,
        check_ratio: float = 0.10,
        threshold: float = 0.20
    ) -> Dict[str, bool]:
        """
        检测全景图顶部和底部的黑色无效区域

        Args:
            pano: 全景图 (H, W, 3)
            check_ratio: 检查区域比例（顶部/底部的百分比）
            threshold: 黑色像素阈值

        Returns:
            包含 'unsafe_upward' 和 'unsafe_downward' 的字典
        """
        h, w = pano.shape[:2]
        check_h = int(h * check_ratio)

        # 检查顶部区域
        top_region = pano[:check_h, :, :]
        top_black_ratio = PanoramaAnalyzer._calculate_black_ratio(top_region)

        # 检查底部区域
        bottom_region = pano[-check_h:, :, :]
        bottom_black_ratio = PanoramaAnalyzer._calculate_black_ratio(bottom_region)

        return {
            'unsafe_upward': top_black_ratio > threshold,
            'unsafe_downward': bottom_black_ratio > threshold,
            'top_black_ratio': top_black_ratio,
            'bottom_black_ratio': bottom_black_ratio
        }

    @staticmethod
    def _calculate_black_ratio(region: np.ndarray) -> float:
        """计算区域中黑色像素的比例"""
        # 定义黑色像素: RGB 值都小于 10
        black_mask = np.all(region < 10, axis=2)
        black_ratio = np.sum(black_mask) / black_mask.size
        return black_ratio


class ViewGenerator:
    """虚拟相机视图生成器"""

    def __init__(self, config: Config):
        self.config = config
        self.rng = np.random.RandomState()

    def set_seed(self, seed: int):
        """设置随机种子（用于多进程）"""
        self.rng = np.random.RandomState(seed)

    def sample_camera_params(
        self,
        safety_flags: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        采样相机参数（支持多种鱼眼模型）

        Args:
            safety_flags: 包含 'unsafe_upward' 和 'unsafe_downward' 的字典

        Returns:
            相机参数字典，包括姿态、FoV、投影类型、鱼眼模型及 KB 系数
        """
        # 决定俯仰角范围
        pitch_type = self.rng.choice(
            ['extreme_up', 'extreme_down', 'normal'],
            p=[
                self.config.pitch_extreme_up_prob,
                self.config.pitch_extreme_down_prob,
                self.config.pitch_normal_prob
            ]
        )

        # 根据安全标志调整采样
        if pitch_type == 'extreme_up' and safety_flags['unsafe_upward']:
            # 回退到正常范围
            pitch_type = 'normal'
        elif pitch_type == 'extreme_down' and safety_flags['unsafe_downward']:
            # 回退到正常范围
            pitch_type = 'normal'

        # 采样俯仰角
        if pitch_type == 'extreme_up':
            pitch = self.rng.uniform(*self.config.pitch_extreme_up_range)
        elif pitch_type == 'extreme_down':
            pitch = self.rng.uniform(*self.config.pitch_extreme_down_range)
        else:
            pitch = self.rng.uniform(*self.config.pitch_normal_range)

        # 采样翻滚角和偏航角
        roll = self.rng.uniform(*self.config.roll_range)
        yaw = self.rng.uniform(*self.config.yaw_range)

        # 采样投影类型和视场角
        is_fisheye = self.rng.random() < self.config.fisheye_probability
        fisheye_model = 'none'  # 默认为非鱼眼
        kb_coeffs = None  # KB 系数初始化为 None

        if is_fisheye:
            fov = self.rng.uniform(*self.config.fov_fisheye_range)

            # 根据权重随机选择鱼眼模型
            fisheye_model = self.rng.choice(
                self.config.fisheye_models,
                p=self.config.fisheye_model_weights
            )

            # 如果是 Kannala-Brandt 模型，采样畸变系数
            if fisheye_model == 'kannala_brandt':
                kb_coeffs = {
                    'k2': self.rng.uniform(*self.config.kb_k2_range),
                    'k3': self.rng.uniform(*self.config.kb_k3_range),
                    'k4': self.rng.uniform(*self.config.kb_k4_range)
                }
        else:
            fov = self.rng.uniform(*self.config.fov_rectilinear_range)

        return {
            'pitch': pitch,
            'roll': roll,
            'yaw': yaw,
            'fov': fov,
            'is_fisheye': bool(is_fisheye),
            'fisheye_model': fisheye_model,
            'kb_coeffs': kb_coeffs,
            'pitch_type': pitch_type
        }

    def generate_view(
        self,
        pano: np.ndarray,
        params: Dict[str, Any]
    ) -> np.ndarray:
        """
        从全景图生成透视视图

        Args:
            pano: 全景图 (H, W, 3)
            params: 相机参数（包括姿态、FoV、投影类型、鱼眼模型等）

        Returns:
            透视图像 (output_h, output_w, 3)
        """
        pano_h, pano_w = pano.shape[:2]

        # 创建映射（支持多种鱼眼模型）
        map_x, map_y = create_perspective_map(
            pano_shape=(pano_h, pano_w),
            output_size=self.config.output_image_size,
            yaw=params['yaw'],
            pitch=params['pitch'],
            roll=params['roll'],
            fov=params['fov'],
            is_fisheye=params['is_fisheye'],
            fisheye_model=params.get('fisheye_model', 'equidistant'),
            kb_coeffs=params.get('kb_coeffs')
        )

        # 使用 cv2.remap 进行重映射（高效且高质量）
        view = cv2.remap(
            pano,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        return view


# ==================== 数据处理管线 ====================
def find_panorama_files(input_root: str) -> List[Path]:
    """递归查找所有全景RGB文件"""
    input_path = Path(input_root)
    pano_files = []

    for area_dir in input_path.glob("area_*"):
        pano_rgb_dir = area_dir / "pano" / "rgb"
        if pano_rgb_dir.exists():
            pano_files.extend(pano_rgb_dir.glob("*_rgb.png"))

    return sorted(pano_files)


def process_single_panorama(
    args: Tuple[Path, Config, int]
) -> List[Dict]:
    """
    处理单个全景图（用于多进程）

    Args:
        args: (pano_path, config, pano_index) 元组

    Returns:
        生成的样本元数据列表
    """
    pano_path, config, pano_index = args

    try:
        # 读取全景图
        pano = cv2.imread(str(pano_path))
        if pano is None:
            logger.error(f"无法读取全景图: {pano_path}")
            return []

        # 分析黑色区域
        analyzer = PanoramaAnalyzer()
        safety_flags = analyzer.detect_black_regions(
            pano,
            check_ratio=config.check_zone_ratio,
            threshold=config.black_pixel_threshold
        )

        # 初始化视图生成器
        generator = ViewGenerator(config)
        generator.set_seed(pano_index * 10000)  # 确保可重复性

        # 生成样本
        samples_metadata = []
        for sample_idx in range(config.samples_per_pano):
            # 采样相机参数
            params = generator.sample_camera_params(safety_flags)

            # 生成视图
            view = generator.generate_view(pano, params)

            # 构建输出文件名
            pano_id = pano_path.stem.replace("_rgb", "")
            filename = f"pano_{pano_index:04d}_crop_{sample_idx:02d}.jpg"

            # 保存图像
            output_image_dir = Path(config.output_root) / "images"
            output_image_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_image_dir / filename

            cv2.imwrite(str(output_path), view, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 记录元数据（包括鱼眼模型和 KB 系数）
            metadata = {
                'filename': filename,
                'source_pano': pano_id,
                'pitch': params['pitch'],
                'roll': params['roll'],
                'yaw': params['yaw'],
                'fov': params['fov'],
                'is_fisheye': params['is_fisheye'],
                'fisheye_model': params.get('fisheye_model', 'none'),
                'kb_k2': params['kb_coeffs']['k2'] if params.get('kb_coeffs') else None,
                'kb_k3': params['kb_coeffs']['k3'] if params.get('kb_coeffs') else None,
                'kb_k4': params['kb_coeffs']['k4'] if params.get('kb_coeffs') else None,
                'top_black_ratio': safety_flags['top_black_ratio'],
                'bottom_black_ratio': safety_flags['bottom_black_ratio']
            }
            samples_metadata.append(metadata)

        logger.info(
            f"处理完成: {pano_path.name} - "
            f"生成 {len(samples_metadata)} 个样本 "
            f"(顶部黑色: {safety_flags['top_black_ratio']:.2%}, "
            f"底部黑色: {safety_flags['bottom_black_ratio']:.2%})"
        )

        return samples_metadata

    except Exception as e:
        logger.error(f"处理全景图时出错 {pano_path}: {str(e)}", exc_info=True)
        return []


def main():
    """主函数"""
    # 初始化配置
    config = Config()

    logger.info("=" * 60)
    logger.info("Stanford2D3D 数据处理管线")
    logger.info("=" * 60)
    logger.info(f"输入目录: {config.input_root}")
    logger.info(f"输出目录: {config.output_root}")
    logger.info(f"每个全景图生成样本数: {config.samples_per_pano}")
    logger.info(f"输出图像尺寸: {config.output_image_size}")
    logger.info(f"工作进程数: {config.num_workers}")
    logger.info("=" * 60)

    # 创建输出目录
    output_path = Path(config.output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "images").mkdir(exist_ok=True)

    # 查找所有全景图
    logger.info("正在搜索全景图文件...")
    pano_files = find_panorama_files(config.input_root)
    logger.info(f"找到 {len(pano_files)} 个全景图文件")

    if len(pano_files) == 0:
        logger.warning("未找到任何全景图文件！请检查输入目录结构。")
        return

    # 准备多进程参数
    process_args = [(pano_file, config, idx) for idx, pano_file in enumerate(pano_files)]

    # 多进程处理
    all_metadata = []
    logger.info(f"开始处理，使用 {config.num_workers} 个进程...")

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        # 提交所有任务
        futures = {
            executor.submit(process_single_panorama, args): args[0]
            for args in process_args
        }

        # 使用 tqdm 显示进度
        with tqdm(total=len(futures), desc="处理全景图") as pbar:
            for future in as_completed(futures):
                try:
                    samples = future.result()
                    all_metadata.extend(samples)
                except Exception as e:
                    pano_path = futures[future]
                    logger.error(f"处理失败: {pano_path} - {str(e)}")
                finally:
                    pbar.update(1)

    # 保存 CSV
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        csv_path = output_path / "labels.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"元数据已保存到: {csv_path}")
        logger.info(f"总共生成 {len(all_metadata)} 个训练样本")

        # 统计信息
        logger.info("=" * 60)
        logger.info("统计信息:")
        logger.info(f"  - 鱼眼样本: {df['is_fisheye'].sum()} ({df['is_fisheye'].sum()/len(df)*100:.1f}%)")
        logger.info(f"  - 针孔样本: {(~df['is_fisheye']).sum()} ({(~df['is_fisheye']).sum()/len(df)*100:.1f}%)")

        # 鱼眼模型分布
        if df['is_fisheye'].sum() > 0:
            logger.info("  - 鱼眼模型分布:")
            fisheye_df = df[df['is_fisheye']]
            model_counts = fisheye_df['fisheye_model'].value_counts()
            for model, count in model_counts.items():
                percentage = count / len(fisheye_df) * 100
                logger.info(f"      • {model}: {count} ({percentage:.1f}%)")

        logger.info(f"  - 俯仰角范围: [{df['pitch'].min():.1f}°, {df['pitch'].max():.1f}°]")
        logger.info(f"  - 视场角范围: [{df['fov'].min():.1f}°, {df['fov'].max():.1f}°]")
        logger.info("=" * 60)
    else:
        logger.warning("未生成任何样本数据！")

    logger.info("处理完成！")


if __name__ == "__main__":
    main()

