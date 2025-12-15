# Stanford2D3D 训练数据生成脚本

## 概述

这是一个强大的、多线程的 Python 数据处理管线，用于从 Stanford2D3D 全景图数据集生成用于相机参数回归的训练数据。

## 功能特性

### ✨ 核心功能

1. **智能黑色区域检测**
   - 自动检测全景图顶部和底部的无效黑色区域
   - 根据检测结果自适应调整采样策略
   - 避免生成包含大量无效像素的训练样本

2. **高级投影算法**
   - ✅ 针孔投影（直线透视，FoV 60°-100°）
   - ✅ 鱼眼投影（等距投影模型，FoV 140°-180°）
   - 精确的数学实现，基于旋转矩阵和球面坐标转换

3. **智能参数采样**
   - 俯仰角分层采样（30% 极端向上 + 30% 极端向下 + 40% 水平）
   - 翻滚角、偏航角、视场角的合理分布
   - 投影类型的均衡采样

4. **高性能并行处理**
   - 使用 `ProcessPoolExecutor` 实现多进程并行
   - 自动根据 CPU 核心数优化工作进程
   - 进度条实时显示处理状态

5. **鲁棒性设计**
   - 完善的异常处理机制
   - 详细的日志记录
   - 自动创建输出目录结构

## 安装依赖

```bash
pip install opencv-python numpy pandas tqdm
```

或使用 requirements.txt：

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python generate_training_dataset.py
```

### 自定义配置

编辑脚本中的 `Config` 类来修改参数：

```python
@dataclass
class Config:
    # 输入输出路径
    input_root: str = r"C:\document\Stanford2D3D"
    output_root: str = r"C:\document\Stanford2D3D\output_dataset"
    
    # 每个全景图生成的样本数
    samples_per_pano: int = 50
    
    # 输出图像尺寸
    output_image_size: Tuple[int, int] = (320, 320)
    
    # 工作进程数（默认：CPU核心数-1）
    num_workers: int = max(1, os.cpu_count() - 1)
```

## 输出结构

```
output_dataset/
├── images/
│   ├── pano_0000_crop_00.jpg
│   ├── pano_0000_crop_01.jpg
│   ├── ...
│   └── pano_XXXX_crop_49.jpg
└── labels.csv
```

### labels.csv 格式

| 列名 | 描述 | 示例 |
|------|------|------|
| filename | 图像文件名 | pano_0000_crop_00.jpg |
| source_pano | 源全景图ID | camera_1a2b3c |
| pitch | 俯仰角（度） | 75.3 |
| roll | 翻滚角（度） | -5.2 |
| yaw | 偏航角（度） | 180.5 |
| fov | 视场角（度） | 90.0 |
| is_fisheye | 是否鱼眼投影 | True/False |
| top_black_ratio | 顶部黑色像素比例 | 0.15 |
| bottom_black_ratio | 底部黑色像素比例 | 0.25 |

## 算法原理

### 投影转换流程

```
输出像素 (u, v)
    ↓
归一化坐标
    ↓
投影类型判断 → [针孔投影] 或 [鱼眼投影]
    ↓
相机坐标系 3D 射线 (x, y, z)
    ↓
应用旋转矩阵 R (Yaw, Pitch, Roll)
    ↓
全局坐标系 3D 射线 (x', y', z')
    ↓
球面坐标 (φ, θ)
    ↓
等距柱状投影像素 (px, py)
    ↓
cv2.remap 插值采样
    ↓
输出图像
```

### 数学公式

#### 欧拉角 → 旋转矩阵

```
R = R_roll × R_pitch × R_yaw
```

#### 针孔投影

```
焦距 f = 0.5 / tan(FoV/2)
射线: (u_norm, v_norm, f)
```

#### 鱼眼投影（等距模型）

```
r = sqrt(u_norm² + v_norm²)
θ = r × (FoV/2)  # 入射角
射线: (sin(θ)cos(φ), sin(θ)sin(φ), cos(θ))
```

## 性能优化建议

1. **内存充足时**：增加 `samples_per_pano` 到 100+
2. **CPU 核心多时**：保持默认 `num_workers` 设置
3. **硬盘 I/O 慢时**：减少 `num_workers` 避免磁盘瓶颈
4. **调试时**：设置 `num_workers = 1` 便于查看错误信息

## 常见问题

### Q: 脚本运行很慢？
A: 检查以下几点：
- 确认使用了多进程（`num_workers > 1`）
- 全景图分辨率是否过高（建议 4096x2048）
- 硬盘写入速度（考虑使用 SSD）

### Q: 生成的图像有黑边？
A: 这是正常现象，表示相机视角超出了全景图的有效范围。可以通过调整采样范围来减少。

### Q: 如何验证投影是否正确？
A: 可以可视化几个样本，检查：
- 直线在针孔投影中应保持直线
- 直线在鱼眼投影中应有适当弯曲
- 俯仰角为 0 时，地平线应在图像中心

## 扩展功能建议

1. **添加深度图生成**：同时处理 `pano/depth/` 文件夹
2. **数据增强**：添加亮度、对比度调整
3. **语义分割**：集成 `pano/semantic/` 数据
4. **质量过滤**：跳过低质量全景图（过度模糊或曝光）

## 技术栈

- **Python 3.7+**
- **OpenCV**: 图像处理和重映射
- **NumPy**: 数值计算和矩阵运算
- **Pandas**: 数据管理和 CSV 导出
- **tqdm**: 进度条显示
- **concurrent.futures**: 多进程并行

## 许可证

本脚本遵循 MIT 许可证。Stanford2D3D 数据集请遵循其原始许可协议。

## 联系方式

如有问题或建议，欢迎提出 Issue 或 Pull Request。

---

**祝您训练顺利！** 🚀

