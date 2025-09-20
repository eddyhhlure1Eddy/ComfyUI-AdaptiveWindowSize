# ComfyUI-AdaptiveWindowSize

🎯 **智能增强节点套件** - 解决WanVideo帧对齐问题 + 精确面部检测裁剪

## 概述

ComfyUI-AdaptiveWindowSize 是一个功能强大的自定义节点扩展，提供两大核心功能：
1. **自适应窗口大小** - 解决WanVideo视频生成中的帧对齐问题
2. **智能面部检测裁剪** - 基于多算法的精确面部检测和裁剪

## 主要功能

### 🔧 自适应窗口算法
- **智能窗口调整**: 自动计算最优窗口大小，最小化浪费帧
- **三种模式选择**:
  - `disabled`: 使用原始固定窗口大小
  - `adaptive`: 自适应调整，当剩余帧小于窗口一半时优化
  - `optimal_fit`: 最优匹配，在±25%范围内寻找最佳窗口大小

### 📊 性能优化
- **零浪费帧**: 消除视频末尾的无效帧
- **精确对齐**: 视频长度精确匹配输入帧数
- **内存优化**: 保持原有的内存管理机制

### 🎯 智能面部检测裁剪
- **多算法融合**: 支持OpenCV Haar、DNN、MediaPipe等算法
- **自动优选**: Hybrid Auto模式自动选择最佳算法组合
- **精确定位**: 结合mask和面部检测的智能裁剪
- **参数可调**: 置信度、优先级、扩展系数等灵活配置
- **NMS去重**: 自动过滤重复检测结果
- **实时反馈**: 详细的检测信息输出

## 安装方法

### 前提条件
确保已安装以下依赖：
- ComfyUI
- ComfyUI-WanVideoWrapper

### 安装步骤

1. **克隆到ComfyUI自定义节点目录**:
```bash
cd ComfyUI/custom_nodes/
git clone <repository-url> ComfyUI-AdaptiveWindowSize
```

2. **重启ComfyUI**:
重启ComfyUI以加载新节点

3. **验证安装**:
在节点列表中查找 `AdaptiveWanVideo` 分类

## 使用方法

### 节点参数说明

#### 必需输入
- **vae**: WanVideo VAE模型
- **width/height**: 视频分辨率 (默认: 832x480)
- **num_frames**: 目标帧数 (默认: 81)
- **frame_window_size**: 基础窗口大小 (默认: 77)
- **adaptive_window_mode**: 自适应模式选择
  - `disabled`: 禁用自适应，使用固定窗口
  - `adaptive`: 智能调整模式
  - `optimal_fit`: 最优匹配模式 (推荐)

#### 可选输入
- **clip_embeds**: CLIP视觉编码
- **ref_images**: 参考图像
- **pose_images**: 姿态图像
- **face_images**: 面部图像
- **bg_images**: 背景图像
- **mask**: 遮罩
- **colormatch**: 颜色匹配方法
- **pose_strength/face_strength**: 强度控制

### 工作流示例

1. **替换原始节点**:
   将工作流中的 `WanVideoAnimateEmbeds` 节点替换为 `AdaptiveWanVideoAnimateEmbeds`

2. **配置自适应模式**:
   - 推荐使用 `optimal_fit` 模式获得最佳效果
   - 对于简单场景可使用 `adaptive` 模式

3. **观察日志输出**:
   节点会输出窗口大小调整信息和浪费帧减少情况

### Smart Image Crop 使用示例

1. **替换原始裁剪节点**:
   将工作流中的 `ImageCropByMaskAndResize` 节点替换为 `SmartImageCropByMaskAndResize`

2. **配置面部检测**:
   ```
   face_detection_mode: hybrid_auto (推荐)
   face_confidence: 0.7
   face_priority: 0.8 (80%依赖面部检测，20%依赖mask)
   face_expansion: 1.3 (面部区域扩展30%)
   ```

3. **算法选择指南**:
   - `disabled`: 禁用面部检测，仅使用mask
   - `opencv_haar`: 快速检测，适合实时处理
   - `opencv_dnn`: 平衡精度和速度，需要模型文件
   - `mediapipe`: 高精度检测，需要安装MediaPipe
   - `hybrid_auto`: 最佳效果，自动选择可用算法

4. **查看检测结果**:
   节点输出包含检测详情：`Method: hybrid_auto, Faces: 2, Algorithms: ['haar', 'mediapipe']`

## 技术原理

### 算法详解

#### Adaptive模式
```python
remainder = total_frames % base_window_size
if remainder > 0 and remainder < base_window_size * 0.5:
    # 调整窗口大小而不是填充帧
    optimal_size = total_frames // num_windows
    optimal_size = ((optimal_size - 1) // 4) * 4 + 1  # 4对齐
```

#### Optimal_fit模式
```python
for size in range(min_size, max_size + 1):
    aligned_size = ((size - 1) // 4) * 4 + 1
    waste = total_frames % aligned_size
    if waste == 0:
        return aligned_size  # 找到完美匹配
```

### 对齐约束
- 保持 `(n-1)//4*4+1` 的帧数对齐模式
- 确保窗口大小不小于基础窗口的50%
- 搜索范围限制在基础窗口的75%-125%

### 面部检测算法原理

#### 算法融合策略
```python
# Hybrid Auto模式工作流程
1. 并行运行多种检测算法
2. 收集所有检测结果
3. 应用NMS去除重复检测
4. 根据置信度排序
5. 与原始mask智能融合
```

#### 智能mask融合
```python
combined_mask = (1 - face_priority) * original_mask + face_priority * face_mask
```

## 性能对比

### 自适应窗口效果
| 模式 | 81帧场景浪费帧 | 100帧场景浪费帧 | 适用场景 |
|------|---------------|----------------|----------|
| disabled | 4帧 | 23帧 | 兼容性优先 |
| adaptive | 0帧 | 0帧 | 平衡模式 |
| optimal_fit | 0帧 | 0帧 | 最优效果 |

### 面部检测性能
| 算法 | 速度 | 精度 | 内存 | 依赖 |
|------|------|------|------|------|
| opencv_haar | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 无 |
| opencv_dnn | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | 模型文件 |
| mediapipe | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | pip install |
| hybrid_auto | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | 自动选择 |

## 故障排除

### 常见问题

1. **节点未显示**:
   - 检查ComfyUI-WanVideoWrapper是否正确安装
   - 确认Python路径配置正确
   - 重启ComfyUI

2. **导入错误**:
   - 检查依赖项是否完整
   - 查看控制台错误信息
   - 确认文件权限

3. **性能问题**:
   - 使用 `optimal_fit` 模式可能略微增加计算时间
   - 对于大型视频，建议使用 `adaptive` 模式

4. **面部检测不准确**:
   - 降低 `face_confidence` 阈值 (尝试0.5-0.6)
   - 使用 `hybrid_auto` 模式获得最佳结果
   - 调整 `face_priority` 平衡mask和检测结果
   - 安装MediaPipe: `pip install mediapipe`

5. **面部检测算法不可用**:
   - MediaPipe错误: 安装mediapipe (`pip install mediapipe`)
   - OpenCV DNN错误: 下载模型文件到 `ComfyUI/models/face_detection/`
   - 使用 `opencv_haar` 作为基础算法 (无需额外依赖)

6. **mask和面部检测冲突**:
   - 调整 `face_priority` 参数 (0.0=仅mask, 1.0=仅面部)
   - 使用 `face_expansion` 控制面部区域大小
   - 检查mask质量和面部检测结果

### 调试信息

启用详细日志输出：
```python
log.info(f"AdaptiveWanAnimate: Window size changed from {original} to {new}")
log.info(f"AdaptiveWanAnimate: Waste frames reduced from {before} to {after}")
```

## 版本历史

### v1.0.0
- 初始发布
- 实现三种自适应窗口模式
- 完整的帧对齐解决方案
- 兼容原有WanVideo工作流

## 贡献指南

欢迎提交Issue和Pull Request来改进这个项目。

### 开发环境设置
1. Fork本项目
2. 创建功能分支
3. 提交更改
4. 创建Pull Request

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 致谢

- ComfyUI团队提供的优秀框架
- WanVideo项目的原始实现
- 社区用户的反馈和建议

## 联系方式

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- ComfyUI Discord社区

---

**注意**: 本节点专为解决WanVideo帧对齐问题设计，确保在使用前已正确安装WanVideoWrapper依赖。