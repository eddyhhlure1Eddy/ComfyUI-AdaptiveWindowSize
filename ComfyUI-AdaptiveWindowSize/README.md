# ComfyUI-AdaptiveWindowSize

🎯 **自适应窗口大小节点** - 解决WanVideo视频生成中的帧对齐问题

## 概述

ComfyUI-AdaptiveWindowSize 是一个专门为WanVideo设计的自定义节点扩展，通过智能调整窗口大小来解决视频生成过程中的帧对齐问题，消除末尾的浪费帧。

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

## 性能对比

| 模式 | 81帧场景浪费帧 | 100帧场景浪费帧 | 适用场景 |
|------|---------------|----------------|----------|
| disabled | 4帧 | 23帧 | 兼容性优先 |
| adaptive | 0帧 | 0帧 | 平衡模式 |
| optimal_fit | 0帧 | 0帧 | 最优效果 |

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