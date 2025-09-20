# 📦 ComfyUI-AdaptiveWindowSize 安装配置指南

## 🚀 快速安装

### 步骤1: 下载节点
```bash
cd ComfyUI/custom_nodes/
git clone <repository-url> ComfyUI-AdaptiveWindowSize
```

### 步骤2: 重启ComfyUI
重启ComfyUI以加载新节点

### 步骤3: 验证安装
在节点列表中查找：
- `AdaptiveWanVideo` 分类下的 "Adaptive WanVideo Animate Embeds"
- `SmartCrop/image` 分类下的 "Smart Image Crop By Mask And Resize"

## 🔧 可选依赖安装

### MediaPipe (推荐 - 提升面部检测精度)
```bash
pip install mediapipe
```

### OpenCV DNN模型 (可选 - 高精度检测)
```bash
# 创建模型目录
mkdir -p ComfyUI/models/face_detection
cd ComfyUI/models/face_detection

# 下载模型文件
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
```

## ✅ 安装验证

### 自动测试
```bash
cd ComfyUI/custom_nodes/ComfyUI-AdaptiveWindowSize
python test_face_detection.py
```

### 手动验证
1. **启动ComfyUI**
2. **查找节点**: 搜索 "Adaptive" 或 "Smart"
3. **创建测试工作流**:
   - 加载图像
   - 添加 "Smart Image Crop By Mask And Resize" 节点
   - 设置 `face_detection_mode` 为 `hybrid_auto`
   - 运行测试

## 📋 功能对照表

| 功能 | 基础安装 | +MediaPipe | +OpenCV DNN |
|------|----------|------------|-------------|
| 自适应窗口大小 | ✅ | ✅ | ✅ |
| Haar级联面部检测 | ✅ | ✅ | ✅ |
| MediaPipe面部检测 | ❌ | ✅ | ✅ |
| DNN面部检测 | ❌ | ❌ | ✅ |
| Hybrid Auto模式 | 部分 | ✅ | ✅ |

## 🎯 推荐配置

### 最小安装 (即插即用)
```
只需基础安装，使用opencv_haar算法
适合: 快速测试、资源受限环境
```

### 推荐安装 (最佳效果)
```bash
pip install mediapipe
```
```
使用hybrid_auto模式获得最佳检测效果
适合: 生产环境、高质量要求
```

### 完整安装 (所有功能)
```bash
pip install mediapipe
# + 下载OpenCV DNN模型文件
```
```
所有算法可用，最大灵活性
适合: 开发测试、专业用户
```

## 🔍 故障排除

### 节点未显示
```bash
# 检查ComfyUI控制台输出
# 确认没有导入错误
# 重启ComfyUI
```

### MediaPipe安装失败
```bash
# 使用conda环境
conda install -c conda-forge mediapipe

# 或使用清华源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple mediapipe
```

### 依赖冲突
```bash
# 检查Python版本 (建议3.8+)
python --version

# 更新pip
pip install --upgrade pip

# 重新安装依赖
pip install --force-reinstall opencv-python
```

## 📞 获取帮助

1. **查看日志**: ComfyUI控制台输出
2. **运行测试**: `python test_face_detection.py`
3. **阅读文档**: [FACE_DETECTION_GUIDE.md](FACE_DETECTION_GUIDE.md)
4. **提交Issue**: GitHub Issues

---

🎉 **安装完成后，享受智能化的视频生成和面部检测体验！**