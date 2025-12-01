# 🛠️ Tools Package

高级增强工具包，为ZSXT项目提供额外功能而不影响核心业务逻辑。

## 📂 Folder Structure

```
tools/
├── __init__.py                      # 包初始化
├── image_comparison.py              # 图像对比和分析
├── batch_manager.py                 # 批量处理管理
├── model_manager.py                 # 模型和配置管理
├── preprocessing_toolkit.py         # 高级预处理工具
├── TOOLS_GUIDE.md                   # 完整文档
├── examples.py                      # 使用示例
└── README.md                        # 本文件
```

## 🚀 Quick Start

### 1. 导入工具

```python
from tools import ImageComparator, BatchProcessor, ModelManager
```

### 2. 使用示例

**图像对比**:
```python
from tools import ImageComparator
metrics = ImageComparator.get_metrics_dict(img1, img2)
print(f"SSIM: {metrics['ssim']}, PSNR: {metrics['psnr']}")
```

**批量处理**:
```python
from tools import BatchProcessor
processor = BatchProcessor()
processor.start_batch('my_batch')
# ... 处理图像 ...
summary = processor.end_batch()
```

**模型管理**:
```python
from tools import ModelManager
manager = ModelManager('./checkpoints')
models = manager.list_models()
```

### 3. 查看完整文档

详见 `TOOLS_GUIDE.md`

### 4. 运行示例

```bash
python tools/examples.py
```

## 📦 Modules Overview

| 模块 | 功能 | 主要类 |
|------|------|--------|
| **image_comparison.py** | 图像对比和分析 | ImageComparator, ImageAnalyzer |
| **batch_manager.py** | 批量处理和报告 | BatchProcessor, ProcessingScheduler, ResultsAnalyzer |
| **model_manager.py** | 模型和配置管理 | ModelManager, ConfigManager, PerformanceProfiler |
| **preprocessing_toolkit.py** | 图像预处理 | ImageEnhancer, ImageAugmenter, ImageOptimizer, ColorCorrection, EdgeDetection |

## 🎯 Use Cases

### 场景 1: 质量评估
```python
from tools import ImageComparator, ImageAnalyzer

# 评估输入质量
quality = ImageAnalyzer.detect_image_quality('input.jpg')

# 评估翻译效果
metrics = ImageComparator.get_metrics_dict(original, translated)
```

### 场景 2: 批量处理
```python
from tools import BatchProcessor, ResultsAnalyzer

processor = BatchProcessor('./results')
processor.start_batch('batch_1')

for img_path in image_files:
    # 处理...
    processor.add_result(img_path.name, True, metrics=...)

batch_log = processor.end_batch()
ResultsAnalyzer.generate_html_report(batch_log, 'report.html')
```

### 场景 3: 模型比较
```python
from tools import ModelManager, PerformanceProfiler

manager = ModelManager()
profiler = PerformanceProfiler()

for model in manager.list_models():
    # 测试模型...
    profiler.add_profile(model['name'], 'config', metrics)

best = profiler.get_best_model('ssim')
```

### 场景 4: 图像增强
```python
from tools import ImageEnhancer, ColorCorrection

# 增强质量
enhanced = ImageEnhancer.enhance_contrast(image)
enhanced = ColorCorrection.white_balance(enhanced)
```

## ✨ Features

✅ **图像对比** - 计算SSIM、PSNR、MSE等指标  
✅ **质量评估** - 检测模糊、亮度、质量问题  
✅ **批量处理** - 管理大规模图像处理  
✅ **模型管理** - 管理多个模型版本和性能  
✅ **配置管理** - 创建和管理配置变体  
✅ **图像增强** - 对比度、锐度、去噪等  
✅ **图像优化** - 调整大小、压缩、归一化  
✅ **颜色校正** - 白平衡、直方图均衡化、CLAHE  
✅ **边界检测** - Canny、Sobel、Laplacian  
✅ **HTML报告** - 自动生成批处理报告  

## 📊 API Reference

### ImageComparator

```python
# 计算指标
mse = ImageComparator.calculate_mse(img1, img2)
ssim = ImageComparator.calculate_ssim(img1, img2)
psnr = ImageComparator.calculate_psnr(img1, img2)

# 获取所有指标
metrics = ImageComparator.get_metrics_dict(img1, img2)

# 创建对比图像
comparison = ImageComparator.create_comparison_image(
    original, translated, method='horizontal'
)
```

### ImageAnalyzer

```python
# 获取图像信息
info = ImageAnalyzer.get_image_info('image.jpg')

# 检测质量
quality = ImageAnalyzer.detect_image_quality('image.jpg')
```

### BatchProcessor

```python
processor = BatchProcessor(output_dir='./results')
processor.start_batch('name')
processor.add_result('file.jpg', True, metrics=...)
summary = processor.end_batch()
```

### ModelManager

```python
manager = ModelManager('./checkpoints')
models = manager.list_models()
model = manager.get_model('gen_best')
comparison = manager.compare_models('model1', 'model2')
```

### ImageEnhancer

```python
enhanced = ImageEnhancer.enhance_contrast(image)
sharpened = ImageEnhancer.enhance_sharpness(image)
denoised = ImageEnhancer.denoise(image)
```

更多API详见 `TOOLS_GUIDE.md`。

## 🔧 Configuration

部分工具支持配置参数:

```python
# 对比度强度
enhanced = ImageEnhancer.enhance_contrast(image, clip_limit=3.0)

# 压缩质量
compressed = ImageOptimizer.compress_image(image, quality=90)

# 输出目录
processor = BatchProcessor(output_dir='./custom_output')
```

## 📈 Performance

| 操作 | 速度 | 内存 |
|------|------|------|
| 图像对比 | ~10ms | 低 |
| 批量处理 | 可扩展 | 中等 |
| 图像增强 | ~50ms | 中等 |
| 模型比较 | ~1ms | 低 |

## 🐛 Troubleshooting

**Q: 导入错误**  
A: 确保在tools文件夹的父目录运行代码，或添加路径:
```python
import sys
sys.path.insert(0, '/path/to/_code_EN')
from tools import *
```

**Q: 文件未找到**  
A: 检查路径和文件名，确保文件存在

**Q: 内存溢出**  
A: 使用较小的batch size或压缩图像

## 📝 Examples

详见 `examples.py` 文件，包含6个完整示例:

1. 图像对比
2. 图像分析
3. 模型管理
4. 图像增强
5. 批量处理
6. 图像优化

运行:
```bash
python tools/examples.py
```

## 🚀 Integration

### 与Web UI集成

在 `app.py` 中使用:

```python
from tools import ImageComparator

metrics = ImageComparator.get_metrics_dict(original, translated)
st.metric("SSIM", metrics['ssim'])
```

### 与CLI集成

在 `inference.py` 中使用:

```python
from tools import ImageAnalyzer

quality = ImageAnalyzer.detect_image_quality(input_path)
if quality['blur_status'] != 'Clear':
    print("Warning: Input image quality is poor")
```

### 与训练脚本集成

在 `train.py` 中使用:

```python
from tools import BatchProcessor

processor = BatchProcessor('./training_results')
processor.start_batch('training')
# ... training loop ...
processor.end_batch()
```

## 📚 Documentation

- **TOOLS_GUIDE.md** - 完整功能文档
- **examples.py** - 使用示例代码
- **源代码注释** - 详细的代码文档

## 🤝 Contributing

欢迎贡献新工具或改进现有功能！

## 📄 License

与主项目相同的许可证

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-30  
**Compatibility**: Python 3.8+, OpenCV 4.0+, NumPy 1.19+
