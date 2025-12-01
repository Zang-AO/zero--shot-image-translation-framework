# 🛠️ Tools Package Documentation

## Overview

`tools/` 文件夹包含一套高级增强功能，用于扩展ZSXT项目的功能，不会影响核心业务逻辑。

---

## 📦 Modules

### 1. **Image Comparison** (`image_comparison.py`)

提供图像比较和分析功能。

#### ImageComparator 类

```python
from tools import ImageComparator

# 计算图像相似度指标
mse = ImageComparator.calculate_mse(img1, img2)      # Mean Squared Error
ssim = ImageComparator.calculate_ssim(img1, img2)    # 结构相似度
psnr = ImageComparator.calculate_psnr(img1, img2)    # 峰值信噪比

# 获取所有指标
metrics = ImageComparator.get_metrics_dict(img1, img2)
# {'mse': 0.0234, 'ssim': 0.9523, 'psnr': 42.31}

# 创建对比图像
comparison = ImageComparator.create_comparison_image(
    original, translated, 
    method='horizontal'  # 'horizontal', 'vertical', 'overlay'
)
```

#### ImageAnalyzer 类

```python
from tools import ImageAnalyzer

# 获取图像信息
info = ImageAnalyzer.get_image_info('image.jpg')
# {'width': 256, 'height': 256, 'channels': 3, 'file_size_mb': 0.15, ...}

# 检测图像质量
quality = ImageAnalyzer.detect_image_quality('image.jpg')
# {'blur_status': 'Clear', 'brightness_status': 'Normal', ...}
```

**应用场景**:
- 评估翻译质量
- 检测输入图像质量问题
- 生成质量报告

---

### 2. **Batch Manager** (`batch_manager.py`)

管理批量处理和生成报告。

#### BatchProcessor 类

```python
from tools import BatchProcessor

# 创建处理器
processor = BatchProcessor(output_dir='./batch_results')

# 开始新批处理
processor.start_batch('my_batch')

# 获取文件列表
images = processor.get_image_files('./images')

# 添加处理结果
processor.add_result(
    filename='image1.jpg',
    success=True,
    output_path='./output/image1.jpg',
    metrics={'ssim': 0.95, 'psnr': 42.31}
)

# 结束批处理并获取报告
summary = processor.end_batch()
# 自动保存到: batch_results/batch_<timestamp>_log.json
```

#### ProcessingScheduler 类

```python
from tools import ProcessingScheduler

scheduler = ProcessingScheduler()

# 添加到队列
scheduler.add_to_queue('image1.jpg', priority=1)
scheduler.add_to_queue('image2.jpg', priority=2)

# 获取下一个任务
next_item = scheduler.get_next_item()

# 标记完成
scheduler.mark_completed(next_item, result={'processed': True})

# 获取统计
stats = scheduler.get_statistics()
```

#### ResultsAnalyzer 类

```python
from tools import ResultsAnalyzer

# 分析批处理结果
analysis = ResultsAnalyzer.analyze_batch_results(batch_log)

# 生成HTML报告
ResultsAnalyzer.generate_html_report(
    batch_log, 
    'batch_report.html'
)
```

**应用场景**:
- 批量处理成千上万的图像
- 自动生成处理报告
- 追踪处理进度和统计

---

### 3. **Model Manager** (`model_manager.py`)

管理模型和配置文件。

#### ModelManager 类

```python
from tools import ModelManager

# 创建管理器
manager = ModelManager(checkpoint_dir='./checkpoints')

# 扫描可用模型
models = manager.list_models()
# [{'name': 'gen_best', 'path': '...', 'size_mb': 34.5, ...}, ...]

# 按标签过滤
best_models = manager.list_models(tag='best')

# 获取模型信息
model = manager.get_model('gen_best')

# 给模型添加标签
manager.tag_model('gen_best', 'production')

# 比较两个模型
comparison = manager.compare_models('gen_best', 'gen_final')
```

#### ConfigManager 类

```python
from tools import ConfigManager

# 管理配置
config_mgr = ConfigManager(config_dir='.')

# 列出所有配置
configs = config_mgr.list_configs()

# 获取配置数据
config = config_mgr.get_config('config')

# 创建配置变体
config_mgr.create_config_variant(
    base_config='config',
    variant_name='config_fast',
    modifications={'batch_size': 8}
)
```

#### PerformanceProfiler 类

```python
from tools import PerformanceProfiler

profiler = PerformanceProfiler()

# 添加性能数据
profiler.add_profile(
    model_name='gen_best',
    config_name='config',
    metrics={'ssim': 0.95, 'inference_time': 0.03}
)

# 获取最佳模型
best = profiler.get_best_model(metric_name='ssim')

# 获取最快模型
fastest = profiler.get_fastest_model()

# 保存性能配置文件
profiler.save_profiles('profiles.json')
```

**应用场景**:
- 管理多个模型版本
- 比较模型性能
- 自动选择最优模型

---

### 4. **Preprocessing Toolkit** (`preprocessing_toolkit.py`)

高级图像预处理功能。

#### ImageEnhancer 类

```python
from tools import ImageEnhancer

enhancer = ImageEnhancer()

# 去噪
denoised = enhancer.denoise(image, method='bilateral')

# 增强对比度
enhanced = enhancer.enhance_contrast(image, clip_limit=2.0)

# 增强锐度
sharpened = enhancer.enhance_sharpness(image, strength=1.5)

# 调整亮度
brighter = enhancer.adjust_brightness(image, value=30)

# 调整饱和度
saturated = enhancer.adjust_saturation(image, value=1.2)
```

#### ImageAugmenter 类

```python
from tools import ImageAugmenter

augmenter = ImageAugmenter()

# 旋转
rotated = augmenter.rotate(image, angle=15)

# 翻转
flipped = augmenter.flip(image, direction='horizontal')

# 透视变换
transformed = augmenter.perspective_transform(image, scale=0.2)

# 弹性变形
elastic = augmenter.elastic_transform(image, alpha=34, sigma=4)
```

#### ImageOptimizer 类

```python
from tools import ImageOptimizer

optimizer = ImageOptimizer()

# 自动调整大小
resized = optimizer.auto_resize(image, target_size=256)

# 压缩图像
compressed = optimizer.compress_image(image, quality=90)

# 归一化
normalized = optimizer.normalize_image(image)  # [0, 1]

# 标准化
standardized = optimizer.standardize_image(image)  # zero mean, unit variance
```

#### ColorCorrection 类

```python
from tools import ColorCorrection

# 白平衡
corrected = ColorCorrection.white_balance(image)

# 直方图均衡化
equalized = ColorCorrection.histogram_equalization(image)

# 自适应直方图均衡化 (CLAHE)
adaptive = ColorCorrection.adaptive_histogram_equalization(image)
```

#### EdgeDetection 类

```python
from tools import EdgeDetection

# Canny 边界检测
edges_canny = EdgeDetection.canny_edge(image)

# Sobel 边界检测
edges_sobel = EdgeDetection.sobel_edge(image)

# Laplacian 边界检测
edges_laplacian = EdgeDetection.laplacian_edge(image)
```

**应用场景**:
- 提高输入图像质量
- 数据增强和变换
- 边界检测和特征提取

---

## 📚 Complete Examples

### Example 1: 质量评估流程

```python
from tools import ImageComparator, ImageAnalyzer

# 评估输入图像质量
quality = ImageAnalyzer.detect_image_quality('input.jpg')
if quality['blur_status'] != 'Clear':
    print("警告: 输入图像质量不佳")

# 比较处理前后的图像
original = cv2.imread('input.jpg')
translated = cv2.imread('output.jpg')
metrics = ImageComparator.get_metrics_dict(original, translated)

print(f"相似度指标: {metrics}")
```

### Example 2: 批量处理和报告

```python
from tools import BatchProcessor, ResultsAnalyzer

processor = BatchProcessor('./results')
processor.start_batch('medical_images')

images = processor.get_image_files('./input_folder')

for img_path in images:
    # 处理图像...
    processor.add_result(
        filename=img_path.name,
        success=True,
        output_path=f'./results/{img_path.name}',
        metrics={'ssim': 0.92}
    )

batch_log = processor.end_batch()

# 生成HTML报告
ResultsAnalyzer.generate_html_report(batch_log, 'report.html')
```

### Example 3: 模型性能比较

```python
from tools import ModelManager, PerformanceProfiler

manager = ModelManager()
profiler = PerformanceProfiler()

models = manager.list_models()

for model in models:
    # 测试模型...
    profiler.add_profile(
        model['name'],
        'config',
        {'ssim': 0.95, 'inference_time': 0.03}
    )

best = profiler.get_best_model('ssim')
fastest = profiler.get_fastest_model()

print(f"最佳模型: {best['model']} (SSIM: {best['metrics']['ssim']})")
print(f"最快模型: {fastest['model']} (Time: {fastest['metrics']['inference_time']})")
```

### Example 4: 图像预处理

```python
from tools import ImageEnhancer, ImageAugmenter, ColorCorrection

image = cv2.imread('input.jpg')

# 增强质量
enhanced = ImageEnhancer.enhance_contrast(image)
enhanced = ColorCorrection.white_balance(enhanced)

# 数据增强
augmented = ImageAugmenter.rotate(enhanced, angle=10)

# 优化
optimized = ImageOptimizer.auto_resize(augmented, target_size=256)

cv2.imwrite('processed.jpg', optimized)
```

---

## 🚀 Quick Start

1. **导入工具包**:
```python
from tools import *
```

2. **使用特定模块**:
```python
from tools import ImageComparator, BatchProcessor
```

3. **查看模块文档**:
```python
import tools
help(tools.ImageComparator)
```

---

## 📊 Integration with Web UI

Web UI (`app.py`) 可以集成这些工具来增强功能:

```python
# 在 Single Image 标签页中使用
from tools import ImageComparator

if original_image is not None and translated_image is not None:
    metrics = ImageComparator.get_metrics_dict(original_image, translated_image)
    st.metric("SSIM", metrics['ssim'])
    st.metric("PSNR", metrics['psnr'])
```

---

## 🔧 Configuration

大多数工具都提供参数来配置行为:

| 工具 | 配置参数 | 说明 |
|------|--------|------|
| ImageEnhancer | clip_limit, strength | 对比度和锐度强度 |
| ImageOptimizer | quality | 压缩质量 (1-100) |
| BatchProcessor | output_dir | 输出目录 |
| ModelManager | checkpoint_dir | 模型存储目录 |

---

## 📝 Best Practices

1. **错误处理**: 始终检查返回值
```python
model = manager.get_model('nonexistent')
if model is None:
    print("Model not found")
```

2. **内存管理**: 大量处理时考虑释放资源
```python
del large_array  # 显式释放
```

3. **日志记录**: 使用 ResultsAnalyzer 追踪结果
```python
processor.end_batch()  # 自动保存日志
```

4. **验证输入**: 检查图像格式和大小
```python
info = ImageAnalyzer.get_image_info(path)
if info['file_size_mb'] > 100:
    print("Image too large")
```

---

## 📞 Support

- 查看模块源代码获取更多详情
- 检查每个类的 docstring
- 查看 `__init__.py` 了解导入结构

---

**Version**: 1.0.0  
**Last Updated**: 2025-11-30
