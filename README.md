# ZSXT - Zero-Shot X-Ray Translation Framework

轻量级、高效的X光无监督跨域翻译框架。

## 🎯 核心脚本

| 脚本 | 功能 |
|------|------|
| **train.py** | 模型训练 |
| **inference.py** | 图像推理 |
| **verify_env.py** | 环境检查 |

## 🚀 快速开始

### 1. 环境检查
```bash
python verify_env.py
```

### 2. 模型训练
```bash
python train.py
```

### 3. 推理处理
```bash
python inference.py --input image.jpg --output output.jpg --gpu
```

## 📁 项目结构

```
_code_EN/
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── verify_env.py             # 环境验证
├── config.yaml               # 配置文件
├── requirements.txt          # 依赖包
├── src/                      # 源代码模块
│   ├── model.py              # 模型定义
│   ├── losses.py             # 损失函数
│   ├── preprocess_pipeline.py
│   └── super_resolution.py
├── tools/                    # 增强工具包
│   └── README.md             # 工具文档
├── docs/                     # 文档和指南
│   ├── README.md             # 项目文档
│   ├── QUICKSTART.md         # 快速开始
│   ├── UI_GUIDE.md           # Web UI指南
│   └── ...
├── checkpoints/              # 模型权重
├── datasets/                 # 数据集
└── generated_images/         # 生成结果
```

## 📦 依赖安装

```bash
pip install -r requirements.txt
```

## 🛠️ 高级工具

项目包含 `tools/` 文件夹，提供 80+ 个增强函数：

```python
from tools import ImageComparator, BatchProcessor

# 图像对比
metrics = ImageComparator.get_metrics_dict(img1, img2)

# 批量处理
processor = BatchProcessor('./results')
```

详见 `tools/README.md`

## 🎨 Web UI (可选)

完整的Web UI已移至 `docs/` 文件夹。启动方式：

```bash
cd docs
python ../run_ui.py
```

## 📚 文档

所有文档都在 `docs/` 文件夹中：

- **START_HERE.md** - 快速开始指南
- **QUICKSTART.md** - 详细使用步骤
- **TOOLS_FEATURES.md** - 工具功能说明
- 以及其他详细文档

## ✨ 主要特性

✅ 轻量级模型 (37.7M 参数)  
✅ 零监督跨域翻译  
✅ 超分辨率支持  
✅ GPU 加速  
✅ 批量处理能力  
✅ 完整工具包  
✅ 可视化 Web UI  

## 🔧 配置

编辑 `config.yaml` 调整：

```yaml
batch_size: 3
num_epochs: 50
learning_rate: 0.0002
img_width: 256
img_height: 256
```

## 📊 模型性能

| 指标 | 值 |
|------|-----|
| Generator 参数 | 34.9M |
| Discriminator 参数 | 2.77M |
| 推理速度 (GPU) | 10-50ms/image |
| 推理速度 (CPU) | 100-500ms/image |

## 🐛 故障排除

**问题: CUDA 错误**
```bash
# 使用 CPU
python inference.py --input image.jpg --cpu
```

**问题: 内存不足**
```bash
# 编辑 config.yaml，降低 batch_size
batch_size: 1
```

## 📖 详细文档

查看 `docs/` 文件夹获取完整文档和指南。

---

**Version**: 1.0.0  
**Updated**: 2025-11-30

