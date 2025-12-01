# 📚 文档和工具导航

这个文件夹包含ZSXT项目的所有文档、Web UI和启动脚本。

## 📖 文档文件

| 文件 | 说明 | 读者 |
|------|------|------|
| **START_HERE.md** | 🚀 快速开始指南 | 新用户必读 |
| **README.md** | 📘 完整项目文档 | 所有用户 |
| **QUICKSTART.md** | ⚡ 详细使用步骤 | 开发者 |
| **UI_GUIDE.md** | 🎨 Web UI 使用指南 | UI用户 |
| **TOOLS_FEATURES.md** | 🛠️ 工具包功能说明 | 高级用户 |
| **PROJECT_OVERVIEW.md** | 📋 项目技术详解 | 研究人员 |
| **ENVIRONMENT_SETUP.md** | ⚙️ 环境配置指南 | 系统管理员 |
| **COMPLETION_REPORT.md** | ✅ 项目完成报告 | 项目管理者 |
| **QUICK_REFERENCE.md** | 📑 快速参考卡 | 快速查询 |

## 🎨 Web UI

### 启动方式

**方式1: Python启动器**
```bash
cd ..
python run_ui.py
```

**方式2: 直接Streamlit**
```bash
cd ..
streamlit run docs/app.py
```

**方式3: 批处理脚本**
```bash
# Windows
docs/start_ui.bat

# Linux/Mac
bash docs/start_ui.sh
```

### Web UI 功能

- 📸 **单张图像处理** - 上传、预览、翻译、下载
- 📁 **批量处理** - 文件夹批处理、进度追踪、批量下载
- ⚙️ **实时配置** - CPU/GPU切换、模型选择、参数调整
- ℹ️ **系统信息** - GPU状态、内存使用、性能指标

详见 **UI_GUIDE.md**

## 🛠️ 工具包

项目的 `tools/` 文件夹包含 80+ 个增强函数。详见 **tools/README.md** 和 **tools/TOOLS_GUIDE.md**

### 快速示例

```python
from tools import ImageComparator, BatchProcessor

# 图像对比
metrics = ImageComparator.get_metrics_dict(img1, img2)

# 批量处理
processor = BatchProcessor('./results')
processor.start_batch('batch_1')
```

## 📚 阅读建议

### 🟢 初级用户

1. START_HERE.md (5 min)
2. QUICKSTART.md (10 min)
3. UI_GUIDE.md (15 min)

**结果**: 能够运行Web UI和基本推理

### 🟡 中级用户

1. README.md (10 min)
2. TOOLS_FEATURES.md (15 min)
3. tools/TOOLS_GUIDE.md (20 min)

**结果**: 能够使用所有功能和工具

### 🔴 高级用户

1. PROJECT_OVERVIEW.md (30 min)
2. 源代码阅读 (60 min)
3. 工具扩展 (可选)

**结果**: 深入理解架构，可扩展功能

## 🔍 快速查询

| 问题 | 查看文件 |
|------|--------|
| 如何快速开始? | START_HERE.md |
| 如何使用Web UI? | UI_GUIDE.md |
| 训练参数是什么? | PROJECT_OVERVIEW.md |
| 工具包有哪些? | TOOLS_FEATURES.md |
| 环境配置问题? | ENVIRONMENT_SETUP.md |
| 命令行使用? | QUICKSTART.md |
| 具体函数用法? | tools/TOOLS_GUIDE.md |

## 📊 文件统计

- **文档数**: 9 个
- **Web UI文件**: 4 个
- **总大小**: ~117 KB
- **完整覆盖**: ✅

## 🚀 快速命令

```bash
# 环境检查
python verify_env.py

# 启动Web UI
python run_ui.py

# 查看文档
cat START_HERE.md

# 运行示例
python tools/examples.py

# 模型训练
python train.py

# 图像推理
python inference.py --input image.jpg --output output.jpg
```

## 💡 常见问题

**Q: 文档太多，从哪里开始?**  
A: 从 `START_HERE.md` 开始，5分钟快速入门。

**Q: 如何启动Web UI?**  
A: 运行 `python run_ui.py`，或查看本文件的Web UI部分。

**Q: 工具包怎么用?**  
A: 查看 `TOOLS_FEATURES.md` 和 `tools/TOOLS_GUIDE.md`。

**Q: 有API文档吗?**  
A: 有的，详见 `tools/TOOLS_GUIDE.md`

**Q: 如何扩展功能?**  
A: 查看源代码或工具包的README。

## 📞 支持

- 📖 阅读文档获取帮助
- 🔍 查看 QUICK_REFERENCE.md 快速查询
- 💻 查看 PROJECT_OVERVIEW.md 了解技术细节
- 🛠️ 查看 tools/README.md 使用工具包

---

**Version**: 1.0.0  
**Updated**: 2025-11-30

**提示**: 所有文件都在 `docs/` 文件夹中，保持主目录的清洁。核心脚本保留在主目录的上级 `_code_EN/`。
