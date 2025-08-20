# NLP\_beginner 项目架构说明

本项目架构参考了 **PyTorch 基础练习**，其中：

* 基础练习中的 **task1, task2, task3** 分别对应本项目中的 **task1, task2 和 task\_transformer**。
* 本项目中的 **task3** 对应原 NLP\_beginner 中的任务3
* 当前 `task1` 和 `task2` 已经按照 **PyTorch 基础练习** 中的要求更改了数据集和划分方式。

---

## 📂 项目结构

### **task2/**

```
task2/
│-- src/                   # 存放源代码文件
│-- glove/                 # 存放预训练的 GloVe Embedding
│-- external_resources/    # 存放数据集
```

---

### **task\_transformer/**

```
task_transformer/
│-- transformer_project/               # task_transformer 的具体实现
│   │-- data_preprocess/               # 存放 dataset 类
│   │   ├── dataset_base.py            # 基类
│   │   ├── dataset_add.py             # 派生类 (加法任务)
│   │   └── dataset_translate.py       # 派生类 (翻译任务)
│   │
│   │-- transformer/                   # Transformer 内部实现
│   │   ├── model.py                   # 框架主结构
│   │   ├── layer.py                   # 组成模型的层
│   │   └── sublayer/                  # 具体子层 (MHA, FFN 等)
│   │
│   │-- train_add.py                   # 子任务1 (加法) 的训练与测试框架
│   │-- train_translate.py             # 子任务2 (德语→英语翻译) 的训练与测试框架
│   │-- train_translate_api.py         # 使用 PyTorch Transformer API 的翻译实现
│   │                                  # (原因：自实现的 Transformer 自回归效果较差，因此用 API 做对比)
│
│-- 其他三个文件夹                      # 存放其他 Transformer 实现 (仅供学习参考)
```

---

## 📥 数据下载 (必读)

本项目依赖外部数据资源，请务必在运行代码之前下载并放置到正确目录。  
由于文件体积较大（部分超过数百 MB），**未包含在 GitHub 仓库中**，需要用户手动下载。

### 1. GloVe Embeddings
- 下载地址：[GloVe 官方页面](https://nlp.stanford.edu/projects/glove/)  
- 本项目使用的版本为 **glove.6B**（基于 Wikipedia 2014 + Gigaword 5 语料训练）。  
- 下载的压缩包 `glove.6B.zip` 大小约 **822 MB**，解压后约 **2 GB**，请确保本地磁盘空间足够。  
- 请解压后，将 **`glove.6B.300d.txt`** 放置到以下两个目录：  
  - `task2/glove/`  
  - `task3/external_resources/glove/`  

> ⚠️ 注意：本项目默认使用 `glove.6B.300d` 版本，如需使用其他维度（50d, 100d, 200d），需自行在代码中调整参数。

### 2. SNLI 数据集
- 下载地址：[SNLI 官方页面](https://nlp.stanford.edu/projects/snli/)  
- 下载后，将解压得到的 `snli_1.0` 文件夹放置到 `task3/external_resources/` 下。


---

✅ 下载完成后，项目目录应包含以下结构（示例）：
```
NLP_beginner/
├── task2/
│   └── glove/
│       ├── glove.6B.50d.txt
│       ├── glove.6B.100d.txt
│       ├── glove.6B.200d.txt
│       └── glove.6B.300d.txt
│
├── task3/
│   └── external_resources/
│       ├── snli_1.0/
│       │   ├── snli_1.0_train.jsonl
│       │   ├── snli_1.0_dev.jsonl
│       │   └── snli_1.0_test.jsonl
│       │
│       └── glove/
│           ├── glove.6B.50d.txt
│           ├── glove.6B.100d.txt
│           ├── glove.6B.200d.txt
│           └── glove.6B.300d.txt

```

---

## 📑 其他说明
* **注意事项**：
  本项目尚未完成,仍有需要完善之处,如其中task_transformer中的Transformer的实现代码仍存在一些问题,使得自回归效果较差,需进一步修改.